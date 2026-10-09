"""The search tools of the unified agent: keyword, value and entity searches, their bans and budget, and the split searches."""

import itertools
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from llama_index.core.llms import ChatMessage

from lakegen.config.experiment import KeywordSearchStrategy
from lakegen.discovery.candidates import format_candidate_context
from lakegen.discovery.terms import keyword_terms, minimal_failing_subsets, split_keywords
from lakegen.discovery.tools import pool
from lakegen.lakes.catalogue_text import clean_catalogue_text
from lakegen.retrieval import (
    EmbeddingGenerationError,
    RetrievalMode,
)
from lakegen.retrieval.factory import get_table_retrieval_service

_SEARCH_STOPWORDS = set(
    "a an the of for in on at to and or by with from as is are was were be "
    "been that this these those what which who how many much per its their "
    "our your not no than then into over under between".split()
)
# Quantifier/aggregation words: they describe how to compute the answer, not
# what the dataset is, so they never occur in a table's title, description or
# publisher and only ever narrow or kill an AND query.
_AGGREGATION_STOPWORDS = set(
    "total average count counts correlation proportion percentage share rank "
    "distinct most largest smallest highest lowest top each combined since "
    "increase decrease change difference number".split()
)

_DECOMPOSE_SYSTEM_PROMPT = """Some questions need SEVERAL tables from an open-data portal. Portals publish the same dataset many times: one file per year, month or snapshot, per edition, per agency or per area, all with almost identical descriptions. A question that compares, combines or spans such files needs each of them, and the files differ only in their distinguishing detail (a date, a year, an edition, an agency, a place) - or the question combines genuinely different datasets.
List the distinct tables the question needs. For each, write one standalone search request in plain words: the dataset's subject words plus THAT table's own distinguishing detail (its date, agency, edition or place). Write a SEPARATE entry for each distinct date or edition the question mentions, even when several appear in the same clause (e.g. "June 2022 and July 2023" needs two entries, one per month) -- never fold two into one request. Name the organisation exactly as the question names it; do not expand an acronym or guess a fuller official name yourself.
Never write the same request twice. If one table is enough, return a single request.
Return JSON: {"tables": [{"detail": "...", "request": "..."}]} with 1 to 4 entries."""

_CANDIDATE_PICK_SYSTEM_PROMPT = """You help a keyword search find the right table(s) in an open-data portal. Every candidate search below is an AND search (a table must contain every word) that the portal already ran; you see how many tables it matches and the first few matches. Pick the searches most likely to surface the table(s) THE QUESTION IS ABOUT: their top matches must name the question's subject, organisation, place and period. Prefer searches whose matches are on-topic over ones that are merely small. Return JSON: {"picks": [numbers]} with 1 to 3 numbers."""

# keyword_search="cascade" (DiscoveryConfig): the searches run one at a time in
# the order written and stop at the first that finds a table, so the ranking is
# the whole strategy.
_KEYWORD_CASCADE_SYSTEM_PROMPT = """You write keyword searches that find one table in an open-data portal's catalogue. Every search is a strict AND search: a table matches only if its catalogue entry (title, description, publisher, tags) contains EVERY word. The searches run one at a time in your order. The first one that finds any table ends the search. One that finds nothing is banned, and so is every later search containing the same failing words, so a word the catalogue does not use costs the whole search it is in.
Write up to {size} searches for the table the REQUEST describes, ranked by importance:
- Search 1 is the one most likely to find exactly that table: the dataset's subject words plus every distinguishing detail the request gives -- its place, region, organisation, period or edition. Keep place and region words exactly as written, short forms included (e.g. "NI", "NYC", "UK"): they are what separates this table from the same dataset published for other areas.
- Calibrate each search for AND: 2 to 5 words, each one a word this table's catalogue entry would itself use. Leave out words that say how the answer is computed (highest, total, average, count, trend) and the question's own name for a value the table merely stores.
- Each later search relaxes the one before in a single step: drop its least important word, or write one word as a catalogue might (a year short, "19" for 2019; a month without its year). Drop generic words and the question's own paraphrase of the subject first. Keep the distinguishing place, region and period longest: a search without them stops on whichever sibling table (same dataset, other area or period) it finds first. The last search is the dataset's core subject alone.
- One word per item, in the portal's language. Prefer the request's own words, then the suggested words. Never expand an acronym or guess an official name. The QUESTION is context only: never add a detail that belongs to a different table of it.
- Never write the same set of words twice.
Return JSON: {{"searches": [["word", ...], ...]}} with 1 to {size} searches, most important first."""


def _extract_json_object(text: str) -> dict:
    """Parse one JSON object out of an LLM response, tolerating code fences and
    leading prose by scanning for the first balanced ``{...}``."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        loaded = None
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                loaded = candidate
                break
        if loaded is None:
            raise
    if not isinstance(loaded, dict):
        raise ValueError("response must be a JSON object")
    return loaded


class SearchMixin:
    """The search methods of ``Phase12ToolsManager``; the manager supplies the attributes they read."""

    def _search_cache_key(self, keywords: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            keyword.casefold() for keyword in keywords if keyword.strip()
        ))

    def _failed_keyword_subset(self, keywords: list[str]) -> frozenset[str] | None:
        """Return the smallest known failed subset contained in ``keywords``."""
        proposed = keyword_terms(keywords)
        matches = [failed for failed in self.state.failed_keyword_combinations
                   if failed <= proposed]
        return min(matches, key=lambda item: (len(item), sorted(item)), default=None)

    def _banned_table_keyword_subset(self, keywords: list[str]) -> frozenset[str] | None:
        """Return the smallest known banned-tables-only subset in ``keywords``.
        AND only narrows, so a superset can match nothing but those tables."""
        proposed = keyword_terms(keywords)
        matches = [banned for banned in self.state.banned_table_keyword_combinations
                   if banned <= proposed]
        return min(matches, key=lambda item: (len(item), sorted(item)), default=None)

    def _ban(self, terms: frozenset[str], attr: str = "failed_keyword_combinations") -> None:
        """Add one failed AND term set to the ``attr`` banlist of state and
        prune its now-redundant supersets."""
        known_sets: list[frozenset[str]] = getattr(self.state, attr)
        if not terms or any(known <= terms for known in known_sets):
            return
        known_sets = [known for known in known_sets if not terms < known]
        known_sets.append(terms)
        known_sets.sort(key=lambda item: (len(item), sorted(item)))
        setattr(self.state, attr, known_sets)

    def _shrink_and_ban(
        self, keywords: list[str], count_fn: Callable[[list[str]], int] | None
    ) -> str | None:
        """Ban a genuinely zero-lexical-hit AND query as its smallest failing
        word subsets when ``count_fn`` can tell them apart, else ban the whole
        query. Returns what was learned, or ``None`` to signal the caller's
        plain fallback wording (no probe was possible or none was conclusive).
        """
        terms = keyword_terms(keywords)
        if count_fn is None or len(terms) < 2:
            self._ban(terms)
            return None
        try:
            failing, alone = minimal_failing_subsets(terms, count_fn)
        except Exception:
            self._ban(terms)
            return None
        alone_text = ", ".join(f"{word} {count}" for word, count in sorted(alone.items()))
        if failing:
            for subset in failing:
                self._ban(subset)
            sets_text = ", ".join("{" + ", ".join(sorted(subset)) + "}" for subset in failing)
            text = f"Smallest word sets that already match nothing: {sets_text}."
            if alone_text:
                text += f" Tables containing each word alone: {alone_text}."
            return text
        self._ban(terms)
        return f"Each word matches tables alone ({alone_text})." if alone_text else None

    def _search_tool_description(self) -> str:
        if self.retrieval_config.mode.value_keywords:
            # Cell-value retrieval matches row contents only, so the agent must be
            # asked for values the rows store; a dataset topic finds little.
            return (
                "Search for tables whose cells contain the supplied values. Pass "
                "`values` as a list of up to 8 values likely to appear verbatim in "
                "the cells of the needed tables, written the way the data would "
                "store them: category labels, place or entity names, codes, or "
                'years (for example ["Brooklyn", "suspension", "2016-17"]). Each '
                "value is matched whole, so a value with spaces stays one entry. "
                "Only these values are searched: not the question, and not titles "
                "or column names. Exactly one initial search is allowed. Use the "
                "bounded metadata and schema previews to shortlist the strongest "
                "candidates, then verify them with inspect_columns before "
                "selecting tables."
            )
        if self.retrieval_config.mode.verbatim_entities:
            # Entities are matched against table content as written, so the agent
            # names them the way the question does rather than describing a topic.
            return (
                "Search for relevant tables using the retrieval strategy configured "
                "by the experiment. Pass `entities` as a list of the specific, "
                "named, canonical entities the question mentions explicitly that "
                "would typically appear verbatim in a database: identifiers, "
                "symbols, codes, or names, each written exactly as it appears in "
                "the question with its original casing and punctuation, and listed "
                'once (for example ["East River", "P.S. 123"]). Do not pass general '
                "concepts or categories; pass an empty list when the question names "
                "none. The tool applies the original question and the configured "
                "retrieval parameters automatically. Exactly one initial search is "
                "allowed. Use the bounded metadata and schema previews to shortlist "
                "the strongest candidates, then verify them with inspect_columns "
                "before selecting tables."
            )
        return (
            "Search for relevant tables using the retrieval strategy configured by "
            "the experiment. Pass `concepts` as a list of 2-4 single-word dataset "
            "concepts in the portal's native language; a multi-word name is "
            "simply its words. The tool applies the original question "
            "automatically. Exactly one initial search is allowed. Shortlist from "
            "the returned previews, then verify with inspect_columns."
        )

    def search_tables(self, concepts_str: str = "") -> str:
        """Search for relevant tables using one or two dataset concepts."""
        return self._search(
            [item.strip() for item in concepts_str.split(" ") if item.strip()]
        )

    def search_keyword_concepts(self, concepts: list[str]) -> str:
        """Search with one or two complete concepts using strict AND.

        Concepts are split into the individual words Solr ANDs together
        (matching WordDelimiterGraphFilter) and deduplicated before the search
        runs, so the banlist keys on what Solr actually matched regardless of
        how the agent grouped or punctuated its concepts.

        With an ``llm``, the configured ``keyword_search`` strategy decides what
        actually runs; the agent's concepts still gate the cache and the
        banlist as a plain search's would.
        """
        listed = concepts or []
        strategy = self.discovery.keyword_search
        mode = self.retrieval_config.mode
        if self.llm is not None:
            if strategy == KeywordSearchStrategy.CASCADE and mode == RetrievalMode.KEYWORD:
                return self._split_search(
                    listed, lambda requests: self._run_keyword_cascades(requests, listed)
                )
            if (
                strategy == KeywordSearchStrategy.DECOMPOSE_PREVIEW
                and mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)
            ):
                return self._split_search(listed, self._run_preview_picks)
        return self._search(split_keywords(listed))

    def search_table_values(self, values: list[str]) -> str:
        """Search for tables whose cells contain the listed values."""
        # Each value stays whole: "East River" is one search value, not two.
        return self._search([
            " ".join(str(value).split())
            for value in values or []
            if str(value).strip()
        ])

    def search_table_entities(self, entities: list[str]) -> str:
        """Search for tables using the entities the question names verbatim."""
        # Each entity stays whole. An empty list is still a search: the question
        # is ranked as usual, with no content search behind it.
        listed = [
            " ".join(str(entity).split())
            for entity in entities or []
            if str(entity).strip()
        ]
        return self._search(listed, entities=listed)

    def _search_limit_reached(self) -> bool:
        outcomes = self.state.search_call_outcomes()
        lexical_retry_limit = (
            self.discovery.max_search_attempts
            + self.discovery.max_zero_result_retries
        )
        kw_hybrid_limit_reached = (
            sum(outcomes) >= self.discovery.max_search_attempts
            or len(outcomes) >= lexical_retry_limit
        )
        if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            return kw_hybrid_limit_reached
        return len(outcomes) >= self.discovery.max_search_attempts

    def _retriever(self):
        """The retrieval service of this run, reporting to the run's observer when it has one."""
        table_dir = [self.csv_dir] if self.retrieval_config.mode.requires_table_dir else []
        if self.retrieval_observer is None:
            return get_table_retrieval_service(self.solr_client, self.retrieval_config, *table_dir)
        return get_table_retrieval_service(
            self.solr_client, self.retrieval_config, *table_dir, observer=self.retrieval_observer,
        )

    def _learn_from_lexical_miss(
        self, retriever, hits: list, keywords: list[str], current: list[str], banned: list[str],
    ) -> tuple[bool, object]:
        """Ban what a keyword or hybrid search proved dead; whether its lexical branch was empty, and what was learned.

        A strict AND lexical branch can fail to match anything while the
        overall search still returns candidates in HYBRID mode, rescued by
        the semantic branch: that miss must still be shrunk and banned, or
        a repeat of the same doomed words looks new every time.
        """
        if self.retrieval_config.mode not in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            return False, None
        lexical_empty = (
            not hits if self.retrieval_config.mode == RetrievalMode.KEYWORD
            else getattr(retriever, "last_lexical_hit_count", None) == 0
        )
        if lexical_empty:
            return True, self._shrink_and_ban(keywords, getattr(retriever, "lexical_match_count", None))
        if not current and banned:
            # Everything local it matched is banned for this question: ban
            # the whole set, but only for this question.
            self._ban(keyword_terms(keywords), "banned_table_keyword_combinations")
        elif not current:
            # Solr matched something real for this AND query, just nothing
            # available locally: ban the whole query as attempted; there is
            # nothing lexically zero to shrink.
            self._ban(keyword_terms(keywords))
        return False, None

    def _run_one_retrieval(
        self, keywords: list[str], entities: list[str] | None = None
    ) -> dict:
        """Run one retrieval call, map it onto local files, fold it into the
        accumulated candidate pool (reciprocal-rank fusion with whatever is
        already there), and ban a zero-lexical-hit AND query as its smallest
        failing word subsets. Appends one entry to state.search_attempts.

        Does not check whether a search may run at all (cache, inspection
        lock, attempt limit) -- the caller decides that once, up front, since
        one caller (the split keyword searches) may need this to run several
        times for what the agent experiences as a single search.
        """
        retriever = self._retriever()
        # Retrieved candidates must first be mapped and de-duplicated against
        # local files.  Request a wider ranked list here and apply the
        # workflow's final top_k only after that mapping below. Widen by
        # the current ban count too (mirrors DIVIDED's _solr_and_search),
        # so a banned hit shrinks the fetch instead of the final visible
        # pool -- otherwise every accumulated ban would silently starve
        # top_k with nothing backfilling it.
        fetch_k = max(self.discovery.fetch_floor, self.retrieval_config.top_k) + len(self.state.excluded_tables)
        retrieval_started = time.monotonic()
        hits = retriever.retrieve(
            question=self.question, keywords=keywords, top_k=fetch_k, lexical_fetch_k=fetch_k,
            q_op="AND", entities=entities,
        )
        mode = self.retrieval_config.mode
        search_mode = "AND" if mode == RetrievalMode.KEYWORD else mode.value
        searched = "the question only" if mode.ranks_question_only else f"concepts {list(keywords)}"
        self._emit_notice(
            f"\n> \U0001f50e **Retrieval:** `{mode.value}` "
            f"\u00b7 {searched} "
            f"\u00b7 {len(hits)} ranked hits in "
            f"{time.monotonic() - retrieval_started:.1f}s\n"
        )
        self.state.keyword_history.append(keywords)
        attempt = len(self.state.keyword_history)
        current_candidates, banned_hits = pool.fold_hits(
            self.state, hits, keywords=keywords, attempt=attempt, all_files=self.all_files,
            top_k=self.retrieval_config.top_k,
        )
        candidates = pool.rank(self.state, self.retrieval_config.top_k)
        visible_candidates = pool.show(self.state, self.discovery.initial_candidates)
        self.state.search_attempts.append({
            "attempt": attempt,
            "call": self.state.search_calls,
            "keywords": list(keywords),
            "search_mode": search_mode,
            "current_candidates": list(current_candidates),
            "accumulated_candidates": list(candidates),
        })
        lexical_empty, learned = self._learn_from_lexical_miss(
            retriever, hits, keywords, current_candidates, banned_hits,
        )
        return {
            "attempt": attempt,
            "keywords": keywords,
            "searched": searched,
            "search_mode": search_mode,
            "current_candidates": current_candidates,
            "banned_hits": banned_hits,
            "candidates": candidates,
            "visible_candidates": visible_candidates,
            "lexical_empty": lexical_empty,
            "learned": learned,
        }

    def _format_search_response(self, result: dict) -> str:
        attempt, searched = result["attempt"], result["searched"]
        current_candidates, candidates = result["current_candidates"], result["candidates"]
        visible_candidates = result["visible_candidates"]
        lexical_empty, learned = result["lexical_empty"], result["learned"]

        if not current_candidates and result["banned_hits"] and not lexical_empty:
            self._record_retrieval_memory_event(
                outcome="banned_tables_only",
                terms=list(result["keywords"]),
                tables=list(result["banned_hits"]),
            )
            return (
                f"Attempt: {attempt}\nSearched: {searched}\n"
                + self._banned_only_note([result])
            )

        if not current_candidates:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            self._record_retrieval_memory_event(
                outcome="zero_results",
                terms=list(result["keywords"]),
            )
            return (
                f"Attempt: {attempt}\nSearched: {searched}\n"
                "No tables found. "
                + (f"{learned} " if learned else "")
                + "This AND keyword combination was added to "
                "the zero-result banlist. Search again with a genuinely "
                "different query.\n"
                + (
                    f"Zero-result banlist: {failed}"
                    if self.retrieval_config.mode in (
                        RetrievalMode.KEYWORD,
                        RetrievalMode.HYBRID,
                    )
                    else ""
                )
            )

        prefix = ""
        if lexical_empty:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            prefix = (
                "No table contains all of these words."
                + (f" {learned}" if learned else "")
                + "\n"
                f"Zero-result banlist: {failed}\n"
                "The candidates below do not match every word (found through "
                "the configured semantic branch instead):\n\n"
            )

        return (
            f"Attempt: {attempt}\nSearched: {searched}\n\n"
            + prefix
            + "Candidates in retrieval order after local-file mapping:\n"
            + format_candidate_context(visible_candidates, self.state.solr_meta)
            + (
                f"\n{len(candidates) - len(visible_candidates)} additional "
                "ranked candidates are available through expand_candidates."
                if len(candidates) > len(visible_candidates)
                else ""
            )
        )

    def _banned_only_note(self, results: list[dict]) -> str:
        """Tell the agent its search(es) found only tables banned for this
        question, and to search for what those tables lack instead."""
        banned_hits = list(dict.fromkeys(
            table for result in results for table in result["banned_hits"]
        ))
        word_sets = list(dict.fromkeys(
            "{" + ", ".join(sorted(keyword_terms(result["keywords"]))) + "}"
            for result in results
            if result["banned_hits"] and not result["lexical_empty"]
        ))
        return (
            "No usable tables: every table found is already banned for this "
            f"question ({', '.join(banned_hits)}). This counts as an empty "
            f"result and {', '.join(word_sets)} "
            f"{'is' if len(word_sets) == 1 else 'are'} now banned for this "
            "question. Search again with different keywords, aimed at the "
            "data those banned tables lack.\n"
        )

    def _search_refusal(self, key: tuple[str, ...], keywords: list[str]) -> str | None:
        """Why a search keyed ``key`` over ``keywords`` must not run, or None."""
        if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            failed_subset = self._failed_keyword_subset(keywords)
            if failed_subset is not None:
                blocked = ", ".join(sorted(failed_subset))
                return (
                    "Search rejected before retrieval: this AND query contains "
                    f"the known zero-result keyword subset {{{blocked}}}. "
                    "Formulate a different query that does not contain that subset."
                )
            banned_subset = self._banned_table_keyword_subset(keywords)
            if banned_subset is not None:
                blocked = ", ".join(sorted(banned_subset))
                return (
                    "Search rejected before retrieval: this AND query contains "
                    f"{{{blocked}}}, which found only tables already banned for "
                    "this question. Search with different keywords, aimed at "
                    "the data those tables lack."
                )
        if key in self.state.search_cache:
            if self.discovery.compact_history:
                # The result is still in context: point to it, don't resend it.
                attempt = self.state.search_cache[key].split("\n", 1)[0]
                return (
                    f"Search skipped: identical concepts were already used ({attempt}, "
                    "shown above). Do not repeat this search."
                )
            return (
                "Search skipped: identical concepts were already used. Do not "
                "repeat this search.\n"
                + self.state.search_cache[key]
            )
        # inspection_counts, not inspection_cache: a carried-forward
        # table's cache entry is seeded onto inspection_cache before this
        # round's first search, so checking inspection_cache here would
        # block search_tables outright the moment anything was carried
        # over. inspection_counts only grows from an inspect_columns call
        # actually made this round, so it still blocks a genuine
        # search-after-judging within the round.
        if self.state.inspection_counts and not self.discovery.search_after_inspection:
            return (
                "Search refinement blocked: a candidate has already been "
                "inspected. Use the existing evidence or one guided expansion."
            )
        if self._search_limit_reached():
            return (
                f"Search limit reached ({self.discovery.max_search_attempts} "
                "attempt(s)). Do not call search_tables again; inspect, expand "
                "if needed, then select."
            )
        return None

    def _search(
        self, supplied_concepts: list[str], entities: list[str] | None = None
    ) -> str:
        """Run the configured retrieval once for the supplied search terms."""
        try:
            keywords = list(supplied_concepts)
            if (
                self.retrieval_config.mode.ranks_question_only
                or self.retrieval_config.mode.verbatim_entities
            ):
                keywords = []
            elif not keywords:
                if self.retrieval_config.mode.value_keywords:
                    return (
                        "No values provided. Search with a list of values likely to "
                        "appear in the cells of the needed tables."
                    )
                return "No concepts provided. Search with one or two dataset concepts."
            if (
                self.retrieval_config.mode.uses_embeddings
                and self.state.semantic_failure is not None
            ):
                return (
                    "Configured retrieval skipped: representation generation already "
                    "failed for this request. " + self.state.semantic_failure
                )
            # Keyed on what the agent supplied, as in every mode. Question-only
            # modes drop the concepts before retrieval, and keying on that empty
            # list told the agent that a different search repeated the first.
            key = self._search_cache_key([*supplied_concepts, *(entities or [])])
            refusal = self._search_refusal(key, keywords)
            if refusal is not None:
                return refusal
            self.state.used_keywords = supplied_concepts
            self.state.search_calls += 1
            result = self._run_one_retrieval(keywords, entities)
            response = self._format_search_response(result)
            self.state.search_cache[key] = response
            return response
        except EmbeddingGenerationError as exc:
            detail = str(exc)
            cause = exc.__cause__
            if cause is not None:
                detail = f"{detail}: {cause}"
            self.state.semantic_failure = detail
            return (
                "Error generating the configured retrieval representation: "
                f"{detail}. The retrieval request has finished and must not be repeated."
            )
        except Exception as exc:
            return f"Error during table retrieval: {exc}."

    # ------------------------------------------------- split keyword searches

    def _chat_json(self, system: str, user: str, *, stage: str) -> dict:
        """One structured LLM call, with a single syntax-repair retry."""
        response = self.llm.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])
        raw = str(response.message.content or "").strip()
        try:
            return _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            repair = self.llm.chat([
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=(
                    user + f"\n\nYour {stage} response was not one valid JSON "
                    "object. Return ONLY the same intended object as valid "
                    "JSON, no prose, no code fences."
                )),
            ])
            return _extract_json_object(str(repair.message.content or "").strip())

    def _decompose_question(self) -> list[str]:
        """Split the ORIGINAL question into 1-4 standalone per-table search
        requests. Falls back to the question itself on any failure."""
        try:
            obj = self._chat_json(
                _DECOMPOSE_SYSTEM_PROMPT, f"QUESTION: {self.question}", stage="decompose"
            )
        except Exception:
            return [self.question]
        subs: list[str] = []
        for table in obj.get("tables") or []:
            request = str((table or {}).get("request") or "").strip()
            if request and request not in subs:
                subs.append(request)
        return subs[:4] or [self.question]

    def _word_frequency(self, word: str) -> int:
        key = word.casefold()
        if key not in self._word_freq_cache:
            response = self.solr_client.select([word], q_op="AND", rows=0)
            self._word_freq_cache[key] = int(
                response.get("response", {}).get("numFound", 0)
            )
        return self._word_freq_cache[key]

    def _and_count(self, words: Sequence[str]) -> int:
        response = self.solr_client.select(list(words), q_op="AND", rows=0)
        return int(response.get("response", {}).get("numFound", 0))

    def _question_search_words(self, text: str, cap: int = 8) -> list[str]:
        """Content words of `text` that occur somewhere in the index, rarest first."""
        seen, out = set(), []
        for word in re.findall(r"[\w'\u2019-]+", text):
            key = word.casefold().strip("'\u2019-")
            if (
                key
                and key not in _SEARCH_STOPWORDS
                and key not in _AGGREGATION_STOPWORDS
                and key not in seen
            ):
                seen.add(key)
                out.append(word.strip("'\u2019-"))
        present = [word for word in out if self._word_frequency(word) > 0]
        present.sort(key=self._word_frequency)
        return present[:cap]

    def _candidate_word_subsets(
        self, words: list[str], *, lo: int = 1, hi: int = 60, cap: int = 8, workers: int = 6
    ) -> list[tuple[list[str], int]]:
        """2-word AND combinations whose match count fits the pool, rarest
        first. Capped to pairs and a small word universe (unlike the lab
        version's 2-3 word / 13-word sweep) since each combination costs one
        live Solr round trip and this runs inline in an interactive tool call.
        """
        combos = list(itertools.combinations(words, 2))
        if not combos:
            return []

        def probe(combo: tuple[str, str]) -> tuple[tuple[str, str], int]:
            return combo, self._and_count(combo)

        out: list[tuple[list[str], int]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for combo, matched in executor.map(probe, combos):
                if lo <= matched <= hi:
                    out.append((list(combo), matched))
        # Same relative order as ranking by summed IDF: for same-size subsets,
        # log(N) per word is a constant, so ranking by ascending summed
        # log(frequency) is equivalent to descending summed IDF, without
        # needing the corpus size N at all.
        out.sort(key=lambda item: sum(
            math.log(max(1, self._word_frequency(word))) for word in item[0]
        ))
        return out[:cap]

    def _preview_candidate_words(self, words: list[str]) -> str:
        response = self.solr_client.select(
            words, q_op="AND", rows=3, fl="title,publisher,description"
        )
        docs = response.get("response", {}).get("docs", [])
        items = []
        for doc in docs:
            description = clean_catalogue_text(doc.get("description"))[:70]
            items.append(f"{(doc.get('publisher') or '')[:22]} | {doc.get('title') or ''} | {description}")
        return " ;; ".join(items)

    def _pick_candidate_word_sets(self, sub_question: str) -> list[list[str]]:
        """1-3 AND-word sets to actually search for one sub-question, chosen
        by the model from a preview of real candidates (match count + top
        matches), not guessed blind."""
        words = self._question_search_words(sub_question)
        if len(words) < 2:
            return [words] if words else []
        subsets = self._candidate_word_subsets(words)
        if not subsets:
            return [words[:3]]
        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                previews = list(executor.map(
                    lambda item: self._preview_candidate_words(item[0]), subsets
                ))
            listing = "\n".join(
                f"{i}. {json.dumps(words_)}  matches {n}: {preview}"
                for i, ((words_, n), preview) in enumerate(zip(subsets, previews), 1)
            )
            obj = self._chat_json(
                _CANDIDATE_PICK_SYSTEM_PROMPT,
                f"QUESTION: {sub_question}\n\nCANDIDATE SEARCHES:\n{listing}",
                stage="candidate pick",
            )
        except Exception:
            return [subsets[0][0]]
        picks = [
            int(pick) for pick in (obj.get("picks") or [])
            if str(pick).isdigit() and 1 <= int(pick) <= len(subsets)
        ][:3]
        chosen = [subsets[pick - 1][0] for pick in picks]
        return chosen or [subsets[0][0]]

    def _format_decompose_response(self, sub_questions: list[str], results: list[dict]) -> str:
        candidates = self.state.all_candidates
        visible_candidates = candidates[: self.state.visible_candidate_count]
        attempts = [result["attempt"] for result in results]
        searched = "; ".join(f"table {i}: {sub}" for i, sub in enumerate(sub_questions, 1))
        any_candidates = any(result["current_candidates"] for result in results)
        all_lexical_empty = all(result["lexical_empty"] for result in results)
        learned_notes = " ".join(
            result["learned"] for result in results if result["learned"]
        )

        if not any_candidates:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            banned_only = [
                result for result in results
                if result["banned_hits"] and not result["lexical_empty"]
            ]
            self._record_retrieval_memory_event(
                outcome="banned_tables_only" if banned_only else "zero_results",
                terms=[word for result in results for word in result["keywords"]],
                tables=[table for result in banned_only for table in result["banned_hits"]],
            )
            header = (
                f"Attempt: {attempts[0]}-{attempts[-1]}\n"
                f"Searched (split by table): {searched}\n"
            )
            if len(banned_only) == len(results):
                return header + self._banned_only_note(banned_only)
            return (
                header
                + (self._banned_only_note(banned_only) if banned_only else "")
                + ("The other split searches found no tables. " if banned_only
                   else "No tables found for any of the split searches. ")
                + (f"{learned_notes} " if learned_notes else "")
                + "These AND combinations were added to the zero-result "
                "banlist. Search again with a genuinely different query.\n"
                + (
                    f"Zero-result banlist: {failed}"
                    if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)
                    else ""
                )
            )

        prefix = ""
        if all_lexical_empty:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            prefix = (
                "No table contains all of the words in any split search."
                + (f" {learned_notes}" if learned_notes else "")
                + "\n"
                f"Zero-result banlist: {failed}\n"
                "The candidates below do not match every word (found through "
                "the configured semantic branch instead):\n\n"
            )

        return (
            f"Attempt: {attempts[0]}-{attempts[-1]}\n"
            f"Searched (split by table): {searched}\n\n"
            + prefix
            + "Candidates in retrieval order after local-file mapping:\n"
            + format_candidate_context(visible_candidates, self.state.solr_meta)
            + (
                f"\n{len(candidates) - len(visible_candidates)} additional "
                "ranked candidates are available through expand_candidates."
                if len(candidates) > len(visible_candidates)
                else ""
            )
        )

    def _split_search(
        self, concepts: list[str], run_requests: Callable[[list[str]], list[dict]]
    ) -> str:
        """Split the ORIGINAL question into its per-table distinguishing-detail
        search requests and let ``run_requests`` run the retrievals for them
        (see ``keyword_search`` in DiscoveryConfig). The agent's own `concepts`
        still gate caching and the zero-result banlist exactly as a plain
        search would, so the tool's contract to the agent is unchanged; only
        what actually gets searched gets smarter.
        """
        try:
            key = self._search_cache_key(concepts)
            refusal = self._search_refusal(key, split_keywords(concepts))
            if refusal is not None:
                return refusal

            self.state.used_keywords = concepts
            self.state.search_calls += 1
            sub_questions = self._decompose_question()
            results = run_requests(sub_questions)

            if not results:
                # Nothing usable was proposed, or every proposal was banned:
                # fall back to the plain single-shot search over the agent's
                # own concepts.
                return self._search(split_keywords(concepts))

            response = self._format_decompose_response(sub_questions, results)
            self.state.search_cache[key] = response
            return response
        except EmbeddingGenerationError as exc:
            detail = str(exc)
            cause = exc.__cause__
            if cause is not None:
                detail = f"{detail}: {cause}"
            self.state.semantic_failure = detail
            return (
                "Error generating the configured retrieval representation: "
                f"{detail}. The retrieval request has finished and must not be repeated."
            )
        except Exception as exc:
            return f"Error during table retrieval: {exc}."

    def _run_preview_picks(self, sub_questions: list[str]) -> list[dict]:
        """decompose_tuned_preview_noacronym, ported from
        experiments/retrieval_lab/arms_retrieval.py: for each request, preview
        a few real AND-word candidates (with match counts and top matches) and
        run every set the model picks."""
        return [
            self._run_one_retrieval(words)
            for sub_question in sub_questions
            for words in self._pick_candidate_word_sets(sub_question)
            if words
        ]

    def _propose_keyword_cascade(self, request: str, concepts: list[str]) -> list[list[str]]:
        """The model's AND word sets for one per-table request, ranked by
        importance and deduplicated by the terms Solr actually ANDs. Empty on
        failure."""
        size = self.discovery.keyword_cascade_size
        try:
            obj = self._chat_json(
                _KEYWORD_CASCADE_SYSTEM_PROMPT.format(size=size),
                f"REQUEST: {request}\nQUESTION: {self.question}\n"
                f"SUGGESTED WORDS: {json.dumps(concepts)}",
                stage="keyword cascade",
            )
        except Exception:
            return []
        cascade: list[list[str]] = []
        seen: set[frozenset[str]] = set()
        for raw in obj.get("searches") or []:
            if not isinstance(raw, list):
                continue
            words = [
                word for word in split_keywords(str(item) for item in raw)
                if word not in _SEARCH_STOPWORDS and word not in _AGGREGATION_STOPWORDS
            ]
            terms = keyword_terms(words)
            if terms and terms not in seen:
                seen.add(terms)
                cascade.append(words)
        return cascade[:size]

    def _run_keyword_cascades(self, sub_questions: list[str], concepts: list[str]) -> list[dict]:
        """For each per-table request, try the model's ranked AND word sets one
        at a time and stop at the first that finds a local table.

        Before each set reaches the portal it is checked against the banlist:
        a set that is banned, or contains a banned subset -- including one an
        earlier step of the same cascade just banned -- is skipped unsearched.
        A set that finds nothing is banned by ``_run_one_retrieval`` (as its
        smallest failing word subsets where those can be told apart). A set
        another table's cascade already ran reuses that outcome, so the pool's
        rank fusion never counts one search twice.
        """
        with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
            cascades = list(executor.map(
                lambda request: self._propose_keyword_cascade(request, concepts),
                sub_questions,
            ))
        results: list[dict] = []
        found_by_terms: dict[frozenset[str], int] = {}
        for table, (request, cascade) in enumerate(zip(sub_questions, cascades), 1):
            steps: list[str] = []
            for step, words in enumerate(cascade, 1):
                terms = keyword_terms(words)
                if terms in found_by_terms:
                    found = found_by_terms[terms]
                    steps.append(f"{step}. {words}: {found} table(s), already searched")
                else:
                    banned = (
                        self._failed_keyword_subset(words)
                        or self._banned_table_keyword_subset(words)
                    )
                    if banned is not None:
                        steps.append(
                            f"{step}. {words}: skipped, contains banned "
                            "{" + ", ".join(sorted(banned)) + "}"
                        )
                        continue
                    result = self._run_one_retrieval(words)
                    results.append(result)
                    found = found_by_terms[terms] = len(result["current_candidates"])
                    steps.append(
                        f"{step}. {words}: {found} table(s)"
                        + (
                            f", only banned tables ({len(result['banned_hits'])})"
                            if not found and result["banned_hits"]
                            else ""
                        )
                    )
                if found:
                    break
            if cascade:
                self._emit_notice(
                    f"\n> \U0001fa9c **Keyword cascade, table {table}:** {request}\n"
                    + "".join(f">    {line}\n" for line in steps)
                )
        return results
