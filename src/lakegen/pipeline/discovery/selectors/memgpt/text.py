"""The word matching the MemGPT selector shares: the question, its requirements, the catalogue."""

from __future__ import annotations

import re
import unicodedata

WORD = re.compile(r"[a-zà-ÿ][a-zà-ÿ'’-]{2,}")
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})

# Words that say nothing about which table holds the data, in the portals' languages.
NOISE = frozenset(
    # English
    "a an the of for in on at to and or by with from as is are was were be been that this these "
    "those what which who whom how many much per its their our your not no than then into over "
    "under between there here did does do total average count counts correlation proportion "
    "percentage share rank distinct most largest smallest highest lowest top each combined since "
    "increase decrease change difference number place places placed held list show give "
    "january february march april may june july august september october november december "
    # French / Italian / Spanish portals
    "des les une dans pour par avec sur est sont quel quelle quels quelles combien chaque selon "
    "della delle degli dello nella nelle nel sono quanti quante quale quali come cada ogni "
    "para los las una por con del que cuál cuáles cuántos cuántas cómo más menos entre sobre "
    "moyenne nombre totale totali medio media numero qué quién quiénes dónde cuándo cuánto cuánta "
    "où quand pourquoi lequel laquelle lesquels chi cosa dove quando quanto quanta perché ogni "
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre "
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre "
    "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre "
    "registró registrado alto bajo mayor menor nivel total promedio".split()
)
# Words every catalogue entry or schema may carry, whatever the table is about.
GENERIC = frozenset(
    "data dataset datasets table tables number numbers value values type types name names list record records "
    "information info total count average mean sum share proportion each per which what where when how many much "
    "dati dato tabella numero valore valori tipo nome elenco données donnée tableau nombre valeur type nom liste "
    "datos dato tabla número valor valores tipo nombre lista conjunto".split()
)


def mentions(text: str, word: str) -> bool:
    """A word occurs in ``text`` as a word start; long words keep only their stem."""
    stem = word[: max(4, len(word) - 3)] if len(word) >= 6 else word.rstrip("s")
    return bool(re.search(rf"(?<![a-zà-ÿ]){re.escape(stem)}", text))


def question_words(question: str) -> list[str]:
    """The question's words, once each, without the ones that say nothing about a table."""
    return [w for w in dict.fromkeys(WORD.findall(question.casefold())) if w not in NOISE]


def content_words(text: str) -> list[str]:
    """The words of a requirement or question that could appear in a catalogue entry or a schema."""
    body = text.split(":", 1)[1] if ":" in text[:16] else text
    words = [w.strip("'’-") for w in WORD.findall(body.casefold())]
    return [w for w in dict.fromkeys(words) if w and w not in NOISE and w not in GENERIC]


def fold(text: str) -> str:
    """Lowercase without accents, for matching across spellings (Cádiz, cadiz); every dash becomes '-'."""
    text = (
        unicodedata.normalize("NFKD", str(text).translate(DASHES))
        .encode("ascii", "ignore")
        .decode()
    )
    return text.casefold()


# ---------------------------------------------------------------- the portal's own place

LOCATION = frozenset(
    "location localisation localizacion ubicacion ubicazione localita city ciudad ciutat citta ville municipio "
    "municipi municipality comune commune town".split()
)
"""Words that say a requirement is about a place, whatever the place."""
_PLACE_STOP = frozenset(
    "de di del della dello la le el il lo los las les the of in en a al at du des d l e y and".split()
)
_OPEN_DATA = re.compile(r"^(.*?)\bopen\s*data\b")


def portal_places(portal: str) -> frozenset[str]:
    """The place an open-data portal is about, from its title.

    "Valencia Open Data portal" -> valencia, "New York City Open Data portal"
    -> new, york, city, nyc, "City of Chicago Open Data portal" -> chicago.
    A title without "Open Data" (a benchmark lake) names no place.

    Args:
        portal: The portal's title.

    Returns:
        The place's words, and their initials when there are several.
    """
    match = _OPEN_DATA.match(fold(portal).strip())
    if not match:
        return frozenset()
    phrase = re.sub(r"^\s*city of\s+", "", match.group(1))
    words = [w for w in re.findall(r"[a-z]+", phrase) if w not in _PLACE_STOP]
    places = set(words)
    if len(words) > 1:
        places.add("".join(w[0] for w in words))
    return frozenset(places)


def only_place(text: str, places: frozenset[str]) -> bool:
    """Whether the text names the portal's place and nothing else (location words aside)."""
    tokens = {t for t in re.findall(r"[a-z]+", fold(text)) if t not in _PLACE_STOP}
    return bool(tokens & (places - LOCATION)) and tokens <= places | LOCATION
