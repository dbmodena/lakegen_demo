import os
from collections import defaultdict
from heapq import heappop, heappush
from typing import Any
import logging

import numpy as np
from tqdm import tqdm

import polars as pl

from ...db import DBHandler
from ...utils import calculate_xash
from .seeker_base import Seeker

TQDM_NCOLS = 120
TQDM_RIGHT_PAD = 31


class MultiColumnOverlap(Seeker):
    def __init__(
        self,
        table: pl.DataFrame,  # pd.DataFrame,
        k: int = 10,
        xash_size: int = 128,
        verbose: bool = False,
    ) -> None:
        super().__init__(k)
        self.table = table  # input_df.copy().astype(str)
        self.xash_size = xash_size
        self.verbose = verbose

        # This is the base SQL query, which needs to be extended
        # to include the inner joins with all the user required
        # columns for the multi-column search
        self.base_sql = """
            SELECT firstcolumn.table_id, firstcolumn.row_id, firstcolumn.super_key, firstcolumn.cell_value,
                    firstcolumn.column_id $OTHER_SELECT_COLUMNS$
            FROM (
                SELECT table_id, row_id, cell_value, column_id, TO_BITSTRING(super_key) AS super_key
                FROM all_tables
                WHERE cell_value IN ($TOKENS$) $ADDITIONALS$
                ) AS firstcolumn $INNERJOINS$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        logger = logging.getLogger(f"blend_logger_{os.getpid()}")

        sql = self.base_sql.replace("$TOPK$", f"{self.k}")

        # The first column is treated outside the loop below
        # self.input[self.input.columns.values[0]]
        firstcolumn_values = self.table.to_series(0).cast(pl.String)
        sql = sql.replace(
            "$TOKENS$",
            db.create_sql_list_str(db.clean_value_collection(firstcolumn_values)),
        )

        # For each column (except the first one) add an inner join to
        # the base SQL query, including a join on its specific values
        innerjoins = ""
        # for column_index in range(1, len(self.input.columns.values)):
        for column_index in range(1, len(self.table.columns)):
            # [self.input.columns.values[column_index]]
            column_values = self.table.to_series(column_index).cast(pl.String)
            column_name = db.random_subquery_name()

            value_collections = db.clean_value_collection(column_values)
            sql_list_str = db.create_sql_list_str(value_collections)

            innerjoins += f"""
                INNER JOIN
                    (
                        SELECT table_id, row_id, cell_value, column_id FROM all_tables
                        WHERE cell_value IN ({sql_list_str})
                            $ADDITIONALS$
                    ) AS col_{column_name}
                ON firstcolumn.table_id = col_{column_name}.table_iD AND firstcolumn.row_id = col_{column_name}.row_id
            """

            # other_select_columns = f' , clm_{column_name}.cell_value, clm_{column_name}.column_id $OTHER_SELECT_COLUMNS$ '
            # sql = sql.replace('$OTHER_SELECT_COLUMNS$', other_select_columns)

        sql = (
            sql.replace("$OTHER_SELECT_COLUMNS$", "")
            .replace("$INNERJOINS$", innerjoins)
            .replace("$ADDITIONALS$", additionals)
        )

        if self.verbose:
            logger.debug("Fetching candidates...")

        candidates = db.execute_and_fetchall(sql)

        if self.verbose:
            logger.debug(f"#candidate rows = {len(candidates)}")

        # Run the MATE specific filters to prune irrelevant results
        results = self.run_filter(
            posting_lists=candidates,
            db=db,
            xash_size=self.xash_size,
            verbose=self.verbose,
        )

        if self.verbose:
            logger.debug(f"#filtered tables = {len(results)}")

        if len(results) == 0:
            return "SELECT * FROM all_tables WHERE 1 = 0;"

        # split the join keys field and map the keys to integers
        results = [
            (table_id, list(map(int, join_keys.split("_"))), joinability_score)
            for table_id, join_keys, joinability_score in results[: self.k]
        ]

        # union the results from all the subqueries
        unioned_queries = " UNION ALL ".join(
            [
                f"(SELECT '{table_id}' AS table_id, {join_keys} AS join_keys, {joinability_score} AS joinability_score)"
                for table_id, join_keys, joinability_score in results[: self.k]
            ]
        )

        return f"""
            SELECT table_id, join_keys, joinability_score
            FROM ({unioned_queries}) AS ResultsSelection
        """

    def cost(self) -> int:
        return 10

    def ml_cost(self, db: DBHandler) -> float:
        # return self._predict_runtime([list(col) for col in self.input.values.T], db)
        raise NotImplementedError("Still to port to Polars")

    def run_filter(
        self,
        posting_lists: list,
        db: DBHandler,
        xash_size: int = 128,
        chunk_size: int = 100_000,
        verbose: bool = False,
    ) -> list[tuple[int, str, float]]:
        logger = logging.getLogger(f"blend_logger_{os.getpid()}")

        # - Preprocessing
        posting_lists_dict = defaultdict(list)
        posting_lists_cand_struct = {}

        for tablerow_superkey in tqdm(
            posting_lists,
            desc="Preprocessing posting lists".ljust(TQDM_RIGHT_PAD, " "),
            disable=not verbose,
            ncols=TQDM_NCOLS,
        ):
            table = tablerow_superkey[0]
            row = tablerow_superkey[1]
            superkey = tablerow_superkey[2]
            token = tablerow_superkey[3]
            colid = tablerow_superkey[4]
            tokens = [
                tablerow_superkey[x] for x in np.arange(5, len(tablerow_superkey), 2)
            ]
            cols = [
                tablerow_superkey[x] for x in np.arange(6, len(tablerow_superkey), 2)
            ]
            posting_lists_dict[table].append((row, superkey, token, colid))
            posting_lists_cand_struct[(table, row)] = [tokens, cols]

        top_joinable_tables = []  # each item includes: Tableid, joinable_rows

        query_columns = self.table.columns

        # Calculate superkey for all input rows
        df = self.table.with_columns(
            self.table.map_rows(
                lambda row: str(self.hash_row_vals(row, xash_size)), pl.String
            )
            .get_column("map")
            .alias("super_key")
        )

        # Get all rows grouped by first token of each row
        g = df.group_by(df.columns[0])
        gd = defaultdict(list)
        for (key,), data in g:
            # gd[str(key[0])] = g.get_group((key[0],)).values
            gd[str(key)] = data.rows()

        candidate_external_row_ids = []
        candidate_external_col_ids = []
        candidate_input_rows = []
        candidate_table_rows = []
        candidate_table_ids = []
        all_pls = 0
        total_approved = 0
        total_match = 0
        overlaps_dict = {}
        super_key_index = list(df.columns).index("super_key")
        # super_key_index = list(input_cpy.columns.values).index("super_key")
        checked_tables = 0
        max_table_check = 10000000

        for tableid in tqdm(
            sorted(
                posting_lists_dict,
                key=lambda k: len(posting_lists_dict[k]),
                reverse=True,
            )[:max_table_check],
            desc="Checking candidate tables".ljust(TQDM_RIGHT_PAD, " "),
            disable=not verbose,
            ncols=TQDM_NCOLS,
        ):
            checked_tables += 1
            if checked_tables == max_table_check:
                # pruned = True
                break
            set_of_rowids = set()
            hitting_PLs = posting_lists_dict[tableid]
            if len(top_joinable_tables) >= self.k and top_joinable_tables[0][0] >= len(
                hitting_PLs
            ):
                # pruned = True
                break
            already_checked_hits = 0

            for hit in tqdm(
                sorted(hitting_PLs),
                desc="Checking Hits".ljust(TQDM_RIGHT_PAD, " "),
                disable=not verbose,
                leave=False,
                ncols=TQDM_NCOLS,
            ):
                if len(top_joinable_tables) >= self.k and (
                    (len(hitting_PLs) - already_checked_hits + len(set_of_rowids))
                    < top_joinable_tables[0][0]
                ):
                    break

                rowid = hit[0]
                superkey = int.from_bytes(hit[1], byteorder="big")
                token = hit[2]
                colid = hit[3]
                relevant_input_rows = gd[token]
                all_pls += len(relevant_input_rows)
                already_checked_hits += 1

                for input_row in relevant_input_rows:
                    if (int(input_row[super_key_index]) | superkey) == superkey:
                        candidate_external_row_ids.append(rowid)
                        set_of_rowids.add(rowid)
                        candidate_external_col_ids.append(colid)
                        candidate_input_rows.append(input_row)
                        candidate_table_ids.append(tableid)
                        candidate_table_rows.append((tableid, rowid))

        if len(candidate_external_row_ids) == 0 and verbose:
            logger.debug("No candidate external row IDs found.")

        if len(candidate_external_row_ids) > 0:
            if verbose:
                logger.debug(
                    f"#Candidate external row IDs: {len(candidate_external_row_ids)}"
                )

            # We get a list of posting lists to evaluate as candidate matches, fetched
            # from the combination of the given table_id and row_id
            candidate_t_ids = [str(t[0]) for t in candidate_table_rows]
            candidate_r_ids = [int(t[1]) for t in candidate_table_rows]

            # contains rowid that each rowid has dict that maps colids to tokenized
            table_row_dict = defaultdict(dict)
            total_fetched_tuples = 0

            for i in tqdm(
                range(0, len(candidate_table_rows) + chunk_size, chunk_size),
                ncols=TQDM_NCOLS,
                desc="Fetching candidate tuples".ljust(TQDM_RIGHT_PAD, " "),
            ):
                query = """
                SELECT 
                    t.table_id, 
                    t.row_id, 
                    t.column_id, 
                    t.cell_value
                FROM all_tables t
                INNER JOIN (
                    SELECT UNNEST(?) AS table_id, UNNEST(?) AS row_id
                ) AS c 
                ON t.table_id = c.table_id AND t.row_id = c.row_id;
                """

                params = (
                    candidate_t_ids[i : i + chunk_size],
                    candidate_r_ids[i : i + chunk_size],
                )

                pls_to_evaluate = db.execute_and_fetchyield(query, params)

                if verbose:
                    logger.debug("Evaluating remaining posting lists (fetch-yield)...")

                for table_id, row_id, col_id, cell_value in tqdm(
                    pls_to_evaluate,
                    desc="Fetching candidate tuples".ljust(TQDM_RIGHT_PAD, " "),
                    ncols=TQDM_NCOLS,
                    disable=True,
                ):
                    # here we are sure that (table_id, row_id) tuples are in candidate_table_rows,
                    # since this condition is used in the above SQL query
                    table_row_dict[(table_id, row_id)][col_id] = cell_value
                    total_fetched_tuples += 1

            if verbose:
                logger.debug(f"Fetched {total_fetched_tuples}")

            for i in tqdm(
                range(len(candidate_table_rows)),
                desc="Evaluating candidate table rows".ljust(TQDM_RIGHT_PAD, " "),
                total=len(candidate_table_rows),
                ncols=TQDM_NCOLS,
                disable=not verbose,
            ):
                if candidate_table_rows[i] not in table_row_dict:
                    continue

                col_dict = table_row_dict[candidate_table_rows[i]]
                match, matched_columns = self.evaluate_rows(
                    candidate_input_rows[i], col_dict, query_columns
                )
                total_approved += 1
                if match:
                    total_match += 1
                    complete_matched_columns = "{}{}".format(
                        str(candidate_external_col_ids[i]), matched_columns
                    )
                    if candidate_table_ids[i] not in overlaps_dict:
                        overlaps_dict[candidate_table_ids[i]] = {}

                    if (
                        complete_matched_columns
                        in overlaps_dict[candidate_table_ids[i]]
                    ):
                        overlaps_dict[candidate_table_ids[i]][
                            complete_matched_columns
                        ] += 1
                    else:
                        overlaps_dict[candidate_table_ids[i]][
                            complete_matched_columns
                        ] = 1

            for tbl in set(candidate_table_ids):
                if tbl in overlaps_dict and len(overlaps_dict[tbl]) > 0:
                    join_keys = max(overlaps_dict[tbl], key=overlaps_dict[tbl].get)
                    joinability_score = overlaps_dict[tbl][join_keys]
                    if self.k <= len(top_joinable_tables):
                        if top_joinable_tables[0][0] < joinability_score:
                            _popped_table = heappop(top_joinable_tables)
                            heappush(
                                top_joinable_tables, [joinability_score, tbl, join_keys]
                            )
                    else:
                        heappush(
                            top_joinable_tables, [joinability_score, tbl, join_keys]
                        )

        # both original code of MATE and BLEND do not
        # return also the join keys, but I found more
        # useful to have them as part of the final results
        return [
            (tableid, join_keys, joinability_score)
            for joinability_score, tableid, join_keys in top_joinable_tables[::-1]
        ]

    def hash_row_vals(self, row: list[Any], xash_size: int = 128) -> int:
        hresult = 0
        for q in row:
            hvalue = calculate_xash(str(q), xash_size)
            hresult = hresult | hvalue
        return hresult

    def evaluate_rows(self, input_row, col_dict, query_columns):
        vals = list(col_dict.values())
        query_cols_arr = np.array(query_columns)
        query_degree = len(query_cols_arr)
        matching_column_order = ""
        for q in query_cols_arr[-(query_degree - 1) :]:
            q_index = list(query_columns).index(q)
            if input_row[q_index] not in vals:
                return False, ""
            else:
                for colid, val in col_dict.items():
                    if val == input_row[q_index]:
                        matching_column_order += "_{}".format(str(colid))
        return True, matching_column_order
