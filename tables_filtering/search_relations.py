import os
import re
from typing import Dict, List, Literal

import polars as pl

from blend import BLEND


def search_single_joins(
    tables_path: str,
    searcher: BLEND | None = None,
    db_path: str | None = None,
    table_ids: List[str] | None = None,
    format: Literal["csv", "parquet"] = "csv",
    join_on_columns: Dict[str, List[int]] | None = None,
    k: int = 3,
    **blend_kwargs,
) -> List[Dict]:
    """
    Run a single-column-join search on the given tables.

    :param tables_path: directory where table (CSV/parquet) files are stored.
    :param searcher: an instance of BLEND searcher. If None, db_path is required to create a new one.
    :param db_path: a path where to store the BLEND database, required if searcher parameter is None.
    :param table_ids: a subset of table IDs (without any type extension) to use for the search. If None, all tables are used.
    :param format: format of tables. Default is CSV.
    :param join_on_columns: a dictionary where, for each table,
                            is provided a list of columns on which the search is performed.
                            If None, the search is done over every column of every tables.
    :param k: number of results to return for each search.
    :param blend_kwargs: keyword arguments passed to blend.create_index.
    :return: a list of dictionaries, where each reports the pair of tablees and columns and the relative overlap.
    """

    # take table IDs
    table_ids = list(filter(
        lambda _id: True if table_ids is None else _id in table_ids,
        map(lambda _id: re.sub(r"(.csv|.parquet)$", "", _id), os.listdir(tables_path)),
    ))

    if not searcher:
        searcher = BLEND(db_path)
        searcher.create_index(data_path=tables_path, **blend_kwargs)

    results = []

    for table_id in table_ids:
        n_rows = blend_kwargs.get("limit_table_rows", None)

        # read the current query table
        match format:
            case "csv":
                df = pl.read_csv(
                    os.path.join(tables_path, table_id + ".csv"), n_rows=n_rows
                )
            case "parquet":
                df = pl.read_parquet(
                    os.path.join(tables_path, table_id + ".parquet"), r_rows=n_rows
                )

        # for each column, perform a signle-join-search
        for column in df.columns:
            # if this column is not in the subset of
            # selected columns for this table, continue
            if (
                join_on_columns
                and table_id in join_on_columns
                and column not in join_on_columns[table_id]
            ):
                continue

            query_values = df.get_column(column).to_list()

            # perform the search with BLEND
            res = searcher.single_column_join_search(query_values, k)

            for res_table_id, res_column_id, overlap in res:
                match format:
                    case "csv":
                        r_df = pl.read_csv(
                            os.path.join(tables_path, res_table_id + ".csv"), n_rows=0
                        )
                    case "parquet":
                        r_df = pl.read_parquet(
                            os.path.join(tables_path, res_table_id + ".parquet"),
                            r_rows=0,
                        )

                results.append(
                    {
                        "left_table": table_id,
                        "left_column": column,
                        "right_table": res_table_id,
                        "right_column": r_df.columns[res_column_id],
                        "overlap": overlap,
                    }
                )

    return results


def main():
    pass


if __name__ == "__main__":
    main()
