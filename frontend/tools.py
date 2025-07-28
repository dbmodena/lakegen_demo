from itertools import chain
import os
import re
from typing import Dict, List
from typing_extensions import Annotated

import polars as pl

from blend import BLEND
from sloth import sloth

from utils import DOWNLOAD_FOLDER, BLEND_DB


async def search_single_joins(
    # tables_path: Annotated[str | None, "Path to the tables directory. If None, default location is used. Default is None."],
    # db_path: Annotated[str | None, "Path to the BLEND database file. If None, default location is used. Default is None."],
    table_ids: Annotated[List[str] | None, "A subset of table IDs to use for the search. If None, all tables are used. Default is None."],
    k: Annotated[int, "Number of results to return for each search. Default is 3."],
    n_rows: Annotated[int, "Number of rows to load for each table. Default is 1000."]
) -> List[Dict]:
    """
    Run a single-column-join search on the given tables.

    :return: a list of dictionaries, where each reports the pair of tablees and columns and the relative overlap.
    """

    s = f" Running Single JOIN Search with {k=} "
    print(s.center(len(s) + 100, '#'))
    print(s)
    print(s.center(len(s) + 100, '#'))
    
    tables_path = None
    db_path = None

    if not tables_path:
        tables_path = DOWNLOAD_FOLDER

    if not db_path:
        db_path = BLEND_DB

    # now we focus only on CSV files...
    format = "csv"

    # FIXED: Store the original table_ids parameter before reassigning
    allowed_table_ids = table_ids
    
    # Get all table files and filter them
    all_files = os.listdir(tables_path)
    table_ids = []
    
    for filename in all_files:
        if filename.endswith(('.csv', '.parquet')):
            table_id = re.sub(r"\.(csv|parquet)$", "", filename)
            # Apply filter if provided
            if allowed_table_ids is None or table_id in allowed_table_ids:
                table_ids.append(table_id)
    
    print(f"Searching in tables: {table_ids}")

    if not searcher:
        print(f"Creating a new BLEND searcher with db_path: {db_path}")
        searcher = BLEND(db_path)
        searcher.create_index(data_path=tables_path, limit_table_rows=n_rows)

    results = []

    for table_id in table_ids:
        print(f"Searching in table {table_id}...")
        print(f"Number of rows to read: {n_rows}")

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

    print(f"Obtained results from JOIN search: ")
    print(results)
    print('#' * (len(s) + 100))

    return results



async def search_unions(
    tables_path: Annotated[str | None, "Path to the tables directory. If None, default location is used. Default is None."] = None,
    db_path: Annotated[str | None, "Path to the BLEND database file. If None, default location is used. Default is None."] = None,
    table_ids: Annotated[List[str] | None, "A subset of table IDs to use for the search. If None, all tables are used. Default is None."] = None,
    k: Annotated[int, "Number of results to return for each search. Default is 3."] = 3,
    n_rows: Annotated[int, "Number of rows to load for each table. Default is 1000."] = 1000,
    # check_headers: Annotated[bool, "If True, use overlapping headers to identify potential unionable tables."] = True,
    check_data: Annotated[bool, "If True, use BLEND combined with SLOTH to identify potential unionable tables. Default is True."] = True
) -> List[Dict]:
    """
    Run a union search on the given tables.

    :return: a list of dictionaries, where each reports the pair of tablees and columns and the relative overlap.
    """

    format = "csv"

    # FIXED: Store the original table_ids parameter before reassigning
    allowed_table_ids = table_ids
    
    # Get all table files and filter them
    all_files = os.listdir(tables_path)
    table_ids = []
    
    for filename in all_files:
        if filename.endswith(('.csv', '.parquet')):
            table_id = re.sub(r"\.(csv|parquet)$", "", filename)
            # Apply filter if provided
            if allowed_table_ids is None or table_id in allowed_table_ids:
                table_ids.append(table_id)
    
    print(f"Searching in tables: {table_ids}")

    if not searcher:
        print(f"Creating a new BLEND searcher with db_path: {db_path}")
        searcher = BLEND(db_path)
        searcher.create_index(data_path=tables_path, limit_table_rows=n_rows)

    results_by_data = []

    if check_data:
        for table_id in table_ids:

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

            # first perform a search with BLEND keywords seeker
            query_values = list(chain(*df.rows()))
            res = searcher.keyword_search(query_values, k)

            # then, for each identified pairs, 
            # check its largest overlap in terms of width 
            for res_table_id, overlap in res:
                match format:
                    case "csv":
                        s_df = pl.read_csv(
                            os.path.join(tables_path, table_id + ".csv"), n_rows=n_rows
                        )
                    case "parquet":
                        s_df = pl.read_parquet(
                            os.path.join(tables_path, table_id + ".parquet"), r_rows=n_rows
                        )
                
                min_w = min(df.shape[1], s_df.shape[1])

                while min_w >= 1:
                    sloth_metrics = {}
                    success, sloth_results, sloth_metrics = sloth(df, s_df, min_w=min_w, metrics=sloth_metrics)
                    
                    if sloth_results == []:
                        # if any result is found, decrease the required width
                        min_w -= 1
                    
                    else:
                        # if a valid overlap is found, then stop the search for the
                        # current table and pass to the next one
                        
                        left_columns = [
                            df.columns[p[0]]
                            for p in sloth_results[0][0]
                        ]

                        right_columns = [
                            s_df.columns[p[1]]
                            for p in sloth_results[0][0]
                        ]

                        results_by_data.append(
                            {
                                "left_table": table_id,
                                "right_table": res_table_id,
                                "left_columns": left_columns,
                                "right_columns": right_columns,
                                "width": sloth_metrics["largest_overlap_width"],
                                "heigth": sloth_metrics["largest_overlap_heigth"],
                                "area": sloth_metrics["largest_overlap_area"] 
                            }
                        )

                        break


    # if check_headers:
    #     raise NotImplementedError("Headers mode not implemented yet.")

    return results_by_data
