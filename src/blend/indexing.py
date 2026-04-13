import logging
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from inspect import cleandoc
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Optional

import polars as pl
from tqdm import tqdm

from .blend import BLEND
from .db import DBHandler
from .utils import init_logger, parse_table


def _db_worker(
    queue: Optional[Queue],
    tmp_path: Optional[Path],
    batch_rows: int,
    db_handler: DBHandler,
):
    """Dedicated consumer process that handles all DB writes.

    Args:
        queue: The queue containing dataframes to write.
        tmp_path: The temporary path for file-based communication (alternative to queue).
        batch_rows: The number of rows to accumulate before writing.
        db_handler: The DBHandler instance.
    """
    dataframes = []
    while True:
        if queue:
            df = queue.get()
            if df is None:  # Poison pill to stop the process
                break

            if not isinstance(df, pl.DataFrame):
                raise TypeError(f"Expected polars.DataFrame, found: {type(df)}")
            dataframes.append(df)
            if sum(_df.height for _df in dataframes) < batch_rows:
                continue

            try:
                db_handler.save_data_to_duckdb(dataframes)
                dataframes.clear()
            except Exception as e:
                print(f"DB Write Error: {e}")

        elif tmp_path:
            time.sleep(0.1)
            stop = False
            files = os.listdir(tmp_path)
            for filename in files:
                if filename == "STOP":
                    os.remove(tmp_path.joinpath(filename))
                    stop = True
                    break
                else:
                    try:
                        file = tmp_path.joinpath(filename)
                        db_handler.save_data_to_duckdb(file)
                        os.remove(tmp_path.joinpath(filename))
                    except Exception as e:
                        print(e)
            if stop:
                break


def _table_parsing_worker(
    table_path: Path,
    load_opts: Optional[dict],
    clean_args: Optional[dict],
    xash_size: int,
    max_cell_length: int,
    db_handler: DBHandler,
    queue: Optional[Queue],
    tmp_path: Optional[Path],
):
    """Parses a table and sends the result to the DB worker.

    Args:
        table_path: Path to the table file.
        load_opts: Options for loading the table.
        clean_args: Options for cleaning the table.
        xash_size: Size of the XASH hash.
        max_cell_length: Maximum length of cell values.
        db_handler: The DBHandler instance.
        queue: Use this queue to send dataframes if provided.
        tmp_path: Use this path to write parquet files if queue is not provided.
    """
    table_id, df = parse_table(
        table_path,
        load_opts,
        clean_args,
        xash_size,
        max_cell_length,
    )

    if isinstance(df, pl.DataFrame):
        if queue is not None:
            queue.put(df)
        elif tmp_path:
            file = tmp_path.joinpath(f"{uuid.uuid4()}.parquet")
            df.write_parquet(file)  # , compression_level=22)
        return table_id, True
    return table_id, False


def index_tables(
    indexer: BLEND,
    tables_path: Path,
    log_stdout: bool = False,
    logfile_path: Optional[Path] = None,
    max_workers: Optional[int] = None,
    load_opts: Optional[dict] = None,
    max_queue_size: Optional[int] = None,
    batch_rows: Optional[int] = None,
    tmp_path: Optional[Path] = None,
) -> tuple:
    """Index all the tables stored under the given tables path.

    It considers the path as a flat folder with only tables.

    Args:
        indexer: A BLEND indexer instance.
        tables_path: The path to the folder containing the tables to index.
        log_stdout: Whether to log to stdout.
        logfile_path: The path to a logfile.
        max_workers: Maximum number of processes to instantiate.
        load_opts: A dictionary with Polars scan csv/parquet/... configuration options. See blend.utils.load_table.
        max_queue_size: Size of the queue when the insertion is queue-based.
        batch_rows: Size of batch insert when the insertion is queue-based.
        tmp_path: Temporary folder where the temporary parquet files representing the parsed tables will be placed
            when the insertion is file-based.

    Returns:
        A tuple with timing for the tables parse and insertion time, support indexes creation time and total time.
    """
    if not tables_path.exists():
        raise FileNotFoundError(f"tables path doesn't exist: {tables_path}")

    if max_queue_size is None:
        max_queue_size = 100
    if batch_rows is None:
        batch_rows = 1_000_000

    init_logger(logfile_path, log_stdout)

    logger = logging.getLogger(f"blend_logger_{os.getpid()}")

    # get IDs of the effective tables
    table_ids = os.listdir(tables_path)

    # drop the main index if already exists
    indexer.db_handler.drop_index_table()

    # create the main index
    indexer.db_handler.create_index_table()

    # Create the Manager and Queue
    manager = Manager()

    if tmp_path:
        queue = None
    else:
        # Optional: limit size to prevent RAM overflow
        queue = manager.Queue(max_queue_size)

    # Start the DB Worker Process
    db_writer = Process(
        target=_db_worker, args=(queue, tmp_path, batch_rows, indexer.db_handler)
    )
    db_writer.start()

    start_t = time.time()

    # TODO: work on Windows? (check mp_context-polars)
    # TODO: Timeout for _process_task/_db_worker?
    try:
        with ProcessPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(
                    _table_parsing_worker,
                    tables_path.joinpath(table_id),
                    load_opts,
                    indexer._clean_args,
                    indexer.xash_size,
                    indexer.max_cell_length,
                    indexer.db_handler,  # ty: ignore
                    queue,
                    tmp_path,
                )
                for table_id in list(table_ids)
            }

            non_empty_tables = 0

            for future in tqdm(
                as_completed(futures),
                desc="Parsing and storing tables: ",
                total=len(table_ids),
                disable=not log_stdout,
            ):
                try:
                    table_id, success = future.result()
                    non_empty_tables += success
                except Exception as e:
                    logger.error(f"[error:{type(e)}][msg:{e}]")

    finally:
        end_ins_t = time.time()

        # Stop the DB worker
        if queue:
            queue.put(None)
        elif tmp_path:
            while len(os.listdir(tmp_path)) != 0:
                continue
            with open(tmp_path.joinpath("STOP"), "w") as file:
                file.write("STOP")
        db_writer.join(30)
        if db_writer.is_alive():
            db_writer.terminate()
            db_writer.join()

    # create indexes
    s = f"""
        Tables ingestion completed.
        Correctly parsed {non_empty_tables}.
        Creating indexes...
        """
    logger.info(cleandoc(s))

    indexer.db_handler.create_column_indexes()
    end_idx_t = time.time()

    logger.debug("Closing DB...")
    indexer.db_handler.close()
    logger.info("Index creation completed.")
    return (end_ins_t - start_t, end_idx_t - end_ins_t, end_idx_t - start_t)
