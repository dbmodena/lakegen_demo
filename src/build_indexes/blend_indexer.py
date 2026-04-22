import os
import sys
import pandas as pd
import polars as pl
from pathlib import Path

class BlendIndexer:
    def __init__(self, csv_dir: Path, db_path: Path):
        self.csv_dir = csv_dir
        self.db_path = db_path

        # Setup blend import
        CURRENT_DIR = Path(__file__).parent.resolve()
        if CURRENT_DIR.name == "build_indexes":
            BASE_DIR = CURRENT_DIR.parent.parent
        else:
            BASE_DIR = CURRENT_DIR.parent
            
        if str(BASE_DIR / "src") not in sys.path:
            sys.path.append(str(BASE_DIR / "src"))
            
        try:
            from blend.blend import BLEND
            self.BLEND_CLASS = BLEND
        except ImportError as e:
            print(f"❌ Critical error: impossible to import BLEND.\nDettaglio: {e}")
            sys.exit(1)

    def build_index(self, specific_files: list[str] = None, silent: bool = False):
        if not silent:
            print("[2/2] Creating DuckDB Index (BLEND) in progress...")
        
        if self.db_path.exists():
            os.remove(self.db_path)

        blend_indexer = self.BLEND_CLASS(db_path=self.db_path)
        blend_indexer.db_handler.create_index_table()

        if specific_files:
            csv_files = [self.csv_dir / f for f in specific_files if (self.csv_dir / f).exists()]
        else:
            csv_files = list(self.csv_dir.glob("*.csv"))
            
        if not csv_files:
            if not silent:
                print("❌ No CSV files found in the specified directory.")
            return

        for csv_path in csv_files:
            nome_file = csv_path.name
            try:
                df = pl.read_csv(str(csv_path), ignore_errors=True, infer_schema_length=0, n_rows=10000)
                
                if len(df.columns) > 1000:
                    df = pl.from_pandas(pd.read_csv(str(csv_path), nrows=10000))
                    if len(df.columns) > 1000:
                        if not silent:
                            print(f"      -> Skipping {nome_file}: too many columns.")
                        continue
                        
                if len(df) == 0: 
                    continue

                dismantled_columns = []
                for col_id, col_name in enumerate(df.columns):
                    colonna = df.select([
                        pl.lit(nome_file).alias("table_id"),
                        pl.lit(col_id).cast(pl.UInt32).alias("column_id"),
                        pl.int_range(0, len(df)).cast(pl.UInt32).alias("row_id"),
                        pl.lit(False).alias("quadrant"),
                        pl.col(col_name).cast(pl.Utf8).alias("cell_value"),
                        pl.lit(b"").alias("super_key")
                    ]).drop_nulls(subset=["cell_value"])
                    dismantled_columns.append(colonna)

                if dismantled_columns:
                    blend_indexer.db_handler.save_data_to_duckdb(pl.concat(dismantled_columns))
                    
            except Exception as e:
                if not silent:
                    print(f"      -> ❌ Error indexing CSV on {nome_file}: {e}")

        blend_indexer.db_handler.create_column_indexes()
        blend_indexer.close()
        if not silent:
            print(f"      -> Columnar DuckDB index created in: {self.db_path}")

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve()

    if CURRENT_DIR.name == "build_indexes":
        BASE_DIR = CURRENT_DIR.parent.parent
    elif CURRENT_DIR.name == "src":
        BASE_DIR = CURRENT_DIR.parent
    else:
        BASE_DIR = CURRENT_DIR

    DATA_DIR = BASE_DIR / "Data"
    CSV_DIR = DATA_DIR / "data_csv"
    DB_PATH = DATA_DIR / "blend_index.db"

    if not DATA_DIR.exists():
        print(f"❌ ERROR: The data folder does not exist in the expected path:\n{DATA_DIR}")
        print("Make sure you have the 'Data' folder at the right level.")
        sys.exit(1)

    indexer = BlendIndexer(csv_dir=CSV_DIR, db_path=DB_PATH)
    indexer.build_index()
