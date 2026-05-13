import sys
from pathlib import Path
sys.path.append(str(Path("src/blend").resolve()))
import blend

db_path = Path("data/blend_index.db")
csv_dir = Path("data/nyc/datasets/csv")

print("Building BLEND index globally...")
indexer = blend.BLEND(db_path=db_path)
opts = {"ignore_errors": True, "infer_schema_length": 0, "n_rows": 10000}
blend.index_tables_seq(indexer, csv_dir, load_opts=opts, log_stdout=True)
print("Done!")
