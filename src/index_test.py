import sys
from pathlib import Path

# ==========================================
# 1. PATH SETUP
# ==========================================
CURRENT_DIR = Path(__file__).parent.resolve()

if CURRENT_DIR.name == "src":
    BASE_DIR = CURRENT_DIR.parent
else:
    BASE_DIR = CURRENT_DIR

sys.path.append(str(BASE_DIR / "src"))

DATA_DIR = BASE_DIR / "Data"
CSV_DIR = DATA_DIR / "data_csv"
JSON_DIR = DATA_DIR / "data_json"
INDEXES_DIR = DATA_DIR / "indexes"
DB_PATH = DATA_DIR / "blend_index.db"

# Import the new separated indexers
from build_indexes.metadata_indexer import MetadataIndexer
from build_indexes.blend_indexer import BlendIndexer

# ==========================================
# 2. INDEXER CLASS
# ==========================================
class DataLakeIndexer:
    def __init__(self, json_dir: Path, csv_dir: Path, db_path: Path, indexes_dir: Path):
        self.metadata_indexer = MetadataIndexer(json_dir=json_dir, indexes_dir=indexes_dir)
        self.blend_indexer = BlendIndexer(csv_dir=csv_dir, db_path=db_path)

    def run(self):
        print("\n=== STARTING LAKEGEN DATA INGESTION PIPELINE ===")
        self.metadata_indexer.build_index()
        self.blend_indexer.build_index()
        print("=== INGESTION COMPLETED SUCCESSFULLY! ===\n")

# ==========================================
# 3. STARTING THE SCRIPT
# ==========================================
if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"❌ ERROR: The data folder does not exist in the expected path:\n{DATA_DIR}")
        print("Make sure you have the 'Data' folder at the right level.")
        sys.exit(1)
        
    indexer = DataLakeIndexer(
        json_dir=JSON_DIR, 
        csv_dir=CSV_DIR, 
        db_path=DB_PATH,
        indexes_dir=INDEXES_DIR
    )
    indexer.run()