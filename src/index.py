import os
import sys
import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. PATH SETUP (Smart Detection)
# ==========================================
CURRENT_DIR = Path(__file__).parent.resolve()

if CURRENT_DIR.name == "src":
    BASE_DIR = CURRENT_DIR.parent
else:
    BASE_DIR = CURRENT_DIR

sys.path.append(str(BASE_DIR / "src"))

try:
    from blend.blend import BLEND
except ImportError as e:
    print(f"❌ Critical error: impossible to import BLEND.\nDettaglio: {e}")
    sys.exit(1)

DATA_DIR = BASE_DIR / "Data"
CSV_DIR = DATA_DIR / "data_csv"
JSON_DIR = DATA_DIR / "data_json"
INDICI_DIR = DATA_DIR / "indices"
DB_PATH = DATA_DIR / "blend_index.db"

INDICI_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

# ==========================================
# 2. INDEXER CLASS
# ==========================================
class DataLakeIndexer:
    def __init__(self, json_dir: Path, csv_dir: Path, db_path: Path):
        self.json_dir = json_dir
        self.csv_dir = csv_dir
        self.db_path = db_path
        self.table_keywords = {}
        self.inverted_index = {}

    def _build_metadata_index(self, top_n_keywords: int = 15):
        print("[1/2] Metadata extraction and TF-IDF calculation in progress...")
        stop_words = list(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        table_names, descriptions = [], []

        for json_filepath in self.json_dir.glob("*.json"):
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if isinstance(metadata, dict) and "recordSet" in metadata:
                    for record in metadata["recordSet"]:
                        table_names.append(record.get("name", "unknown_table"))
                        t_desc = record.get("description", "")
                        col_text = "".join([f"{field.get('name', '')} {field.get('description', '')} " for field in record.get("field", [])])
                        descriptions.append(f"{t_desc} {col_text}")
                elif isinstance(metadata, list):
                    for item in metadata:
                        table_names.append(item.get('name', item.get('title', 'unknown_table')))
                        descriptions.append(item.get('description', ''))
            except Exception as e:
                print(f"      -> Warning: Error reading {json_filepath.name}: {e}")
                continue

        if not descriptions:
            print("❌ No valid metadata found in JSON. The text index will be empty.")
            return

        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        feature_names = np.array(vectorizer.get_feature_names_out())

        for i, table_name in enumerate(table_names):
            row = tfidf_matrix.getrow(i).toarray()[0]
            top_indices = row.argsort()[-top_n_keywords:][::-1]
            raw_keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            unique_lemmatized = list(set([lemmatizer.lemmatize(kw) for kw in raw_keywords]))
            self.table_keywords[table_name] = unique_lemmatized

            for kw in unique_lemmatized:
                self.inverted_index.setdefault(kw, []).append(table_name)

        print(f"      -> Indexed metadata for {len(self.table_keywords)} tables.")

        with open(INDICI_DIR / "table_keywords.json", "w", encoding="utf-8") as f:
            json.dump(self.table_keywords, f, ensure_ascii=False, indent=2)
        with open(INDICI_DIR / "inverted_index.json", "w", encoding="utf-8") as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=2)
            
        print(f"      -> Textual indices saved in: {INDICI_DIR}")

    def _build_blend_index(self):
        print("[2/2] Creating DuckDB Index (BLEND) in progress...")
        
        if self.db_path.exists():
            os.remove(self.db_path)

        blend_indexer = BLEND(db_path=self.db_path)
        blend_indexer.db_handler.create_index_table()

        csv_files = list(self.csv_dir.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in Data/data_csv/")
            return

        for csv_path in csv_files:
            nome_file = csv_path.name
            try:
                df = pl.read_csv(str(csv_path), ignore_errors=True, infer_schema_length=0, n_rows=10000)
                
                if len(df.columns) > 1000:
                    df = pl.from_pandas(pd.read_csv(str(csv_path), nrows=10000))
                    if len(df.columns) > 1000:
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
                print(f"      -> ❌ Error indexing CSV on {nome_file}: {e}")

        blend_indexer.db_handler.create_column_indexes()
        blend_indexer.close()
        print(f"      -> Columnar DuckDB index created in: {self.db_path}")

    def run(self):
        print("\n=== STARTING LAKEGEN DATA INGESTION PIPELINE ===")
        self._build_metadata_index()
        self._build_blend_index()
        print("=== INGESTION COMPLETED SUCCESSFULLY! ===\n")

# ==========================================
# 3. STARTING THE SCRIPT
# ==========================================
if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"❌ ERROR: The data folder does not exist in the expected path:\n{DATA_DIR}")
        print("Make sure you have the 'Data' folder at the right level.")
        sys.exit(1)
        
    indexer = DataLakeIndexer(json_dir=JSON_DIR, csv_dir=CSV_DIR, db_path=DB_PATH)
    indexer.run()