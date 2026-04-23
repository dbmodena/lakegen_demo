import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class MetadataIndexer:
    def __init__(self, json_dir: Path, indexes_dir: Path):
        self.json_dir = json_dir
        self.indexes_dir = indexes_dir
        self.table_keywords = {}
        self.inverted_index = {}

        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)

    def build_index(self, top_n_keywords: int = 15):
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
                        if "dataset_id" in item:
                            dataset_id = item.get("dataset_id", "unknown")
                            table_names.append(f"{dataset_id}.csv")
                            t_desc = item.get("description") or ""
                            t_title = item.get("title") or ""
                            col_text = " ".join([
                                f"{col.get('name', '')} {col.get('label', '')} {col.get('description') or ''}" 
                                for col in item.get("columns", [])
                            ])
                            descriptions.append(f"{t_title} {t_desc} {col_text}")
                        else:
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
            top_indixes = row.argsort()[-top_n_keywords:][::-1]
            raw_keywords = [feature_names[idx] for idx in top_indixes if row[idx] > 0]
            unique_lemmatized = list(set([lemmatizer.lemmatize(kw) for kw in raw_keywords]))
            self.table_keywords[table_name] = unique_lemmatized

            for kw in unique_lemmatized:
                self.inverted_index.setdefault(kw, []).append(table_name)

        print(f"      -> Indexed metadata for {len(self.table_keywords)} tables.")

        with open(self.indexes_dir / "table_keywords.json", "w", encoding="utf-8") as f:
            json.dump(self.table_keywords, f, ensure_ascii=False, indent=2)
        with open(self.indexes_dir / "inverted_index.json", "w", encoding="utf-8") as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=2)
            
        print(f"      -> Textual indexes saved in: {self.indexes_dir}")

if __name__ == "__main__":
    import sys
    CURRENT_DIR = Path(__file__).parent.resolve()

    if CURRENT_DIR.name == "build_indexes":
        BASE_DIR = CURRENT_DIR.parent.parent
    elif CURRENT_DIR.name == "src":
        BASE_DIR = CURRENT_DIR.parent
    else:
        BASE_DIR = CURRENT_DIR

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
        
    from src.utils import DATA_DIR, JSON_DIR, INDEXES_DIR

    if not DATA_DIR.exists():
        print(f"❌ ERROR: The data folder does not exist in the expected path:\n{DATA_DIR}")
        print("Make sure you have the 'Data' folder at the right level.")
        sys.exit(1)

    indexer = MetadataIndexer(json_dir=JSON_DIR, indexes_dir=INDEXES_DIR)
    indexer.build_index()
