import pandas as pd
import duckdb
import os
import uuid
import io
import json
import sqlite3
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# Constants
STORAGE_PATH = "./data_lake"
DB_PATH = "metadata.db"

class DataIngestor:
    """Handles raw data ingestion from various sources."""
    
    @staticmethod
    def read_file(file_obj, file_type: str) -> pd.DataFrame:
        try:
            if file_type == "csv":
                return pd.read_csv(file_obj)
            elif file_type == "excel":
                return pd.read_excel(file_obj)
            elif file_type == "json":
                return pd.read_json(file_obj)
            elif file_type == "pdf":
                # Basic PDF text extraction
                reader = PdfReader(file_obj)
                text = [page.extract_text() for page in reader.pages]
                return pd.DataFrame({"content": text, "page": range(1, len(text) + 1)})
            else:
                raise ValueError(f"Unsupported format: {file_type}")
        except Exception as e:
            raise RuntimeError(f"Ingestion failed: {str(e)}")

class StorageManager:
    """Manages Parquet storage and DuckDB interactions (The 'Delta Lake' layer)."""
    
    def __init__(self):
        os.makedirs(STORAGE_PATH, exist_ok=True)
        self.con = duckdb.connect(database=':memory:') # In-memory DuckDB for speed

    def save_to_bronze(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Saves raw data to Parquet."""
        file_path = f"{STORAGE_PATH}/{dataset_name}.parquet"
        df.to_parquet(file_path, index=False)
        return file_path

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        file_path = f"{STORAGE_PATH}/{dataset_name}.parquet"
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return pd.DataFrame()

    def execute_sql(self, query: str, dataset_name: str) -> pd.DataFrame:
        """Runs SQL queries on the Parquet file directly."""
        file_path = f"{STORAGE_PATH}/{dataset_name}.parquet"
        # DuckDB allows querying parquet files directly
        query = query.replace("CURRENT_TABLE", f"'{file_path}'")
        return self.con.execute(query).df()

class SearchIndexer:
    """Handles Vector Search and Indexing."""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

    def index_data(self, df: pd.DataFrame, collection_name: str, key_col: str):
        """Indexes a specific text column for semantic search."""
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(name=collection_name, embedding_function=self.embed_fn)
        
        # Convert non-string data for indexing
        documents = df[key_col].astype(str).tolist()
        ids = [str(i) for i in df.index.tolist()]
        metadatas = df.to_dict(orient='records')
        
        # Batch add to avoid payload limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            
    def search(self, query: str, collection_name: str, n_results=5):
        collection = self.client.get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=n_results)
        return results
