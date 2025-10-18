import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# VECTOR STORE
# -------------------------------

VECTOR_FILE = "./data/vector_store.npy"
METADATA_FILE = "./data/metadata.json"
model = SentenceTransformer('all-MiniLM-L6-v2')

class VectorStore:
    def __init__(self, vector_file=VECTOR_FILE, metadata_file=METADATA_FILE):
        self.vector_file = vector_file
        self.metadata_file = metadata_file
        self.embeddings = []
        self.metadata = []

        self.load()

    def load(self):
        if os.path.exists(self.vector_file):
            self.embeddings = np.load(self.vector_file)
        else:
            self.embeddings = np.empty((0, 384))  # for MiniLM-L6-v2

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def save(self):
        np.save(self.vector_file, self.embeddings)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_source(self, source_name, source_type, text_data, chunks):
        if chunks is None:
            chunks = chunk_text(text_data)
            
        vectors = model.encode(chunks)
        metadata_entries = [
            {
                "source_name": source_name,
                "source_type": source_type,
                "chunk_index": i,
                "text": chunks[i],
                "timestamp": get_timestamp()
            }
            for i in range(len(chunks))
        ]
        if len(self.embeddings) == 0:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack((self.embeddings, vectors))
        self.metadata.extend(metadata_entries)
        self.save()

    def query(self, text, top_k=3):
        if len(self.embeddings) == 0:
            return []

        query_vec = model.encode([text])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in indices:
            results.append({
                "similarity": float(sims[idx]),
                "metadata": self.metadata[idx]
            })
        return results
    
# -------------------------------
# UTILITIES
# -------------------------------
def chunk_text(text, chunk_size=500):
    """Split large text into chunks for better embeddings."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")