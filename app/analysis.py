import pandas as pd
import glob
import os
import json
from datetime import datetime
import pandas as pd
import feedparser
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.store.vector import VectorStore
from sklearn.metrics.pairwise import cosine_similarity

EXCEL_IMPORT_DIR = "./data/excel/"
PDF_IMPORT_DIR = "./data/pdf/"
TEXT_IMPORT_DIR = "./data/text/"

VECTOR_FILE = "./data/vector_store.npy"
METADATA_FILE = "./data/metadata.json"
EMBED_MODEL = "text-embedding-3-small"
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------- DATA READERS ----------

def read_excels(data_dir=EXCEL_IMPORT_DIR):
    try:
        print("Reading Excel files...")
        paths = glob.glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)
        all_chunks = []
        print(paths)
        for p in paths:
            try:
                metadata = pd.read_excel(p, nrows=1, header=None)
                metadata_text = metadata.to_string(index=False, header=False)
                df = pd.read_excel(p, skiprows=1)
                print(f"Read {p} with shape {df.shape}")

                chunks = []
                chunks.append(metadata_text)
                year_cols = [col for col in df.columns if col not in ["Country", "%Growth", "S.No."]]
                for i, row in df.iterrows():
                    chunk = f"Country: {row['Country']}"
                    for year in year_cols:
                        value = row[year] if pd.notna(row[year]) else 0
                        chunk += f"Fiscal Year {year}: {value}"
                    if "%Growth" in df.columns:
                        value = row['%Growth'] if pd.notna(row['%Growth']) else 0
                        chunk += f", Growth Percentage: {value}"
                    print(f"Processing row {i}: {chunk}")
                    chunks.append(chunk)
                chunks.append("End of this Document.")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {p}: {e}")
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    except Exception as e:
        print(f"Error in read_excels: {e}")

def read_excel_file(p):
    """
    Read a single Excel file and return a list of text chunks.
    Errors are caught and an empty list is returned on failure.
    """
    try:
        metadata = pd.read_excel(p, nrows=1, header=None)
        metadata_text = metadata.to_string(index=False, header=False)
        df = pd.read_excel(p, skiprows=1)
        print(f"Read {p} with shape {df.shape}")

        chunks = [metadata_text]
        year_cols = [col for col in df.columns if col not in ["Country", "%Growth", "S.No."]]
        for i, row in df.iterrows():
            chunk = f"Country: {row['Country']}"
            for year in year_cols:
                value = row[year] if pd.notna(row[year]) else 0
                chunk += f" Fiscal Year {year}: {value}"
            if "%Growth" in df.columns:
                value = row['%Growth'] if pd.notna(row['%Growth']) else 0
                chunk += f", Growth Percentage: {value}"
            print(f"Processing row {i}: {chunk}")
            chunks.append(chunk)
        chunks.append("End of this Document.")
        return chunks
    except Exception as e:
        print(f"Error reading {p}: {e}")
        return []
    
def read_pdf(path=PDF_IMPORT_DIR):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_rss(url):
    feed = feedparser.parse(url)
    return "\n".join([entry.title + " " + entry.summary for entry in feed.entries])

def read_text_file(path=TEXT_IMPORT_DIR):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_linkedin_post(post_text):
    return post_text.strip()


def recreate_vector_base_for_all_docs():
    # read_excels()
    store = VectorStore()
    # Add PDFs
    for file in os.listdir(PDF_IMPORT_DIR):
        if file.endswith(".pdf"):
            text = read_pdf(os.path.join(PDF_IMPORT_DIR, file))
            store.add_source(file, "pdf", text)

    # Add Excels
    for file in os.listdir(EXCEL_IMPORT_DIR):
        if file.endswith(".xlsx"):
            chunks = read_excel_file(os.path.join(EXCEL_IMPORT_DIR, file))
            store.add_source(file, "excel", None, chunks)

# -------- UTILITY ----------
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def retrieve_top_k(vectorstore, query_text, top_k=3):
    """
    Retrieve top-K relevant chunks from your custom VectorStore.
    Returns both the text and metadata for source tracking.
    """
    # Use the VectorStore's query function
    results = vectorstore.query(query_text, top_k=top_k)
    
    # Extract text and metadata
    context_chunks = [r['metadata']['text'] for r in results]
    metadata_list = [r['metadata'] for r in results]
    
    # Optional: log sources
    print("\nRetrieved Contexts:")
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata_list)):
        print(f"[{i+1}] Source: {meta.get('source', 'Unknown')} | Added: {meta.get('added_on', 'N/A')}")
        print(f"→ {chunk}...\n")
    
    return context_chunks, metadata_list

def get_similarity(client, text, index_path="./data/data.faiss", top_k=6):
        index = faiss.read_index(index_path)
        with open(index_path + ".meta.json", "r", encoding="utf-8") as f:
            meta_raw = json.load(f)

        # Embed query text
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        query_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
        
        # normalize to match IndexFlatIP built from normalized vectors
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_vec = query_vec / norms

        # Search FAISS index for top-K similar vectors
        distances, indices = index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(meta_raw):
                continue
            meta_entry = meta_raw[idx]
            results.append({
                "score": float(score),
                "metadata": meta_entry
            })

        # sims = cosine_similarity(query_vec, query_vec)[0]
        # indices = np.argsort(sims)[-top_k:][::-1]

        return results

def assemble_context(context_chunks):
    """
    Concatenates retrieved chunks into a single prompt-ready context string.
    """
    return "\n\n".join(context_chunks)
