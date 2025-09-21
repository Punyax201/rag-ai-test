from __future__ import annotations

import os
import re
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from analysis import read_excels

import numpy as np
from openai import OpenAI

try:
    import faiss  # faiss-cpu
except ImportError as e:
    raise SystemExit("faiss-cpu not installed. Run: pip install faiss-cpu")

# ---------- Config ----------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 600 
DEFAULT_CHUNK_OVERLAP = 80

USER_PROMPT = (
    "You are a helpful assistant answering based only on the provided context. "
    "Follow these steps strictly:\n\n"
    "1. First, write a short summary (max 100 words) of the context or scenario in natural language. "
    "   - Include a mild, context-appropriate greeting only once.\n"
    "2. On a new line, explicitly list the relevant source(s) in square brackets, e.g. [Source 1, Source 3].\n"
    "3. Then, provide the final answer to the question. "
    "   - Be concise, factual, and grounded in the context. "
    "   - If the answer is not present in the context, reply with: 'I don't know.'\n\n"
    "4. Highlight key figures such as amounts, important names, locations as bold html text.\n"
    "5. Text marked as bold should be wrapped in <b> and </b> tags.\n"
    "Context:\n---\n{context}\n---\n"
    "Question: {question}"
)



# ---------- Data structures ----------
@dataclass
class DocChunk:
    text: str
    meta: Dict

# ---------- Utilities ----------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def read_txt_md_files(data_dir: str) -> List[Tuple[str, str]]:
    """Return list of (path, text) for .txt/.md files."""
    paths = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True) + \
            glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    docs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((p, f.read()))
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    return docs

# TODO: add PDF support if needed
# import fitz  # PyMuPDF
# def read_pdfs(data_dir: str):
#     for p in glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True):
#         doc = fitz.open(p)
#         text = "\n\n".join(page.get_text("text") for page in doc)
#         yield (p, text)

def read_pdfs(data_dir: str) -> List[Tuple[str, str]]:
    """Return list of (path, text) for .pdf files in the given directory."""
    paths = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    docs = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                docs.append((p, text))
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    return docs


def recursive_chunk(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Chunk text preferring paragraph/sentence boundaries, with overlap."""
    text = text.replace("\r", "")
    # Split by double newline first (paragraphs)
    paras = text.split("\n\n")
    chunks = []
    buff = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        # If paragraph is too large, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(buff) + len(sent) + 1 <= chunk_size:
                    buff = (buff + " " + sent).strip()
                else:
                    if buff:
                        chunks.append(buff)
                    # start new with overlap from end of previous
                    buff_tail = buff[-overlap:] if overlap and len(buff) > overlap else ""
                    buff = (buff_tail + " " + sent).strip()
        else:
            if len(buff) + len(para) + 2 <= chunk_size:
                buff = (buff + "\n\n" + para).strip()
            else:
                if buff:
                    chunks.append(buff)
                buff_tail = buff[-overlap:] if overlap and len(buff) > overlap else ""
                buff = (buff_tail + "\n\n" + para).strip()
    if buff:
        chunks.append(buff)
    return [normalize_ws(c) for c in chunks]

# ---------- Embeddings & Index ----------

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")
    return OpenAI()


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # OpenAI API accepts up to ~2048 inputs per call; we batch conservatively
    BATCH = 128
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [np.array(e.embedding, dtype="float32") for e in resp.data]
        vectors.extend(vecs)
    return np.vstack(vectors)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, meta: List[DocChunk], path: str):
    faiss.write_index(index, path)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump([{"text": dc.text, "meta": dc.meta} for dc in meta], f, ensure_ascii=False)


def load_index(path: str) -> Tuple[faiss.IndexFlatIP, List[DocChunk]]:
    index = faiss.read_index(path)
    with open(path + ".meta.json", "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
    meta = [DocChunk(m["text"], m["meta"]) for m in meta_raw]
    return index, meta


# ---------- Prompt construction & generation ----------
SYSTEM_PROMPT = (
    "You are a sharp, intelligent Astro Physicist Assistant. "
    "Answer **only** from the provided context. If the answer is not in the context, say you don't know. "
    "Be concise and safe; avoid rumours or unverified claims."
)


def build_context(hits: List[DocChunk]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h.meta.get("source", "unknown")
        title = h.meta.get("title", os.path.basename(src))
        blocks.append(f"[Source {i}: {title}]\n{h.text}")
    return "\n\n".join(blocks)


def ask_gpt(client: OpenAI, question: str, context_text: str) -> str:
    messages = [
         {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.replace("{context}", context_text).replace("{question}", question)},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        # temperature=0.2,
        max_completion_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ---------- Retrieval ----------

def retrieve(client: OpenAI, index: faiss.IndexFlatIP, meta: List[DocChunk], query: str, top_k: int = 5) -> List[DocChunk]:
    q_emb = embed_texts(client, [query])
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    hits = []
    for idx in I[0]:
        if int(idx) < 0:  # safety
            continue
        hits.append(meta[int(idx)])
    return hits

# ------------ PDF --------------

# Step 1: Extract text from PDF
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Step 2: Split into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Step 3: Build embeddings & index
def build_index(pdf_path, index_path="pdf_index"):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(index_path)  # Save for reuse
    return db

# Step 4: Load index and query with RAG
def query_pdf(query, index_path="pdf_index"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(index_path, embeddings)

    docs = db.similarity_search(query, k=3)  # Retrieve top 3 chunks
    context = "\n".join([d.page_content for d in docs])

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# ---------- CLI ops ----------

def op_ingest(args):
    client = get_client()
    all_txt_docs: List[Tuple[str, str]] = []
    all_txt_docs.extend(read_txt_md_files(args.data_dir))

    all_pdf_docs: List[Tuple[str, str]] = []
    all_pdf_docs.extend(read_pdfs(f"{args.data_dir}/pdf"))
    # for p, t in read_pdfs(args.data_dir): all_docs.append((p, t))  # enable if you add PDF loader

    # if not all_txt_docs and all_pdf_docs:
    #     raise SystemExit(f"No .txt/.md/ .pdf files found in {args.data_dir}")

    chunks: List[DocChunk] = []
    for path, text in all_pdf_docs:
        title = os.path.splitext(os.path.basename(path))[0]
        for ch in recursive_chunk(text, args.chunk_size, args.chunk_overlap):
            chunks.append(DocChunk(text=ch, meta={"source": path, "title": title}))

    #  Excel Data
    excel_chunks = read_excels()
    chunks.extend([DocChunk(text=ch, meta={"source": "excel_data", "title": "Excel Data for Export"}) for ch in excel_chunks])

    print(f"Prepared {len(chunks)} chunks. Embedding…")
    embs = embed_texts(client, [c.text for c in chunks])
    index = build_faiss_index(embs)
    save_index(index, chunks, args.index_path)
    print(f"Index saved to: {args.index_path} (+ .meta.json)")


def op_ask(args):
    client = get_client()
    index, meta = load_index(args.index_path)
    hits = retrieve(client, index, meta, args.query, args.top_k)
    context = build_context(hits)
    answer = ask_gpt(client, args.query, context)

    # Pretty print with sources
    print("\n==== Answer ====")
    print(answer)
    print("\n==== Sources ====")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.meta.get('source')}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=False)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--index_path", type=str, default="./data/data.faiss")
    args = parser.parse_args()
    op_ingest(args)

    # Only run once to build index

    # ap_ask = sub.add_parser("ask", help="Query the index and get an answer")
    # ap_ask.add_argument("--index_path", default="./data/data.faiss",)
    # ap_ask.add_argument("--top_k", type=int, default=5)
    # ap_ask.add_argument("query", type=str, nargs="?", default="What is the capital of France?")
    # qargs = ap_ask.parse_args()
    # op_ask(qargs)

def query_handler(query: str, top_k: int = 5, index_path: str = "./data/data.faiss"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default="./data/data.faiss")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("query", type=str, nargs="?", default=query)
    # args = parser.parse_args()
    client = get_client()
    index, meta = load_index(index_path)
    hits = retrieve(client, index, meta, query.query, top_k)
    context = build_context(hits)
    answer = ask_gpt(client, query.query, context)
    sources = [h.meta.get("source") for h in hits]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    main()