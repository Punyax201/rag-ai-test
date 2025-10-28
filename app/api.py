from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from app.core import load_index, build_context, ask_gpt, retrieve, get_client, query_handler, op_ingest

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE etc.
    allow_headers=["*"],        # allow all headers
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    index_path: Optional[str] = "./data/data.faiss"
    historical_context: Optional[object]

class IngestResponse(BaseModel):
    message: str
    # index_path: Optional[str]

@app.post("/ask")
def ask_endpoint(request: QueryRequest):
    try:
        # client = get_client()
        # index, meta = load_index(request.index_path)
        # hits = retrieve(client, index, meta, request.query, request.top_k)
        # context = build_context(hits)
        response = query_handler(request)
        # sources = [h.meta.get("source") for h in hits]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/update_vector_base")
async def update_vector_base_endpoint(files: list[UploadFile], chunk_size: Optional[int] = 600,
    chunk_overlap: Optional[int] = 80,
    index_path: Optional[str] = "./data/data.faiss"):
    try:

        # Prepare target directories under ./data
        base_data_dir = "./data"
        pdf_dir = os.path.join(base_data_dir, "pdf")
        excel_dir = os.path.join(base_data_dir, "excel")
        text_dir = os.path.join(base_data_dir, "text")
        other_dir = os.path.join(base_data_dir, "temp")

        # Create directories if they don't exist
        for d in (pdf_dir, excel_dir, text_dir, other_dir):
            os.makedirs(d, exist_ok=True)

        file_paths = []
        # Save files into folders based on extension
        for file in files:
            filename = file.filename or "unnamed"
            ext = os.path.splitext(filename)[1].lower()
            if ext in (".pdf",):
                dest_dir = pdf_dir
            elif ext in (".xlsx", ".xls"):
                dest_dir = excel_dir
            elif ext in (".txt", ".md"):
                dest_dir = text_dir
            else:
                dest_dir = other_dir

            file_location = os.path.join(dest_dir, filename)
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
            file_paths.append(file_location)

        # Pass base data dir so op_ingest can look into ./data/pdf, ./data/excel, ./data/text
        args = {
            "file_paths": file_paths,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "index_path": index_path,
            "data_dir": base_data_dir,
        }
        response = op_ingest(args)
        return IngestResponse(message="Ingestion completed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "RAG AI v1 API is running."}
