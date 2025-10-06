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
        client = get_client()
        index, meta = load_index(request.index_path)
        hits = retrieve(client, index, meta, request.query, request.top_k)
        context = build_context(hits)
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

        # Save uploaded files to a temporary directory
        temp_upload_dir = "./data/temp"
        os.makedirs(temp_upload_dir, exist_ok=True)
        file_paths = []
        for file in files:
            file_location = os.path.join(temp_upload_dir, file.filename)
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
            file_paths.append(file_location)
        
        args = {
            "file_paths": file_paths,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "index_path": index_path,
            "data_dir": temp_upload_dir,
        }
        response = op_ingest(args)
        return IngestResponse(message="Ingestion completed successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "RAG AI API is running."}
