from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from core import load_index, build_context, ask_gpt, retrieve, get_client, query_handler, update_vector_base

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
    
@app.post("/update")
def ask_endpoint(request: QueryRequest):
    try:
        response = update_vector_base(request.dict())
        # sources = [h.meta.get("source") for h in hits]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "RAG AI API is running."}
