from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from chroma.chroma_connection import get_chroma_collection
from pydantic import BaseModel
from typing import Optional

from api.v1.router import router

class RequestBody(BaseModel):
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]

app = FastAPI(
    title="TikTok Geo-Compliance System API",
    description="FastAPI backend for PDF document ingestion and compliance analysis",
    version="1.0.0"
)

# Add CORS middleware to allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "message": "TikTok Geo-Compliance System API is running"}