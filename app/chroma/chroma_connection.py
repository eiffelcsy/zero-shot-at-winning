import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import Depends
from dotenv import load_dotenv
import os

load_dotenv()

_client: ClientAPI | None = None
_collection: Collection | None = None

def get_chroma_client() -> ClientAPI:
    global _client
    if _client is None:
        # Check if we're using cloud or local ChromaDB
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        
        if chroma_api_key and os.getenv("CHROMA_TENANT") and os.getenv("CHROMA_DATABASE"):
            # Use cloud client
            _client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE")
            )
        else:
            # Use local HTTP client for Docker container
            _client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
    return _client

def get_chroma_collection(client: ClientAPI = Depends(get_chroma_client)) -> Collection:
    global _collection
    if _collection is None:
        _collection = client.get_or_create_collection(
            name="rag_collection",
        )
    return _collection