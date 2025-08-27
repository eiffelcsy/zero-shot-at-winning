from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from rag.ingestion.pipeline import PDFIngestionPipeline
from chroma.chroma_connection import get_chroma_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/api/v1", tags=["pdf-upload"])

# Response models
class UploadResponse(BaseModel):
    status: str
    message: str
    results: List[Dict[str, Any]]
    total_files: int
    successful: int
    failed: int

class PipelineStatsResponse(BaseModel):
    pipeline_config: Dict[str, Any]
    storage_stats: Dict[str, Any]
    status: str

# Initialize the PDF ingestion pipeline
pipeline = PDFIngestionPipeline(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-3-large",
    collection_name="regulation_kb"
)

@router.post("/upload-pdfs", response_model=UploadResponse)
async def upload_pdf_files(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = 1000,
    chunk_overlap: Optional[int] = 200
) -> UploadResponse:
    """
    Upload and process PDF files through the ingestion pipeline.
    
    This endpoint:
    1. Accepts multiple PDF files from the Streamlit frontend
    2. Processes them through the PDF ingestion pipeline
    3. Stores the extracted and chunked text in ChromaDB
    4. Returns detailed processing results
    
    Args:
        files: List of PDF files to upload and process
        chunk_size: Optional custom chunk size (default: 1000)
        chunk_overlap: Optional custom chunk overlap (default: 200)
    
    Returns:
        UploadResponse with processing results for each file
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate that all files are PDFs
    invalid_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            invalid_files.append(file.filename)
    
    if invalid_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file types. Only PDF files are allowed. Invalid files: {', '.join(invalid_files)}"
        )
    
    logger.info(f"Starting batch PDF upload and processing for {len(files)} files")
    
    try:
        # If custom parameters are provided, create a new pipeline instance
        if chunk_size != 1000 or chunk_overlap != 200:
            custom_pipeline = PDFIngestionPipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model="text-embedding-3-large",
                collection_name="regulation_kb"
            )
            results = custom_pipeline.process_batch(files)
        else:
            # Use the default pipeline
            results = pipeline.process_batch(files)
        
        # Calculate summary statistics
        successful_count = sum(1 for result in results if result.get('status') == 'success')
        failed_count = len(results) - successful_count
        
        # Determine overall status
        overall_status = "success" if failed_count == 0 else "partial_success" if successful_count > 0 else "error"
        
        # Create response message
        if overall_status == "success":
            message = f"Successfully processed all {len(files)} PDF files"
        elif overall_status == "partial_success":
            message = f"Processed {successful_count} out of {len(files)} files successfully"
        else:
            message = f"Failed to process all {len(files)} PDF files"
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
        
        return UploadResponse(
            status=overall_status,
            message=message,
            results=results,
            total_files=len(files),
            successful=successful_count,
            failed=failed_count
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in PDF upload endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during PDF processing: {str(e)}"
        )

@router.get("/upload-stats", response_model=PipelineStatsResponse)
async def get_upload_stats() -> PipelineStatsResponse:
    """
    Get statistics about the PDF upload pipeline and ChromaDB storage.
    
    Returns:
        PipelineStatsResponse with current pipeline configuration and storage stats
    """
    try:
        stats = pipeline.get_pipeline_stats()
        return PipelineStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting pipeline stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving pipeline statistics: {str(e)}"
        )

@router.delete("/clear-documents")
async def clear_document_collection() -> Dict[str, Any]:
    """
    Clear all documents from the ChromaDB collection.
    
    Warning: This will permanently delete all uploaded PDF documents and their embeddings.
    
    Returns:
        Dictionary with operation status and details
    """
    try:
        result = pipeline.clear_collection()
        if result['status'] == 'success':
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('message', 'Failed to clear collection'))
    except Exception as e:
        logger.error(f"Error clearing document collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing document collection: {str(e)}"
        )