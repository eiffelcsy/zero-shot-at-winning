from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
from rag.ingestion.pipeline import PDFIngestionPipeline
from chroma.chroma_connection import get_chroma_collection
from agents.orchestrator import ComplianceOrchestrator
from agents.feedback.learning import LearningAgent
import os
import io

# Use the enhanced logging system that automatically saves to files
from logs.logging_config import ensure_logging_setup, get_logger

# Ensure logging is set up for the API router
ensure_logging_setup()
logger = get_logger("api.router")

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
    collection_stats: Dict[str, Any]
    embedding_model: str

class ComplianceResponse(BaseModel):
    status: str = Field(description="Analysis status: 'success', 'partial', or 'error'")
    message: str = Field(description="Human-readable summary of the analysis")
    analysis_id: str = Field(description="Unique identifier for this analysis")
    timestamp: str = Field(description="ISO timestamp of when analysis was completed")
    
    # Agent-specific results
    screening_result: Optional[Dict[str, Any]] = Field(default=None, description="Screening agent analysis")
    research_result: Optional[Dict[str, Any]] = Field(default=None, description="Research agent analysis")
    validation_result: Optional[Dict[str, Any]] = Field(default=None, description="Validation agent analysis")
    
    # Overall workflow results
    agents_completed: List[str] = Field(description="List of agents that completed successfully")
    workflow_status: str = Field(description="Overall workflow status")
    confidence_score: Optional[float] = Field(default=None, description="Overall confidence score")
    
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")

class OriginalResult(BaseModel):
    flag: Optional[str] = Field(default=None, description="Original yes/no/unknown decision")
    risk_level: Optional[str] = Field(default=None, description="Original risk level label")


class CorrectionData(BaseModel):
    correct_flag: Optional[str] = Field(default=None, description="Corrected yes/no (optional)")
    correct_risk_level: Optional[str] = Field(default=None, description="Corrected risk level (optional)")
    original_result: Optional[OriginalResult] = None


class FeedbackRequest(BaseModel):
    analysis_id: str
    feedback_type: Literal["positive", "negative"]
    feedback_text: Optional[str] = ""
    timestamp: Optional[str] = None
    state: Dict[str, Any]  # Full current state object from the client


class FeedbackResponse(BaseModel):
    status: Literal["success", "error"]
    message: str
    analysis_id: str
    learning_report: Optional[Dict[str, Any]] = None

# Log API router initialization
logger.info("=== Initializing API Router ===")

# Initialize the PDF ingestion pipeline
pipeline = PDFIngestionPipeline(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-3-large",
    collection_name="regulation_kb"
)
logger.info("PDF ingestion pipeline initialized")

# Initialize the compliance orchestrator for LangGraph multi-agent workflow
# Use the new combined memory system that includes both TikTok terminology and ALL fewshot examples
try:
    # Initialize with combined memory system (TikTok + all fewshot examples)
    compliance_orchestrator = ComplianceOrchestrator(use_combined_memory=True)
    logger.info("Compliance orchestrator initialized with combined memory system (TikTok + all fewshot examples)")
except Exception as e:
    logger.error(f"Failed to initialize combined memory system: {e}")
    logger.warning("Falling back to orchestrator without memory overlay")
    compliance_orchestrator = ComplianceOrchestrator(memory_overlay="", use_combined_memory=False)

# ================================================
# REGULATION UPLOAD ENDPOINTS
# ================================================

@router.post("/upload-pdfs", response_model=UploadResponse)
async def upload_pdf_files(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(1000),
    chunk_overlap: Optional[int] = Form(200)
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
        metadata: JSON string containing regulation metadata for each file
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
    
    # Parse metadata if provided
    files_metadata = {}
    if metadata:
        try:
            import json
            files_metadata = json.loads(metadata)
            logger.info(f"Parsed metadata for {len(files_metadata)} files: {files_metadata}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")
    else:
        logger.warning("No metadata provided in request")
    
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
            results = custom_pipeline.process_batch(files, files_metadata)
        else:
            # Use the default pipeline
            results = pipeline.process_batch(files, files_metadata)
        
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
    
# ================================================
# COMPLIANCE ENDPOINTS
# ================================================

@router.post("/compliance/check", response_model=ComplianceResponse)
async def check_compliance(
    title: str = Form(...),
    description: str = Form(...),
    document: Optional[UploadFile] = File(None)
):
    """
    Enhanced compliance check endpoint with optional document upload
    
    Args:
        title: Feature name/title
        description: Feature description
        document: Optional feature document (PDF, TXT, DOCX, MD)
    
    Returns:
        ComplianceResponse with analysis results
    """
    try:
        logger.info(f"Starting compliance check for feature: {title}")
        
        # Process uploaded document if provided
        document_content = None
        document_metadata = None
        
        if document:
            logger.info(f"Processing uploaded document: {document.filename}")
            
            # Read document content
            document_bytes = await document.read()
            
            # Extract text based on file type
            document_content = await process_document(document_bytes, document.filename)
            document_metadata = {
                "filename": document.filename,
                "size": len(document_bytes),
                "content_type": document.content_type
            }
        
        # Call the LangGraph multi-agent compliance analysis system
        analysis_result = await run_compliance_analysis(
            title=title,
            description=description,
            document_content=document_content,
            document_metadata=document_metadata
        )
        
        logger.info(f"Compliance analysis completed for feature: {title}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in compliance check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Compliance analysis failed: {str(e)}")


@router.post("/compliance/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accepts client-side snapshot state + feedback; applies LearningAgent updates.
    """
    try:
        logger.info(f"Feedback received: analysis_id={feedback.analysis_id}, type={feedback.feedback_type}")


        # Basic validation
        if not feedback.state:
            raise HTTPException(status_code=400, detail="Missing `state` in request body.")


        # Build LearningAgent input from request
        la_state = build_learning_state_from_request(feedback)


        # Run learning
        agent = LearningAgent()  # reads PG_CONN_STRING, writes Postgres + JSONLs
        learning_result = await agent.process(la_state)


        logger.info(f"Learning applied for analysis_id={feedback.analysis_id}")
        return FeedbackResponse(
            status="success",
            message="Feedback applied; memory updated.",
            analysis_id=feedback.analysis_id,
            learning_report=learning_result.get("learning_report"),
        )


    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feedback processing failed")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {e}")



# # ================================================
# # REGULATION UPLOAD ENDPOINTS
# # ================================================

# @router.post("/upload-pdfs", response_model=Dict[str, Any])
# async def upload_regulation_pdfs(files: List[UploadFile] = File(...)):
#     """
#     Batch upload PDF regulation documents
    
#     Args:
#         files: List of PDF files to upload and process
    
#     Returns:
#         Upload results with statistics
#     """
#     try:
#         logger.info(f"Starting batch upload of {len(files)} PDF files")
        
#         upload_results = await process_pdf_batch_upload(files)
        
#         logger.info(f"Batch upload completed: {upload_results['successful']} successful, {upload_results['failed']} failed")
#         return upload_results
        
#     except Exception as e:
#         logger.error(f"Error in batch PDF upload: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")


# @router.get("/upload-stats", response_model=Dict[str, Any])
# async def get_upload_statistics():
#     """
#     Get statistics about the PDF upload pipeline
    
#     Returns:
#         Pipeline configuration and storage statistics
#     """
#     try:
#         logger.info("Fetching upload pipeline statistics")
        
#         stats = await get_pipeline_stats()
        
#         logger.info("Upload statistics retrieved successfully")
#         return stats
        
#     except Exception as e:
#         logger.error(f"Error fetching upload stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")


# # ================================================
# # ANALYTICS ENDPOINTS
# # ================================================

# @router.get("/analytics/summary", response_model=Dict[str, Any])
# async def get_analytics_summary():
#     """
#     Get summary analytics about compliance analyses
    
#     Returns:
#         Summary statistics and trends
#     """
#     try:
#         logger.info("Fetching analytics summary")
        
#         summary = await get_compliance_analytics_summary()
        
#         logger.info("Analytics summary retrieved successfully")
#         return summary
        
#     except Exception as e:
#         logger.error(f"Error fetching analytics summary: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


# @router.get("/analytics/history", response_model=List[Dict[str, Any]])
# async def get_analysis_history(
#     limit: Optional[int] = 100,
#     offset: Optional[int] = 0,
#     filter_flagged: Optional[bool] = None
# ):
#     """
#     Get historical compliance analyses
    
#     Args:
#         limit: Maximum number of records to return
#         offset: Number of records to skip
#         filter_flagged: If True, only return flagged analyses; if False, only non-flagged
    
#     Returns:
#         List of historical analyses
#     """
#     try:
#         logger.info(f"Fetching analysis history: limit={limit}, offset={offset}")
        
#         history = await get_compliance_history(limit, offset, filter_flagged)
        
#         logger.info(f"Retrieved {len(history)} historical analyses")
#         return history
        
#     except Exception as e:
#         logger.error(f"Error fetching analysis history: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


# # ================================================
# # ADMIN ENDPOINTS
# # ================================================

# @router.get("/admin/system-status", response_model=Dict[str, Any])
# async def get_system_status():
#     """
#     Get system health and status information
    
#     Returns:
#         System status including agent health, database connectivity, etc.
#     """
#     try:
#         logger.info("Checking system status")
        
#         status = await check_system_health()
        
#         logger.info("System status check completed")
#         return status
        
#     except Exception as e:
#         logger.error(f"Error checking system status: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")


# @router.post("/admin/retrain-agents", response_model=StatusResponse)
# async def retrain_agents():
#     """
#     Trigger retraining of compliance analysis agents based on feedback
    
#     Returns:
#         Status of the retraining process
#     """
#     try:
#         logger.info("Starting agent retraining process")
        
#         retraining_result = await trigger_agent_retraining()
        
#         logger.info("Agent retraining process initiated")
#         return StatusResponse(
#             status="success",
#             message="Agent retraining process started successfully",
#             data=retraining_result
#         )
        
#     except Exception as e:
#         logger.error(f"Error initiating agent retraining: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Agent retraining failed: {str(e)}")


# # ================================================
# # DUMMY IMPLEMENTATION FUNCTIONS
# # ================================================
# # These are placeholder functions that should be implemented with your actual logic

async def process_document(document_bytes: bytes, filename: str) -> str:
    """
    Process uploaded document and extract text content
    
    Supports:
    - PDF text extraction using PyPDF2
    - Plain text and markdown handling
    - Basic document type validation
    """
    logger.info(f"Processing document {filename}")
    
    file_extension = filename.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        try:
            # Use PDF processor from the existing pipeline
            from app.rag.ingestion.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            
            # Create a temporary file-like object from bytes
            pdf_file_obj = io.BytesIO(document_bytes)
            text_content = processor._extract_text_from_file(pdf_file_obj, filename)
            logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
            return text_content
        except Exception as e:
            logger.error(f"Failed to process PDF {filename}: {str(e)}")
            return f"Error processing PDF {filename}: {str(e)}"
    
    elif file_extension in ['txt', 'md']:
        try:
            content = document_bytes.decode('utf-8')
            logger.info(f"Successfully processed text file {filename}")
            return content
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode text file {filename}: {str(e)}")
            return f"Error decoding text file {filename}: {str(e)}"
    
    else:
        error_msg = f"Unsupported file type: {file_extension}"
        logger.warning(error_msg)
        return error_msg


async def run_compliance_analysis(
    title: str, 
    description: str, 
    document_content: Optional[str] = None,
    document_metadata: Optional[Dict] = None
) -> ComplianceResponse:
    """
    Run the LangGraph multi-agent compliance analysis system
    
    This orchestrates the full workflow with:
    - Screening Agent: Initial risk assessment
    - Research Agent: RAG-powered research 
    - Validation Agent: Final decision validation
    - Learning Agent: Feedback integration
    """
    logger.info(f"=== Starting Compliance Analysis ===")
    logger.info(f"Feature: {title}")
    logger.info(f"Description length: {len(description)} characters")
    logger.info(f"Document content provided: {document_content is not None}")
    logger.info(f"Document metadata: {document_metadata}")
    
    try:
        # Prepare context documents if provided
        context_documents = None
        if document_content:
            context_documents = {
                "content": document_content,
                "metadata": document_metadata or {}
            }
            logger.info(f"Context documents prepared: {len(document_content)} characters")
        else:
            logger.info("No context documents provided")
        
        # Log orchestrator status
        logger.info(f"Orchestrator memory overlay length: {len(compliance_orchestrator.memory_overlay) if compliance_orchestrator.memory_overlay else 0}")
        if compliance_orchestrator.memory_overlay and "TIKTOK TERMINOLOGY REFERENCE" in compliance_orchestrator.memory_overlay:
            logger.info("Orchestrator has TikTok terminology context")
        else:
            logger.warning("Orchestrator missing TikTok terminology context")
        
        # Execute the LangGraph multi-agent workflow
        logger.info("Executing LangGraph multi-agent workflow...")
        orchestrator_result = await compliance_orchestrator.analyze_feature(
            feature_name=title,
            feature_description=description,
            context_documents=context_documents
        )
        logger.info("LangGraph workflow execution completed")
        
        # Log orchestrator results
        logger.info(f"Orchestrator result keys: {list(orchestrator_result.keys())}")
        logger.info(f"Agents completed: {orchestrator_result.get('agents_completed', [])}")
        
        # Map orchestrator result to API response format
        # Extract agent-specific results for detailed UI display with safe access
        screening_analysis = orchestrator_result.get("screening_analysis")
        research_analysis = orchestrator_result.get("research_analysis")
        validation_analysis = orchestrator_result.get("validation_analysis")
        
        logger.info(f"Screening analysis present: {screening_analysis is not None}")
        logger.info(f"Research analysis present: {research_analysis is not None}")
        logger.info(f"Validation analysis present: {validation_analysis is not None}")
        
        # Safely extract agents completed list
        agents_completed = orchestrator_result.get("agents_completed", [])
        if not isinstance(agents_completed, list):
            agents_completed = []
            logger.warning("Agents completed is not a list, defaulting to empty")
        
        # Determine agent statuses with error handling
        agent_statuses = {
            "screening": "completed" if "screening" in agents_completed else "pending",
            "research": "completed" if "research" in agents_completed else "pending",
            "validation": "completed" if "validation" in agents_completed else "pending"
        }
        logger.info(f"Agent statuses: {agent_statuses}")
        
        # Handle workflow completion status
        workflow_completed = orchestrator_result.get("workflow_completed", False)
        if workflow_completed is None:
            workflow_completed = False
            logger.warning("Workflow completed is None, defaulting to False")
        
        # Safely extract related regulations
        related_regulations = orchestrator_result.get("related_regulations", [])
        if not isinstance(related_regulations, list):
            related_regulations = []
            logger.warning("Related regulations is not a list, defaulting to empty")
        
        # Safely extract applicable jurisdictions
        applicable_jurisdictions = orchestrator_result.get("applicable_jurisdictions", [])
        if not isinstance(applicable_jurisdictions, list):
            applicable_jurisdictions = []
            logger.warning("Applicable jurisdictions is not a list, defaulting to empty")
        
        # Log confidence score
        confidence_score = float(orchestrator_result.get("confidence_score", 0.0))
        logger.info(f"Confidence score: {confidence_score}")
        
        logger.info("=== Compliance Analysis Completed Successfully ===")
        
        return ComplianceResponse(
            status="success",
            message=f"Compliance analysis for '{title}' completed successfully.",
            analysis_id=str(uuid.uuid4()), # Generate a new ID for each analysis
            timestamp=datetime.now().isoformat(),
            
            # Agent-specific results
            screening_result=screening_analysis,
            research_result=research_analysis,
            validation_result=validation_analysis,
            
            # Overall workflow results
            agents_completed=agents_completed,
            workflow_status="completed",
            confidence_score=confidence_score,
            
            error=orchestrator_result.get("error")
        )
        
    except Exception as e:
        logger.error(f"=== Compliance Analysis Failed ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return ComplianceResponse(
            status="error",
            message=f"Compliance analysis failed for '{title}': {str(e)}",
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            
            # Agent-specific error state
            screening_result=None,
            research_result=None,
            validation_result=None,
            
            # Overall workflow results
            agents_completed=[],
            workflow_status="failed",
            confidence_score=0.0,
            
            error=str(e)
        )


# async def process_feedback(feedback: FeedbackRequest) -> FeedbackResponse:
#     """
#     DUMMY: Process feedback for agent learning
    
#     TODO: Implement actual feedback processing:
#     - Store feedback in database
#     - Update agent training data
#     - Trigger incremental learning if needed
#     """
#     logger.info(f"DUMMY: Processing {feedback.feedback_type} feedback for analysis {feedback.analysis_id}")
    
#     # Simulate feedback processing
#     feedback_id = str(uuid.uuid4())
    
#     # TODO: Store feedback in database
#     # TODO: Update agent learning parameters
#     # TODO: If negative feedback, add to correction dataset
    
#     return FeedbackResponse(
#         feedback_id=feedback_id,
#         status="processed",
#         message=f"Feedback received and will be used to improve analysis quality. Type: {feedback.feedback_type}",
#         analysis_id=feedback.analysis_id,
#         timestamp=datetime.now().isoformat()
#     )


# async def process_pdf_batch_upload(files: List[UploadFile]) -> Dict[str, Any]:
#     """
#     DUMMY: Process batch PDF upload through RAG pipeline
    
#     TODO: Implement actual PDF processing:
#     - Extract text from PDFs
#     - Chunk documents appropriately
#     - Generate embeddings
#     - Store in vector database (ChromaDB)
#     """
#     logger.info(f"DUMMY: Processing batch upload of {len(files)} PDFs")
    
#     results = []
#     successful = 0
#     failed = 0
    
#     for file in files:
#         try:
#             # Simulate processing each file
#             file_bytes = await file.read()
            
#             # TODO: Extract text from PDF
#             # TODO: Chunk text appropriately
#             # TODO: Generate embeddings
#             # TODO: Store in ChromaDB
            
#             chunks_processed = len(file_bytes) // 1000  # Dummy chunk calculation
            
#             results.append({
#                 "filename": file.filename,
#                 "status": "success",
#                 "chunks_processed": chunks_processed,
#                 "size_bytes": len(file_bytes)
#             })
#             successful += 1
            
#         except Exception as e:
#             results.append({
#                 "filename": file.filename,
#                 "status": "failed",
#                 "error": str(e)
#             })
#             failed += 1
    
#     return {
#         "status": "success" if failed == 0 else ("partial_success" if successful > 0 else "failed"),
#         "message": f"Processed {successful} files successfully, {failed} failed",
#         "total_files": len(files),
#         "successful": successful,
#         "failed": failed,
#         "results": results,
#         "timestamp": datetime.now().isoformat()
#     }


# async def get_pipeline_stats() -> Dict[str, Any]:
#     """
#     DUMMY: Get PDF processing pipeline statistics
    
#     TODO: Implement actual pipeline stats:
#     - ChromaDB collection info
#     - Embedding model details
#     - Processing configuration
#     """
#     logger.info("DUMMY: Fetching pipeline statistics")
    
#     return {
#         "pipeline_config": {
#             "chunk_size": 1000,
#             "chunk_overlap": 200,
#             "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#             "collection_name": "regulations_collection"
#         },
#         "storage_stats": {
#             "document_count": 156,  # Dummy count
#             "total_chunks": 2847,   # Dummy count
#             "status": "healthy",
#             "last_updated": datetime.now().isoformat()
#         }
#     }


# async def get_compliance_analytics_summary() -> Dict[str, Any]:
#     """
#     DUMMY: Get compliance analytics summary
    
#     TODO: Implement actual analytics:
#     - Query database for analysis statistics
#     - Calculate trends and patterns
#     - Generate insights
#     """
#     logger.info("DUMMY: Fetching compliance analytics summary")
    
#     return {
#         "total_analyses": 245,
#         "flagged_analyses": 89,
#         "flagged_percentage": 36.3,
#         "average_confidence": 0.847,
#         "risk_distribution": {
#             "low": 156,
#             "medium": 67,
#             "high": 22
#         },
#         "feedback_stats": {
#             "positive_feedback": 198,
#             "negative_feedback": 23,
#             "context_requests": 24
#         },
#         "trends": {
#             "analyses_this_week": 47,
#             "trend_direction": "increasing",
#             "accuracy_improvement": 0.12
#         },
#         "timestamp": datetime.now().isoformat()
#     }


# async def get_compliance_history(
#     limit: int, 
#     offset: int, 
#     filter_flagged: Optional[bool]
# ) -> List[Dict[str, Any]]:
#     """
#     DUMMY: Get compliance analysis history
    
#     TODO: Implement actual database query:
#     - Query analyses table with pagination
#     - Apply filters
#     - Return formatted results
#     """
#     logger.info(f"DUMMY: Fetching compliance history with limit={limit}, offset={offset}")
    
#     # Dummy historical data
#     dummy_analyses = []
#     for i in range(min(limit, 20)):  # Return up to 20 dummy records
#         flagged = (i % 3 == 0)  # Every third analysis is flagged
        
#         if filter_flagged is not None and flagged != filter_flagged:
#             continue
            
#         dummy_analyses.append({
#             "analysis_id": f"analysis_dummy_{i + offset}",
#             "timestamp": datetime.now().isoformat(),
#             "title": f"Feature Analysis {i + offset + 1}",
#             "description": f"Analysis of feature with various compliance considerations...",
#             "flag": "yes" if flagged else "no",
#             "confidence": 0.75 + (i % 10) * 0.02,
#             "risk_level": "High" if flagged else "Low",
#             "has_feedback": (i % 5 == 0)  # Every fifth analysis has feedback
#         })
    
#     return dummy_analyses


# async def check_system_health() -> Dict[str, Any]:
#     """
#     DUMMY: Check system health status
    
#     TODO: Implement actual health checks:
#     - Database connectivity
#     - Agent service status
#     - Vector database status
#     - API dependencies
#     """
#     logger.info("DUMMY: Performing system health check")
    
#     return {
#         "status": "healthy",
#         "components": {
#             "database": {
#                 "status": "healthy",
#                 "response_time_ms": 45,
#                 "last_check": datetime.now().isoformat()
#             },
#             "vector_database": {
#                 "status": "healthy",
#                 "collection_count": 1,
#                 "document_count": 156,
#                 "last_check": datetime.now().isoformat()
#             },
#             "llm_agents": {
#                 "status": "healthy",
#                 "screening_agent": "ready",
#                 "research_agent": "ready", 
#                 "validation_agent": "ready",
#                 "last_check": datetime.now().isoformat()
#             },
#             "external_apis": {
#                 "status": "healthy",
#                 "regulation_apis": "accessible",
#                 "last_check": datetime.now().isoformat()
#             }
#         },
#         "uptime_seconds": 86400,  # Dummy uptime
#         "version": "1.0.0",
#         "timestamp": datetime.now().isoformat()
#     }


# async def trigger_agent_retraining() -> Dict[str, Any]:
#     """
#     DUMMY: Trigger agent retraining process
    
#     TODO: Implement actual retraining:
#     - Collect recent feedback data
#     - Update agent parameters
#     - Retrain models incrementally
#     - Update agent configurations
#     """
#     logger.info("DUMMY: Triggering agent retraining process")
    
#     return {
#         "retraining_id": str(uuid.uuid4()),
#         "status": "initiated",
#         "feedback_samples": 47,  # Dummy count
#         "estimated_completion": "2025-01-15T10:30:00Z",
#         "agents_affected": [
#             "screening_agent",
#             "research_agent", 
#             "validation_agent"
#         ],
#         "timestamp": datetime.now().isoformat()
#     }

def to_learning_user_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """
    Only map to LearningAgent format:
      - is_correct: 'yes' if correction_data.correct_flag != original_result.flag, else 'no'
      - notes: exactly feedback_text
    """

    feedback_type = (req.feedback_type or "").strip()
    is_correct = "yes" if (feedback_type == "positive") else "no"
    return {
        "is_correct": is_correct,
        "notes": (req.feedback_text or "").strip()
    }


def build_learning_state_from_request(req: FeedbackRequest) -> Dict[str, Any]:
    """
    Extract the pieces LearningAgent expects from the provided state object.
      Required keys for LearningAgent:
        - feature_name
        - feature_description
        - screening_analysis (dict)
        - research_analysis (dict)
        - validation_analysis (dict using the NEW ValidationOutput schema)
        - user_feedback (normalized)
    """
    s = req.state or {}
    print(to_learning_user_feedback(req))
    return {
        "feature_name": s.get("feature_name", ""),
        "feature_description": s.get("feature_description", ""),
        "screening_analysis": s.get("screening_analysis", {}) or {},
        "research_analysis": s.get("research_analysis", {}) or {},
        "validation_analysis": s.get("validation_analysis", {}) or {},
        "user_feedback": to_learning_user_feedback(req),
    }
