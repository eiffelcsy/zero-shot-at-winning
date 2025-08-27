import logging
from typing import Dict, Any, List
from fastapi import UploadFile

from rag.ingestion.pdf_processor import PDFProcessor, PDFValidationError
from rag.ingestion.text_chunker import TextChunker
from rag.ingestion.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class PDFIngestionPipeline:
    """
    End-to-end pipeline for PDF document ingestion into ChromaDB.
    
    This pipeline handles the complete workflow:
    1. PDF processing and text extraction
    2. Text chunking with metadata
    3. Embedding generation
    4. Storage in ChromaDB
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-large",
        collection_name: str = "rag_collection",
        batch_size: int = 300
    ):
        """
        Initialize the PDF ingestion pipeline.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            embedding_model: OpenAI embedding model to use
            collection_name: ChromaDB collection name
            batch_size: Number of chunks to store in ChromaDB per batch
        """
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.vector_storage = VectorStorage(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        self.batch_size = batch_size
        
        logger.info(f"Initialized PDFIngestionPipeline with chunk_size={chunk_size}, "
                   f"overlap={chunk_overlap}, model={embedding_model}, batch_size={batch_size}")
    
    def process_pdf(self, pdf_file: UploadFile) -> Dict[str, Any]:
        """
        Process a single uploaded PDF file through the complete ingestion pipeline.
        
        Args:
            pdf_file: FastAPI UploadFile object from Streamlit file uploader
            
        Returns:
            Dictionary containing processing results and statistics
        """
        try:
            logger.info(f"Starting PDF processing pipeline for: {getattr(pdf_file, 'filename', pdf_file)}")
            
            # Step 1: Extract text and metadata from uploaded PDF
            extracted_text = self.pdf_processor.load_pdf(pdf_file)
            metadata = self.pdf_processor.extract_metadata(pdf_file)
            
            # Step 2: Chunk the extracted text
            source_id = metadata.get('filename', 'unknown_document')
            chunks = self.text_chunker.chunk_text(extracted_text, source_id=source_id)
            
            if not chunks:
                return {
                    'status': 'warning',
                    'message': 'No text chunks generated from PDF',
                    'filename': source_id,
                    'chunks_processed': 0
                }
            
            # Step 3: Generate embeddings and store in ChromaDB
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.vector_storage.generate_embeddings(chunk_texts)
            
            # Prepare metadata for storage
            chunk_metadatas = []
            for chunk in chunks:
                chunk_metadata = {
                    **metadata,  # Include original PDF metadata
                    **chunk.metadata,  # Include chunk-specific metadata
                    'processing_pipeline': 'PDFIngestionPipeline',
                    'chunk_content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content
                }
                chunk_metadatas.append(chunk_metadata)
            
            # Step 4: Store chunks with embeddings in batches
            document_ids = self.vector_storage.store_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                batch_size=self.batch_size
            )
            
            # Create success result
            result = {
                'status': 'success',
                'filename': source_id,
                'chunks_processed': len(chunks),
                'document_ids': document_ids,
                'text_length': len(extracted_text),
                'metadata': metadata,
                'chunk_stats': self.text_chunker.get_chunk_stats(chunks),
                'processing_details': {
                    'pdf_pages': metadata.get('page_count', 'unknown'),
                    'text_extraction_successful': True,
                    'embeddings_generated': len(embeddings),
                    'storage_successful': True
                }
            }
            
            logger.info(f"Successfully processed PDF {source_id}: {len(chunks)} chunks stored")
            return result
            
        except (ValueError, PDFValidationError) as e:
            error_msg = f"PDF processing error: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'filename': getattr(pdf_file, 'filename', str(pdf_file)),
                'chunks_processed': 0
            }
        except Exception as e:
            error_msg = f"Unexpected error in PDF processing pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'filename': getattr(pdf_file, 'filename', str(pdf_file)),
                'chunks_processed': 0
            }
    
    def process_batch(self, pdf_files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Process multiple uploaded PDF files through the ingestion pipeline.
        
        Args:
            pdf_files: List of FastAPI UploadFile objects from Streamlit file uploader
            
        Returns:
            List of dictionaries containing processing results for each file
        """
        if not pdf_files:
            logger.warning("Empty PDF file list provided for batch processing")
            return []
        
        logger.info(f"Starting batch processing of {len(pdf_files)} PDF files")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                logger.info(f"Processing batch item {i+1}/{len(pdf_files)}: "
                           f"{getattr(pdf_file, 'filename', pdf_file)}")
                
                result = self.process_pdf(pdf_file)
                results.append(result)
                
                if result['status'] == 'success':
                    successful_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'error': f"Batch processing error: {str(e)}",
                    'filename': getattr(pdf_file, 'filename', str(pdf_file)),
                    'chunks_processed': 0
                }
                results.append(error_result)
                failed_count += 1
                logger.error(f"Failed to process PDF {i+1}: {str(e)}")
        
        logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
        
        # Add batch summary to each result
        for result in results:
            result['batch_summary'] = {
                'total_files': len(pdf_files),
                'successful': successful_count,
                'failed': failed_count
            }
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current pipeline state and ChromaDB collection.
        
        Returns:
            Dictionary containing pipeline and storage statistics
        """
        try:
            collection_stats = self.vector_storage.get_collection_stats()
            
            return {
                'pipeline_config': {
                    'chunk_size': self.text_chunker.chunk_size,
                    'chunk_overlap': self.text_chunker.overlap,
                    'embedding_model': self.vector_storage.embedding_model,
                    'collection_name': self.vector_storage.collection_name,
                    'batch_size': self.batch_size
                },
                'storage_stats': collection_stats,
                'status': 'operational'
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {
                'pipeline_config': {
                    'chunk_size': self.text_chunker.chunk_size,
                    'chunk_overlap': self.text_chunker.overlap,
                    'embedding_model': self.vector_storage.embedding_model,
                    'collection_name': self.vector_storage.collection_name,
                    'batch_size': self.batch_size
                },
                'storage_stats': {'error': str(e)},
                'status': 'error'
            }
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the ChromaDB collection.
        
        Returns:
            Dictionary containing operation results
        """
        try:
            success = self.vector_storage.clear_collection()
            return {
                'status': 'success' if success else 'error',
                'message': 'Collection cleared successfully' if success else 'Failed to clear collection',
                'collection_name': self.vector_storage.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return {
                'status': 'error',
                'message': f'Error clearing collection: {str(e)}',
                'collection_name': self.vector_storage.collection_name
            }
