# API Documentation

## Overview
This API provides endpoints for automated geo-regulation compliance analysis using a multi-agent system. The system analyzes features to determine if they require location-specific compliance logic and provides audit-ready reasoning.

**Base URL:** `http://localhost:8000/api/v1`  
**Content-Type:** `application/json` (except for file uploads)

### Endpoint Categories
- **Compliance Endpoints**
  - `POST /compliance/check` - Enhanced with document upload support
  - `POST /compliance/feedback` - Supports positive, negative, and context feedback types
- **Upload Endpoints**
  - `POST /upload-pdfs` - Batch PDF processing with detailed results
  - `DELETE /clear-documents` - Clear all documents from the ChromaDB collection

---

## Compliance Endpoints

### 1. Check Feature Compliance
**Endpoint:** `POST /compliance/check`  

**Description:** Analyzes a feature to determine if it requires geo-specific compliance logic. Supports optional document upload.  

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| title | string | Yes | Feature name (1-200 characters) |
| description | string | Yes | Detailed feature description (1-5000 characters) |
| document | file | No | Optional feature document (PDF, TXT, DOCX, MD) |

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/compliance/check" \
  -F "title=Curfew login blocker with ASL and GH for Utah minors" \
  -F "description=To comply with the Utah Social Media Regulation Act, we are implementing a curfew-based login restriction for users under 18 in Utah. The feature uses geolocation to determine user location and applies time-based access restrictions." \
  -F "document=@feature_spec.pdf"
```
**Response Schema:**
```bash
{
  "status": "string",
  "message": "string",
  "analysis_id": "string",
  "timestamp": "2025-08-31T12:34:56Z",
  "screening_result": { ... },
  "research_result": { ... },
  "validation_result": { ... },
  "agents_completed": [ "string" ],
  "workflow_status": "string",
  "confidence_score": "float",
  "error": "string"
}
```

**Success Response Examples:**
```bash
{
  "status": "success",
  "message": "Compliance check completed",
  "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2025-08-31T12:34:56Z",
  "screening_result": { ... },
  "research_result": { ... },
  "validation_result": { ... },
  "agents_completed": ["ScreeningAgent", "ResearchAgent", "ValidationAgent"],
  "workflow_status": "completed",
  "confidence_score": 0.92,
  "error": null
}
```

**Error Responses:**
- 400 Bad Request
```bash
{"detail": "Title and description are required fields"}
```
- 500 Internal Server Error
```bash
{"detail": "Compliance analysis failed: {specific error message}"}
```

### 2. Submit Feedback
**Endpoint:** `POST /compliance/feedback`  

**Description:** Submit feedback on a compliance analysis to improve the multi-agent system.

**Content-Type:** `application/json`

**Request Schema:**
```bash
{
  "analysis_id": "string",
  "feedback_type": "string",
  "feedback_text": "string",
  "timestamp": "2025-01-15T14:35:00.000Z"
  "state": {
      "feature_name": "string",
      "feature_description": "string",
      "screening_analysis": "string",
      "research_analysis": "string",
      "validation_analysis": "string"
      }
}
```

**Request Examples:**
- Positive Feedback
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "feedback_type": "positive",
  "timestamp": "2025-01-15T14:35:00.000Z"
  "state": {
      "feature_name": "string",
      "feature_description": "string",
      "screening_analysis": "string",
      "research_analysis": "string",
      "validation_analysis": "string"
      }
}
```
- Negative Feedback with Reasoning
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "feedback_type": "negative",
  "feedback_text": "This feature should NOT require geo-compliance because it doesn't process location data or implement region-specific restrictions.",
  "timestamp": "2025-01-15T14:35:00.000Z",
  "state": {
            "feature_name": "string",
            "feature_description": "string",
            "screening_analysis": "string",
            "research_analysis": "string",
            "validation_analysis": "string"
            }
}
```

### 3. Batch Upload PDF Regulations
**Endpoint:** `POST /upload-pdfs`  

**Description:** Upload and process multiple PDF files through the ingestion pipeline.

**Parameters:**
| Parameter      | Type              | Required | Default | Description                    |
| -------------- | ----------------- | -------- | ------- | ------------------------------ |
| files          | List\[UploadFile] | Yes      | -       | List of PDF files              |
| metadata       | string            | No       | null    | JSON string with file metadata |
| chunk\_size    | integer           | No       | 1000    | Custom chunk size              |
| chunk\_overlap | integer           | No       | 200     | Custom chunk overlap           |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "metadata={\"document1.pdf\": {\"type\": \"regulation\", \"category\": \"safety\"}, \"document2.pdf\": {\"type\": \"policy\", \"category\": \"compliance\"}}" \
  -F "chunk_size=1200" \
  -F "chunk_overlap=150"
```
**Success Response Example:**
```bash
{
  "status": "success",
  "message": "Successfully processed all 2 PDF files",
  "results": [
    {
      "filename": "document1.pdf",
      "status": "success",
      "chunks_created": 45,
      "processing_time": 12.5,
      "file_size": 1024000,
      "metadata": {"type": "regulation","category": "safety"}
    }
  ],
  "total_files": 2,
  "successful": 2,
  "failed": 0
}
```
**Error Response Example:**
```bash
{
    "detail": "No files provided"
}
```

### 4. Clear Document Collection
**Endpoint:** `DELETE /clear-documents`

**Description:** Permanently delete all documents from ChromaDB.

**Success Response:**
```bash
{
  "status": "success",       
  "message": "All documents cleared successfully",  
  "deleted_count": 42
}
```

**Error Response:**
```bash
{
    "detail": "Error clearing document collection: {specific error message}"
}
```
