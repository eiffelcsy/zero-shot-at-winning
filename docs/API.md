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
| title | string | Yes | Feature name/title (1-200 characters) |
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
  "analysis_id": "string",
  "flag": "string", 
  "confidence_score": 0.92,
  "risk_level": "string",
  "reasoning": "string",
  "related_regulations": ["string"],
  "agent_details": {
    "screening_agent_status": "completed",
    "research_agent_status": "completed",
    "validation_agent_status": "completed",
    "document_processed": true,
    "document_metadata": {
      "filename": "string",
      "size": 0,
      "content_type": "string"
    }
  },
  "timestamp": "2025-01-15T14:30:22.123Z"
}
```

**Success Response Examples:**
- Feature Requires Compliance
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "flag": "yes",
  "confidence_score": 0.92,
  "risk_level": "High",
  "reasoning": "This feature requires geo-specific compliance logic because it reads user location data to enforce region-specific copyright restrictions. The blocking of downloads based on geographic location directly relates to Utah's specific age verification requirements.",
  "related_regulations": [
    "EU Digital Service Act (DSA) - Article 14",
    "Utah Social Media Regulation Act",
    "GDPR - Location data processing requirements"
  ],
  "agent_details": {
    "screening_agent_status": "completed",
    "research_agent_status": "completed",
    "validation_agent_status": "completed",
    "document_processed": true,
    "document_metadata": {
      "filename": "feature_spec.pdf",
      "size": 245760,
      "content_type": "application/pdf"
    }
  },
  "timestamp": "2025-01-15T14:30:22.123Z"
}
```
- Feature Does Not Require Compliance
```bash
{
  "analysis_id": "analysis_20250115_143045_2",
  "flag": "no",
  "confidence_score": 0.78,
  "risk_level": "Low",
  "reasoning": "This feature does not require geo-specific compliance logic because it operates uniformly across regions without location dependencies or region-specific data processing requirements.",
  "related_regulations": [
    "General platform terms of service",
    "Standard content policies"
  ],
  "agent_details": {
    "screening_agent_status": "completed",
    "research_agent_status": "completed",
    "validation_agent_status": "completed",
    "document_processed": false
  },
  "timestamp": "2025-01-15T14:30:45.456Z"
}
```
**Error Responses:**
- 400 Bad Request
```bash
{"detail": "Title and description are required fields"}
```
- 500 Internal Server Error
```bash
{"detail": "Compliance analysis failed: Agent orchestration error"}
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
  "correction_data": {
    "correct_flag": "string",
    "correct_risk_level": "string",
    "original_result": {}
  },
  "timestamp": "2025-01-15T14:35:00.000Z"
}
```

**Request Examples:**
- Positive Feedback
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "feedback_type": "positive",
  "timestamp": "2025-01-15T14:35:00.000Z"
}
```
- Negative Feedback with Correction
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "feedback_type": "negative",
  "feedback_text": "This feature should NOT require geo-compliance because it doesn't process location data or implement region-specific restrictions.",
  "correction_data": {
    "correct_flag": "no",
    "correct_risk_level": "Low",
    "original_result": {
      "flag": "yes",
      "risk_level": "High"
    }
  },
  "timestamp": "2025-01-15T14:35:00.000Z"
}
```
- Context Improvement Request
```bash
{
  "analysis_id": "analysis_20250115_143022_1",
  "feedback_type": "needs_context",
  "feedback_text": "Analysis should consider specific industry regulations and cross-border data transfer requirements.",
  "timestamp": "2025-01-15T14:35:00.000Z"
}
```
- Success Response:
```bash
{
  "feedback_id": "feedback_uuid_12345",
  "status": "processed",
  "message": "Feedback received and will be used to improve analysis quality. Type: negative",
  "analysis_id": "analysis_20250115_143022_1",
  "timestamp": "2025-01-15T14:35:22.123Z"
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

### 4. Clear Document Collection
**Endpoint:** `DELETE /clear-documents`

**Description:** Permanently delete all documents from ChromaDB.

**Success Response:**
```bash
{
  "status": "success",
  "message": "Successfully cleared all documents from collection",
  "documents_deleted": 156,
  "chunks_deleted": 8934,
  "timestamp": "2024-08-28T10:30:00Z"
}
```
