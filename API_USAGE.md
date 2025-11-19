# RayLM API Server Usage

## Overview
This is a simplified FastAPI server with only the `/submit` endpoint for document ingestion.

## Installation

First, install the dependencies:

```bash
pip install -e .
```

## Running the Server

### Method 1: Direct Python execution
```bash
cd /localhome/local-jdyer/raylm
python src/api.py
```

### Method 2: With command-line options
```bash
python src/api.py --host 0.0.0.0 --port 8000 --log-level info
```

### Method 3: Using the installed script (after `pip install -e .`)
```bash
raylm-api --host 0.0.0.0 --port 8000
```

### Development mode with auto-reload
```bash
python src/api.py --reload
```

## Command-Line Options

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Log level - debug, info, warning, error (default: info)

## API Endpoints

### 1. Root Endpoint
**GET /** - Service information

```bash
curl http://localhost:8000/
```

### 2. Health Check
**GET /health** - Health check endpoint

```bash
curl http://localhost:8000/health
```

### 3. Submit Document
**POST /submit** - Submit a document for processing

```bash
curl -X POST http://localhost:8000/submit \
  -F "file=@/path/to/document.pdf"
```

Example response:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "document_type": "pdf",
  "file_size": 12345,
  "status": "submitted",
  "message": "File 'document.pdf' successfully submitted for processing",
  "timestamp": 1700000000.0
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Test the endpoint with a PDF file:
```bash
# Assuming you have a test PDF
curl -X POST http://localhost:8000/submit \
  -F "file=@test.pdf" \
  | jq .
```

## Supported File Types

Currently supports:
- PDF (.pdf)
- Text (.txt)
- HTML (.html)

Unknown file types will be treated as PDF by default.

