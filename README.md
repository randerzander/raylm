# raylm

Distributed PDF, HTML, and text file parsing and embedding pipeline using Ray Data, NVIDIA NeMo, and LanceDB.

## Overview

Multi-stage pipeline for document processing with automatic file type detection:

**Text Files (.txt):**
1. **Chunk** - Split on ~4096 token boundaries
2. **Embed** - Generate vectors via NVIDIA NeMo Retriever
3. **Store** - Save to LanceDB

**HTML Files (.html, .htm):**
1. **Convert** - HTML to markdown with MarkItDown
2. **Embed** - Markdown to vectors via NVIDIA NeMo Retriever
3. **Store** - Save to LanceDB

**PDFs (.pdf):**
1. **Split** - PDFs into individual pages
2. **Render** - Pages to JPEG images
3. **Parse** - Images to markdown via NVIDIA NeMo Parse
4. **Embed** - Markdown to vectors via NVIDIA NeMo Retriever
5. **Store** - Vectors in LanceDB

All stages run in parallel using Ray Data streaming pipeline.

## Features

- **Unified pipeline**: Mix PDFs, HTML, and text files in the same directory
- **Ray Data streaming**: Lazy execution with backpressure management
- **Parallel processing**: Configurable actor pools for embeddings
- **Bulk vector storage**: Efficient batch writes to LanceDB
- **RAG query interface**: Query with Nemotron LLM
- **Rich progress bars**: Visual pipeline execution tracking

## Requirements

- Python 3.8+
- NVIDIA API key

## ⚠️ Important: Nemotron Wheels Setup

**Before building the Docker image or installing the project**, you must create a `nemotron-wheels/` directory in the project root containing the following private wheel files:

```
nemotron-wheels/
├── nemotron_page_elements_v3-3.0.0-py3-none-any.whl
├── nemotron_graphic_elements_v1-1.0.0-py3-none-any.whl
├── nemotron_table_structure_v1-1.0.0-py3-none-any.whl
└── nemotron_ocr-1.0.0-py3-none-any.whl
```

### Required Wheels:

1. **nemotron-page-elements-v3** (>= 3.0.0) - Page element detection model
2. **nemotron-graphic-elements-v1** (>= 1.0.0) - Graphic element detection model  
3. **nemotron-table-structure-v1** (>= 1.0.0) - Table structure detection model
4. **nemotron-ocr** (>= 1.0.0) - OCR model

**Note:** These wheels are not publicly available and must be obtained separately. The Docker build will fail if this directory is missing or incomplete.

### Verification:

```bash
# Verify all required wheels are present
ls nemotron-wheels/*.whl

# Should show 4 .whl files
```

Once these wheels become publicly available, this requirement will be removed.

## Installation

### Option 1: Docker (Recommended for Production)

```bash
# Build the Docker image (ensure nemotron-wheels/ is present first)
docker build -t raylm-api:latest .

# Run the container
docker run -d --name raylm-api -p 8000:8000 raylm-api:latest

# Or use docker-compose
docker-compose up -d
```

See [QUICKSTART.md](QUICKSTART.md) and [DOCKER_USAGE.md](DOCKER_USAGE.md) for more details.

### Option 2: Local Installation

Using uv (recommended):

```bash
# Install nemotron wheels first
pip install nemotron-wheels/*.whl

# Install project
uv sync
```

Or manually:

```bash
# Install nemotron wheels first
pip install nemotron-wheels/*.whl

# Install other dependencies
pip install ray pandas rich pypdfium2 requests openai lancedb pillow markitdown fastapi uvicorn
```

## Setup

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

## Usage

### FastAPI Server

Start the document ingestion API server:

```bash
# Direct execution
python src/api.py

# Or with custom options
python src/api.py --host 0.0.0.0 --port 8000 --log-level info

# Or using the installed script
raylm-api --port 8000
```

Submit documents via HTTP:

```bash
# Submit a document
curl -X POST http://localhost:8000/submit \
  -F "file=@document.pdf"

# Check health
curl http://localhost:8000/health
```

Interactive API docs available at: http://localhost:8000/docs

See [API_USAGE.md](API_USAGE.md) for complete API documentation.

### Process Files (Batch Pipeline)

Place PDFs, HTML, and/or text files in `data/` directory. Files are automatically processed based on extension:

```bash
python ingestor.py
```

For debugging without Ray:

```bash
python experiments/single_process_ingestor.py
```

### Query Documents

```bash
python query.py "What is this document about?"
```

Or interactive mode:

```bash
python query.py
```

## Output Structure

```
extracts/<file_name>/
├── pages_jpg/         # JPEG images (PDFs only)
├── pages_json/        # API responses (PDFs only)
├── pages_md/          # Markdown text (PDFs only)
├── html_md/           # Converted markdown (HTML only)
├── text_chunks/       # Text chunks (TXT only)
└── embeddings/        # Embedding vectors (all types)

lancedb/               # Vector database
scratch/               # Intermediate PDFs
```

## LanceDB Schema

Table: `document_embeddings`

- **source_id** (string) - Source filename (no extension)
- **chunk_sequence** (int) - Page/chunk number
- **text** (string) - Markdown or text content
- **vector** (list[float]) - 4096-dim embedding

## Query Script

The `query.py` script implements RAG:

1. Embeds user query with NeMo Retriever
2. Searches LanceDB for top 5 similar chunks
3. Generates answer with Nemotron Nano 9B using retrieved context

Output includes reasoning, answer, and source citations.

## Performance

Ray Data vs Single-Process (8 pages):
- **Single-process**: ~14.4s (0.55 pages/sec)
- **Ray Data**: ~22s (0.36 pages/sec)

For small workloads (<50 pages), single-process is faster. Ray Data's parallelism benefits large workloads (100+ pages).

## Architecture

- `ingestor.py` - Ray Data pipeline orchestrator
- `utils.py` - Pure functions for processing (split, render, parse, embed)
- `query.py` - RAG query interface
- `experiments/` - Alternative implementations (single-process, actor-based)

## Example Output

```
=== Summary ===
Files processed: 0 text files, 0 HTML files, 2 PDFs
Total chunks (pages): 8
Extracts saved to: extracts/
Page PDFs saved to: scratch/
LanceDB saved to: lancedb/ (8 records)

Timing:
  End-to-End - Total: 22.45s, Throughput: 0.36 items/sec
```

## License

Apache License 2.0
