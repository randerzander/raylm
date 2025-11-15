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

## Installation

Using uv (recommended):

```bash
uv sync
```

Or manually:

```bash
pip install ray pandas rich pypdfium2 requests openai lancedb pillow markitdown
```

## Setup

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

## Usage

### Process Files

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
