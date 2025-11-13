# raylm

Distributed PDF and text file parsing and embedding pipeline using Ray, NVIDIA NeMo, and LanceDB.

## Overview

Multi-stage pipeline for PDF and text file processing:

**Text Files:**
1. **Chunk** - Split on ~4096 token boundaries
2. **Embed** - Generate vectors via NVIDIA NeMo Retriever
3. **Store** - Save to LanceDB

**PDFs:**
1. **Split** - PDFs into individual pages
2. **Render** - Pages to JPEG images
3. **Parse** - Images to markdown via NVIDIA NeMo Parse
4. **Embed** - Markdown to vectors via NVIDIA NeMo Retriever
5. **Store** - Vectors in LanceDB

All stages run in parallel using Ray.

## Features

- Parallel processing with Ray actors
- Text file chunking (~4096 token boundaries)
- Bulk vector storage with LanceDB
- Per-file output organization
- RAG query interface with Nemotron LLM
- Throughput metrics (items/sec)

## Requirements

- Python 3.8+
- NVIDIA API key

## Installation

Using uv:

```bash
uv add ray pypdfium2 requests openai lancedb pillow sentencepiece
```

Or pip:

```bash
pip install ray pypdfium2 requests openai lancedb pillow sentencepiece
```

## Setup

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

## Usage

### Process Files

Place PDFs and/or text files in `data/` directory:

```bash
python ray_pdf_parser.py
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
# For PDFs:
├── pages_jpg/         # JPEG images
├── pages_json/        # API responses
├── pages_md/          # Markdown text
└── pages_embeddings/  # Embedding vectors

# For text files:
├── text_chunks/       # Text chunks
└── text_embeddings/   # Embedding vectors

lancedb/               # Vector database
scratch/               # Intermediate PDFs
```

## LanceDB Schema

Table: `document_embeddings`

- **source_id** (string) - PDF or text filename
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

- **Text chunking**: Fast, character-based (~4 chars/token)
- **Embeddings**: ~1s per chunk/page
- **PDF parsing**: ~5s per page
- **Throughput**: ~0.5 items/sec end-to-end

Ray's parallel processing makes end-to-end time much faster than sequential processing.

## Example Output

```
=== Summary ===
functional_validation.pdf: 5 pages
multimodal_test.pdf: 3 pages

Total: 2 PDFs, 8 pages
LanceDB: lancedb/ (8 records)

Timing:
  End-to-End - Total: 17.02s, Throughput: 0.47 pages/sec
  Parsing - Total: 38.05s, Average: 4.76s per page
  Embeddings - Total: 9.59s, Average: 1.20s per page
```

## License

Apache License 2.0
