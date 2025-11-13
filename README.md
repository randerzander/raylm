# raylm

Distributed PDF parsing pipeline using Ray and NVIDIA NeMo for document extraction.

## Overview

This project provides a scalable pipeline for processing PDF documents:
1. **Split** PDFs into individual pages
2. **Render** pages to high-quality JPEG images
3. **Parse** images using NVIDIA's NeMo Parse API to extract structured content

All stages run in parallel using Ray for maximum throughput.

## Features

- **Parallel processing** with Ray actors for all stages
- **Per-PDF organization** - outputs organized by source document
- **High-quality rendering** - 2x scale factor for sharp images
- **Structured extraction** - JSON and Markdown outputs
- **Progress tracking** - detailed timing and statistics

## Requirements

- Python 3.8+
- Ray
- pypdfium2
- requests
- NVIDIA API key (for NeMo Parse)

## Installation

```bash
pip install ray pypdfium2 requests
```

## Setup

Set your NVIDIA API key:

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

## Usage

Place PDF files in the `data/` directory, then run:

```bash
python ray_pdf_parser.py
```

## Output Structure

```
extracts/
├── document1/
│   ├── pages_jpg/      # Rendered JPEG images
│   ├── pages_json/     # Full API responses
│   └── pages_md/       # Extracted markdown
└── document2/
    ├── pages_jpg/
    ├── pages_json/
    └── pages_md/
```

Intermediate files are saved in `scratch/` (single-page PDFs).

## How It Works

### Stage 1: PDF Splitting
Each PDF is split into individual single-page PDFs using `PDFSplitterActor`. This allows downstream stages to process pages independently.

### Stage 2: JPEG Rendering
Each page PDF is rendered to a high-quality JPEG using `PDFRendererActor` with 2x scaling for optimal OCR quality.

### Stage 3: Model Parsing
Each JPEG is sent to NVIDIA's NeMo Parse API using `ModelParserActor`. The API returns structured content in both JSON (full response) and Markdown (extracted text) formats.

## Performance

All stages run in parallel:
- Multiple PDFs processed simultaneously
- All pages within PDFs processed in parallel
- Ray handles actor scheduling and resource management

Typical processing time: ~2-3 seconds per page (model API call).

## License

MIT
