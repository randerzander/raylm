"""Utility functions for PDF and text processing with Ray Data."""

import os
import base64
import json
import mimetypes
import requests
import time
import pypdfium2 as pdfium
from pathlib import Path
from openai import OpenAI
from markitdown import MarkItDown
import lancedb
import pandas as pd


NVAI_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

TOOLS = [
    "markdown_bbox",
    "markdown_no_bbox",
    "detection_only",
]


def read_image_as_base64(path):
    """Read an image file and encode it as base64."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    return b64, mime


def generate_content(task_id, b64_str, mime):
    """Generate content for NVIDIA API request."""
    if task_id < 0 or task_id >= len(TOOLS):
        raise ValueError(f"task_id should be within [0, {len(TOOLS)-1}]")
    tool_name = TOOLS[task_id]
    media_tag = f'<img src="data:{mime};base64,{b64_str}" />'
    content = f"{media_tag}"
    tool_spec = [{"type": "function", "function": {"name": tool_name}}]
    return content, tool_spec, tool_name


def split_pdf(row):
    """Map function to split a PDF into individual pages."""
    pdf_path = Path(row["pdf_path"])
    output_dir = Path(row["output_dir"])
    
    # Create output directory for this PDF
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the PDF
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    
    print(f"Splitting {pdf_path.name}: {num_pages} pages")
    
    page_files = []
    for i in range(num_pages):
        # Create a new PDF with just this page
        new_pdf = pdfium.PdfDocument.new()
        new_pdf.import_pages(pdf, pages=[i])
        
        # Save the single-page PDF
        output_file = pdf_output_dir / f"page_{i+1:03d}.pdf"
        new_pdf.save(output_file)
        new_pdf.close()
        
        page_files.append({
            "page_file": str(output_file),
            "source_filename": pdf_path.stem,
            "page_number": i + 1
        })
    
    pdf.close()
    
    return page_files


def render_to_jpeg(row):
    """Map function to render a single-page PDF to JPEG."""
    pdf_path = Path(row["page_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    scale = row.get("scale", 2.0)
    
    # Create output directory for this PDF's pages
    pdf_output_dir = output_dir / source_filename / "pages_jpg"
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the PDF
    pdf = pdfium.PdfDocument(pdf_path)
    
    # Render the page (should only be one page)
    page = pdf[0]
    pil_image = page.render(scale=scale).to_pil()
    
    # Save as JPEG with same name structure
    output_file = pdf_output_dir / f"{pdf_path.stem}.jpg"
    pil_image.save(output_file, "JPEG", quality=95)
    
    pdf.close()
    
    return {
        "jpeg_file": str(output_file),
        "source_filename": source_filename,
        "page_number": row["page_number"]
    }


def chunk_text(row):
    """Map function to chunk text files using character-based chunking."""
    text_path = Path(row["text_path"])
    output_dir = Path(row["output_dir"])
    max_tokens = row.get("max_tokens", 4096)
    
    # Create output directory for this text file
    source_filename = text_path.stem
    text_output_dir = output_dir / source_filename / "text_chunks"
    text_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read text content
    with open(text_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    
    # Simple character-based chunking (~4 chars per token average)
    chunk_size = max_tokens * 4
    chunks = []
    
    for i in range(0, len(text_content), chunk_size):
        chunk = text_content[i:i + chunk_size]
        if chunk.strip():
            chunks.append({
                "chunk_text": chunk,
                "chunk_number": len(chunks) + 1,
                "source_filename": source_filename
            })
    
    # Save chunks and return metadata
    chunk_files = []
    for i, chunk_data in enumerate(chunks):
        chunk_file = text_output_dir / f"chunk_{i+1:03d}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk_data["chunk_text"])
        chunk_files.append({
            "chunk_file": str(chunk_file),
            "source_filename": source_filename,
            "chunk_number": i + 1,
            "text": chunk_data["chunk_text"]
        })
    
    return chunk_files


def convert_html_to_markdown(row):
    """Map function to convert HTML files to markdown using markitdown."""
    html_path = Path(row["html_path"])
    output_dir = Path(row["output_dir"])
    
    # Create output directory for this HTML file
    source_filename = html_path.stem
    html_output_dir = output_dir / source_filename / "html_md"
    html_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert HTML to markdown
    md_converter = MarkItDown()
    result = md_converter.convert(str(html_path))
    markdown_content = result.text_content
    
    # Save markdown
    md_file = html_output_dir / f"{source_filename}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return [{
        "md_file": str(md_file),
        "source_filename": source_filename,
        "text": markdown_content
    }]


def parse_image(row):
    """Map function to parse a JPEG image using NVIDIA API."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    
    image_path = Path(row["jpeg_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    task_id = row.get("task_id", 1)
    
    # Create output directories for this PDF's results
    json_output_dir = output_dir / source_filename / "pages_json"
    md_output_dir = output_dir / source_filename / "pages_md"
    
    json_output_dir.mkdir(parents=True, exist_ok=True)
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Read and encode image
    b64_str, mime = read_image_as_base64(image_path)
    content, tool_spec, tool_name = generate_content(task_id, b64_str, mime)
    
    inputs = {
        "model": "nvidia/nemotron-parse",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "tools": tool_spec,
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
        "max_tokens": 1024,
    }
    
    # Call model
    start_time = time.time()
    response = requests.post(NVAI_URL, headers=headers, json=inputs, timeout=120)
    response.raise_for_status()
    model_time = time.time() - start_time
    
    result = response.json()
    
    # Save JSON result
    json_file = json_output_dir / f"{image_path.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Extract and save markdown
    md_file = None
    markdown_content = ""
    try:
        tool_call = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [{}])[0]
        arguments_str = tool_call.get("function", {}).get("arguments", "")
        if arguments_str:
            arguments = json.loads(arguments_str)
            markdown_content = arguments[0].get("text", "")
            if markdown_content:
                md_file = md_output_dir / f"{image_path.stem}.md"
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
    except Exception as e:
        print(f"Warning: Could not extract markdown from {image_path.name}: {e}")
    
    return {
        "md_file": str(md_file) if md_file else None,
        "source_filename": source_filename,
        "page_number": row["page_number"],
        "text": markdown_content,
        "model_time": model_time
    }


class EmbeddingBatcher:
    """Stateful map function to batch embedding requests."""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    
    def __call__(self, batch):
        """Process a batch of rows."""
        # Convert to pandas for easier handling
        df = pd.DataFrame(batch)
        
        # Get output_dir
        if "output_dir" in df.columns:
            output_dir = Path(df["output_dir"].iloc[0])
        else:
            output_dir = Path("extracts")
        
        # Filter rows with valid text
        df = df[df["text"].notna() & (df["text"] != "")]
        
        if len(df) == 0:
            return pd.DataFrame({
                "embedding": pd.Series([], dtype=object),
                "embedding_file": pd.Series([], dtype=object),
                "embedding_time": pd.Series([], dtype=float)
            })
        
        texts = df["text"].tolist()
        
        # Generate embeddings for batch
        start_time = time.time()
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"] * len(texts), "input_type": "query", "truncate": "NONE"}
            )
            embedding_time = time.time() - start_time
            
            # Save each embedding
            embeddings = []
            embedding_files = []
            embedding_times = []
            
            for idx, (embedding_data, (_, row)) in enumerate(zip(response.data, df.iterrows())):
                source_name = row.get("source_filename")
                
                # Determine output directory - all go to same embeddings folder
                embedding_output_dir = output_dir / source_name / "embeddings"
                embedding_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Determine filename
                if pd.notna(row.get("md_file")):
                    filename = Path(row["md_file"]).stem
                elif pd.notna(row.get("chunk_file")):
                    filename = Path(row["chunk_file"]).stem
                else:
                    filename = f"embedding_{idx}"
                
                # Save embedding
                embedding_file = embedding_output_dir / f"{filename}.json"
                with open(embedding_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "embedding": embedding_data.embedding,
                        "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                        "text": row["text"]
                    }, f, ensure_ascii=False, indent=2)
                
                embeddings.append(embedding_data.embedding)
                embedding_files.append(str(embedding_file))
                embedding_times.append(embedding_time / len(texts))
            
            # Add results to dataframe
            df = df.copy()
            df["embedding"] = embeddings
            df["embedding_file"] = embedding_files
            df["embedding_time"] = embedding_times
            
            return df
        
        except Exception as e:
            print(f"Error generating embeddings for batch: {e}")
            # Return dataframe with None values
            df = df.copy()
            df["embedding"] = None
            df["embedding_file"] = None
            df["embedding_time"] = 0.0
            return df


def write_to_lancedb_batch(batch, db_path):
    """Map batches function to write embedding data to LanceDB."""
    # Convert to pandas if not already
    if not isinstance(batch, pd.DataFrame):
        df = pd.DataFrame(batch)
    else:
        df = batch
    
    records = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get("embedding_file")) or row.get("embedding_file") is None:
            continue
        
        try:
            # Load embedding data
            with open(row["embedding_file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Get source info
            source_name = row.get("source_filename")
            chunk_sequence = row.get("page_number") or row.get("chunk_number") or 1
            
            # Prepare record
            record = {
                "source_id": source_name,
                "chunk_sequence": chunk_sequence,
                "text": data["text"],
                "vector": data["embedding"]
            }
            records.append(record)
        except Exception as e:
            print(f"Error loading embedding data: {e}")
            continue
    
    if not records:
        return pd.DataFrame({"success": [0], "failed": [0]})
    
    try:
        # Connect and write all records at once
        db = lancedb.connect(db_path)
        table_name = "document_embeddings"
        
        try:
            table = db.open_table(table_name)
            table.add(records)
        except Exception:
            # Table doesn't exist, create it
            table = db.create_table(table_name, data=records)
        
        return pd.DataFrame({"success": [len(records)], "failed": [0]})
    except Exception as e:
        print(f"Error writing to LanceDB: {e}")
        return pd.DataFrame({"success": [0], "failed": [len(records)]})
