import os
import sys
import base64
import json
import mimetypes
import requests
import ray
import pypdfium2 as pdfium
from pathlib import Path
from openai import OpenAI
import lancedb
import sentencepiece as spm
import time


nvai_url = "https://integrate.api.nvidia.com/v1/chat/completions"

tools = [
    "markdown_bbox",
    "markdown_no_bbox",
    "detection_only",
]


def _read_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    return b64, mime


def _generate_content(task_id, b64_str, mime):
    if task_id < 0 or task_id >= len(tools):
        raise ValueError(f"task_id should be within [0, {len(tools)-1}]")
    tool_name = tools[task_id]
    media_tag = f'<img src="data:{mime};base64,{b64_str}" />'
    content = f"{media_tag}"
    tool_spec = [{"type": "function", "function": {"name": tool_name}}]
    return content, tool_spec, tool_name


@ray.remote
class PDFSplitterActor:
    """Ray actor to split a PDF into individual pages."""
    
    def split_pdf(self, pdf_path, output_dir):
        """Split PDF into individual page PDFs."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
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
            
            page_files.append(str(output_file))
        
        pdf.close()
        
        return {
            "pdf_name": pdf_path.name,
            "num_pages": num_pages,
            "page_files": page_files
        }


@ray.remote
class PDFRendererActor:
    """Ray actor to render a single-page PDF to JPEG."""
    
    def render_to_jpeg(self, pdf_path, output_dir, pdf_name, scale=2.0):
        """Render a single-page PDF to JPEG."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Create output directory for this PDF's pages
        pdf_output_dir = output_dir / pdf_name / "pages_jpg"
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
        
        return str(output_file)


@ray.remote
class TextChunkerActor:
    """Ray actor to chunk text files using sentencepiece tokenizer."""
    
    def __init__(self):
        # Download and use a pretrained sentencepiece model
        # For simplicity, we'll use character-based chunking as fallback
        self.tokenizer = None
    
    def chunk_text(self, text_path, output_dir, max_tokens=4096):
        """Chunk a text file into ~4096 token segments."""
        text_path = Path(text_path)
        output_dir = Path(output_dir)
        
        # Create output directory for this text file
        text_name = text_path.stem
        text_output_dir = output_dir / text_name / "text_chunks"
        text_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read text content
        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
        # Simple character-based chunking (~4 chars per token average)
        # This is a rough approximation without a trained tokenizer
        chunk_size = max_tokens * 4  # ~4 chars per token
        chunks = []
        
        for i in range(0, len(text_content), chunk_size):
            chunk = text_content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Save chunks
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = text_output_dir / f"chunk_{i+1:03d}.txt"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)
            chunk_files.append(str(chunk_file))
        
        return {
            "text_name": text_path.name,
            "num_chunks": len(chunks),
            "chunk_files": chunk_files
        }


@ray.remote
def generate_embeddings_batch(content_paths_and_names, output_dir, batch_size=32):
    """Generate embeddings for multiple content files in batches using NVIDIA API.
    
    Args:
        content_paths_and_names: List of (content_path, source_name) tuples
        output_dir: Output directory for embeddings
        batch_size: Number of texts to embed in a single API call
    
    Returns:
        dict with 'results' list and 'num_requests' count
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    
    output_dir = Path(output_dir)
    results = []
    num_requests = 0
    
    # Process in batches
    for batch_start in range(0, len(content_paths_and_names), batch_size):
        batch = content_paths_and_names[batch_start:batch_start + batch_size]
        
        # Read all content files in batch
        batch_texts = []
        batch_metadata = []
        
        for content_path, source_name in batch:
            content_path = Path(content_path)
            
            try:
                with open(content_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                
                if text_content.strip():
                    batch_texts.append(text_content)
                    batch_metadata.append({
                        "content_path": content_path,
                        "source_name": source_name,
                        "text": text_content
                    })
                else:
                    results.append({
                        "content_file": str(content_path),
                        "embedding_file": None,
                        "embedding_time": 0
                    })
            except Exception as e:
                print(f"Error reading {content_path}: {e}")
                results.append({
                    "content_file": str(content_path),
                    "embedding_file": None,
                    "embedding_time": 0
                })
        
        if not batch_texts:
            continue
        
        # Generate embeddings for batch
        start_time = time.time()
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
            )
            num_requests += 1  # Count API request
            embedding_time = time.time() - start_time
            
            # Save each embedding
            for i, (embedding_data, metadata) in enumerate(zip(response.data, batch_metadata)):
                content_path = metadata["content_path"]
                source_name = metadata["source_name"]
                
                # Determine output directory based on content type
                if "text_chunks" in str(content_path):
                    embedding_output_dir = output_dir / source_name / "text_embeddings"
                else:
                    embedding_output_dir = output_dir / source_name / "pages_embeddings"
                
                embedding_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save embedding
                embedding_file = embedding_output_dir / f"{content_path.stem}.json"
                with open(embedding_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "embedding": embedding_data.embedding,
                        "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                        "text": metadata["text"]
                    }, f, ensure_ascii=False, indent=2)
                
                results.append({
                    "content_file": str(content_path),
                    "embedding_file": str(embedding_file),
                    "embedding_time": embedding_time / len(batch_texts)  # Amortize time across batch
                })
        
        except Exception as e:
            print(f"Error generating embeddings for batch: {e}")
            for metadata in batch_metadata:
                results.append({
                    "content_file": str(metadata["content_path"]),
                    "embedding_file": None,
                    "embedding_time": 0
                })
    
    return {
        "results": results,
        "num_requests": num_requests
    }


@ray.remote
def write_to_lancedb(embedding_results, source_tasks, db_path):
    """Write all embedding data to LanceDB in a single bulk operation.
    
    Args:
        embedding_results: List of embedding result dicts
        source_tasks: List of (file_path, source_name) tuples
        db_path: Path to LanceDB
    """
    records = []
    
    for embedding_result, (_, source_name) in zip(embedding_results, source_tasks):
        if not embedding_result.get("embedding_file"):
            continue
        
        try:
            # Load embedding data
            with open(embedding_result["embedding_file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract chunk/page number from filename
            source_file = embedding_result.get("md_file") or embedding_result.get("chunk_file")
            if source_file:
                source_path = Path(source_file)
                # Extract number from filename (e.g., "page_001" -> 1 or "chunk_002" -> 2)
                stem = source_path.stem
                if "_" in stem:
                    num_str = stem.split("_")[-1]
                    chunk_sequence = int(num_str)
                else:
                    chunk_sequence = 1
            else:
                chunk_sequence = 1
            
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
        return {"success": 0, "failed": 0}
    
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
        
        return {"success": len(records), "failed": 0}
    except Exception as e:
        print(f"Error writing to LanceDB: {e}")
        return {"success": 0, "failed": len(records)}


@ray.remote
class ModelParserActor:
    """Ray actor to parse a JPEG image using NVIDIA API."""
    
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
    
    def parse_image(self, image_path, output_dir, pdf_name, task_id=1):
        """Parse an image using NVIDIA API and save results."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        
        # Create output directories for this PDF's results
        json_output_dir = output_dir / pdf_name / "pages_json"
        md_output_dir = output_dir / pdf_name / "pages_md"
        
        json_output_dir.mkdir(parents=True, exist_ok=True)
        md_output_dir.mkdir(parents=True, exist_ok=True)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Read and encode image
        b64_str, mime = _read_image_as_base64(image_path)
        content, tool_spec, tool_name = _generate_content(task_id, b64_str, mime)
        
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
        response = requests.post(nvai_url, headers=headers, json=inputs, timeout=120)
        response.raise_for_status()
        model_time = time.time() - start_time
        
        result = response.json()
        
        # Save JSON result
        json_file = json_output_dir / f"{image_path.stem}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Extract and save markdown
        md_file = None
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
            "image": str(image_path),
            "json_file": str(json_file),
            "md_file": str(md_file) if md_file else None,
            "model_time": model_time
        }


def process_files(data_dir="data", scratch_dir="scratch", output_dir="extracts", db_path="lancedb"):
    """Process all PDFs and text files using Ray actors."""
    data_dir = Path(data_dir)
    scratch_dir = Path(scratch_dir)
    output_dir = Path(output_dir)
    db_path = Path(db_path)
    
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF and text files
    pdf_files = sorted(data_dir.glob("*.pdf"))
    txt_files = sorted(data_dir.glob("*.txt"))
    
    # Initialize request counters
    total_parse_requests = 0
    total_embedding_requests = 0
    
    if not pdf_files and not txt_files:
        print(f"No PDF or text files found in {data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs and {len(txt_files)} text files to process\n")
    
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        _metrics_export_port=None,  # Disable metrics export
    )
    
    # Start timing
    pipeline_start = time.time()
    
    try:
        all_embedding_results = []
        all_source_tasks = []
        total_chunks = 0
        
        # Process text files first
        if txt_files:
            print("=== Stage 1a: Chunking text files ===")
            chunker_actors = [TextChunkerActor.remote() for _ in txt_files]
            
            chunk_futures = [
                actor.chunk_text.remote(str(txt_path), str(output_dir))
                for actor, txt_path in zip(chunker_actors, txt_files)
            ]
            
            chunk_results = ray.get(chunk_futures)
            
            # Collect all chunk files with their text names
            text_tasks = []
            for result in chunk_results:
                text_name = Path(result["text_name"]).stem
                total_chunks += result["num_chunks"]
                for chunk_file in result["chunk_files"]:
                    text_tasks.append((chunk_file, text_name))
            
            print(f"Chunked {len(txt_files)} text files into {total_chunks} chunks\n")
            
            # Stage 1b: Generate embeddings for text chunks (batched)
            if text_tasks:
                print("=== Stage 1b: Embedding text chunks ===")
                
                # Batch all text tasks into fewer API calls
                embedding_future = generate_embeddings_batch.remote(text_tasks, str(output_dir), batch_size=32)
                text_embedding_data = ray.get(embedding_future)
                text_embedding_results = text_embedding_data["results"]
                total_embedding_requests += text_embedding_data["num_requests"]
                
                successful_text_embeddings = sum(1 for r in text_embedding_results if r["embedding_file"])
                total_text_embedding_time = sum(r["embedding_time"] for r in text_embedding_results)
                
                print(f"Generated {successful_text_embeddings}/{len(text_tasks)} text embeddings ({text_embedding_data['num_requests']} API requests)\n")
                
                all_embedding_results.extend(text_embedding_results)
                all_source_tasks.extend(text_tasks)
        else:
            total_text_embedding_time = 0
        
        # Process PDFs
        total_pages = 0
        if pdf_files:
            # Stage 2: Split PDFs into single-page PDFs
            print("=== Stage 2: Splitting PDFs ===")
            splitter_actors = [PDFSplitterActor.remote() for _ in pdf_files]
            
            split_futures = [
                actor.split_pdf.remote(str(pdf_path), str(scratch_dir))
                for actor, pdf_path in zip(splitter_actors, pdf_files)
            ]
            
            split_results = ray.get(split_futures)
            
            # Collect all page files with their PDF names
            all_tasks = []
            for result in split_results:
                pdf_name = Path(result["pdf_name"]).stem
                total_pages += result["num_pages"]
                for page_file in result["page_files"]:
                    all_tasks.append((page_file, pdf_name))
            
            print(f"Split complete: {len(all_tasks)} total pages\n")
            
            # Stage 3: Render each single-page PDF to JPEG
            print("=== Stage 3: Rendering JPEGs ===")
            renderer_actors = [PDFRendererActor.remote() for _ in all_tasks]
            
            render_futures = [
                actor.render_to_jpeg.remote(page_file, str(output_dir), pdf_name)
                for actor, (page_file, pdf_name) in zip(renderer_actors, all_tasks)
            ]
            
            jpeg_files = ray.get(render_futures)
            
            print(f"Rendered {len(jpeg_files)} JPEG images\n")
            
            # Stage 4: Parse each JPEG with model
            print("=== Stage 4: Parsing with model ===")
            parser_actors = [ModelParserActor.remote() for _ in all_tasks]
            
            parse_futures = [
                actor.parse_image.remote(jpeg_file, str(output_dir), pdf_name)
                for actor, (jpeg_file, (_, pdf_name)) in zip(parser_actors, zip(jpeg_files, all_tasks))
            ]
            
            parse_results = ray.get(parse_futures)
            
            total_parse_requests += len(parse_results)  # Count parsing API requests
            total_model_time = sum(r["model_time"] for r in parse_results)
            avg_model_time = total_model_time / len(parse_results) if parse_results else 0
            
            print(f"Parsed {len(parse_results)} images ({len(parse_results)} API requests)\n")
            
            # Stage 5: Generate embeddings for markdown files (batched)
            print("=== Stage 5: Generating PDF embeddings ===")
            
            # Collect all markdown files with their PDF names
            md_tasks = []
            for result, (_, pdf_name) in zip(parse_results, all_tasks):
                if result.get("md_file"):
                    md_tasks.append((result["md_file"], pdf_name))
            
            if md_tasks:
                # Batch all markdown tasks into fewer API calls
                embedding_future = generate_embeddings_batch.remote(md_tasks, str(output_dir), batch_size=32)
                embedding_data = ray.get(embedding_future)
                embedding_results = embedding_data["results"]
                total_embedding_requests += embedding_data["num_requests"]
                
                successful_embeddings = sum(1 for r in embedding_results if r["embedding_file"])
                total_embedding_time = sum(r["embedding_time"] for r in embedding_results)
                avg_embedding_time = total_embedding_time / len(embedding_results) if embedding_results else 0
                
                print(f"Generated {successful_embeddings}/{len(md_tasks)} embeddings ({embedding_data['num_requests']} API requests)\n")
                
                all_embedding_results.extend(embedding_results)
                all_source_tasks.extend(md_tasks)
            else:
                embedding_results = []
                total_embedding_time = 0
                avg_embedding_time = 0
        else:
            total_model_time = 0
            avg_model_time = 0
            total_embedding_time = 0
            avg_embedding_time = 0
            split_results = []
            print("No PDF files to process\n")
        
        # Stage 6: Write all embeddings to LanceDB (single bulk write)
        print("=== Stage 6: Writing to LanceDB ===")
        
        if all_embedding_results and any(r["embedding_file"] for r in all_embedding_results):
            lancedb_future = write_to_lancedb.remote(all_embedding_results, all_source_tasks, str(db_path))
            lancedb_result = ray.get(lancedb_future)
            
            print(f"Wrote {lancedb_result['success']} records to LanceDB\n")
        else:
            lancedb_result = {"success": 0, "failed": 0}
            print("No embeddings to write to LanceDB\n")
        
        # Calculate end-to-end timing
        pipeline_end = time.time()
        total_pipeline_time = pipeline_end - pipeline_start
        total_items = total_pages + total_chunks
        throughput = total_items / total_pipeline_time if total_pipeline_time > 0 else 0
        
        # Print summary
        print("=== Summary ===")
        
        if txt_files:
            for result in chunk_results:
                text_name = Path(result["text_name"]).stem
                print(f"{result['text_name']}: {result['num_chunks']} chunks")
                print(f"  Output: {output_dir}/{text_name}/")
        
        if pdf_files:
            for result in split_results:
                pdf_name = Path(result["pdf_name"]).stem
                print(f"{result['pdf_name']}: {result['num_pages']} pages")
                print(f"  Output: {output_dir}/{pdf_name}/")
        
        print(f"\nTotal: {len(txt_files)} text files ({total_chunks} chunks), {len(pdf_files)} PDFs ({total_pages} pages)")
        if pdf_files:
            print(f"Page PDFs saved to: {scratch_dir}/")
        print(f"Extracts saved to: {output_dir}/")
        if lancedb_result["success"] > 0:
            print(f"LanceDB saved to: {db_path}/ ({lancedb_result['success']} records)")
        
        print(f"\nTiming:")
        print(f"  End-to-End - Total: {total_pipeline_time:.2f}s, Throughput: {throughput:.2f} items/sec")
        
        if txt_files and total_text_embedding_time > 0:
            avg_text_embedding = total_text_embedding_time / total_chunks if total_chunks > 0 else 0
            print(f"  Text Embeddings - Total: {total_text_embedding_time:.2f}s, Average: {avg_text_embedding:.2f}s per chunk")
        
        if pdf_files:
            print(f"  PDF Parsing - Total: {total_model_time:.2f}s, Average: {avg_model_time:.2f}s per page")
            if embedding_results and total_embedding_time > 0:
                print(f"  PDF Embeddings - Total: {total_embedding_time:.2f}s, Average: {avg_embedding_time:.2f}s per page")
        
        print(f"\nAPI Requests:")
        if total_parse_requests > 0:
            print(f"  Parse requests: {total_parse_requests}")
        if total_embedding_requests > 0:
            print(f"  Embedding requests: {total_embedding_requests}")
        print(f"  Total requests: {total_parse_requests + total_embedding_requests}")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    process_files()
