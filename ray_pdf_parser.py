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
class EmbeddingGeneratorActor:
    """Ray actor to generate embeddings for markdown pages using NVIDIA API."""
    
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    
    def generate_embedding(self, md_path, output_dir, pdf_name):
        """Generate embeddings for a markdown file and save results."""
        md_path = Path(md_path)
        output_dir = Path(output_dir)
        
        # Create output directory for embeddings
        embedding_output_dir = output_dir / pdf_name / "pages_embeddings"
        embedding_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
        if not text_content.strip():
            print(f"Warning: Empty markdown file {md_path.name}")
            return {
                "md_file": str(md_path),
                "embedding_file": None,
                "embedding_time": 0
            }
        
        # Generate embedding
        start_time = time.time()
        try:
            response = self.client.embeddings.create(
                input=[text_content],
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
            )
            embedding = response.data[0].embedding
            embedding_time = time.time() - start_time
            
            # Save embedding
            embedding_file = embedding_output_dir / f"{md_path.stem}.json"
            with open(embedding_file, "w", encoding="utf-8") as f:
                json.dump({
                    "embedding": embedding,
                    "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                    "text": text_content
                }, f, ensure_ascii=False, indent=2)
            
            return {
                "md_file": str(md_path),
                "embedding_file": str(embedding_file),
                "embedding_time": embedding_time
            }
        except Exception as e:
            print(f"Error generating embedding for {md_path.name}: {e}")
            return {
                "md_file": str(md_path),
                "embedding_file": None,
                "embedding_time": time.time() - start_time
            }


@ray.remote
def write_to_lancedb(embedding_results, md_tasks, db_path):
    """Write all embedding data to LanceDB in a single bulk operation."""
    records = []
    
    for embedding_result, (_, pdf_name) in zip(embedding_results, md_tasks):
        if not embedding_result.get("embedding_file"):
            continue
        
        try:
            # Load embedding data
            with open(embedding_result["embedding_file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract page number from filename (e.g., "page_001.json" -> 1)
            md_path = Path(embedding_result["md_file"])
            page_str = md_path.stem.split("_")[-1]
            page_number = int(page_str)
            
            # Prepare record
            record = {
                "source_id": pdf_name,
                "chunk_sequence": page_number,
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


def process_pdfs(data_dir="data", scratch_dir="scratch", output_dir="extracts", db_path="lancedb"):
    """Process all PDFs using Ray actors: split then render to JPEGs."""
    data_dir = Path(data_dir)
    scratch_dir = Path(scratch_dir)
    output_dir = Path(output_dir)
    db_path = Path(db_path)
    
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process\n")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Start timing
    pipeline_start = time.time()
    
    try:
        # Stage 1: Split PDFs into single-page PDFs
        print("=== Stage 1: Splitting PDFs ===")
        splitter_actors = [PDFSplitterActor.remote() for _ in pdf_files]
        
        split_futures = [
            actor.split_pdf.remote(str(pdf_path), str(scratch_dir))
            for actor, pdf_path in zip(splitter_actors, pdf_files)
        ]
        
        split_results = ray.get(split_futures)
        
        # Collect all page files with their PDF names
        all_tasks = []
        total_pages = 0
        for result in split_results:
            pdf_name = Path(result["pdf_name"]).stem
            total_pages += result["num_pages"]
            for page_file in result["page_files"]:
                all_tasks.append((page_file, pdf_name))
        
        print(f"Split complete: {len(all_tasks)} total pages\n")
        
        # Stage 2: Render each single-page PDF to JPEG
        print("=== Stage 2: Rendering JPEGs ===")
        renderer_actors = [PDFRendererActor.remote() for _ in all_tasks]
        
        render_futures = [
            actor.render_to_jpeg.remote(page_file, str(output_dir), pdf_name)
            for actor, (page_file, pdf_name) in zip(renderer_actors, all_tasks)
        ]
        
        jpeg_files = ray.get(render_futures)
        
        print(f"Rendered {len(jpeg_files)} JPEG images\n")
        
        # Stage 3: Parse each JPEG with model
        print("=== Stage 3: Parsing with model ===")
        parser_actors = [ModelParserActor.remote() for _ in all_tasks]
        
        parse_futures = [
            actor.parse_image.remote(jpeg_file, str(output_dir), pdf_name)
            for actor, (jpeg_file, (_, pdf_name)) in zip(parser_actors, zip(jpeg_files, all_tasks))
        ]
        
        parse_results = ray.get(parse_futures)
        
        total_model_time = sum(r["model_time"] for r in parse_results)
        avg_model_time = total_model_time / len(parse_results) if parse_results else 0
        
        print(f"Parsed {len(parse_results)} images\n")
        
        # Stage 4: Generate embeddings for each markdown file
        print("=== Stage 4: Generating embeddings ===")
        
        # Collect all markdown files with their PDF names
        md_tasks = []
        for result, (_, pdf_name) in zip(parse_results, all_tasks):
            if result.get("md_file"):
                md_tasks.append((result["md_file"], pdf_name))
        
        if md_tasks:
            embedding_actors = [EmbeddingGeneratorActor.remote() for _ in md_tasks]
            
            embedding_futures = [
                actor.generate_embedding.remote(md_file, str(output_dir), pdf_name)
                for actor, (md_file, pdf_name) in zip(embedding_actors, md_tasks)
            ]
            
            embedding_results = ray.get(embedding_futures)
            
            successful_embeddings = sum(1 for r in embedding_results if r["embedding_file"])
            total_embedding_time = sum(r["embedding_time"] for r in embedding_results)
            avg_embedding_time = total_embedding_time / len(embedding_results) if embedding_results else 0
            
            print(f"Generated {successful_embeddings}/{len(md_tasks)} embeddings\n")
        else:
            embedding_results = []
            total_embedding_time = 0
            avg_embedding_time = 0
            print("No markdown files to process\n")
        
        # Stage 5: Write embeddings to LanceDB (single bulk write)
        print("=== Stage 5: Writing to LanceDB ===")
        
        if embedding_results and any(r["embedding_file"] for r in embedding_results):
            lancedb_future = write_to_lancedb.remote(embedding_results, md_tasks, str(db_path))
            lancedb_result = ray.get(lancedb_future)
            
            print(f"Wrote {lancedb_result['success']} records to LanceDB\n")
        else:
            lancedb_result = {"success": 0, "failed": 0}
            print("No embeddings to write to LanceDB\n")
        
        # Calculate end-to-end timing
        pipeline_end = time.time()
        total_pipeline_time = pipeline_end - pipeline_start
        pages_per_second = total_pages / total_pipeline_time if total_pipeline_time > 0 else 0
        
        # Print summary
        print("=== Summary ===")
        for result in split_results:
            pdf_name = Path(result["pdf_name"]).stem
            print(f"{result['pdf_name']}: {result['num_pages']} pages")
            print(f"  Output: {output_dir}/{pdf_name}/")
        
        print(f"\nTotal: {len(pdf_files)} PDFs, {total_pages} pages")
        print(f"Page PDFs saved to: {scratch_dir}/")
        print(f"Extracts saved to: {output_dir}/<pdf_name>/{{pages_jpg,pages_json,pages_md,pages_embeddings}}/")
        if lancedb_result["success"] > 0:
            print(f"LanceDB saved to: {db_path}/ ({lancedb_result['success']} records)")
        print(f"\nTiming:")
        print(f"  End-to-End - Total: {total_pipeline_time:.2f}s, Throughput: {pages_per_second:.2f} pages/sec")
        print(f"  Parsing - Total: {total_model_time:.2f}s, Average: {avg_model_time:.2f}s per page")
        if embedding_results:
            print(f"  Embeddings - Total: {total_embedding_time:.2f}s, Average: {avg_embedding_time:.2f}s per page")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    process_pdfs()
