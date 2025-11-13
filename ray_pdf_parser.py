import os
import sys
import base64
import json
import mimetypes
import requests
import ray
import pypdfium2 as pdfium
from pathlib import Path
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


def process_pdfs(data_dir="data", scratch_dir="scratch", output_dir="extracts"):
    """Process all PDFs using Ray actors: split then render to JPEGs."""
    data_dir = Path(data_dir)
    scratch_dir = Path(scratch_dir)
    output_dir = Path(output_dir)
    
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
        for result in split_results:
            pdf_name = Path(result["pdf_name"]).stem
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
        
        # Print summary
        print("=== Summary ===")
        total_pages = 0
        for result in split_results:
            pdf_name = Path(result["pdf_name"]).stem
            print(f"{result['pdf_name']}: {result['num_pages']} pages")
            print(f"  Output: {output_dir}/{pdf_name}/")
            total_pages += result['num_pages']
        
        print(f"\nTotal: {len(pdf_files)} PDFs, {total_pages} pages")
        print(f"Page PDFs saved to: {scratch_dir}/")
        print(f"Extracts saved to: {output_dir}/<pdf_name>/{{pages_jpg,pages_json,pages_md}}/")
        print(f"\nModel timing:")
        print(f"  Total: {total_model_time:.2f}s")
        print(f"  Average per page: {avg_model_time:.2f}s")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    process_pdfs()
