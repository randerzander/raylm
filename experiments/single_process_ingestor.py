"""Simple single-process ingestor - no Ray, just plain Python loops."""

import sys
from pathlib import Path
import time

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import chunk_text, convert_html_to_markdown, split_pdf, render_to_jpeg, parse_image


def process_files(data_dir="data", scratch_dir="scratch", output_dir="extracts"):
    """Process files sequentially without Ray."""
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / data_dir
    scratch_dir = project_root / scratch_dir
    output_dir = project_root / output_dir
    
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for files in: {data_dir}")
    
    # Find all files
    all_files = sorted(data_dir.glob("*"))
    
    if not all_files:
        print(f"No files found in {data_dir}")
        return
    
    pipeline_start = time.time()
    total_files = 0
    total_chunks = 0
    
    for file_path in all_files:
        if not file_path.is_file():
            continue
        
        total_files += 1
        file_start = time.time()
        print(f"\nProcessing {file_path.name}...")
        
        # Route to appropriate handler
        if file_path.suffix.lower() == '.txt':
            # Text: chunk it
            row = {"text_path": str(file_path), "output_dir": str(output_dir)}
            chunks = chunk_text(row)
            total_chunks += len(chunks)
            file_time = time.time() - file_start
            print(f"  -> Generated {len(chunks)} text chunks ({file_time:.2f}s)")
            
        elif file_path.suffix.lower() in ['.html', '.htm']:
            # HTML: convert to markdown
            row = {"html_path": str(file_path), "output_dir": str(output_dir)}
            result = convert_html_to_markdown(row)
            total_chunks += len(result)
            file_time = time.time() - file_start
            print(f"  -> Converted to markdown ({file_time:.2f}s)")
            
        elif file_path.suffix.lower() == '.pdf':
            # PDF: split -> render -> parse
            # Step 1: Split into pages
            row = {"pdf_path": str(file_path), "output_dir": str(scratch_dir)}
            pages = split_pdf(row)
            print(f"  -> Split into {len(pages)} pages")
            
            # Step 2-3: Render each page and parse
            for page in pages:
                page_start = time.time()
                
                # Render to JPEG
                render_row = {**page, "output_dir": str(output_dir)}
                jpeg_result = render_to_jpeg(render_row)
                
                # Parse with model
                parse_row = {**jpeg_result, "output_dir": str(output_dir)}
                parsed = parse_image(parse_row)
                
                page_time = time.time() - page_start
                if parsed.get("text"):
                    total_chunks += 1
                    print(f"  -> Page {page['page_number']}: extracted {len(parsed['text'])} chars ({page_time:.2f}s)")
            
            file_time = time.time() - file_start
            print(f"  -> Total file time: {file_time:.2f}s")
        else:
            print(f"  -> Skipped (unsupported file type)")
    
    pipeline_time = time.time() - pipeline_start
    throughput = total_chunks / pipeline_time if pipeline_time > 0 else 0
    
    print(f"\n=== Summary ===")
    print(f"Processed {total_files} files")
    print(f"Total chunks (pages): {total_chunks}")
    print(f"Total time: {pipeline_time:.2f}s")
    print(f"Throughput: {throughput:.2f} chunks/sec")


if __name__ == "__main__":
    process_files()
