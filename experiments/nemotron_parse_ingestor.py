"""Main Ray Data ingestion pipeline for PDFs and text files."""

import ray
import ray.data
from pathlib import Path
import time

from stages import (
    split_pdf,
    render_to_jpeg,
    chunk_text,
    convert_html_to_markdown,
    parse_image,
    EmbeddingBatcher,
    write_to_lancedb_batch
)


def process_files(data_dir="data", scratch_dir="scratch", output_dir="extracts", db_path="lancedb"):
    """Process all PDFs, HTML, and text files using Ray Data.
    
    Files are automatically categorized by extension:
    - .pdf files are processed as PDFs
    - .html files are converted to markdown
    - .txt files are processed as text
    """
    data_dir = Path(data_dir)
    scratch_dir = Path(scratch_dir)
    output_dir = Path(output_dir)
    db_path = Path(db_path)
    
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files and categorize by extension
    all_files = sorted(data_dir.glob("*"))
    pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf' and f.is_file()]
    html_files = [f for f in all_files if f.suffix.lower() in ['.html', '.htm'] and f.is_file()]
    txt_files = [f for f in all_files if f.suffix.lower() == '.txt' and f.is_file()]
    
    if not pdf_files and not html_files and not txt_files:
        print(f"No PDF, HTML, or text files found in {data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs, {len(html_files)} HTML files, and {len(txt_files)} text files in {data_dir}\n")
    
    # Initialize Ray with conservative settings
    ray.init(
        ignore_reinit_error=True,
        _temp_dir="/tmp/ray",
        object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
        logging_level="ERROR",  # Reduce logging noise
        _metrics_export_port=None,  # Disable metrics export
    )
    
    # Enable rich progress bars
    ray.data.DataContext.get_current().enable_rich_progress_bars = True
    
    # Start timing
    pipeline_start = time.time()
    
    try:
        # Process text files
        text_embedding_ds = None
        
        if txt_files:
            print("=== Stage 1: Processing text files ===")
            
            # Create dataset from text files
            text_ds = ray.data.from_items([
                {"text_path": str(txt_path), "output_dir": str(output_dir)}
                for txt_path in txt_files
            ])
            
            # Chunk text files (flat_map because each file becomes multiple chunks)
            chunked_ds = text_ds.flat_map(chunk_text)
            
            # Add output_dir to each row for embedding generation
            chunked_ds = chunked_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            
            # Generate embeddings (use map_batches for efficient batching)
            # Use fewer actors to reduce memory pressure
            text_embedding_ds = chunked_ds.map_batches(
                EmbeddingBatcher,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(size=2)
            )
            
            print(f"Text embedding pipeline created\n")
        
        # Process HTML files
        html_embedding_ds = None
        
        if html_files:
            print("=== Stage 2: Processing HTML files ===")
            
            # Create dataset from HTML files
            html_ds = ray.data.from_items([
                {"html_path": str(html_path), "output_dir": str(output_dir)}
                for html_path in html_files
            ])
            
            # Convert HTML to markdown (flat_map in case we want to handle multiple outputs later)
            html_md_ds = html_ds.flat_map(convert_html_to_markdown)
            
            # Add output_dir to each row for embedding generation
            html_md_ds = html_md_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            
            # Generate embeddings (use map_batches for efficient batching)
            # Use fewer actors to reduce memory pressure
            html_embedding_ds = html_md_ds.map_batches(
                EmbeddingBatcher,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(size=2)
            )
            
            print(f"HTML embedding pipeline created\n")
        
        # Process PDFs
        pdf_embedding_ds = None
        
        if pdf_files:
            print("=== Stage 3: Processing PDF files ===")
            
            # Create dataset from PDF files
            pdf_ds = ray.data.from_items([
                {"pdf_path": str(pdf_path), "output_dir": str(scratch_dir)}
                for pdf_path in pdf_files
            ])
            
            # Split PDFs into pages (flat_map because each PDF becomes multiple pages)
            pages_ds = pdf_ds.flat_map(split_pdf)
            
            # Render pages to JPEG
            pages_ds = pages_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            jpeg_ds = pages_ds.map(render_to_jpeg)
            
            # Parse images with model
            jpeg_ds = jpeg_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            parsed_ds = jpeg_ds.map(parse_image)
            
            # Filter out rows without markdown
            parsed_ds = parsed_ds.filter(lambda row: row["md_file"] is not None and row["text"])
            
            # Generate embeddings for parsed markdown
            # Use fewer actors to reduce memory pressure
            pdf_embedding_ds = parsed_ds.map_batches(
                EmbeddingBatcher,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(size=2)
            )
            
            print(f"PDF embedding pipeline created\n")
        
        # Combine all embeddings
        print("=== Stage 4: Writing to LanceDB ===")
        
        # Collect all embedding datasets
        embedding_datasets = []
        if text_embedding_ds:
            embedding_datasets.append(text_embedding_ds)
        if html_embedding_ds:
            embedding_datasets.append(html_embedding_ds)
        if pdf_embedding_ds:
            embedding_datasets.append(pdf_embedding_ds)
        
        # Union all datasets
        if len(embedding_datasets) > 1:
            all_embeddings_ds = embedding_datasets[0]
            for ds in embedding_datasets[1:]:
                all_embeddings_ds = all_embeddings_ds.union(ds)
        elif len(embedding_datasets) == 1:
            all_embeddings_ds = embedding_datasets[0]
        else:
            all_embeddings_ds = None
        
        if all_embeddings_ds:
            # Write to LanceDB in batches
            lancedb_results = all_embeddings_ds.map_batches(
                lambda batch: write_to_lancedb_batch(batch, str(db_path)),
                batch_size=100
            )
            
            # Execute pipeline and collect results (this triggers lazy execution)
            print("Executing pipeline...")
            results = lancedb_results.take_all()
            
            # Results are rows from DataFrame, extract success/failed counts
            total_success = sum(r["success"] if isinstance(r["success"], int) else r["success"][0] for r in results)
            total_failed = sum(r["failed"] if isinstance(r["failed"], int) else r["failed"][0] for r in results)
            
            print(f"Wrote {total_success} records to LanceDB\n")
        else:
            total_success = 0
            total_failed = 0
            print("No embeddings to write to LanceDB\n")
        
        # Calculate end-to-end timing
        pipeline_end = time.time()
        total_pipeline_time = pipeline_end - pipeline_start
        throughput = total_success / total_pipeline_time if total_pipeline_time > 0 else 0
        
        # Print summary
        print("=== Summary ===")
        print(f"Files processed: {len(txt_files)} text files, {len(html_files)} HTML files, {len(pdf_files)} PDFs")
        print(f"Total chunks (pages): {total_success}")
        print(f"Extracts saved to: {output_dir}/")
        if pdf_files:
            print(f"Page PDFs saved to: {scratch_dir}/")
        if total_success > 0:
            print(f"LanceDB saved to: {db_path}/ ({total_success} records)")
        
        print(f"\nTiming:")
        print(f"  End-to-End - Total: {total_pipeline_time:.2f}s, Throughput: {throughput:.2f} items/sec")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    process_files()
