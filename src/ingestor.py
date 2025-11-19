"""Main Ray Data ingestion pipeline for PDFs and text files."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
import ray.data
import time
import numpy as np

from stages import (
    split_pdf,
    render_to_jpeg,
    chunk_text,
    convert_html_to_markdown,
    page_elements,
    extract_page_elements,
    process_table_structure,
    process_chart_elements,
    process_ocr,
    EmbeddingBatcher,
    LocalEmbeddingBatcher,
    write_to_lancedb_batch,
    PageElementsActor,
    TableStructureActor,
    ChartElementsActor
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
        _system_config={
            "event_stats": False,
            "metrics_report_interval_ms": 0,
        }
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
            # text_embedding_ds = chunked_ds.map_batches(
            #     EmbeddingBatcher,
            #     batch_size=32,
            #     compute=ray.data.ActorPoolStrategy(size=2)
            # )
            text_embedding_ds = None
            
            print(f"Text chunking pipeline created\n")
        
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
            # html_embedding_ds = html_md_ds.map_batches(
            #     EmbeddingBatcher,
            #     batch_size=32,
            #     compute=ray.data.ActorPoolStrategy(size=2)
            # )
            html_embedding_ds = None
            
            print(f"HTML conversion pipeline created\n")
        
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
            
            # Render pages to JPEG and extract text (single pass)
            pages_ds = pages_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            jpeg_ds = pages_ds.map(render_to_jpeg)
            
            # Create page text dataset for embeddings (just add label to existing data)
            page_text_ds = jpeg_ds.map(lambda row: {
                **row,
                "label": "page_text",
                "output_dir": str(output_dir)
            })
            
            # Parse images with local page-elements-v3 model (reuse jpeg_ds)
            # Use single actor to avoid multiple model copies in GPU memory
            jpeg_ds = jpeg_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            parsed_ds = jpeg_ds.map_batches(
                PageElementsActor,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                num_gpus=1
            )
            
            # Filter out rows without detections
            parsed_ds = parsed_ds.filter(lambda row: row["num_detections"] > 0 and row["text"])
            
            # Extract sub-images for each detected element
            parsed_ds = parsed_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            elements_ds = parsed_ds.flat_map(extract_page_elements)
            
            # Materialize elements to avoid recomputing the pipeline for each branch
            elements_ds = elements_ds.materialize()
            
            # Filter elements to only tables
            tables_ds = elements_ds.filter(lambda row: row["label"] == "table")
            
            # Process table structure for table elements only
            tables_ds = tables_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            table_structure_ds = tables_ds.map_batches(
                TableStructureActor,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                num_gpus=1
            )
            
            # Filter elements to only charts
            charts_ds = elements_ds.filter(lambda row: row["label"] == "chart")
            
            # Process chart elements for chart elements only
            charts_ds = charts_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            chart_elements_ds = charts_ds.map_batches(
                ChartElementsActor,
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                num_gpus=1
            )
            
            # Filter elements for infographics only (directly from step 4)
            infographics_ds = elements_ds.filter(lambda row: row["label"] in ["infographic", "graphic"])
            infographics_ds = infographics_ds.map(lambda row: {**row, "output_dir": str(output_dir)})
            
            # Run OCR on all structured elements (tables with structure, charts with elements, infographics)
            # Union the three branches together for OCR
            print(f"Running OCR on tables, charts, and infographics...")
            ocr_input_ds = table_structure_ds.union(chart_elements_ds).union(infographics_ds)
            ocr_ds = ocr_input_ds.map(process_ocr)
            
            # Prepare OCR results for embedding
            # For tables, use table_md if available, otherwise use ocr_text
            # For charts and infographics, use ocr_text
            def prepare_ocr_for_embedding(row):
                label = row.get("label", "unknown")
                text = ""
                
                if label == "table" and row.get("table_md"):
                    # Use markdown representation for tables
                    text = row["table_md"]
                else:
                    # Use OCR text for charts and infographics
                    text = row.get("ocr_text", "")
                
                return {
                    **row,
                    "text": text,
                    "output_dir": str(output_dir)
                }
            
            ocr_ds = ocr_ds.map(prepare_ocr_for_embedding)
            
            # Union page text with OCR results for embedding
            all_pdf_content_ds = page_text_ds.union(ocr_ds)
            
            # Generate embeddings for all PDF content (page text + OCR results)
            # Use local model with 1 GPU
            pdf_embedding_ds = all_pdf_content_ds.map_batches(
                LocalEmbeddingBatcher,
                fn_constructor_kwargs={"batch_size": 32, "device": "cuda:0"},
                batch_size=32,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                num_gpus=1
            )
            
            print(f"PDF parsing pipeline created\n")
        
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
            # Execute pipeline - iterate over embeddings to count types and write to LanceDB
            print("Executing pipeline...")
            total_success = 0
            total_failed = 0
            content_type_counts = {}
            
            # Iterate over embeddings dataset batches
            for batch in all_embeddings_ds.iter_batches(batch_size=100):
                # Tally content types from this batch
                if "label" in batch:
                    for label in batch["label"]:
                        label_str = str(label)
                        content_type_counts[label_str] = content_type_counts.get(label_str, 0) + 1
                
                # Write this batch to LanceDB
                result_batch = write_to_lancedb_batch(batch, str(db_path))
                batch_success = result_batch["success"][0] if isinstance(result_batch["success"][0], (int, np.integer)) else int(result_batch["success"][0])
                batch_failed = result_batch["failed"][0] if isinstance(result_batch["failed"][0], (int, np.integer)) else int(result_batch["failed"][0])
                total_success += batch_success
                total_failed += batch_failed
            
            print(f"Wrote {total_success} records to LanceDB")
            print(f"Content type breakdown: {dict(sorted(content_type_counts.items()))}\n")
        else:
            total_success = 0
            total_failed = 0
            content_type_counts = {}
            print("No embeddings to write to LanceDB\n")
        
        # Calculate end-to-end timing
        pipeline_end = time.time()
        total_pipeline_time = pipeline_end - pipeline_start
        throughput = total_success / total_pipeline_time if total_pipeline_time > 0 else 0
        
        # Print summary
        print("=== Summary ===")
        print(f"Files processed: {len(txt_files)} text files, {len(html_files)} HTML files, {len(pdf_files)} PDFs")
        print(f"Total records embedded: {total_success}")
        if content_type_counts:
            print(f"  By content type:")
            for content_type, count in sorted(content_type_counts.items()):
                print(f"    - {content_type}: {count}")
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
    process_files(data_dir="/localhome/local-jdyer/data")
