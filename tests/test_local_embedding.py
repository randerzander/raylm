#!/usr/bin/env python3
"""Test script for LocalEmbeddingBatcher."""

import pandas as pd
from stages import LocalEmbeddingBatcher
from pathlib import Path
import json
import tempfile

def test_local_embedding_batcher():
    """Test the LocalEmbeddingBatcher with sample data."""
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create sample batch data matching the expected format
        batch_data = {
            "text": [
                "how much protein should a female eat",
                "summit define",
                "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day."
            ],
            "source_filename": ["test_doc.pdf", "test_doc.pdf", "test_doc.pdf"],
            "page_number": [1, 1, 2],
            "label": ["page_text", "page_text", "page_text"],
            "output_dir": [str(output_dir), str(output_dir), str(output_dir)]
        }
        
        # Create batcher instance
        print("Initializing LocalEmbeddingBatcher...")
        batcher = LocalEmbeddingBatcher(batch_size=32, device="cuda:0")
        
        # Process batch
        print("\nProcessing batch...")
        result_df = batcher(batch_data)
        
        # Verify results
        print(f"\nProcessed {len(result_df)} rows")
        print(f"Columns: {list(result_df.columns)}")
        
        # Check embedding files were created
        for idx, row in result_df.iterrows():
            embedding_file = row["embedding_file"]
            if embedding_file and embedding_file != "None":
                print(f"\nRow {idx}:")
                print(f"  Embedding file: {embedding_file}")
                print(f"  Embedding time: {row['embedding_time']:.4f}s")
                
                # Load and verify embedding
                with open(embedding_file, "r") as f:
                    data = json.load(f)
                    
                print(f"  Model: {data['model']}")
                print(f"  Text preview: {data['text'][:50]}...")
                print(f"  Embedding shape: ({len(data['embedding'])})")
                print(f"  First 5 values: {data['embedding'][:5]}")
        
        print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_local_embedding_batcher()
