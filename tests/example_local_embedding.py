#!/usr/bin/env python3
"""Example usage of LocalEmbeddingBatcher in Ray Data pipeline.

This example shows how to replace EmbeddingBatcher with LocalEmbeddingBatcher
for local inference using the nvidia/llama-nemotron-embed-1b-v2 model.
"""

import ray
import ray.data
from stages import LocalEmbeddingBatcher

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create sample dataset
data = [
    {
        "text": "how much protein should a female eat",
        "source_filename": "nutrition.pdf",
        "page_number": 1,
        "label": "page_text",
        "output_dir": "extracts"
    },
    {
        "text": "summit define",
        "source_filename": "dictionary.pdf", 
        "page_number": 1,
        "label": "page_text",
        "output_dir": "extracts"
    },
    {
        "text": "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "source_filename": "nutrition.pdf",
        "page_number": 2,
        "label": "page_text",
        "output_dir": "extracts"
    }
]

# Create Ray Dataset
ds = ray.data.from_items(data)

# Generate embeddings using LocalEmbeddingBatcher
# Replace EmbeddingBatcher with LocalEmbeddingBatcher in your pipeline:
# 
# Before (API-based):
#   embedding_ds = ds.map_batches(
#       EmbeddingBatcher,
#       batch_size=32,
#       compute=ray.data.ActorPoolStrategy(size=2)
#   )
#
# After (local inference):
embedding_ds = ds.map_batches(
    LocalEmbeddingBatcher,
    fn_constructor_kwargs={"device": "cuda:0"},
    batch_size=32,
    compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
    num_gpus=1  # Allocate 1 GPU per actor
)

# Materialize the dataset to trigger execution
print("Generating embeddings...")
result = embedding_ds.materialize()

# Show results
print(f"\nGenerated {result.count()} embeddings")
print("\nSample results:")
for row in result.take(3):
    print(f"  - {row['source_filename']}, page {row['page_number']}: {row['embedding_file']}")

print("\n✓ Example completed successfully!")

# Clean up
ray.shutdown()
