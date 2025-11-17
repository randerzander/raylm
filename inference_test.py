import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from pathlib import Path
from PIL import Image

from model import define_model
from utils import plot_sample, postprocess_preds_page_element, reformat_for_plotting

# Configuration
BATCH_SIZE = 8

# Input and output directories
images_dir = Path("/raid/rgelhausen/multiformat_docs/images")
output_dir = Path("./inference_outputs")
output_dir.mkdir(exist_ok=True)

# Load model once
print("Loading model...")
start_time = time.time()
model = define_model("page_element_v3")
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.2f}s")

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions])

print(f"Found {len(image_files)} images to process")
print(f"Using batch size: {BATCH_SIZE}")

# Timing variables
total_preprocess_time = 0
total_inference_time = 0
total_postprocess_time = 0
total_viz_time = 0
total_images_processed = 0

# Open JSONL file for writing all results
jsonl_path = output_dir / "results.jsonl"
overall_start = time.time()

with open(jsonl_path, "w") as jsonl_file:
    
    # Process images in batches
    for batch_start in range(0, len(image_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(image_files))
        batch_paths = image_files[batch_start:batch_end]
        
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} ({len(batch_paths)} images)")
        print(f"{'='*60}")
        
        # Load and preprocess batch
        preprocess_start = time.time()
        batch_images = []
        batch_orig_sizes = []
        
        for img_path in batch_paths:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            batch_images.append(img_array)
            batch_orig_sizes.append(img_array.shape)
        
        # Preprocess all images in batch
        batch_tensors = []
        for img in batch_images:
            x = model.preprocess(img)
            batch_tensors.append(x)
        
        # Stack into single batch tensor
        batch_tensor = torch.stack(batch_tensors)
        preprocess_time = time.time() - preprocess_start
        total_preprocess_time += preprocess_time
        
        # Batched inference
        inference_start = time.time()
        with torch.inference_mode():
            batch_preds = model(batch_tensor, batch_orig_sizes)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        print(f"Preprocess: {preprocess_time:.3f}s | Inference: {inference_time:.3f}s | {inference_time/len(batch_paths):.3f}s per image")
        
        # Process each image's results
        for img_path, img, preds in zip(batch_paths, batch_images, batch_preds):
            postprocess_start = time.time()
            
            # Post-processing
            boxes, labels, scores = postprocess_preds_page_element(preds, model.thresholds_per_class, model.labels)
            
            # Convert numeric labels to string labels
            label_names = [model.labels[int(label)] for label in labels]
            
            # Combine into single structure
            detections = []
            for box, label, score in zip(boxes, label_names, scores):
                detections.append({
                    "label": label,
                    "box": box.tolist() if hasattr(box, 'tolist') else box,
                    "score": float(score)
                })
            
            # Create result dict for this image
            result = {
                "image": str(img_path),
                "image_name": img_path.name,
                "detections": detections
            }
            
            # Write to JSONL file
            jsonl_file.write(json.dumps(result) + "\n")
            
            postprocess_time = time.time() - postprocess_start
            total_postprocess_time += postprocess_time
            
            # Create output subdirectory for visualization
            viz_start = time.time()
            img_output_dir = output_dir / img_path.stem
            img_output_dir.mkdir(exist_ok=True)
            
            # Plot
            boxes_plot, confs = reformat_for_plotting(boxes, labels, scores, img.shape, model.num_classes)
            
            plt.figure(figsize=(15, 10))
            plot_sample(img, boxes_plot, confs, labels=model.labels)
            plt.savefig(img_output_dir / "visualization.png")
            plt.close()
            
            viz_time = time.time() - viz_start
            total_viz_time += viz_time
            
            total_images_processed += 1
            print(f"  {img_path.name}: {len(detections)} detections")

overall_time = time.time() - overall_start

print(f"\n{'='*60}")
print(f"TIMING SUMMARY")
print(f"{'='*60}")
print(f"Model loading:       {model_load_time:7.2f}s")
print(f"Total preprocessing: {total_preprocess_time:7.2f}s ({total_preprocess_time/total_images_processed:.3f}s per image)")
print(f"Total inference:     {total_inference_time:7.2f}s ({total_inference_time/total_images_processed:.3f}s per image)")
print(f"Total postprocess:   {total_postprocess_time:7.2f}s ({total_postprocess_time/total_images_processed:.3f}s per image)")
print(f"Total visualization: {total_viz_time:7.2f}s ({total_viz_time/total_images_processed:.3f}s per image)")
print(f"{'='*60}")
print(f"Overall time:        {overall_time:7.2f}s")
print(f"Images processed:    {total_images_processed}")
print(f"Throughput:          {total_images_processed/overall_time:.2f} images/sec")
print(f"{'='*60}")
print(f"\nResults saved to {jsonl_path}")
