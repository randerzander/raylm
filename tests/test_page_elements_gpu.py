#!/usr/bin/env python3
"""Test page_elements stage to verify GPU usage."""

import sys
from pathlib import Path
import time
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stages import page_elements


def check_gpu_usage():
    """Check current GPU usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(',')
                if len(parts) == 2:
                    gpu_util = int(parts[0].strip())
                    mem_used = int(parts[1].strip())
                    gpu_info.append((gpu_util, mem_used))
            return gpu_info
    except Exception as e:
        print(f"Error checking GPU: {e}")
    return None


def main():
    # Find test images
    test_images = list(Path("extracts/multimodal_test/pages_jpg").glob("*.jpg"))
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    print(f"Testing with: {test_images[0]}")
    
    # Check initial GPU state
    print("\n=== Initial GPU State ===")
    initial_gpu = check_gpu_usage()
    if initial_gpu:
        for i, (util, mem) in enumerate(initial_gpu):
            print(f"GPU {i}: {util}% utilization, {mem} MB memory")
    else:
        print("Could not check GPU (nvidia-smi not available or no GPUs)")
    
    # Prepare test data
    test_row = {
        "jpeg_file": str(test_images[0]),
        "output_dir": "extracts",
        "source_filename": "test.pdf",
        "page_number": 1
    }
    
    print("\n=== Running page_elements ===")
    start_time = time.time()
    
    # Run inference
    result = page_elements(test_row)
    
    elapsed = time.time() - start_time
    
    print(f"\nProcessing time: {elapsed:.3f}s")
    print(f"Detections: {result['num_detections']}")
    print(f"Model time: {result.get('model_time', 0):.3f}s")
    
    # Check GPU state during/after inference
    print("\n=== GPU State After Inference ===")
    final_gpu = check_gpu_usage()
    if final_gpu:
        for i, (util, mem) in enumerate(final_gpu):
            print(f"GPU {i}: {util}% utilization, {mem} MB memory")
            if initial_gpu and i < len(initial_gpu):
                mem_diff = mem - initial_gpu[i][1]
                print(f"  Memory delta: +{mem_diff} MB")
    
    # Run a second inference to see sustained GPU usage
    print("\n=== Running second inference to check GPU caching ===")
    start_time = time.time()
    result2 = page_elements(test_row)
    elapsed2 = time.time() - start_time
    
    print(f"Processing time (2nd run): {elapsed2:.3f}s")
    print(f"Model time (2nd run): {result2.get('model_time', 0):.3f}s")
    print(f"Speedup from caching: {elapsed/elapsed2:.2f}x")
    
    # Final GPU check
    print("\n=== Final GPU State ===")
    final_gpu2 = check_gpu_usage()
    if final_gpu2:
        for i, (util, mem) in enumerate(final_gpu2):
            print(f"GPU {i}: {util}% utilization, {mem} MB memory")
    
    # Summary
    print("\n=== Summary ===")
    if initial_gpu and final_gpu2:
        total_mem_increase = sum(f[1] - i[1] for f, i in zip(final_gpu2, initial_gpu))
        if total_mem_increase > 100:  # More than 100MB increase suggests GPU usage
            print("✅ GPU appears to be in use (significant memory increase)")
        else:
            print("⚠️  GPU memory increase minimal - may be using CPU")
    else:
        print("⚠️  Could not verify GPU usage (nvidia-smi not available)")
    
    print(f"\nProcessing speed: {1/elapsed:.2f} images/sec")


if __name__ == "__main__":
    main()
