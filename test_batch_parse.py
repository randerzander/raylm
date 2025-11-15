import os
import base64
import json
import requests
from pathlib import Path


def read_image_as_base64(path):
    """Read an image file and return base64 encoded string with mime type."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return b64, "image/jpeg"


def test_batch_parse():
    """Test if NVIDIA NeMo Parse API supports multiple images in one request."""
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable not set")
        return
    
    nvai_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    # Find JPEG files in extracts/multimodal_test/pages_jpg/
    jpg_dir = Path("extracts/multimodal_test/pages_jpg")
    if not jpg_dir.exists():
        print(f"Error: Directory {jpg_dir} does not exist")
        return
    
    jpg_files = sorted(jpg_dir.glob("*.jpg"))[:3]  # Test with first 3 images
    
    if not jpg_files:
        print(f"Error: No JPEG files found in {jpg_dir}")
        return
    
    print(f"Testing batch parse with {len(jpg_files)} images:")
    for jpg in jpg_files:
        print(f"  - {jpg.name}")
    print()
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Test 1: Multiple images in a single message
    print("=== Test 1: Multiple <img> tags in one message ===")
    media_tags = []
    for jpg_file in jpg_files:
        b64_str, mime = read_image_as_base64(jpg_file)
        media_tags.append(f'<img src="data:{mime};base64,{b64_str}" />')
    
    content = "\n".join(media_tags)
    
    tool_spec = [{
        "type": "function",
        "function": {"name": "markdown_no_bbox"}
    }]
    
    inputs = {
        "model": "nvidia/nemotron-parse",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "tools": tool_spec,
        "tool_choice": {"type": "function", "function": {"name": "markdown_no_bbox"}},
        "max_tokens": 4096,
    }
    
    try:
        print(f"Sending request with {len(jpg_files)} images in one message...")
        response = requests.post(nvai_url, headers=headers, json=inputs, timeout=120)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Request succeeded!")
            
            # Check how many tool calls we got back
            tool_calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
            print(f"Number of tool calls returned: {len(tool_calls)}")
            
            # Try to extract markdown from each tool call
            for i, tool_call in enumerate(tool_calls):
                arguments_str = tool_call.get("function", {}).get("arguments", "")
                if arguments_str:
                    arguments = json.loads(arguments_str)
                    markdown_content = arguments[0].get("text", "") if arguments else ""
                    print(f"\nTool call {i+1} - Markdown length: {len(markdown_content)} chars")
                    print(f"First 200 chars: {markdown_content[:200]}")
            
            # Save full response
            with open("batch_parse_test_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nFull response saved to: batch_parse_test_result.json")
        else:
            print("✗ Request failed!")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80 + "\n")
    
    # Test 2: Multiple messages (unlikely to work but worth trying)
    print("=== Test 2: Multiple messages (one image each) ===")
    
    messages = []
    for jpg_file in jpg_files:
        b64_str, mime = read_image_as_base64(jpg_file)
        media_tag = f'<img src="data:{mime};base64,{b64_str}" />'
        messages.append({
            "role": "user",
            "content": media_tag
        })
    
    inputs2 = {
        "model": "nvidia/nemotron-parse",
        "messages": messages,
        "tools": tool_spec,
        "tool_choice": {"type": "function", "function": {"name": "markdown_no_bbox"}},
        "max_tokens": 4096,
    }
    
    try:
        print(f"Sending request with {len(messages)} messages (one image each)...")
        response = requests.post(nvai_url, headers=headers, json=inputs2, timeout=120)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Request succeeded!")
            
            tool_calls = result.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
            print(f"Number of tool calls returned: {len(tool_calls)}")
            
            with open("batch_parse_test_result2.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"Full response saved to: batch_parse_test_result2.json")
        else:
            print("✗ Request failed!")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("\nConclusion:")
    print("If Test 1 returned multiple tool calls, batching is supported!")
    print("If Test 1 returned only one tool call, batching is NOT supported.")
    print("If Test 2 succeeded, conversation-style batching might work.")


if __name__ == "__main__":
    test_batch_parse()
