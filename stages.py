"""Utility functions for PDF and text processing with Ray Data."""

import os
import base64
import json
import mimetypes
import requests
import time
import pypdfium2 as pdfium
from pathlib import Path
from openai import OpenAI
from markitdown import MarkItDown
import lancedb
import pandas as pd
import torch
import numpy as np
from PIL import Image
from nemotron_table_structure_v1 import define_model, YoloXWrapper


NVAI_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

TOOLS = [
    "markdown_bbox",
    "markdown_no_bbox",
    "detection_only",
]


def get_source_dir_name(source_filename):
    """Get directory name from source filename (strips extension if present)."""
    return Path(source_filename).stem


def read_image_as_base64(path):
    """Read an image file and encode it as base64."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    return b64, mime


def generate_content(task_id, b64_str, mime):
    """Generate content for NVIDIA API request."""
    if task_id < 0 or task_id >= len(TOOLS):
        raise ValueError(f"task_id should be within [0, {len(TOOLS)-1}]")
    tool_name = TOOLS[task_id]
    media_tag = f'<img src="data:{mime};base64,{b64_str}" />'
    content = f"{media_tag}"
    tool_spec = [{"type": "function", "function": {"name": tool_name}}]
    return content, tool_spec, tool_name


def extract_subimage(image_path, bbox, output_path=None):
    """Extract a sub-image from an image given a bounding box.
    
    Args:
        image_path: Path to the source image file
        bbox: Bounding box as [x1, y1, x2, y2] or [x_min, y_min, x_max, y_max]
        output_path: Optional path to save the extracted sub-image
        
    Returns:
        PIL.Image: The extracted sub-image, or None if bbox is invalid
    """
    img = Image.open(image_path)
    
    # Ensure bbox coordinates are integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validate bounding box
    if x1 >= x2 or y1 >= y2:
        return None
    
    # Clamp to image bounds
    width, height = img.size
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Check again after clamping
    if x1 >= x2 or y1 >= y2:
        return None
    
    # Crop the image
    sub_img = img.crop((x1, y1, x2, y2))
    
    # Save if output path is provided and image is valid
    if output_path and sub_img.size[0] > 0 and sub_img.size[1] > 0:
        sub_img.save(output_path)
    
    return sub_img


def split_pdf(row):
    """Map function to split a PDF into individual pages."""
    pdf_path = Path(row["pdf_path"])
    output_dir = Path(row["output_dir"])
    
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
        
        page_files.append({
            "page_file": str(output_file),
            "source_filename": pdf_path.name,
            "page_number": i + 1
        })
    
    pdf.close()
    
    return page_files


def render_to_jpeg(row):
    """Map function to render a single-page PDF to JPEG and extract text."""
    pdf_path = Path(row["page_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    scale = row.get("scale", 2.0)
    
    # Use stem for directory (without extension)
    source_dir_name = Path(source_filename).stem
    
    # Create output directories for this PDF's pages
    pdf_output_dir = output_dir / source_dir_name / "pages_jpg"
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    text_output_dir = output_dir / source_dir_name / "pages_text"
    text_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the PDF
    pdf = pdfium.PdfDocument(pdf_path)
    
    # Render the page (should only be one page)
    page = pdf[0]
    pil_image = page.render(scale=scale).to_pil()
    
    # Extract text from page
    textpage = page.get_textpage()
    page_text = textpage.get_text_range()
    
    # Save as JPEG with same name structure
    output_file = pdf_output_dir / f"{pdf_path.stem}.jpg"
    pil_image.save(output_file, "JPEG", quality=95)
    
    # Save text
    text_file = text_output_dir / f"{pdf_path.stem}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(page_text)
    
    pdf.close()
    
    return {
        "jpeg_file": str(output_file),
        "text_file": str(text_file),
        "source_filename": source_filename,
        "page_number": row["page_number"],
        "text": page_text
    }


def chunk_text(row):
    """Map function to chunk text files using character-based chunking."""
    text_path = Path(row["text_path"])
    output_dir = Path(row["output_dir"])
    max_tokens = row.get("max_tokens", 4096)
    
    # Create output directory for this text file
    source_filename = text_path.stem
    text_output_dir = output_dir / source_filename / "text_chunks"
    text_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read text content
    with open(text_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    
    # Simple character-based chunking (~4 chars per token average)
    chunk_size = max_tokens * 4
    chunks = []
    
    for i in range(0, len(text_content), chunk_size):
        chunk = text_content[i:i + chunk_size]
        if chunk.strip():
            chunks.append({
                "chunk_text": chunk,
                "chunk_number": len(chunks) + 1,
                "source_filename": source_filename
            })
    
    # Save chunks and return metadata
    chunk_files = []
    for i, chunk_data in enumerate(chunks):
        chunk_file = text_output_dir / f"chunk_{i+1:03d}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk_data["chunk_text"])
        chunk_files.append({
            "chunk_file": str(chunk_file),
            "source_filename": source_filename,
            "chunk_number": i + 1,
            "text": chunk_data["chunk_text"]
        })
    
    return chunk_files


def convert_html_to_markdown(row):
    """Map function to convert HTML files to markdown using markitdown."""
    html_path = Path(row["html_path"])
    output_dir = Path(row["output_dir"])
    
    # Create output directory for this HTML file
    source_filename = html_path.stem
    html_output_dir = output_dir / source_filename / "html_md"
    html_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert HTML to markdown
    md_converter = MarkItDown()
    result = md_converter.convert(str(html_path))
    markdown_content = result.text_content
    
    # Save markdown
    md_file = html_output_dir / f"{source_filename}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return [{
        "md_file": str(md_file),
        "source_filename": source_filename,
        "text": markdown_content
    }]


def page_elements(row):
    """Map function to parse a JPEG image using local page-elements-v3 model."""
    from model import define_model
    from utils import postprocess_preds_page_element
    
    image_path = Path(row["jpeg_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    source_dir_name = get_source_dir_name(source_filename)
    
    # Create output directories for this PDF's results
    json_output_dir = output_dir / source_dir_name / "pages_json"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (this will be cached by the actor)
    if not hasattr(page_elements, "model"):
        model = define_model("page_element_v3")
        # Force GPU device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        page_elements.model = model.to(device).eval()
        print(f"Page elements model loaded on device: {device}")
    
    model = page_elements.model
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    start_time = time.time()
    x = model.preprocess(img_array)
    x = x.unsqueeze(0)  # Add batch dimension
    
    with torch.inference_mode():
        preds = model(x, [orig_size])[0]
    
    model_time = time.time() - start_time
    
    # Post-process predictions
    boxes, labels, scores = postprocess_preds_page_element(
        preds, model.thresholds_per_class, model.labels
    )
    
    # Convert to detections list
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = model.labels[int(label)]
        detections.append({
            "label": label_name,
            "box": box.tolist() if hasattr(box, 'tolist') else box,
            "score": float(score)
        })
    
    # Save detections as JSON
    json_file = json_output_dir / f"{image_path.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "detections": detections,
            "num_detections": len(detections),
            "image": str(image_path)
        }, f, ensure_ascii=False, indent=2)
    
    # Create text summary for embedding
    text_summary = f"Page {row['page_number']} with {len(detections)} elements: " + \
                   ", ".join([f"{d['label']} ({d['score']:.2f})" for d in detections[:5]])
    
    return {
        "json_file": str(json_file),
        "jpeg_file": str(image_path),
        "source_filename": source_filename,
        "page_number": row["page_number"],
        "detections": detections,
        "num_detections": len(detections),
        "text": text_summary,
        "model_time": model_time
    }


def extract_page_elements(row):
    """Map function to extract sub-images for all detected elements from a page.
    
    Takes the output from page_elements and extracts each detected element
    as a separate image file. Handles both normalized (0-1) and pixel coordinates.
    """
    image_path = Path(row["jpeg_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    source_dir_name = get_source_dir_name(source_filename)
    page_number = row["page_number"]
    detections = row.get("detections", [])
    
    if not detections:
        return []
    
    # Get image dimensions for denormalization
    img = Image.open(image_path)
    img_width, img_height = img.size
    img.close()
    
    # Create output directory for element images
    elements_output_dir = output_dir / source_dir_name / "page_elements"
    elements_output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_elements = []
    
    for idx, detection in enumerate(detections):
        label = detection["label"]
        bbox = detection["box"]
        score = detection["score"]
        
        # Check if coordinates are normalized (0-1 range)
        if all(0 <= coord <= 1 for coord in bbox):
            # Denormalize to pixel coordinates
            x1 = bbox[0] * img_width
            y1 = bbox[1] * img_height
            x2 = bbox[2] * img_width
            y2 = bbox[3] * img_height
            pixel_bbox = [x1, y1, x2, y2]
        else:
            pixel_bbox = bbox
        
        # Create filename for this element
        element_filename = f"page_{page_number:03d}_element_{idx:03d}_{label}.jpg"
        element_path = elements_output_dir / element_filename
        
        # Extract sub-image
        sub_img = extract_subimage(image_path, pixel_bbox, str(element_path))
        
        # Skip if extraction failed
        if sub_img is None:
            continue
        
        extracted_elements.append({
            "element_image": str(element_path),
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": idx,
            "label": label,
            "bbox": bbox,
            "score": score
        })
    
    return extracted_elements


def process_table_structure(row):
    """Map function to process table images and detect structure.
    
    Uses nemotron-table-structure-v1 model to detect table structure
    including rows, columns, and cells.
    """
    from nemotron_table_structure_v1 import Exp, YoloXWrapper
    from nemotron_table_structure_v1.utils import postprocess_preds_table_structure
    
    element_image_path = Path(row["element_image"])
    output_dir = Path(row.get("output_dir", "extracts"))
    source_filename = row["source_filename"]
    source_dir_name = get_source_dir_name(source_filename)
    page_number = row["page_number"]
    element_index = row["element_index"]
    
    # Preserve original element metadata
    original_label = row.get("label", "table")
    original_bbox = row.get("bbox", None)
    original_score = row.get("score", None)
    
    # Create output directory for table structure results
    table_output_dir = output_dir / source_dir_name / "table_structure"
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (this will be cached by the actor)
    if not hasattr(process_table_structure, "model"):
        # Initialize config
        exp = Exp()
        # Force GPU device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exp.device = device
        
        # Update weights path to absolute path
        weights_path = Path(__file__).parent.parent / "nemotron-table-structure-v1" / "weights.pth"
        if weights_path.exists():
            exp.ckpt = str(weights_path)
        else:
            # Try relative to nemotron package location
            import nemotron_table_structure_v1
            pkg_path = Path(nemotron_table_structure_v1.__file__).parent.parent
            weights_path = pkg_path / "weights.pth"
            if weights_path.exists():
                exp.ckpt = str(weights_path)
        
        # Create model
        model = exp.get_model()
        ckpt = torch.load(exp.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        
        model = YoloXWrapper(model, exp)
        process_table_structure.model = model.eval().to(device)
        print(f"Table structure model loaded on device: {device}")
    
    model = process_table_structure.model
    
    # Load and preprocess image
    img = Image.open(element_image_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    start_time = time.time()
    x = model.preprocess(img_array)
    
    with torch.inference_mode():
        preds = model(x, orig_size)[0]
    
    model_time = time.time() - start_time
    
    # Post-process predictions
    boxes, labels, scores = postprocess_preds_table_structure(
        preds, model.threshold, model.labels
    )
    
    # Convert to detections list
    table_elements = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = model.labels[int(label)]
        table_elements.append({
            "label": label_name,
            "bbox": box.tolist() if hasattr(box, 'tolist') else box,
            "score": float(score)
        })
    
    # Save table structure as JSON
    json_filename = f"page_{page_number:03d}_element_{element_index:03d}_table_structure.json"
    json_file = table_output_dir / json_filename
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_image": str(element_image_path),
            "detections": table_elements,
            "num_detections": len(table_elements),
            "page_number": page_number,
            "element_index": element_index
        }, f, ensure_ascii=False, indent=2)
    
    return {
        "table_structure_json": str(json_file),
        "element_image": str(element_image_path),
        "source_filename": source_filename,
        "page_number": page_number,
        "element_index": element_index,
        "label": original_label,
        "bbox": original_bbox,
        "score": original_score,
        "table_elements": table_elements,
        "num_table_elements": len(table_elements),
        "model_time": model_time
    }


def process_chart_elements(row):
    """Map function to process chart images and detect graphic elements.
    
    Uses nemotron-graphic-elements-v1 model to detect chart elements
    including titles, axis labels, legends, and data annotations.
    """
    from nemotron_graphic_elements_v1 import Exp, YoloXWrapper
    from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element
    
    element_image_path = Path(row["element_image"])
    output_dir = Path(row.get("output_dir", "extracts"))
    source_filename = row["source_filename"]
    source_dir_name = get_source_dir_name(source_filename)
    page_number = row["page_number"]
    element_index = row["element_index"]
    
    # Preserve original element metadata
    original_label = row.get("label", "chart")
    original_bbox = row.get("bbox", None)
    original_score = row.get("score", None)
    
    # Create output directory for chart element results
    chart_output_dir = output_dir / source_dir_name / "chart_elements"
    chart_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (this will be cached by the actor)
    if not hasattr(process_chart_elements, "model"):
        # Initialize config
        exp = Exp()
        # Force GPU device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exp.device = device
        
        # Update weights path to absolute path
        weights_path = Path(__file__).parent.parent / "nemotron-graphic-elements-v1" / "weights.pth"
        if weights_path.exists():
            exp.ckpt = str(weights_path)
        else:
            # Try relative to nemotron package location
            import nemotron_graphic_elements_v1
            pkg_path = Path(nemotron_graphic_elements_v1.__file__).parent.parent
            weights_path = pkg_path / "weights.pth"
            if weights_path.exists():
                exp.ckpt = str(weights_path)
        
        # Create model
        model = exp.get_model()
        ckpt = torch.load(exp.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        
        model = YoloXWrapper(model, exp)
        process_chart_elements.model = model.eval().to(device)
        print(f"Chart elements model loaded on device: {device}")
    
    model = process_chart_elements.model
    
    # Load and preprocess image
    img = Image.open(element_image_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    start_time = time.time()
    x = model.preprocess(img_array)
    
    with torch.inference_mode():
        preds = model(x, orig_size)[0]
    
    model_time = time.time() - start_time
    
    # Post-process predictions
    boxes, labels, scores = postprocess_preds_graphic_element(
        preds, model.threshold, model.labels
    )
    
    # Convert to detections list
    chart_elements = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = model.labels[int(label)]
        chart_elements.append({
            "label": label_name,
            "bbox": box.tolist() if hasattr(box, 'tolist') else box,
            "score": float(score)
        })
    
    # Save chart elements as JSON
    json_filename = f"page_{page_number:03d}_element_{element_index:03d}_chart_elements.json"
    json_file = chart_output_dir / json_filename
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_image": str(element_image_path),
            "detections": chart_elements,
            "num_detections": len(chart_elements),
            "page_number": page_number,
            "element_index": element_index
        }, f, ensure_ascii=False, indent=2)
    
    return {
        "chart_elements_json": str(json_file),
        "element_image": str(element_image_path),
        "source_filename": source_filename,
        "page_number": page_number,
        "element_index": element_index,
        "label": original_label,
        "bbox": original_bbox,
        "score": original_score,
        "chart_elements": chart_elements,
        "num_chart_elements": len(chart_elements),
        "model_time": model_time
    }


def generate_table_markdown(table_elements, text_detections):
    """Generate markdown table from table structure and OCR text detections.
    
    Args:
        table_elements: List of detected table structure elements (cells, rows, columns)
        text_detections: List of OCR text detections with bounding boxes
        
    Returns:
        Markdown string representation of the table
    """
    # Organize table structure
    cells = [e for e in table_elements if e["label"] == "cell"]
    rows = [e for e in table_elements if e["label"] == "row"]
    columns = [e for e in table_elements if e["label"] == "column"]
    
    if not cells:
        # No cells detected, return simple text
        return " ".join([d["text_prediction"]["text"] for d in text_detections])
    
    # Sort cells by position (top to bottom, left to right)
    cells_sorted = sorted(cells, key=lambda c: (c["bbox"][1], c["bbox"][0]))
    
    # Assign OCR text to cells based on bounding box overlap
    cell_texts = []
    for cell in cells_sorted:
        cell_bbox = cell["bbox"]
        # Find OCR text that overlaps with this cell
        matching_texts = []
        for ocr_det in text_detections:
            # Extract bounding box from OCR detection
            points = ocr_det["bounding_box"]["points"]
            ocr_bbox = [
                points[0]["x"],
                points[0]["y"],
                points[2]["x"],
                points[2]["y"]
            ]
            
            # Calculate overlap (simple center-point test)
            ocr_center_x = (ocr_bbox[0] + ocr_bbox[2]) / 2
            ocr_center_y = (ocr_bbox[1] + ocr_bbox[3]) / 2
            
            # Check if OCR center is within cell bounds
            if (cell_bbox[0] <= ocr_center_x <= cell_bbox[2] and
                cell_bbox[1] <= ocr_center_y <= cell_bbox[3]):
                matching_texts.append(ocr_det["text_prediction"]["text"])
        
        cell_texts.append(" ".join(matching_texts) if matching_texts else "")
    
    # Estimate number of columns from structure or cell layout
    num_cols = len(columns) if columns else max(1, int(len(cells) ** 0.5))
    
    # Create markdown table
    markdown_lines = []
    for i in range(0, len(cell_texts), num_cols):
        row_cells = cell_texts[i:i+num_cols]
        # Pad row if needed
        while len(row_cells) < num_cols:
            row_cells.append("")
        markdown_lines.append("| " + " | ".join(row_cells) + " |")
        
        # Add header separator after first row
        if i == 0:
            markdown_lines.append("|" + "|".join([" --- " for _ in range(num_cols)]) + "|")
    
    return "\n".join(markdown_lines)


def process_ocr(row):
    """Map function to perform OCR on element images.
    
    Uses nemoretriever-ocr-v1 model running at localhost:8009 to extract
    text from images.
    """
    element_image_path = Path(row["element_image"])
    output_dir = Path(row.get("output_dir", "extracts"))
    source_filename = row["source_filename"]
    source_dir_name = get_source_dir_name(source_filename)
    page_number = row["page_number"]
    element_index = row["element_index"]
    
    # Extract metadata from upstream stages
    label = row.get("label", "unknown")
    element_bbox = row.get("bbox", None)
    element_score = row.get("score", None)
    
    # Table structure metadata (if available)
    table_elements = row.get("table_elements", None)
    table_structure_json = row.get("table_structure_json", None)
    
    # Chart elements metadata (if available)
    chart_elements = row.get("chart_elements", None)
    chart_elements_json = row.get("chart_elements_json", None)
    
    # Create output directory for OCR results
    ocr_output_dir = output_dir / source_dir_name / "ocr"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and encode image as base64
    with open(element_image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Prepare request
    invoke_url = "http://localhost:8009/v1/infer"
    headers = {
        "Accept": "application/json"
    }
    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        ]
    }
    
    # Call OCR service
    start_time = time.time()
    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        model_time = time.time() - start_time
        
        # Extract OCR text from response
        # Response format: {"data": [{"index": 0, "text_detections": [...]}]}
        ocr_text = ""
        text_detections = []
        if "data" in result and len(result["data"]) > 0:
            text_detections = result["data"][0].get("text_detections", [])
            # Extract text from each detection and combine
            texts = [detection["text_prediction"]["text"] for detection in text_detections]
            ocr_text = " ".join(texts)
        
        # Generate table markdown if this is a table with structure
        table_md = None
        if label == "table" and table_elements is not None and text_detections:
            table_md = generate_table_markdown(table_elements, text_detections)
        
        # Save OCR result as JSON with metadata
        json_filename = f"page_{page_number:03d}_element_{element_index:03d}_{label}_ocr.json"
        json_file = ocr_output_dir / json_filename
        
        ocr_metadata = {
            "source_filename": source_filename,
            "source_image": str(element_image_path),
            "page_number": page_number,
            "element_index": element_index,
            "element_type": label,
            "element_bbox": element_bbox,
            "element_score": element_score,
            "ocr_text": ocr_text,
            "full_response": result
        }
        
        # Add table markdown if generated
        if table_md is not None:
            ocr_metadata["table_md"] = table_md
        
        # Add table structure metadata if available
        if table_elements is not None:
            ocr_metadata["table_structure"] = {
                "table_elements": table_elements,
                "table_structure_json": table_structure_json
            }
        
        # Add chart elements metadata if available
        if chart_elements is not None:
            ocr_metadata["chart_structure"] = {
                "chart_elements": chart_elements,
                "chart_elements_json": chart_elements_json
            }
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_metadata, f, ensure_ascii=False, indent=2)
        
        # Also save as plain text
        txt_filename = f"page_{page_number:03d}_element_{element_index:03d}_{label}_ocr.txt"
        txt_file = ocr_output_dir / txt_filename
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        
        return {
            "ocr_json": str(json_file),
            "ocr_txt": str(txt_file),
            "element_image": str(element_image_path),
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": element_index,
            "label": label,
            "ocr_text": ocr_text,
            "table_md": table_md,
            "model_time": model_time,
            "success": True
        }
    
    except Exception as e:
        print(f"OCR failed for {element_image_path}: {e}")
        return {
            "ocr_json": None,
            "ocr_txt": None,
            "element_image": str(element_image_path),
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": element_index,
            "label": label,
            "ocr_text": "",
            "table_md": None,
            "model_time": 0.0,
            "success": False,
            "error": str(e)
        }


def create_table_markdown(row):
    """Map function to create markdown table from table structure and OCR results.
    
    Combines table structure detection (rows/columns/cells) with OCR text to
    create a markdown representation of the table.
    """
    import json
    
    # Get inputs from both table structure and OCR results
    table_structure_json = row.get("table_structure_json")
    ocr_json = row.get("ocr_json")
    output_dir = Path(row.get("output_dir", "extracts"))
    source_filename = row["source_filename"]
    page_number = row["page_number"]
    element_index = row["element_index"]
    
    if not table_structure_json or not ocr_json:
        return {
            "markdown_file": None,
            "markdown_text": "",
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": element_index,
            "success": False,
            "error": "Missing table structure or OCR data"
        }
    
    # Create output directory for markdown tables
    markdown_output_dir = output_dir / source_filename / "table_markdown"
    markdown_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load table structure
        with open(table_structure_json, "r") as f:
            table_data = json.load(f)
        
        # Load OCR results
        with open(ocr_json, "r") as f:
            ocr_data = json.load(f)
        
        # Extract detections
        table_detections = table_data.get("detections", [])
        ocr_detections = ocr_data.get("full_response", {}).get("data", [{}])[0].get("text_detections", [])
        
        # Organize table structure
        cells = [d for d in table_detections if d["label"] == "cell"]
        rows = [d for d in table_detections if d["label"] == "row"]
        columns = [d for d in table_detections if d["label"] == "column"]
        
        # Sort rows and columns by position
        rows.sort(key=lambda r: r["bbox"][1])  # Sort by y position
        columns.sort(key=lambda c: c["bbox"][0])  # Sort by x position
        
        # Create a simple table from OCR text and structure
        # For now, use a heuristic approach based on cell positions
        if cells and ocr_detections:
            # Sort cells by position (top to bottom, left to right)
            cells_sorted = sorted(cells, key=lambda c: (c["bbox"][1], c["bbox"][0]))
            
            # Assign OCR text to cells based on bounding box overlap
            cell_texts = []
            for cell in cells_sorted:
                cell_bbox = cell["bbox"]
                # Find OCR text that overlaps with this cell
                matching_texts = []
                for ocr_det in ocr_detections:
                    ocr_bbox = [
                        ocr_det["bounding_box"]["points"][0]["x"],
                        ocr_det["bounding_box"]["points"][0]["y"],
                        ocr_det["bounding_box"]["points"][2]["x"],
                        ocr_det["bounding_box"]["points"][2]["y"]
                    ]
                    # Check if OCR text is within cell bounds
                    if (ocr_bbox[0] >= cell_bbox[0] and ocr_bbox[2] <= cell_bbox[2] and
                        ocr_bbox[1] >= cell_bbox[1] and ocr_bbox[3] <= cell_bbox[3]):
                        matching_texts.append(ocr_det["text_prediction"]["text"])
                
                cell_texts.append(" ".join(matching_texts) if matching_texts else "")
            
            # Estimate number of columns from structure
            num_cols = len(columns) if columns else max(1, int(len(cells) ** 0.5))
            num_rows = len(rows) if rows else max(1, len(cells) // num_cols)
            
            # Create markdown table
            markdown_lines = []
            for i in range(0, len(cell_texts), num_cols):
                row_cells = cell_texts[i:i+num_cols]
                # Pad row if needed
                while len(row_cells) < num_cols:
                    row_cells.append("")
                markdown_lines.append("| " + " | ".join(row_cells) + " |")
                
                # Add header separator after first row
                if i == 0:
                    markdown_lines.append("|" + "|".join([" --- " for _ in range(num_cols)]) + "|")
            
            markdown_text = "\n".join(markdown_lines)
        else:
            # Fallback: just use OCR text
            markdown_text = ocr_data.get("ocr_text", "")
        
        # Save markdown file
        markdown_filename = f"page_{page_number:03d}_element_{element_index:03d}_table.md"
        markdown_file = markdown_output_dir / markdown_filename
        
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        return {
            "markdown_file": str(markdown_file),
            "markdown_text": markdown_text,
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": element_index,
            "success": True
        }
    
    except Exception as e:
        print(f"Failed to create markdown for table: {e}")
        return {
            "markdown_file": None,
            "markdown_text": "",
            "source_filename": source_filename,
            "page_number": page_number,
            "element_index": element_index,
            "success": False,
            "error": str(e)
        }


def parse_image(row):
    """Map function to parse a JPEG image using NVIDIA API."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")
    
    image_path = Path(row["jpeg_file"])
    output_dir = Path(row["output_dir"])
    source_filename = row["source_filename"]
    task_id = row.get("task_id", 1)
    
    # Create output directories for this PDF's results
    json_output_dir = output_dir / source_filename / "pages_json"
    md_output_dir = output_dir / source_filename / "pages_md"
    
    json_output_dir.mkdir(parents=True, exist_ok=True)
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Read and encode image
    b64_str, mime = read_image_as_base64(image_path)
    content, tool_spec, tool_name = generate_content(task_id, b64_str, mime)
    
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
    response = requests.post(NVAI_URL, headers=headers, json=inputs, timeout=120)
    response.raise_for_status()
    model_time = time.time() - start_time
    
    result = response.json()
    
    # Save JSON result
    json_file = json_output_dir / f"{image_path.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Extract and save markdown
    md_file = None
    markdown_content = ""
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
        "md_file": str(md_file) if md_file else None,
        "source_filename": source_filename,
        "page_number": row["page_number"],
        "text": markdown_content,
        "model_time": model_time
    }


class EmbeddingBatcher:
    """Stateful map function to batch embedding requests."""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
    
    def __call__(self, batch):
        """Process a batch of rows."""
        # Convert to pandas for easier handling
        df = pd.DataFrame(batch)
        
        # Get output_dir
        if "output_dir" in df.columns:
            output_dir = Path(df["output_dir"].iloc[0])
        else:
            output_dir = Path("extracts")
        
        # Filter rows with valid text
        df = df[df["text"].notna() & (df["text"] != "")]
        
        if len(df) == 0:
            print(f"EmbeddingBatcher: Batch had no valid text rows after filtering")
            return pd.DataFrame({
                "embedding_file": pd.Series([], dtype=object),
                "embedding_time": pd.Series([], dtype=float)
            })
        
        print(f"EmbeddingBatcher: Processing {len(df)} rows with valid text")
        texts = df["text"].tolist()
        
        # Generate embeddings for batch
        start_time = time.time()
        try:
            response = self.client.embeddings.create(
                input=texts,
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"] * len(texts), "input_type": "query", "truncate": "NONE"}
            )
            embedding_time = time.time() - start_time
            
            # Save each embedding
            embedding_files = []
            embedding_times = []
            
            for idx, (embedding_data, (_, row)) in enumerate(zip(response.data, df.iterrows())):
                source_name = row.get("source_filename")
                
                # Use stem for directory (without extension)
                source_dir_name = get_source_dir_name(source_name)
                
                # Determine output directory - all go to same embeddings folder
                embedding_output_dir = output_dir / source_dir_name / "embeddings"
                embedding_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Determine filename based on content type
                label = row.get("label", "unknown")
                page_num = int(row.get("page_number", 0))
                
                if label == "page_text":
                    # For page text, just use page number
                    filename = f"page_{page_num:03d}_text"
                else:
                    # For elements (tables/charts/infographics), include element index
                    elem_idx = int(row.get("element_index", idx))
                    filename = f"page_{page_num:03d}_element_{elem_idx:03d}_{label}"
                
                # Save embedding
                embedding_file = embedding_output_dir / f"{filename}.json"
                with open(embedding_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "embedding": embedding_data.embedding,
                        "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                        "text": row["text"]
                    }, f, ensure_ascii=False, indent=2)
                
                # Don't store embedding in dataframe, just the file path
                embedding_files.append(str(embedding_file))
                embedding_times.append(embedding_time / len(texts))
            
            # Add results to dataframe
            df = df.copy()
            df["embedding_file"] = embedding_files
            df["embedding_time"] = embedding_times
            
            return df
        
        except Exception as e:
            print(f"Error generating embeddings for batch: {e}")
            # Return dataframe with None values
            df = df.copy()
            df["embedding_file"] = None
            df["embedding_time"] = 0.0
            return df


def write_to_lancedb_batch(batch, db_path):
    """Map batches function to write embedding data to LanceDB."""
    # Convert to pandas if not already
    if not isinstance(batch, pd.DataFrame):
        df = pd.DataFrame(batch)
    else:
        df = batch
    
    records = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get("embedding_file")) or row.get("embedding_file") is None:
            continue
        
        try:
            # Load embedding data
            with open(row["embedding_file"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Get source info
            source_name = row.get("source_filename")
            chunk_sequence = row.get("page_number") or row.get("chunk_number") or 1
            
            # Prepare record
            record = {
                "source_id": source_name,
                "chunk_sequence": chunk_sequence,
                "text": data["text"],
                "vector": data["embedding"]
            }
            records.append(record)
        except Exception as e:
            print(f"Error loading embedding data: {e}")
            continue
    
    if not records:
        return pd.DataFrame({"success": [0], "failed": [0]})
    
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
        
        return pd.DataFrame({"success": [len(records)], "failed": [0]})
    except Exception as e:
        print(f"Error writing to LanceDB: {e}")
        return pd.DataFrame({"success": [0], "failed": [len(records)]})
