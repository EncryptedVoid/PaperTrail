#!/usr/bin/env python3
"""
Simple Ollama HTML Converter
Uses qwen2.5vl:7b to recreate PDF documents as HTML.

Requirements:
    pip install ollama pillow PyMuPDF
    ollama pull qwen2.5vl:7b

Usage:
    python simple_ollama_converter.py input.pdf output_directory
"""

import base64
import io
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import ollama
from PIL import Image

# Hardcoded model - no detection bullshit
MODEL_NAME = "qwen2.5vl:7b"


def pdf_page_to_image(pdf_path, page_num=0, dpi=150):
    """Convert a PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Render page to image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")

    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))
    doc.close()

    return img


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def convert_with_qwen(image):
    """Convert image to HTML using Qwen2.5-VL vision model."""

    # Convert image to base64
    img_b64 = image_to_base64(image)

    # Better prompt to avoid repetition and get complete output
    prompt = """Create HTML that recreates this document exactly. Focus on:

1. Extract ALL visible text content
2. Position elements using absolute positioning
3. Include table borders and lines
4. Use simple, clean CSS - NO nested classes or repetition
5. Keep CSS concise and efficient
6. Make sure the HTML is COMPLETE

Return ONLY valid HTML with embedded CSS. Do not repeat CSS rules. Make it concise but accurate."""

    try:
        print(f"Sending image to {MODEL_NAME}...")
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images=[img_b64],
            options={
                "temperature": 0.0,  # Zero temperature for consistency
                "num_predict": 4096,  # More tokens
                "repeat_penalty": 1.2,  # Prevent repetition
                "top_k": 10,  # More focused
                "top_p": 0.9,
            },
        )

        return response["response"]

    except Exception as e:
        print(f"Error with Ollama: {e}")
        return None


def main():
    input_file = r"/src/diagnostic_tools\2024 T4.pdf"
    output_dir = r"/src/diagnostic_tools"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {MODEL_NAME}")

    try:
        # Convert PDF to image
        print(f"Converting PDF page to image...")
        if input_file.lower().endswith(".pdf"):
            image = pdf_page_to_image(
                input_file, page_num=0, dpi=200
            )  # Higher DPI for better quality
        else:
            # Assume it's an image file
            image = Image.open(input_file)

        print(f"Image size: {image.size}")

        # Get HTML from Qwen2.5-VL
        html_content = convert_with_qwen(image)

        if html_content:
            # Clean up the response (remove markdown if present)
            html_content = html_content.strip()
            if html_content.startswith("```html"):
                html_content = html_content[7:]
            if html_content.startswith("```"):
                html_content = html_content[3:]
            if html_content.endswith("```"):
                html_content = html_content[:-3]
            html_content = html_content.strip()

            # Save HTML file
            base_name = Path(input_file).stem
            output_file = output_path / f"{base_name}_qwen2.5vl.html"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"✅ Success! Generated HTML: {output_file}")
            print(f"📖 Open {output_file} in your browser to see the result.")

        else:
            print("❌ Failed to generate HTML from Qwen2.5-VL vision model.")
            print(f"Make sure you have the model installed: ollama pull qwen2.5vl:7b")

    except Exception as e:
        print(f"Error: {e}")
        if "not found" in str(e):
            print(f"\n🚨 Model not found! Run this first:")
            print(f"   ollama pull qwen2.5vl:7b")
        sys.exit(1)


if __name__ == "__main__":
    main()
