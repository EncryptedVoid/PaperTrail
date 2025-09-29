#!/usr/bin/env python3
"""
PDF to HTML Converter - Fixed Version
Converts PDF to HTML while preserving approximate layout and positioning.
Handles dark backgrounds and inverted pages properly.

Usage:
    python pdf_to_html_fixed.py input.pdf output_directory
"""

import base64
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF


def detect_page_colors(page):
    """Detect if page has white text on dark background."""
    try:
        text_data = page.get_text("dict")
        white_text_count = 0
        total_text_count = 0

        for block in text_data.get("blocks", [])[:10]:  # Check first 10 blocks
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        color = span.get("color", 0)
                        total_text_count += 1
                        # Check if text is white/very light
                        if color > 0xE0E0E0:  # Light text
                            white_text_count += 1

        # If more than 50% of text is light colored, assume dark background
        return (
            white_text_count > (total_text_count * 0.5)
            if total_text_count > 0
            else False
        )
    except:
        return False


def pdf_to_html(input_pdf, output_dir):
    """Convert PDF to HTML with preserved positioning."""

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the PDF
    doc = fitz.open(input_pdf)

    # Get the base name for output file
    base_name = Path(input_pdf).stem
    output_file = output_path / f"{base_name}.html"

    html_content = []
    html_content.append(
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Converted PDF</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px;
            background: #f0f0f0;
        }
        .page { 
            background: white; 
            margin-bottom: 20px; 
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: relative;
            min-height: 800px;
        }
        .text-block { 
            position: absolute; 
            font-size: 12px;
            line-height: 1.2;
            z-index: 10;
        }
        .page-number {
            text-align: center;
            color: #666;
            margin: 10px 0;
            font-weight: bold;
        }
        .page-border {
            position: absolute;
            z-index: 2;
        }
    </style>
</head>
<body>"""
    )

    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Detect if this page has inverted colors
        is_dark_page = detect_page_colors(page)
        page_bg = "black" if is_dark_page else "white"
        default_text_color = "white" if is_dark_page else "black"

        print(
            f"Page {page_num + 1}: {'Dark background' if is_dark_page else 'Light background'}"
        )

        html_content.append(f'<div class="page-number">Page {page_num + 1}</div>')
        html_content.append(
            f'<div class="page" style="width: {page_width}px; height: {page_height}px; background-color: {page_bg};">'
        )

        # Skip large background images that cause inversion
        try:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Skip very small images
                    if len(image_bytes) < 100:
                        continue

                    # Get basic image info
                    img_width = base_image.get("width", 100)
                    img_height = base_image.get("height", 100)

                    # Skip large images that cover most of the page (likely backgrounds)
                    if img_width > page_width * 0.7 and img_height > page_height * 0.7:
                        print(
                            f"Skipping large background image: {img_width}x{img_height}"
                        )
                        continue

                    # For smaller images, try to place them
                    img_b64 = base64.b64encode(image_bytes).decode()
                    img_ext = base_image["ext"]

                    # Estimate position (fallback)
                    x_pos = 50 + img_index * 120
                    y_pos = 50 + img_index * 100

                    html_content.append(
                        f'<img style="position: absolute; left: {x_pos}px; top: {y_pos}px; '
                        f'max-width: 200px; max-height: 200px; z-index: 1;" '
                        f'src="data:image/{img_ext};base64,{img_b64}" />'
                    )
                except Exception as e:
                    print(f"Warning: Could not process image {img_index}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not extract images from page {page_num + 1}: {e}")

        # Extract text with formatting
        try:
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:  # Only process non-empty text
                                bbox = span["bbox"]
                                x = bbox[0]
                                y = bbox[1]
                                font_size = span["size"]
                                font_flags = span["flags"]
                                color = span.get("color", 0)
                                font_name = span.get("font", "Arial")

                                # Escape HTML characters
                                text = (
                                    text.replace("&", "&amp;")
                                    .replace("<", "&lt;")
                                    .replace(">", "&gt;")
                                )

                                # Build style
                                styles = [
                                    f"left: {x}px",
                                    f"top: {y}px",
                                    f"font-size: {font_size}px",
                                    "z-index: 10",
                                ]

                                # Handle text color
                                if color != 0:
                                    hex_color = f"#{color:06x}"
                                    styles.append(f"color: {hex_color}")
                                else:
                                    # Use default color based on page background
                                    styles.append(f"color: {default_text_color}")

                                # Add formatting
                                if font_flags & 16:  # Bold (2^4)
                                    styles.append("font-weight: bold")
                                if font_flags & 2:  # Italic (2^1)
                                    styles.append("font-style: italic")

                                styles.append(
                                    f"font-family: '{font_name}', Arial, sans-serif"
                                )

                                style_string = "; ".join(styles)

                                html_content.append(
                                    f'<div class="text-block" style="{style_string};">{text}</div>'
                                )
        except Exception as e:
            print(f"Warning: Could not extract text from page {page_num + 1}: {e}")

        # Try to extract simple lines/borders
        try:
            drawings = page.get_drawings()
            for drawing in drawings:
                if "items" in drawing:
                    for item in drawing["items"]:
                        try:
                            if item[0] == "l" and len(item) >= 5:  # Line
                                x1, y1, x2, y2 = item[1:5]
                                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                                    html_content.append(
                                        f'<div class="page-border" style="left: {min(x1,x2)}px; top: {y1}px; '
                                        f'width: {abs(x2-x1)}px; height: 1px; background-color: gray;"></div>'
                                    )
                                else:  # Vertical line
                                    html_content.append(
                                        f'<div class="page-border" style="left: {x1}px; top: {min(y1,y2)}px; '
                                        f'width: 1px; height: {abs(y2-y1)}px; background-color: gray;"></div>'
                                    )
                        except:
                            continue
        except:
            pass

        html_content.append("</div>")  # Close page div

    html_content.append(
        """</body>
</html>"""
    )

    # Write HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"Converted PDF to HTML: {output_file}")
    print(f"Pages processed: {len(doc)}")

    # Close the PDF
    doc.close()

    return output_file


def main():
    input_pdf = r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\diagnostic_tools\65wr7tysf7yu3.pdf"
    output_dir = r"/src/diagnostic_tools"

    # Check if input file exists
    if not os.path.exists(input_pdf):
        print(f"Error: Input file '{input_pdf}' not found.")
        sys.exit(1)

    # Check if input is a PDF
    if not input_pdf.lower().endswith(".pdf"):
        print("Error: Input file must be a PDF.")
        sys.exit(1)

    try:
        result_file = pdf_to_html(input_pdf, output_dir)
        print(f"\n✅ Success! Open {result_file} in your browser to see the result.")

    except Exception as e:
        print(f"Error converting PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
