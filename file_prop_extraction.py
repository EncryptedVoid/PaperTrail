from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union
import mimetypes

# Document libraries
from docx import Document
import openpyxl
from pptx import Presentation

# Image libraries
from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif  # For HEIC/HEIF support

try:
    import pyexiv2

    HAS_PYEXIV2 = True
except ImportError:
    HAS_PYEXIV2 = False
    print("pyexiv2 not available - will use basic EXIF only")
# PDF libraries
try:
    import fitz  # pymupdf

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

pillow_heif.register_heif_opener()

# Supported formats
IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
    ".heif",
    ".raw",
    ".cr2",
    ".nef",
    ".arw",
}

DOCUMENT_FORMATS = {".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}


def _get_basic_info(filepath: Path) -> Dict[str, Any]:
    """Extract basic file information"""
    try:
        stat = filepath.stat()
        mime_type, _ = mimetypes.guess_type(str(filepath))

        return {
            "filename": filepath.name,
            "path": str(filepath.absolute()),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": filepath.suffix.lower(),
            "mime_type": mime_type,
        }
    except Exception as e:
        return {"basic_info_error": str(e)}


def _extract_image_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract comprehensive image metadata"""
    result = {"type": "image"}

    try:
        # Basic image info with PIL
        with Image.open(filepath) as img:
            result["image_info"] = {
                "dimensions": img.size,
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "format": img.format,
                "has_transparency": img.mode in ("RGBA", "LA")
                or "transparency" in img.info,
            }

            # Basic EXIF from PIL
            exif_dict = {}
            if hasattr(img, "_getexif") and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value

            result["basic_exif"] = exif_dict

    except Exception as e:
        result["image_error"] = str(e)

    # Advanced metadata with pyexiv2 (if available)
    if HAS_PYEXIV2:
        try:
            img_meta = pyexiv2.Image(str(filepath))
            result["advanced_metadata"] = {
                "exif": img_meta.read_exif(),
                "iptc": img_meta.read_iptc(),
                "xmp": img_meta.read_xmp(),
            }
            img_meta.close()
        except Exception as e:
            result["advanced_metadata_error"] = str(e)

    return result


def _extract_document_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract document metadata based on file type"""
    result = {"type": "document"}

    if filepath.suffix.lower() == ".pdf":
        result.update(_extract_pdf_metadata(filepath))
    elif filepath.suffix.lower() == ".docx":
        result.update(_extract_docx_metadata(filepath))
    elif filepath.suffix.lower() == ".xlsx":
        result.update(_extract_xlsx_metadata(filepath))
    elif filepath.suffix.lower() == ".pptx":
        result.update(_extract_pptx_metadata(filepath))
    else:
        result["warning"] = "Document format not fully supported"

    return result


def _extract_pdf_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract PDF metadata"""
    result = {}

    # Try pymupdf first (better metadata)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(filepath))
            result["pdf_info"] = {
                "page_count": doc.page_count,
                "is_encrypted": doc.needs_pass,
                "metadata": doc.metadata,
                "toc": doc.get_toc(),  # Table of contents
            }
            doc.close()
        except Exception as e:
            result["pymupdf_error"] = str(e)

    # Try pdfplumber for text info
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(filepath) as pdf:
                result["text_info"] = {
                    "page_count": len(pdf.pages),
                    "has_text": bool(
                        pdf.pages[0].extract_text() if pdf.pages else False
                    ),
                }
        except Exception as e:
            result["pdfplumber_error"] = str(e)

    return result


def _extract_docx_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract Word document metadata"""
    try:
        doc = Document(str(filepath))
        props = doc.core_properties

        return {
            "document_info": {
                "title": props.title,
                "author": props.author,
                "subject": props.subject,
                "created": props.created.isoformat() if props.created else None,
                "modified": props.modified.isoformat() if props.modified else None,
                "last_modified_by": props.last_modified_by,
                "revision": props.revision,
                "category": props.category,
                "comments": props.comments,
                "keywords": props.keywords,
                "language": props.language,
                "paragraph_count": len(doc.paragraphs),
                "has_tables": len(doc.tables) > 0,
                "table_count": len(doc.tables),
            }
        }
    except Exception as e:
        return {"docx_error": str(e)}


def _extract_xlsx_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract Excel metadata"""
    try:
        wb = openpyxl.load_workbook(str(filepath), data_only=True)
        props = wb.properties

        return {
            "spreadsheet_info": {
                "title": props.title,
                "creator": props.creator,
                "created": props.created.isoformat() if props.created else None,
                "modified": props.modified.isoformat() if props.modified else None,
                "last_modified_by": props.lastModifiedBy,
                "subject": props.subject,
                "description": props.description,
                "keywords": props.keywords,
                "category": props.category,
                "sheet_names": wb.sheetnames,
                "sheet_count": len(wb.worksheets),
            }
        }
    except Exception as e:
        return {"xlsx_error": str(e)}


def _extract_pptx_metadata(filepath: Path) -> Dict[str, Any]:
    """Extract PowerPoint metadata"""
    try:
        prs = Presentation(str(filepath))
        props = prs.core_properties

        return {
            "presentation_info": {
                "title": props.title,
                "author": props.author,
                "created": props.created.isoformat() if props.created else None,
                "modified": props.modified.isoformat() if props.modified else None,
                "last_modified_by": props.last_modified_by,
                "subject": props.subject,
                "category": props.category,
                "comments": props.comments,
                "keywords": props.keywords,
                "slide_count": len(prs.slides),
            }
        }
    except Exception as e:
        return {"pptx_error": str(e)}


def extract_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Main method to extract all metadata from a file"""
    filepath = Path(filepath)

    if not filepath.exists():
        return {"error": "File not found"}

    # Get basic file info
    result = _get_basic_info(filepath)

    # Route to appropriate extractor
    if filepath.suffix.lower() in IMAGE_FORMATS:
        result.update(_extract_image_metadata(filepath))
    elif filepath.suffix.lower() in DOCUMENT_FORMATS:
        result.update(_extract_document_metadata(filepath))
    else:
        result["warning"] = f"Unsupported format: {filepath.suffix}"

    return result


# Simple usage
def main():
    # Extract from multiple files
    files = [
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (1).pdf",
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (3).pdf",
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (5).pdf",
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\2024 WORK PERMIT.JPG",
    ]
    for file in files:
        if Path(file).exists():
            metadata = extract_metadata(file)
            print(f"\n--- {file} ---")
            print(metadata)
            print()


if __name__ == "__main__":
    main()
