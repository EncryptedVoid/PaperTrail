import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from config import TIKA_APP_JAR_PATH
from utilities.dependancy_ensurance import ensure_apache_tika


def is_password_protected(file_path: Path):
    """
    Check if a file is password-protected.

    Supports: ZIP, PDF, DOCX, XLSX, PPTX, RAR, 7Z

    Args:
                    file_path: Path to the file

    Returns:
                    bool: True if password-protected, False otherwise
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    # ZIP files
    if ext == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            for info in zf.infolist():
                if info.flag_bits & 0x1:
                    return True
        return False

    # PDF files
    elif ext == ".pdf":
        try:
            import pymupdf  # aka fitz

            doc = pymupdf.open(file_path)
            is_protected = doc.needs_pass or doc.is_encrypted
            doc.close()
            return is_protected
        except ImportError:
            print("Install pymupdf: pip install pymupdf")
            return False

    # Office files (DOCX, XLSX, PPTX)
    elif ext in [".docx", ".xlsx", ".pptx", ".docm", ".xlsm", ".pptm"]:
        try:
            import msoffcrypto

            with open(file_path, "rb") as f:
                office_file = msoffcrypto.OfficeFile(f)
                return office_file.is_encrypted()
        except ImportError:
            print("Install msoffcrypto-tool: pip install msoffcrypto-tool")
            return False

    # RAR files
    elif ext == ".rar":
        try:
            import rarfile

            with rarfile.RarFile(file_path) as rf:
                return any(info.needs_password() for info in rf.infolist())
        except ImportError:
            print("Install rarfile: pip install rarfile")
            return False

    # 7Z files
    elif ext == ".7z":
        try:
            import py7zr

            with py7zr.SevenZipFile(file_path, "r") as archive:
                return archive.needs_password()
        except ImportError:
            print("Install py7zr: pip install py7zr")
            return False

    return False  # Unsupported format


def is_file_empty(file_path: Path) -> bool:
    """
    Check if a file is empty (0 bytes).

    Args:
                    file_path: Path to the file

    Returns:
                    True if file is 0 bytes, False otherwise
    """
    return file_path.stat().st_size == 0


def is_file_corrupted(file_path: Path) -> bool:
    """
    Check if a file is corrupted by attempting to open/parse it.
    Supports: Images (PNG, JPG, GIF, etc.), ZIP, PDF, and generic binary validation.

    Args:
                    file_path: Path to the file

    Returns:
                    True if file appears corrupted, False if it can be opened successfully

    Note: This is not exhaustive - some corruption may not be detectable,
    and unknown file types will return False (assumed not corrupted).
    """
    if not file_path.exists():
        return True

    if file_path.stat().st_size == 0:
        return True  # Empty file is considered corrupted

    try:
        # Detect file type by reading magic bytes
        with open(file_path, "rb") as f:
            header = f.read(16)

        # Check ZIP files (including Office formats)
        if header[:4] == b"PK\x03\x04" or header[:4] == b"PK\x05\x06":
            return _check_zip_corruption(file_path)

        # Check PDF files
        if header[:4] == b"%PDF":
            return _check_pdf_corruption(file_path)

        # Check image files
        if _is_image_header(header):
            return _check_image_corruption(file_path)

        # For unknown types, do basic validation
        return _check_basic_corruption(file_path)

    except Exception:
        return True  # If we can't even read the file, it's corrupted


def _check_zip_corruption(file_path: Path) -> bool:
    """Check if ZIP file is corrupted."""
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            # Test the ZIP file integrity
            bad_file = zf.testzip()
            return bad_file is not None
    except (zipfile.BadZipFile, EOFError, RuntimeError):
        return True
    except Exception:
        return True


def _check_pdf_corruption(file_path: Path) -> bool:
    """Check if PDF file is corrupted."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()

            # Basic PDF validation
            if not content.startswith(b"%PDF"):
                return True

            # Check for EOF marker
            if b"%%EOF" not in content[-1024:]:
                return True

            # Check for basic PDF structure
            if b"trailer" not in content:
                return True

        return False
    except Exception:
        return True


def _is_image_header(header: bytes) -> bool:
    """Check if header matches common image formats."""
    image_signatures = [
        b"\xff\xd8\xff",  # JPEG
        b"\x89PNG\r\n\x1a\n",  # PNG
        b"GIF87a",  # GIF87a
        b"GIF89a",  # GIF89a
        b"BM",  # BMP
        b"II\x2a\x00",  # TIFF (little-endian)
        b"MM\x00\x2a",  # TIFF (big-endian)
    ]
    return any(header.startswith(sig) for sig in image_signatures)


def _check_image_corruption(file_path: Path) -> bool:
    """Check if image file is corrupted."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity

        # Second pass to fully load the image
        with Image.open(file_path) as img:
            img.load()

        return False
    except Exception:
        return True


def _check_basic_corruption(file_path: Path) -> bool:
    """Basic corruption check for unknown file types."""
    try:
        with open(file_path, "rb") as f:
            # Try to read the entire file
            data = f.read()

            # If file is very small but not empty, might be truncated
            if len(data) < 16:
                return True

        return False
    except Exception:
        return True


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file's type is in the provided list of allowed extensions.
    Uses Apache Tika CLI to detect file type based on content.

    Args:
                    file_path: Path to the file

    Returns:
                    True if file type is in the list, False otherwise
    """
    ensure_apache_tika()

    try:
        # Run Tika command-line to detect MIME type
        result = subprocess.run(
            ["java", "-jar", str(TIKA_APP_JAR_PATH), "--detect", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return False

        # Get the MIME type from output
        mime_type = result.stdout.strip()

        # Map common MIME types to extensions
        mime_to_ext = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/msword": ".doc",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.ms-powerpoint": ".ppt",
            "text/plain": ".txt",
            "text/html": ".html",
            "text/csv": ".csv",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "application/zip": ".zip",
            "application/x-rar-compressed": ".rar",
            "application/x-7z-compressed": ".7z",
            "application/json": ".json",
            "application/xml": ".xml",
            "video/mp4": ".mp4",
            "video/x-msvideo": ".avi",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "application/x-msdownload": ".exe",
            "application/x-dosexec": ".exe",
        }

        # Get the extension from MIME type
        detected_ext = mime_to_ext.get(mime_type, "").lower()

        # Normalize the allowed extensions list
        normalized_extensions = []
        for ext in UNSUPPORTED_EXTENSIONS:
            ext_lower = ext.lower()
            if not ext_lower.startswith("."):
                ext_lower = "." + ext_lower
            normalized_extensions.append(ext_lower)

        return detected_ext in normalized_extensions

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
