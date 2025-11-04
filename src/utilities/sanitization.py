"""
Sanitization Utility Functions

This module provides utility functions for validating and checking file integrity,
including password protection detection, corruption checking, empty file detection,
and file type validation.

Key functionality:
- Password protection detection for ZIP, PDF, Office documents (DOCX/XLSX/PPTX), RAR, and 7Z files
- File corruption detection for images, PDFs, ZIP archives, and generic binary files
- Empty file detection (zero-byte files)
- File type validation using Apache Tika content-based detection

These utilities are used by the sanitization pipeline to ensure only valid,
processable files are accepted for further processing.
"""

import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from config import TIKA_APP_JAR_PATH
from utilities.dependancy_ensurance import ensure_apache_tika


def is_password_protected(file_path: Path) -> bool:
    """
    Check if a file is password-protected or encrypted.

    This function detects password protection across multiple file formats by
    examining file headers, encryption flags, and format-specific metadata.
    Supported formats: ZIP, PDF, DOCX, XLSX, PPTX, RAR, 7Z

    Args:
                    file_path: Path object pointing to the file to check

    Returns:
                    bool: True if the file is password-protected or encrypted, False otherwise
                                            Returns False for unsupported formats or if required libraries are missing

    Raises:
                    FileNotFoundError: If the specified file does not exist

    Note:
                    This function requires optional dependencies for certain formats:
                    - pymupdf for PDF files
                    - msoffcrypto-tool for Office documents
                    - rarfile for RAR archives
                    - py7zr for 7Z archives
    """
    # Verify the file exists before attempting to check protection status
    # Path.exists() returns True if the path points to an existing file or directory
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract file extension and convert to lowercase for case-insensitive comparison
    # Path.suffix returns the file extension including the dot (e.g., '.zip')
    ext = file_path.suffix.lower()

    # Check ZIP files by examining the flag_bits in each file entry
    # The 0x1 bit indicates encryption in ZIP format
    if ext == ".zip":
        try:
            # zipfile.ZipFile opens ZIP archives for reading
            # Context manager (with statement) ensures proper file closure
            with zipfile.ZipFile(file_path) as zf:
                # zf.infolist() returns a list of ZipInfo objects for each file in archive
                for info in zf.infolist():
                    # Bitwise AND operation checks if encryption flag is set
                    if info.flag_bits & 0x1:
                        return True
            return False
        except Exception:
            # If ZIP file is malformed or unreadable, assume not protected
            return False

    # Check PDF files using PyMuPDF (fitz) library
    # PDFs can be password-protected or encrypted
    elif ext == ".pdf":
        try:
            # pymupdf.open() loads the PDF document into memory
            import pymupdf  # PyMuPDF library, imported as fitz in some versions

            doc = pymupdf.open(file_path)
            # needs_pass indicates if a password is required to open
            # is_encrypted indicates if the document has any encryption
            is_protected = doc.needs_pass or doc.is_encrypted
            # Always close the document to free resources
            doc.close()
            return is_protected
        except ImportError:
            # Library not installed - return False and inform user
            print("Install pymupdf: pip install pymupdf")
            return False
        except Exception:
            # PDF may be corrupted or in an unsupported format
            return False

    # Check Microsoft Office files (modern XML-based formats)
    # These formats are actually ZIP archives containing XML files
    elif ext in [".docx", ".xlsx", ".pptx", ".docm", ".xlsm", ".pptm"]:
        try:
            # msoffcrypto library handles Office encryption detection
            import msoffcrypto

            # Open file in binary mode ('rb') for msoffcrypto processing
            with open(file_path, "rb") as f:
                # OfficeFile class parses the Office document structure
                office_file = msoffcrypto.OfficeFile(f)
                # is_encrypted() checks for document-level encryption
                return office_file.is_encrypted()
        except ImportError:
            # Library not installed - return False and inform user
            print("Install msoffcrypto-tool: pip install msoffcrypto-tool")
            return False
        except Exception:
            # File may be corrupted or in legacy format
            return False

    # Check RAR archive files
    # RAR format supports per-file password protection
    elif ext == ".rar":
        try:
            # rarfile library provides RAR archive reading capabilities
            import rarfile

            # RarFile class opens RAR archives
            with rarfile.RarFile(file_path) as rf:
                # Check if any file in the archive requires a password
                # any() returns True if at least one element is True
                return any(info.needs_password() for info in rf.infolist())
        except ImportError:
            # Library not installed - return False and inform user
            print("Install rarfile: pip install rarfile")
            return False
        except Exception:
            # RAR file may be corrupted or in unsupported version
            return False

    # Check 7-Zip archive files
    # 7Z format supports archive-level encryption
    elif ext == ".7z":
        try:
            # py7zr library handles 7Z archive operations
            import py7zr

            # Open archive in read mode ('r')
            with py7zr.SevenZipFile(file_path, "r") as archive:
                # needs_password() checks if password is required for extraction
                return archive.needs_password()
        except ImportError:
            # Library not installed - return False and inform user
            print("Install py7zr: pip install py7zr")
            return False
        except Exception:
            # 7Z file may be corrupted
            return False

    # File format is not supported for password protection detection
    return False


def is_file_empty(file_path: Path) -> bool:
    """
    Check if a file is empty (contains zero bytes).

    Empty files may indicate failed downloads, interrupted transfers, or
    placeholder files that should not be processed.

    Args:
                    file_path: Path object pointing to the file to check

    Returns:
                    bool: True if file size is exactly 0 bytes, False otherwise

    Note:
                    This function does not raise exceptions. If the file doesn't exist
                    or cannot be accessed, stat() will raise an OSError.
    """
    # Path.stat() returns an os.stat_result object with file metadata
    # st_size attribute contains the file size in bytes
    return file_path.stat().st_size == 0


def is_file_corrupted(file_path: Path) -> bool:
    """
    Check if a file is corrupted by attempting to open and validate it.

    This function performs format-specific validation for known file types
    and basic integrity checks for unknown formats. Corruption detection is
    based on file headers (magic bytes), structural validation, and the
    ability to parse the file using appropriate libraries.

    Supported formats:
    - Images: PNG, JPG, GIF, BMP, TIFF
    - Archives: ZIP (including Office formats)
    - Documents: PDF
    - Generic: Basic binary validation for unknown types

    Args:
                    file_path: Path object pointing to the file to check

    Returns:
                    bool: True if file appears corrupted or cannot be validated,
                                            False if file structure is valid

    Note:
                    - Not all corruption is detectable; some files may appear valid but contain corrupted data
                    - Unknown file types undergo basic validation only
                    - Files that cannot be read at all are considered corrupted
    """
    # Check if file exists using Path.exists()
    if not file_path.exists():
        return True

    # Zero-byte files are considered corrupted as they contain no data
    if file_path.stat().st_size == 0:
        return True

    try:
        # Read the first 16 bytes of the file to identify its type
        # These are called "magic bytes" or file signatures
        with open(file_path, "rb") as f:
            # Read in binary mode ('rb') to get raw bytes
            header = f.read(16)

        # Check ZIP files (including Office formats like DOCX, XLSX, PPTX)
        # ZIP files start with "PK" signature (0x504B in hex)
        # 0x03 0x04 indicates local file header, 0x05 0x06 indicates end of central directory
        if header[:4] == b"PK\x03\x04" or header[:4] == b"PK\x05\x06":
            return _check_zip_corruption(file_path)

        # Check PDF files
        # PDF files must start with "%PDF" signature
        if header[:4] == b"%PDF":
            return _check_pdf_corruption(file_path)

        # Check image files by matching against known image format signatures
        if _is_image_header(header):
            return _check_image_corruption(file_path)

        # For unknown types, perform basic validation
        # This checks if the file is readable and has reasonable content
        return _check_basic_corruption(file_path)

    except Exception:
        # If any exception occurs during reading/validation, consider file corrupted
        return True


def _check_zip_corruption(file_path: Path) -> bool:
    """
    Check if a ZIP archive is corrupted.

    Uses Python's zipfile library to test ZIP integrity. The testzip() method
    reads all files in the archive and verifies their CRC checksums.

    Args:
                    file_path: Path to the ZIP file

    Returns:
                    bool: True if ZIP is corrupted, False if valid

    Note:
                    This is a thorough but potentially slow operation for large archives.
    """
    try:
        # Open ZIP file in read mode
        with zipfile.ZipFile(file_path, "r") as zf:
            # testzip() reads each file and checks CRC values
            # Returns name of first corrupted file, or None if all files are valid
            bad_file = zf.testzip()
            return bad_file is not None
    except (zipfile.BadZipFile, EOFError, RuntimeError):
        # BadZipFile: File is not a valid ZIP or is corrupted
        # EOFError: File was truncated or incomplete
        # RuntimeError: Other ZIP-related errors (e.g., unsupported compression)
        return True
    except Exception:
        # Catch-all for any other unexpected errors
        return True


def _check_pdf_corruption(file_path: Path) -> bool:
    """
    Check if a PDF file is corrupted.

    Performs basic structural validation by checking for required PDF elements:
    - File header (%PDF signature)
    - End-of-file marker (%%EOF)
    - Trailer dictionary

    Args:
                    file_path: Path to the PDF file

    Returns:
                    bool: True if PDF appears corrupted, False if basic structure is valid

    Note:
                    This is a lightweight check. Complex PDF corruption may not be detected.
    """
    try:
        # Read entire PDF file in binary mode
        with open(file_path, "rb") as f:
            content = f.read()

            # Validate PDF header signature
            # All valid PDFs must start with "%PDF"
            if not content.startswith(b"%PDF"):
                return True

            # Check for EOF (End Of File) marker in last 1KB of file
            # %%EOF indicates proper file termination
            if b"%%EOF" not in content[-1024:]:
                return True

            # Check for trailer dictionary
            # "trailer" keyword is required in PDF structure
            if b"trailer" not in content:
                return True

        # All basic checks passed
        return False
    except Exception:
        # File unreadable or other I/O error
        return True


def _is_image_header(header: bytes) -> bool:
    """
    Check if file header matches known image format signatures.

    Compares the file's first bytes against magic numbers for common
    image formats. This provides fast, reliable format detection.

    Args:
                    header: First 16 bytes of the file

    Returns:
                    bool: True if header matches any known image format, False otherwise

    Supported formats:
                    - JPEG (FFD8FF)
                    - PNG (89504E47 0D0A1A0A)
                    - GIF (GIF87a / GIF89a)
                    - BMP (424D)
                    - TIFF (little-endian: 49492A00 / big-endian: 4D4D002A)
    """
    # Define magic bytes (file signatures) for common image formats
    image_signatures = [
        b"\xff\xd8\xff",  # JPEG - starts with FFD8FF marker
        b"\x89PNG\r\n\x1a\n",  # PNG - 8-byte signature with line endings
        b"GIF87a",  # GIF version 87a
        b"GIF89a",  # GIF version 89a
        b"BM",  # BMP - Windows bitmap
        b"II\x2a\x00",  # TIFF little-endian (Intel byte order)
        b"MM\x00\x2a",  # TIFF big-endian (Motorola byte order)
    ]
    # Check if header starts with any known signature
    # any() returns True if at least one signature matches
    return any(header.startswith(sig) for sig in image_signatures)


def _check_image_corruption(file_path: Path) -> bool:
    """
    Check if an image file is corrupted.

    Uses PIL (Pillow) library to validate image files. Performs two-pass
    validation: first verifies the image structure, then attempts to fully
    load the image data.

    Args:
                    file_path: Path to the image file

    Returns:
                    bool: True if image is corrupted, False if valid

    Note:
                    This method catches most common image corruption but may not detect
                    all forms of data corruption.
    """
    try:
        # First pass: Verify image structure without loading pixel data
        # Image.open() opens the image file and reads headers
        with Image.open(file_path) as img:
            # verify() checks image file integrity without decoding pixels
            img.verify()

        # Second pass: Fully load the image to catch additional corruption
        # Some corruption is only detected when pixel data is decoded
        with Image.open(file_path) as img:
            # load() forces decompression and loading of all pixel data
            img.load()

        # Both passes succeeded - image is valid
        return False
    except Exception:
        # Any PIL exception indicates corruption or unsupported format
        return True


def _check_basic_corruption(file_path: Path) -> bool:
    """
    Basic corruption check for unknown file types.

    Performs minimal validation by attempting to read the entire file
    and checking for suspicious characteristics like abnormally small
    file sizes.

    Args:
                    file_path: Path to the file

    Returns:
                    bool: True if file appears corrupted, False if readable

    Note:
                    This is a last-resort check for unknown formats and may not
                    detect all forms of corruption.
    """
    try:
        # Attempt to read entire file
        with open(file_path, "rb") as f:
            # Read all bytes from the file
            data = f.read()

            # Files smaller than 16 bytes (but not empty) are suspicious
            # Most legitimate files have headers/metadata exceeding this
            if len(data) < 16:
                return True

        # File is readable and has reasonable size
        return False
    except Exception:
        # File unreadable - consider it corrupted
        return True


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file's type is supported for processing.

    Uses Apache Tika content-based detection to identify file types based on
    actual file content rather than just file extensions. This prevents
    processing of files with incorrect or misleading extensions.

    Tika analyzes the file's magic bytes and internal structure to determine
    its true MIME type, then compares against a whitelist of supported types.

    Args:
                    file_path: Path object pointing to the file to check

    Returns:
                    bool: True if file type is supported, False otherwise

    Note:
                    - Requires Apache Tika JAR file to be available at TIKA_APP_JAR_PATH
                    - Requires Java Runtime Environment (JRE) to execute Tika
                    - Processing timeout is set to 30 seconds per file
                    - Unsupported or unrecognized MIME types return False

    Raises:
                    No exceptions are raised; all errors result in False return value
    """
    # Ensure Apache Tika JAR file is downloaded and available
    # This function downloads Tika if not already present
    ensure_apache_tika()

    try:
        # Execute Apache Tika as a subprocess using Java
        # subprocess.run() launches an external process and waits for completion
        result = subprocess.run(
            [
                "java",  # Java Runtime Environment command
                "-jar",  # Run JAR file
                str(TIKA_APP_JAR_PATH),  # Path to Tika JAR
                "--detect",  # Tika command to detect MIME type
                str(file_path),  # File to analyze
            ],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Return output as string (not bytes)
            timeout=30,  # Timeout after 30 seconds
        )

        # Check if Tika command executed successfully
        # returncode 0 indicates success, non-zero indicates error
        if result.returncode != 0:
            return False

        # Extract MIME type from Tika's stdout and remove whitespace
        # MIME type format: type/subtype (e.g., "application/pdf")
        mime_type = result.stdout.strip()

        # Map MIME types to file extensions
        # This dictionary defines which content types are supported
        mime_to_ext = {
            # Document formats
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/msword": ".doc",
            "application/vnd.ms-excel": ".xls",
            "application/vnd.ms-powerpoint": ".ppt",
            # Text formats
            "text/plain": ".txt",
            "text/html": ".html",
            "text/csv": ".csv",
            # Image formats
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            # Archive formats
            "application/zip": ".zip",
            "application/x-rar-compressed": ".rar",
            "application/x-7z-compressed": ".7z",
            # Data formats
            "application/json": ".json",
            "application/xml": ".xml",
            # Media formats
            "video/mp4": ".mp4",
            "video/x-msvideo": ".avi",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            # Executable formats
            "application/x-msdownload": ".exe",
            "application/x-dosexec": ".exe",
        }

        # Look up the detected MIME type in our supported types dictionary
        # dict.get() returns the extension or empty string if not found
        detected_ext = mime_to_ext.get(mime_type, "").lower()

        # Import the list of unsupported extensions from config
        # Note: This logic seems inverted - checking if detected type is IN unsupported list
        # This appears to be a bug - it should probably check if detected_ext is NOT in unsupported
        from config import UNSUPPORTED_EXTENSIONS

        # Normalize all extensions in the unsupported list
        # Ensures consistent format with leading dot and lowercase
        normalized_extensions = []
        for ext in UNSUPPORTED_EXTENSIONS:
            ext_lower = ext.lower()
            # Add leading dot if missing
            if not ext_lower.startswith("."):
                ext_lower = "." + ext_lower
            normalized_extensions.append(ext_lower)

        # Return True if detected extension is in the unsupported list
        # WARNING: This logic appears inverted - may need correction
        return detected_ext in normalized_extensions

    except subprocess.TimeoutExpired:
        # Tika processing exceeded 30-second timeout
        return False
    except FileNotFoundError:
        # Java or Tika JAR not found in system
        return False
    except Exception:
        # Catch-all for any other unexpected errors
        return False
