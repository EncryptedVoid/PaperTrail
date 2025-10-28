import email
import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
import zipfile
from pathlib import Path

from config import (
	ARCHIVE_TYPES ,
	AUDIO_TYPES ,
	CODE_EXTENSIONS ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	IMAGE_TYPES ,
	TEXT_TYPES ,
	TIKA_APP_JAR_PATH ,
	VIDEO_TYPES ,
)
from visual_processor import VisualProcessor


def is_bookmark_file(file_path: Path) -> bool:
    """
    Detects if an HTML file is a bookmark export with high accuracy.

    Args:
                    file_path (str): The HTML content to check

    Returns:
                    bool: True if it's a bookmark file, False otherwise
    """
    with open("bookmarks.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Required: Netscape bookmark DOCTYPE
    has_netscape_doctype = bool(
        re.search(r"<!DOCTYPE\s+NETSCAPE-Bookmark-file-1>", html_content, re.IGNORECASE)
    )

    # Required: Definition list structure with links
    has_dl_structure = bool(re.search(r"<DL\s*>", html_content, re.IGNORECASE))

    # Required: DT entries with anchor tags
    has_dt_anchors = bool(
        re.search(r"<DT>\s*<A\s+[^>]*HREF\s*=", html_content, re.IGNORECASE)
    )

    # Required: ADD_DATE attribute (specific to bookmark exports)
    has_add_date = bool(
        re.search(r'ADD_DATE\s*=\s*["\']?\d+["\']?', html_content, re.IGNORECASE)
    )

    # All conditions must be met for absolute certainty
    return has_netscape_doctype and has_dl_structure and has_dt_anchors and has_add_date


def is_anki_deck(file_path: Path):
    """
    Detect if a file is an Anki deck with high accuracy.
    Supports: .apkg, .colpkg, .anki2, .anki21, and formatted text/CSV files.

    Args:
                    file_path: Path to the file to check

    Returns:
                    bool: True if file is detected as Anki deck, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    if not os.path.isfile(file_path):
        return False

    file_ext = Path(file_path).suffix.lower()

    # Check 1: APKG/COLPKG format (ZIP-based packages)
    if file_ext in [".apkg", ".colpkg"]:
        return _is_valid_anki_package(file_path)

    # Check 2: Direct Anki2/Anki21 database files
    if file_ext in [".anki2", ".anki21"]:
        return _is_valid_anki_db(file_path)

    # Check 3: CSV/TXT files with Anki-style card format
    if file_ext in [".csv", ".txt", ".tsv"]:
        return _is_anki_text_format(file_path)

    # Check 4: No extension or unknown - probe content
    # Try to detect by content signature
    try:
        # Check if it's a ZIP file (APKG/COLPKG without extension)
        if _is_valid_anki_package(file_path):
            return True

        # Check if it's a SQLite database (Anki2 without extension)
        if _is_valid_anki_db(file_path):
            return True

        # Check if it's text-based Anki format
        if _is_anki_text_format(file_path):
            return True
    except:
        pass

    return False


def is_backup_codes_file(file_path: Path):
    """
    Detect if a file contains 2FA backup/recovery codes.

    Real-world formats:
    - Google: 8 digits (e.g., 12345678)
    - Discord: 8 digits
    - Twitter: 12 alphanumeric chars (e.g., a1b2c3d4e5f6)
    - GitHub: 8-16 alphanumeric chars

    Args:
                    file_path: Path to file

    Returns:
                    bool: True if backup codes detected
    """

    # Quick check: filename contains backup code keywords
    filename = os.path.basename(file_path).lower()
    filename_keywords = ["backup", "recovery", "2fa", "codes", "twofactor", "mfa"]
    if any(keyword in filename for keyword in filename_keywords):
        return True

    # Backup codes files are always small (< 50KB)
    if os.path.getsize(file_path) > 50 * 1024:
        return False

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except:
        return False

    # Pattern: 8-16 character alphanumeric codes (with optional dashes/spaces)
    # Matches: 12345678, abcd1234, a1b2-c3d4-e5f6, ABCD EFGH IJKL
    code_pattern = (
        r"\b[A-Za-z0-9]{4}[\s\-]?[A-Za-z0-9]{4}(?:[\s\-]?[A-Za-z0-9]{4,8})?\b"
    )
    codes = re.findall(code_pattern, content)

    # Filter: Must have at least 4 unique characters (not 11111111)
    codes = [c for c in codes if len(set(c.replace("-", "").replace(" ", ""))) >= 4]

    # Need at least 5 codes (most platforms give 8-10)
    if len(codes) < 5:
        return False

    # Check for backup code keywords (optional but strong signal)
    keywords = r"backup|recovery|2fa|two.factor|verification|emergency|authentication"
    has_keywords = bool(re.search(keywords, content, re.IGNORECASE))

    # Count non-empty lines
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return False

    # Code density: At least 50% of lines should contain codes
    lines_with_codes = sum(1 for line in lines if any(code in line for code in codes))
    code_density = lines_with_codes / len(lines)

    # Decision: High code density OR (medium density + keywords)
    if code_density >= 0.5:
        return True
    elif code_density >= 0.3 and has_keywords:
        return True

    return False


def is_financial_document(
    file_path: Path, visual_processor: VisualProcessor, logger: logging.Logger
) -> bool:
    """
    Determine if a document contains financial, invoice, or purchase-related data.

    Handles multiple file formats:
    - Plain text formats (.txt, .eml, .csv, etc.): Direct text extraction
    - Images (.jpg, .png, etc.): Visual text extraction via VisualProcessor
    - PDFs: Visual text extraction via VisualProcessor

    Args:
                    file_path: Path to the document to analyze
                    visual_processor: Instantiated VisualProcessor instance
                    logger: Logger instance for tracking

    Returns:
                    bool: True if document contains financial data, False otherwise
    """
    logger.info(f"Analyzing document for financial content: {file_path.name}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    try:
        # Extract text based on file type
        text = _extract_text_by_type(file_path, visual_processor, logger)

        if not text or len(text.strip()) < 10:
            logger.warning(f"No meaningful text extracted from {file_path.name}")
            return False

        # Analyze for financial content
        is_financial = _contains_financial_content(text, logger)

        logger.info(f"Financial content detected: {is_financial}")
        return is_financial

    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return False


def _extract_text_by_type(
    file_path: Path, visual_processor: VisualProcessor, logger: logging.Logger
) -> str:
    """
    Extract text from file based on its type.

    Args:
                    file_path: Path to file
                    visual_processor: VisualProcessor instance
                    logger: Logger instance

    Returns:
                    Extracted text content
    """
    ext = file_path.suffix.lower()

    # Plain text formats - read directly
    if ext in [".txt", ".csv", ".json", ".xml", ".log"]:
        logger.debug(f"Reading plain text file: {ext}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            return ""

    # Email files - parse email format
    elif ext == ".eml":
        logger.debug("Parsing email file")
        return _extract_from_email(file_path, logger)

    # Images and PDFs - use VisualProcessor
    elif ext in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        logger.debug(f"Using VisualProcessor for {ext} file")
        try:
            return visual_processor.extract_text(file_path)
        except Exception as e:
            logger.error(f"VisualProcessor extraction failed: {e}")
            return ""

    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


def _extract_from_email(file_path: Path, logger: logging.Logger) -> str:
    """
    Extract text content from .eml email file.

    Args:
                    file_path: Path to .eml file
                    logger: Logger instance

    Returns:
                    Email text content (subject + body)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f)

        # Extract subject
        subject = msg.get("Subject", "")

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            except:
                body = str(msg.get_payload())

        # Combine subject and body
        text = f"{subject}\n\n{body}"
        logger.debug(f"Extracted {len(text)} characters from email")
        return text

    except Exception as e:
        logger.error(f"Failed to parse email: {e}")
        return ""


def _contains_financial_content(text: str, logger: logging.Logger) -> bool:
    """
    Analyze text to determine if it contains financial/invoice/purchase content.

    Uses keyword matching and pattern detection:
    - Financial keywords (invoice, receipt, payment, etc.)
    - Currency symbols and amounts
    - Common financial patterns

    Args:
                    text: Text content to analyze
                    logger: Logger instance

    Returns:
                    bool: True if financial content detected
    """
    text_lower = text.lower()

    # Financial keywords (grouped by category for scoring)
    invoice_keywords = ["invoice", "receipt", "bill", "statement", "quote", "quotation"]

    payment_keywords = [
        "payment",
        "paid",
        "due",
        "owing",
        "balance",
        "transaction",
        "purchase",
        "order",
        "sale",
        "refund",
    ]

    financial_terms = [
        "total",
        "subtotal",
        "amount",
        "price",
        "cost",
        "fee",
        "charge",
        "tax",
        "vat",
        "gst",
        "discount",
        "credit",
        "debit",
    ]

    business_terms = [
        "customer",
        "vendor",
        "supplier",
        "billing",
        "account number",
        "reference number",
        "po number",
        "order number",
    ]

    # Count keyword matches
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    payment_score = sum(1 for kw in payment_keywords if kw in text_lower)
    financial_score = sum(1 for kw in financial_terms if kw in text_lower)
    business_score = sum(1 for kw in business_terms if kw in text_lower)

    total_keyword_score = (
        invoice_score + payment_score + financial_score + business_score
    )

    logger.debug(
        f"Keyword scores - Invoice: {invoice_score}, Payment: {payment_score}, "
        f"Financial: {financial_score}, Business: {business_score}"
    )

    # Pattern detection
    pattern_score = 0

    # Currency symbols with amounts (e.g., $123.45, €50.00, £99)
    currency_pattern = r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    if re.search(currency_pattern, text):
        pattern_score += 2
        logger.debug("Currency pattern detected")

    # Amount patterns (e.g., "Total: 123.45", "Amount: $50")
    amount_pattern = r"(?:total|amount|price|cost|subtotal|balance)[\s:]+\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    if re.search(amount_pattern, text_lower):
        pattern_score += 2
        logger.debug("Amount pattern detected")

    # Date patterns (common in invoices)
    date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}"
    if re.search(date_pattern, text):
        pattern_score += 1
        logger.debug("Date pattern detected")

    # Invoice number patterns (e.g., "INV-12345", "Invoice #123")
    invoice_number_pattern = r"(?:invoice|receipt|bill|ref(?:erence)?)[#\s:-]*\w+\d+"
    if re.search(invoice_number_pattern, text_lower):
        pattern_score += 2
        logger.debug("Invoice number pattern detected")

    total_score = total_keyword_score + pattern_score

    logger.info(
        f"Financial analysis - Keyword score: {total_keyword_score}, "
        f"Pattern score: {pattern_score}, Total: {total_score}"
    )

    # Decision threshold: need strong evidence
    # At least 3 relevant keywords OR 2 keywords + patterns
    if invoice_score >= 1 and total_score >= 4:
        return True
    elif total_keyword_score >= 3 and pattern_score >= 2:
        return True
    elif total_keyword_score >= 5:
        return True

    return False


def _is_valid_anki_package(file_path):
    """Check if file is a valid APKG or COLPKG package."""
    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            files = zip_file.namelist()

            # APKG must contain collection.anki2 or collection.anki21
            has_collection = any(
                f in files
                for f in ["collection.anki2", "collection.anki21", "collection.anki21b"]
            )

            # COLPKG contains collection.anki21 and other files
            has_media = "media" in files or any("media" in f for f in files)

            if has_collection:
                # Verify the collection file is actually a SQLite database
                for collection_name in [
                    "collection.anki2",
                    "collection.anki21",
                    "collection.anki21b",
                ]:
                    if collection_name in files:
                        # Extract and verify it's a valid Anki database
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(zip_file.read(collection_name))
                            tmp_path = tmp.name

                        try:
                            is_valid = _is_valid_anki_db(tmp_path)
                            os.unlink(tmp_path)
                            return is_valid
                        except:
                            os.unlink(tmp_path)
                            continue

            return False
    except (zipfile.BadZipFile, OSError):
        return False


def _is_valid_anki_db(file_path: Path):
    """Check if file is a valid Anki SQLite database."""
    try:
        # Check SQLite magic number
        with open(file_path, "rb") as f:
            header = f.read(16)
            if not header.startswith(b"SQLite format 3"):
                return False

        # Try to open as SQLite and check for Anki-specific tables
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}

        # Anki databases must have these core tables
        required_tables = {"col", "notes", "cards"}

        # Check if all required tables exist
        has_required = required_tables.issubset(tables)

        if has_required:
            # Additional verification: check col table structure
            cursor.execute("PRAGMA table_info(col);")
            col_columns = {row[1] for row in cursor.fetchall()}

            # Anki's col table should have specific columns
            anki_col_columns = {"crt", "mod", "scm", "ver", "dty", "usn", "ls"}
            has_anki_structure = anki_col_columns.issubset(col_columns)

            conn.close()
            return has_anki_structure

        conn.close()
        return False

    except (sqlite3.DatabaseError, OSError, Exception):
        return False


def _is_anki_text_format(file_path: Path):
    """Check if text/CSV file follows Anki import format."""
    try:
        # Read first few lines to analyze
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(min(100, sum(1 for _ in f) + 1))]

        if not lines or len(lines) < 1:
            return False

        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            return False

        # Anki text format characteristics:
        # 1. Tab or semicolon separated fields (at least 2 fields per line)
        # 2. Consistent delimiter usage
        # 3. Typically 2+ fields (front, back, optional extras)

        delimiter_counts = {"\t": 0, ";": 0, ",": 0}

        # Count delimiters in each line
        for line in lines:
            for delim in delimiter_counts:
                count = line.count(delim)
                if count > 0:
                    delimiter_counts[delim] += 1

        # Find most common delimiter
        primary_delim = max(delimiter_counts, key=delimiter_counts.get)

        # At least 70% of lines should have the delimiter
        if delimiter_counts[primary_delim] < len(lines) * 0.7:
            return False

        # Check if lines have consistent number of fields (2-10 typical for flashcards)
        field_counts = []
        valid_lines = 0

        for line in lines:
            fields = line.split(primary_delim)
            # Filter out empty fields
            fields = [f.strip() for f in fields if f.strip()]

            if len(fields) >= 2:  # Minimum: front and back
                field_counts.append(len(fields))
                valid_lines += 1

        # At least 70% should be valid card-like lines
        if valid_lines < len(lines) * 0.7:
            return False

        # Check consistency - most lines should have similar field count
        if field_counts:
            from collections import Counter

            count_freq = Counter(field_counts)
            most_common_count, frequency = count_freq.most_common(1)[0]

            # At least 60% should have the same field count
            if frequency / len(field_counts) >= 0.6 and 2 <= most_common_count <= 20:
                return True

        return False

    except (OSError, UnicodeDecodeError, Exception):
        return False


def is_supported(file_path: Path):
    if (
        file_path in EMAIL_TYPES
        or file_path in DOCUMENT_TYPES
        or file_path in IMAGE_TYPES
        or file_path in VIDEO_TYPES
        or file_path in AUDIO_TYPES
        or file_path in TEXT_TYPES
        or file_path in ARCHIVE_TYPES
    ):
        return True

    return False


def is_book(file_path: Path, isbn_threshold=4, overall_threshold=5):
    """
    Detects whether a document (PDF, EPUB, DOC) is a book or personal document.

    Args:
                    file_path: Path to the document file
                    isbn_threshold: Score to auto-return True if ISBN found
                    overall_threshold: Minimum score needed to classify as book

    Returns:
                    bool: True if document is likely a book, False otherwise
    """

    # EPUB files are almost always books
    if file_path.suffix.lower() == ".epub":
        return True

    score = 0

    # Extract metadata and content using Tika
    metadata = _get_tika_metadata(file_path)
    content_sample = _get_tika_content(file_path, max_length=50000)  # First ~50KB

    if not content_sample:
        return False

    # Check for ISBN (very strong indicator)
    if _contains_isbn(content_sample):
        score += isbn_threshold

    # Check metadata for publisher
    if _has_publisher_metadata(metadata):
        score += 3

    # Check for professional creation tools
    if _is_professional_tool(metadata):
        score += 2

    # Check for chapter structure
    if _has_chapter_structure(content_sample):
        score += 2

    # Check for copyright notice
    if _has_copyright_notice(content_sample):
        score += 2

    # Check page count
    page_count = _get_page_count(metadata)
    if page_count > 50:
        score += 1
    if page_count > 150:
        score += 1

    # Check for table of contents
    if _has_table_of_contents(content_sample):
        score += 1

    # Check for bibliography/references
    if _has_bibliography(content_sample):
        score += 1

    return score >= overall_threshold


def _get_tika_metadata(file_path):
    """Extract metadata using Apache Tika."""
    try:
        result = subprocess.run(
            ["java", "-jar", TIKA_APP_JAR_PATH, "--json", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}


def _get_tika_content(file_path, max_length=50000):
    """Extract text content using Apache Tika."""
    try:
        result = subprocess.run(
            ["java", "-jar", TIKA_APP_JAR_PATH, "--text", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout[:max_length]
        return ""
    except Exception as e:
        print(f"Error extracting content: {e}")
        return ""


def _contains_isbn(text):
    """Check if text contains ISBN number."""
    # ISBN-10 or ISBN-13 patterns
    isbn_pattern = (
        r"ISBN[-:\s]*(97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]"
    )
    return bool(re.search(isbn_pattern, text, re.IGNORECASE))


def _has_publisher_metadata(metadata):
    """Check if metadata contains publisher information."""
    if not metadata:
        return False

    publisher_fields = ["publisher", "dc:publisher", "Publisher", "meta:publisher"]
    for field in publisher_fields:
        if field in metadata and metadata[field]:
            return True

    # Check for ISBN in metadata
    isbn_fields = ["isbn", "ISBN", "dc:identifier"]
    for field in isbn_fields:
        if field in metadata and metadata[field]:
            isbn_value = str(metadata[field])
            if "isbn" in isbn_value.lower() or re.search(r"97[89]\d{10}", isbn_value):
                return True

    return False


def _is_professional_tool(metadata):
    """Check if document was created with professional publishing tools."""
    if not metadata:
        return False

    professional_tools = [
        "Adobe InDesign",
        "InDesign",
        "LaTeX",
        "XeTeX",
        "LuaTeX",
        "Calibre",
        "Scribus",
        "QuarkXPress",
        "FrameMaker",
        "Microsoft Publisher",
        "Sigil",
        "Vellum",
    ]

    creator_fields = [
        "creator",
        "producer",
        "Application",
        "Creator",
        "Producer",
        "pdf:producer",
        "xmp:CreatorTool",
        "meta:creator",
    ]

    for field in creator_fields:
        if field in metadata and metadata[field]:
            creator = str(metadata[field])
            for tool in professional_tools:
                if tool.lower() in creator.lower():
                    return True

    return False


def _has_chapter_structure(text):
    """Check for chapter-based structure."""
    # Look for multiple chapter headings
    chapter_pattern = (
        r"(?:^|\n)\s*(?:Chapter|CHAPTER)\s+(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five)"
    )
    matches = re.findall(chapter_pattern, text[:20000])  # Check first portion
    return len(matches) >= 3


def _has_copyright_notice(text):
    """Check for copyright notices typical of published books."""
    copyright_patterns = [
        r"(?:©|Copyright|COPYRIGHT)\s*(?:19|20)\d{2}",
        r"All rights reserved",
        r"Published by",
        r"No part of this (?:book|publication) may be reproduced",
    ]

    # Check first 5000 characters (typical location of copyright page)
    sample = text[:5000]

    for pattern in copyright_patterns:
        if re.search(pattern, sample, re.IGNORECASE):
            return True

    return False


def _get_page_count(metadata):
    """Extract page count from metadata."""
    if not metadata:
        return 0

    page_fields = ["xmpTPg:NPages", "meta:page-count", "Page-Count", "pageCount"]

    for field in page_fields:
        if field in metadata:
            try:
                return int(metadata[field])
            except (ValueError, TypeError):
                continue

    return 0


def _has_table_of_contents(text):
    """Check for table of contents."""
    toc_patterns = [
        r"(?:^|\n)\s*(?:Table of Contents|CONTENTS|TABLE OF CONTENTS)",
        r"(?:^|\n)\s*(?:Chapter|Part)\s+\d+.*?\d+\s*$",  # TOC entry pattern
    ]

    sample = text[:10000]  # TOC usually near beginning

    for pattern in toc_patterns:
        if re.search(pattern, sample, re.MULTILINE | re.IGNORECASE):
            return True

    return False


def _has_bibliography(text):
    """Check for bibliography or references section."""
    biblio_patterns = [
        r"(?:^|\n)\s*(?:Bibliography|BIBLIOGRAPHY|References|REFERENCES|Works Cited)",
        r"\[\d+\]\s+\w+.*?(?:19|20)\d{2}",  # Citation pattern like [1] Author et al. 2020
    ]

    # Check last 20000 characters (bibliography usually at end)
    sample = text[-20000:]

    for pattern in biblio_patterns:
        matches = re.findall(pattern, sample, re.IGNORECASE)
        if len(matches) >= 2:  # Need multiple citations/references
            return True

    return False


def is_code(file_path: Path):
    if file_path.suffix in CODE_EXTENSIONS:
        return True

    return False
