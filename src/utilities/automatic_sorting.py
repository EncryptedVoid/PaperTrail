"""
Automatic file type detection and classification utilities.

This module provides functions for detecting and classifying various file types
based on content analysis, metadata inspection, and pattern matching. Supports
detection of bookmarks, Anki decks, backup codes, books, financial documents,
and source code files.

Key Features:
- Content-based file type detection
- Metadata analysis for classification
- Pattern matching for specific file formats
- Support for multiple file formats and encodings
"""

import email
import json
import logging
import os
import re
import sqlite3
import subprocess
import tempfile
import time
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
    Detect if an HTML file is a browser bookmark export with high accuracy.

    Analyzes HTML structure for Netscape bookmark file format markers including
    DOCTYPE declaration, definition list structure, anchor tags, and timestamp
    attributes. All conditions must be met for positive detection.

    Args:
                    file_path: Path object pointing to the HTML file to analyze

    Returns:
                    bool: True if file matches bookmark export format, False otherwise

    Detection Criteria:
                    - Netscape bookmark DOCTYPE declaration
                    - Definition list (DL) structure
                    - DT entries with anchor (A) tags containing HREF attributes
                    - ADD_DATE attributes (unique to bookmark exports)

    Note:
                    All four conditions must be met to ensure high detection accuracy
                    and avoid false positives from regular HTML files.
    """
    try:
        # Read the HTML file content with UTF-8 encoding
        # Using 'r' mode for text files, errors='ignore' handles encoding issues
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
    except Exception as e:
        # Return False if file cannot be read (permissions, encoding errors, etc.)
        return False

    # Check for required Netscape bookmark DOCTYPE declaration
    # re.IGNORECASE flag makes the search case-insensitive
    has_netscape_doctype = bool(
        re.search(r"<!DOCTYPE\s+NETSCAPE-Bookmark-file-1>", html_content, re.IGNORECASE)
    )

    # Check for definition list (DL) structure used in bookmark hierarchy
    # DL tags contain bookmark folders and items
    has_dl_structure = bool(re.search(r"<DL\s*>", html_content, re.IGNORECASE))

    # Check for DT (definition term) entries with anchor tags
    # DT tags mark individual bookmarks, A tags contain the actual links
    has_dt_anchors = bool(
        re.search(r"<DT>\s*<A\s+[^>]*HREF\s*=", html_content, re.IGNORECASE)
    )

    # Check for ADD_DATE attribute specific to bookmark exports
    # ADD_DATE contains Unix timestamp when bookmark was created
    has_add_date = bool(
        re.search(r'ADD_DATE\s*=\s*["\']?\d+["\']?', html_content, re.IGNORECASE)
    )

    # All conditions must be met for absolute certainty
    # Boolean AND operation ensures strict matching
    return has_netscape_doctype and has_dl_structure and has_dt_anchors and has_add_date


def is_anki_deck(file_path: Path) -> bool:
    """
    Detect if a file is an Anki flashcard deck with high accuracy.

    Supports multiple Anki formats including packaged decks (.apkg, .colpkg),
    database files (.anki2, .anki21), and formatted text/CSV files that match
    Anki's import format.

    Args:
                    file_path: Path object pointing to the file to analyze

    Returns:
                    bool: True if file is detected as Anki deck, False otherwise

    Supported Formats:
                    - .apkg: Anki deck package (ZIP-based)
                    - .colpkg: Anki collection package (ZIP-based)
                    - .anki2: Anki 2.0 SQLite database
                    - .anki21: Anki 2.1 SQLite database
                    - .csv/.txt/.tsv: Text files with Anki card format
                    - Extension-less files: Content-based detection

    Detection Methods:
                    - ZIP package validation for .apkg/.colpkg
                    - SQLite database structure validation for .anki2/.anki21
                    - Pattern matching for text-based card formats
    """
    # Verify file exists using os.path for compatibility
    if not os.path.exists(file_path):
        return False

    # Ensure path points to a file, not a directory
    if not os.path.isfile(file_path):
        return False

    # Extract file extension and normalize to lowercase for comparison
    file_ext = Path(file_path).suffix.lower()

    # Check 1: APKG/COLPKG format (ZIP-based packages)
    # These are the most common Anki export formats
    if file_ext in [".apkg", ".colpkg"]:
        return _is_valid_anki_package(file_path)

    # Check 2: Direct Anki2/Anki21 database files
    # These are SQLite databases used by Anki internally
    if file_ext in [".anki2", ".anki21"]:
        return _is_valid_anki_db(file_path)

    # Check 3: CSV/TXT files with Anki-style card format
    # Users can create Anki decks from plain text or CSV files
    if file_ext in [".csv", ".txt", ".tsv"]:
        return _is_anki_text_format(file_path)

    # Check 4: No extension or unknown - probe content
    # Try to detect by examining file content signatures
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
        # Silently fail if content detection raises exceptions
        pass

    return False


def is_backup_codes_file(file_path: Path) -> bool:
    """
    Detect if a file contains 2FA backup/recovery codes.

    Analyzes file content for patterns matching common 2FA backup code formats
    used by Google, Discord, Twitter, GitHub, and other services. Uses keyword
    matching, pattern recognition, and code density analysis.

    Args:
                    file_path: Path object pointing to the file to analyze

    Returns:
                    bool: True if backup codes detected, False otherwise

    Common Formats:
                    - Google: 8 digits (e.g., 12345678)
                    - Discord: 8 digits
                    - Twitter: 12 alphanumeric characters (e.g., a1b2c3d4e5f6)
                    - GitHub: 8-16 alphanumeric characters

    Detection Strategy:
                    1. Quick filename keyword check
                    2. File size validation (must be under 50KB)
                    3. Pattern matching for code formats
                    4. Code density analysis (percentage of lines containing codes)
                    5. Keyword presence for additional confidence
    """
    # Quick check: filename contains backup code keywords
    # os.path.basename() extracts filename from full path
    filename = os.path.basename(file_path).lower()
    filename_keywords = ["backup", "recovery", "2fa", "codes", "twofactor", "mfa"]

    # any() returns True if at least one keyword is found in filename
    if any(keyword in filename for keyword in filename_keywords):
        return True

    # Backup codes files are always small (< 50KB)
    # os.path.getsize() returns file size in bytes
    if os.path.getsize(file_path) > 50 * 1024:
        return False

    # Attempt to read file content as text
    try:
        # errors='ignore' skips invalid UTF-8 sequences
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except:
        # Return False if file cannot be read as text
        return False

    # Pattern: 8-16 character alphanumeric codes (with optional dashes/spaces)
    # \b marks word boundaries, [A-Za-z0-9] matches alphanumeric characters
    # [\s\-]? allows optional space or dash separators
    code_pattern = (
        r"\b[A-Za-z0-9]{4}[\s\-]?[A-Za-z0-9]{4}(?:[\s\-]?[A-Za-z0-9]{4,8})?\b"
    )
    # re.findall() returns list of all pattern matches
    codes = re.findall(code_pattern, content)

    # Filter: Must have at least 4 unique characters (not 11111111)
    # set() creates collection of unique characters
    # len(set()) counts unique characters after removing separators
    codes = [c for c in codes if len(set(c.replace("-", "").replace(" ", ""))) >= 4]

    # Need at least 5 codes (most platforms give 8-10)
    if len(codes) < 5:
        return False

    # Check for backup code keywords (optional but strong signal)
    # re.IGNORECASE makes search case-insensitive
    keywords = r"backup|recovery|2fa|two.factor|verification|emergency|authentication"
    has_keywords = bool(re.search(keywords, content, re.IGNORECASE))

    # Count non-empty lines for density calculation
    # List comprehension filters out empty/whitespace-only lines
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return False

    # Code density: At least 50% of lines should contain codes
    # sum() with generator counts lines containing at least one code
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

    Analyzes document content using text extraction and pattern matching to identify
    financial documents such as invoices, receipts, statements, and purchase orders.
    Handles multiple file formats including text, images, PDFs, and emails.

    Args:
                    file_path: Path object pointing to the document to analyze
                    visual_processor: VisualProcessor instance for OCR/text extraction
                    logger: Logger instance for tracking analysis progress

    Returns:
                    bool: True if document contains financial data, False otherwise

    Supported Formats:
                    - Plain text: .txt, .csv, .json, .xml, .log
                    - Email: .eml
                    - Images: .jpg, .jpeg, .png, .bmp, .tiff, .webp
                    - PDF: .pdf (uses visual processing for text extraction)

    Detection Method:
                    1. Extract text based on file type
                    2. Analyze for financial keywords and patterns
                    3. Score based on keyword categories and pattern matches
                    4. Apply threshold for positive classification
    """
    # Record start time for performance tracking
    start_time = time.time()
    logger.info(f"Analyzing document for financial content: {file_path.name}")

    # Verify file exists before processing
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    try:
        # Extract text based on file type
        # Different file types require different extraction methods
        text = _extract_text_by_type(file_path, visual_processor, logger)

        # Validate extracted text has meaningful content
        # Minimum 10 characters required for analysis
        if not text or len(text.strip()) < 10:
            logger.warning(f"No meaningful text extracted from {file_path.name}")
            return False

        # Analyze for financial content using keyword and pattern matching
        is_financial = _contains_financial_content(text, logger)

        # Calculate processing time
        elapsed_time = time.time() - start_time
        logger.info(f"Financial content detected: {is_financial}")
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

        return is_financial

    except Exception as e:
        # Log error with full exception details using exc_info
        logger.error(f"Error analyzing {file_path.name}: {e}", exc_info=True)
        return False


def _extract_text_by_type(
    file_path: Path, visual_processor: VisualProcessor, logger: logging.Logger
) -> str:
    """
    Extract text from file based on its type.

    Routes to appropriate extraction method based on file extension.
    Supports plain text files, emails, images, and PDFs.

    Args:
                    file_path: Path object pointing to the file
                    visual_processor: VisualProcessor instance for OCR
                    logger: Logger instance for tracking extraction

    Returns:
                    str: Extracted text content, empty string if extraction fails
    """
    # Extract file extension and normalize to lowercase
    ext = file_path.suffix.lower()

    # Plain text formats - read directly
    # These formats contain human-readable text without encoding
    if ext in [".txt", ".csv", ".json", ".xml", ".log"]:
        logger.debug(f"Reading plain text file: {ext}")
        try:
            # errors='ignore' handles invalid UTF-8 sequences gracefully
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.debug(f"Extracted {len(text)} characters from plain text file")
            return text
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            return ""

    # Email files - parse email format
    # .eml files use RFC 822 message format
    elif ext == ".eml":
        logger.debug("Parsing email file")
        return _extract_from_email(file_path, logger)

    # Images and PDFs - use VisualProcessor
    # VisualProcessor performs OCR (Optical Character Recognition)
    elif ext in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        logger.debug(f"Using VisualProcessor for {ext} file")
        try:
            # VisualProcessor.extract_text() performs OCR on images/PDFs
            text = visual_processor.extract_text(file_path)
            logger.debug(f"Extracted {len(text)} characters via OCR")
            return text
        except Exception as e:
            logger.error(f"VisualProcessor extraction failed: {e}")
            return ""

    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


def _extract_from_email(file_path: Path, logger: logging.Logger) -> str:
    """
    Extract text content from .eml email file.

    Parses RFC 822 email format and extracts subject and body content.
    Handles both plain text and multipart messages.

    Args:
                    file_path: Path object pointing to .eml file
                    logger: Logger instance for tracking extraction

    Returns:
                    str: Combined email subject and body text
    """
    try:
        # Open and parse email file using Python's email library
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # email.message_from_file() parses RFC 822 format
            msg = email.message_from_file(f)

        # Extract subject line from email headers
        # get() returns empty string if header not found
        subject = msg.get("Subject", "")

        # Extract body content - handle multipart messages
        body = ""
        # is_multipart() checks if email has multiple parts (HTML, plain text, attachments)
        if msg.is_multipart():
            # walk() iterates through all message parts recursively
            for part in msg.walk():
                # get_content_type() returns MIME type (e.g., 'text/plain')
                content_type = part.get_content_type()
                # Only process plain text parts, skip HTML and attachments
                if content_type == "text/plain":
                    try:
                        # get_payload(decode=True) decodes base64/quoted-printable
                        body += part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                    except:
                        # Silently skip parts that fail to decode
                        pass
        else:
            # Simple message with single part
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            except:
                # Fallback to string representation if decoding fails
                body = str(msg.get_payload())

        # Combine subject and body with separator
        text = f"{subject}\n\n{body}"
        logger.debug(f"Extracted {len(text)} characters from email")
        return text

    except Exception as e:
        logger.error(f"Failed to parse email: {e}")
        return ""


def _contains_financial_content(text: str, logger: logging.Logger) -> bool:
    """
    Analyze text to determine if it contains financial/invoice/purchase content.

    Uses multi-level scoring system combining keyword matching and pattern detection.
    Keywords are grouped into categories (invoice, payment, financial, business) and
    scored separately. Pattern matching identifies currency symbols, amounts, dates,
    and invoice numbers.

    Args:
                    text: Text content to analyze
                    logger: Logger instance for tracking analysis

    Returns:
                    bool: True if financial content detected based on scoring thresholds

    Scoring System:
                    - Invoice keywords: High value indicators (invoice, receipt, bill)
                    - Payment keywords: Transaction indicators (payment, paid, purchase)
                    - Financial terms: Money-related words (total, amount, price)
                    - Business terms: Commercial context (customer, vendor, billing)
                    - Pattern matches: Currency symbols, amounts, dates, invoice numbers

    Thresholds:
                    - Minimum 3 keywords + 2 patterns, OR
                    - Minimum 5 keywords, OR
                    - 1+ invoice keyword + 4+ total score
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Financial keywords grouped by category for granular scoring
    # Invoice keywords are strongest indicators
    invoice_keywords = ["invoice", "receipt", "bill", "statement", "quote", "quotation"]

    # Payment keywords indicate financial transactions
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

    # Financial terms commonly appear in monetary documents
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

    # Business terms provide commercial context
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

    # Count keyword matches using sum with generator expression
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    payment_score = sum(1 for kw in payment_keywords if kw in text_lower)
    financial_score = sum(1 for kw in financial_terms if kw in text_lower)
    business_score = sum(1 for kw in business_terms if kw in text_lower)

    # Calculate total keyword score across all categories
    total_keyword_score = (
        invoice_score + payment_score + financial_score + business_score
    )

    logger.debug(
        f"Keyword scores - Invoice: {invoice_score}, Payment: {payment_score}, "
        f"Financial: {financial_score}, Business: {business_score}"
    )

    # Pattern detection for common financial formats
    pattern_score = 0

    # Currency symbols with amounts (e.g., $123.45, €50.00, £99)
    # \s* allows optional whitespace, \d+ matches one or more digits
    # (?:,\d{3})* matches optional thousands separators
    # (?:\.\d{2})? matches optional decimal part
    currency_pattern = r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    if re.search(currency_pattern, text):
        pattern_score += 2
        logger.debug("Currency pattern detected")

    # Amount patterns (e.g., "Total: 123.45", "Amount: $50")
    # (?:...) creates non-capturing group
    amount_pattern = r"(?:total|amount|price|cost|subtotal|balance)[\s:]+\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?"
    if re.search(amount_pattern, text_lower):
        pattern_score += 2
        logger.debug("Amount pattern detected")

    # Date patterns common in invoices (MM/DD/YYYY or YYYY-MM-DD)
    date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}"
    if re.search(date_pattern, text):
        pattern_score += 1
        logger.debug("Date pattern detected")

    # Invoice number patterns (e.g., "INV-12345", "Invoice #123")
    # \w+ matches alphanumeric characters, \d+ ensures numeric component
    invoice_number_pattern = r"(?:invoice|receipt|bill|ref(?:erence)?)[#\s:-]*\w+\d+"
    if re.search(invoice_number_pattern, text_lower):
        pattern_score += 2
        logger.debug("Invoice number pattern detected")

    # Calculate total score combining keywords and patterns
    total_score = total_keyword_score + pattern_score

    logger.info(
        f"Financial analysis - Keyword score: {total_keyword_score}, "
        f"Pattern score: {pattern_score}, Total: {total_score}"
    )

    # Decision threshold: need strong evidence across multiple dimensions
    # At least 1 invoice keyword + 4 total score indicates financial doc
    if invoice_score >= 1 and total_score >= 4:
        return True
    # Or 3+ keywords + 2+ patterns (strong combined evidence)
    elif total_keyword_score >= 3 and pattern_score >= 2:
        return True
    # Or 5+ keywords regardless of patterns (keyword density)
    elif total_keyword_score >= 5:
        return True

    return False


def _is_valid_anki_package(file_path: Path) -> bool:
    """
    Check if file is a valid APKG or COLPKG package.

    APKG/COLPKG files are ZIP archives containing Anki collection database
    and media files. Validates ZIP structure and verifies collection database.

    Args:
                    file_path: Path to file to check

    Returns:
                    bool: True if valid Anki package, False otherwise
    """
    try:
        # Open file as ZIP archive
        # zipfile.ZipFile() raises BadZipFile if not a valid ZIP
        with zipfile.ZipFile(file_path, "r") as zip_file:
            # namelist() returns list of all files in archive
            files = zip_file.namelist()

            # APKG must contain collection database file
            # Different Anki versions use different filenames
            has_collection = any(
                f in files
                for f in ["collection.anki2", "collection.anki21", "collection.anki21b"]
            )

            # COLPKG contains collection and media folder
            has_media = "media" in files or any("media" in f for f in files)

            if has_collection:
                # Verify the collection file is actually a SQLite database
                # Iterate through possible collection filenames
                for collection_name in [
                    "collection.anki2",
                    "collection.anki21",
                    "collection.anki21b",
                ]:
                    if collection_name in files:
                        # Extract collection file to temporary location for validation
                        # tempfile.NamedTemporaryFile() creates secure temp file
                        # delete=False prevents auto-deletion on close
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            # zip_file.read() extracts file content as bytes
                            tmp.write(zip_file.read(collection_name))
                            tmp_path = tmp.name

                        try:
                            # Validate extracted file is valid Anki database
                            is_valid = _is_valid_anki_db(tmp_path)
                            # os.unlink() deletes temporary file
                            os.unlink(tmp_path)
                            return is_valid
                        except:
                            # Clean up temp file even if validation fails
                            os.unlink(tmp_path)
                            continue

            return False
    except (zipfile.BadZipFile, OSError):
        # Not a valid ZIP file or file access error
        return False


def _is_valid_anki_db(file_path: Path) -> bool:
    """
    Check if file is a valid Anki SQLite database.

    Validates SQLite database structure by checking for Anki-specific tables
    and schema. Anki uses SQLite to store flashcard collections.

    Args:
                    file_path: Path to database file

    Returns:
                    bool: True if valid Anki database, False otherwise
    """
    try:
        # Check SQLite magic number (file signature)
        # SQLite files start with "SQLite format 3" in first 16 bytes
        with open(file_path, "rb") as f:
            header = f.read(16)
            # startswith() checks byte sequence at start of file
            if not header.startswith(b"SQLite format 3"):
                return False

        # Open as SQLite database
        # sqlite3.connect() establishes database connection
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        # Query for all table names
        # sqlite_master is special table containing database schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # Create set of table names for efficient membership testing
        tables = {row[0] for row in cursor.fetchall()}

        # Anki databases must have these core tables
        # col: collection metadata
        # notes: flashcard notes (content)
        # cards: individual flashcards (scheduling info)
        required_tables = {"col", "notes", "cards"}

        # issubset() checks if all required tables exist
        has_required = required_tables.issubset(tables)

        if has_required:
            # Additional verification: check col table structure
            # PRAGMA table_info() returns column information
            cursor.execute("PRAGMA table_info(col);")
            # Extract column names from result tuples
            col_columns = {row[1] for row in cursor.fetchall()}

            # Anki's col table should have specific columns
            # These columns are present in all Anki versions
            anki_col_columns = {"crt", "mod", "scm", "ver", "dty", "usn", "ls"}
            # Verify Anki-specific columns are present
            has_anki_structure = anki_col_columns.issubset(col_columns)

            # Close database connection
            conn.close()
            return has_anki_structure

        conn.close()
        return False

    except (sqlite3.DatabaseError, OSError, Exception):
        # Database corruption, file access error, or other exception
        return False


def _is_anki_text_format(file_path: Path) -> bool:
    """
    Check if text/CSV file follows Anki import format.

    Anki can import flashcards from plain text or CSV files with tab or
    semicolon delimiters. Validates file structure for consistent delimiter
    usage and appropriate field counts.

    Args:
                    file_path: Path to text file

    Returns:
                    bool: True if file matches Anki text format, False otherwise
    """
    try:
        # Read first few lines to analyze format
        # Reading only 100 lines for efficiency
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Create list of first 100 lines or all lines if file is shorter
            # sum(1 for _ in f) counts total lines
            lines = [f.readline() for _ in range(min(100, sum(1 for _ in f) + 1))]

        # Validate file has content
        if not lines or len(lines) < 1:
            return False

        # Remove empty lines - list comprehension filters whitespace-only lines
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            return False

        # Anki text format characteristics:
        # 1. Tab or semicolon separated fields (at least 2 fields per line)
        # 2. Consistent delimiter usage across lines
        # 3. Typically 2+ fields (front, back, optional extras)

        # Initialize delimiter counts
        # Track usage frequency of each delimiter type
        delimiter_counts = {"\t": 0, ";": 0, ",": 0}

        # Count delimiters in each line
        for line in lines:
            for delim in delimiter_counts:
                # str.count() returns number of non-overlapping occurrences
                count = line.count(delim)
                if count > 0:
                    delimiter_counts[delim] += 1

        # Find most common delimiter
        # max() with key parameter finds delimiter with highest count
        primary_delim = max(delimiter_counts, key=delimiter_counts.get)

        # At least 70% of lines should have the delimiter
        # Ensures consistent format throughout file
        if delimiter_counts[primary_delim] < len(lines) * 0.7:
            return False

        # Check if lines have consistent number of fields
        # Flashcards typically have 2-10 fields
        field_counts = []
        valid_lines = 0

        for line in lines:
            # Split line by primary delimiter
            fields = line.split(primary_delim)
            # Filter out empty fields - strip() removes whitespace
            fields = [f.strip() for f in fields if f.strip()]

            # Minimum: front and back of flashcard
            if len(fields) >= 2:
                field_counts.append(len(fields))
                valid_lines += 1

        # At least 70% should be valid card-like lines
        if valid_lines < len(lines) * 0.7:
            return False

        # Check consistency - most lines should have similar field count
        if field_counts:
            # collections.Counter counts frequency of each field count
            from collections import Counter

            count_freq = Counter(field_counts)
            # most_common(1) returns [(count, frequency)] for most common count
            most_common_count, frequency = count_freq.most_common(1)[0]

            # At least 60% should have the same field count
            # 2-20 fields is reasonable range for flashcards
            if frequency / len(field_counts) >= 0.6 and 2 <= most_common_count <= 20:
                return True

        return False

    except (OSError, UnicodeDecodeError, Exception):
        # File access error or encoding issues
        return False


def is_supported(file_path: Path) -> bool:
    """
    Check if file type is in list of supported formats.

    Validates file extension against known type lists from config.
    Supported types include emails, documents, images, videos, audio,
    text files, and archives.

    Args:
                    file_path: Path object to check

    Returns:
                    bool: True if file type is supported, False otherwise
    """
    # Get file extension for comparison
    ext = file_path.suffix.lower()

    # Check extension against all supported type lists from config
    # Using 'in' operator to test membership in each list
    if (
        ext in EMAIL_TYPES
        or ext in DOCUMENT_TYPES
        or ext in IMAGE_TYPES
        or ext in VIDEO_TYPES
        or ext in AUDIO_TYPES
        or ext in TEXT_TYPES
        or ext in ARCHIVE_TYPES
    ):
        return True

    return False


def is_book(
    file_path: Path, isbn_threshold: int = 4, overall_threshold: int = 5
) -> bool:
    """
    Detect whether a document (PDF, EPUB, DOC) is a book or personal document.

    Uses multi-factor scoring system analyzing metadata, content structure,
    and professional indicators. EPUB files are automatically classified as books.
    For other formats, combines evidence from ISBN, publisher metadata, chapter
    structure, copyright notices, and other book-specific markers.

    Args:
                    file_path: Path object pointing to document file
                    isbn_threshold: Score to auto-return True if ISBN found (default: 4)
                    overall_threshold: Minimum score needed to classify as book (default: 5)

    Returns:
                    bool: True if document is likely a book, False otherwise

    Scoring System:
                    - ISBN present: +4 points (auto-pass)
                    - Publisher metadata: +3 points
                    - Professional creation tool: +2 points
                    - Chapter structure: +2 points
                    - Copyright notice: +2 points
                    - Page count > 50: +1 point
                    - Page count > 150: +1 additional point
                    - Table of contents: +1 point
                    - Bibliography/references: +1 point
    """
    # Record start time for performance tracking
    start_time = time.time()

    # EPUB files are almost always books
    # EPUB is dedicated e-book format, rarely used for personal documents
    if file_path.suffix.lower() == ".epub":
        return True

    # Initialize score accumulator
    score = 0

    # Extract metadata and content using Apache Tika
    # Tika is Java-based tool for content analysis and metadata extraction
    metadata = _get_tika_metadata(file_path)
    # Extract first ~50KB of content for analysis
    content_sample = _get_tika_content(file_path, max_length=50000)

    # Cannot analyze without content
    if not content_sample:
        return False

    # Check for ISBN (very strong indicator)
    # ISBN is unique identifier used exclusively for published books
    if _contains_isbn(content_sample):
        score += isbn_threshold

    # Check metadata for publisher information
    # Professional publishers add metadata to published books
    if _has_publisher_metadata(metadata):
        score += 3

    # Check for professional creation tools
    # Books are typically created with specialized publishing software
    if _is_professional_tool(metadata):
        score += 2

    # Check for chapter structure
    # Multiple chapters indicate book organization
    if _has_chapter_structure(content_sample):
        score += 2

    # Check for copyright notice
    # Professional books include copyright pages
    if _has_copyright_notice(content_sample):
        score += 2

    # Check page count from metadata
    # Books typically have many pages
    page_count = _get_page_count(metadata)
    if page_count > 50:
        score += 1
    if page_count > 150:
        score += 1  # Additional point for substantial length

    # Check for table of contents
    # Structured documents like books have TOCs
    if _has_table_of_contents(content_sample):
        score += 1

    # Check for bibliography/references
    # Academic and non-fiction books include citations
    if _has_bibliography(content_sample):
        score += 1

    # Calculate processing time
    elapsed_time = time.time() - start_time

    # Score must meet threshold to classify as book
    is_book_result = score >= overall_threshold
    return is_book_result


def _get_tika_metadata(file_path: Path) -> dict:
    """
    Extract metadata using Apache Tika.

    Tika is universal content analysis tool that extracts metadata from
    various file formats. Returns metadata as dictionary.

    Args:
                    file_path: Path to file

    Returns:
                    dict: Metadata dictionary, empty dict if extraction fails
    """
    try:
        # Run Tika as subprocess
        # subprocess.run() executes external command
        # --json flag outputs metadata in JSON format
        result = subprocess.run(
            ["java", "-jar", TIKA_APP_JAR_PATH, "--json", str(file_path)],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Return output as string instead of bytes
            timeout=30,  # Kill process after 30 seconds
        )
        # returncode=0 indicates successful execution
        if result.returncode == 0:
            # json.loads() parses JSON string to dictionary
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        # Return empty dict if Tika execution fails
        return {}


def _get_tika_content(file_path: Path, max_length: int = 50000) -> str:
    """
    Extract text content using Apache Tika.

    Uses Tika to extract plain text from document. Limits output to
    max_length characters for efficiency.

    Args:
                    file_path: Path to file
                    max_length: Maximum characters to extract (default: 50000)

    Returns:
                    str: Extracted text content, empty string if extraction fails
    """
    try:
        # Run Tika with --text flag for content extraction
        result = subprocess.run(
            ["java", "-jar", TIKA_APP_JAR_PATH, "--text", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Slice string to max_length using [:max_length]
            return result.stdout[:max_length]
        return ""
    except Exception as e:
        return ""


def _contains_isbn(text: str) -> bool:
    """
    Check if text contains ISBN number.

    ISBN (International Standard Book Number) is unique identifier for books.
    Supports both ISBN-10 and ISBN-13 formats with various separator styles.

    Args:
                    text: Text to search

    Returns:
                    bool: True if ISBN found, False otherwise
    """
    # ISBN-10 or ISBN-13 patterns
    # 97[89] matches ISBN-13 prefix (978 or 979)
    # [-\s]? allows optional dash or space separators
    # [\dXx] allows digits or X (check digit in ISBN-10)
    isbn_pattern = (
        r"ISBN[-:\s]*(97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]"
    )
    # bool() converts match object to True/False
    return bool(re.search(isbn_pattern, text, re.IGNORECASE))


def _has_publisher_metadata(metadata: dict) -> bool:
    """
    Check if metadata contains publisher information.

    Examines various metadata fields that might contain publisher info
    or ISBN. Different tools and formats use different field names.

    Args:
                    metadata: Metadata dictionary from Tika

    Returns:
                    bool: True if publisher info found, False otherwise
    """
    if not metadata:
        return False

    # Check standard publisher fields
    # Different metadata standards use different field names
    publisher_fields = ["publisher", "dc:publisher", "Publisher", "meta:publisher"]
    for field in publisher_fields:
        # Check if field exists and has non-empty value
        if field in metadata and metadata[field]:
            return True

    # Check for ISBN in metadata
    # ISBN in metadata strongly suggests published book
    isbn_fields = ["isbn", "ISBN", "dc:identifier"]
    for field in isbn_fields:
        if field in metadata and metadata[field]:
            # Convert to string for pattern matching
            isbn_value = str(metadata[field])
            # Check for "isbn" keyword or ISBN-13 pattern (97[89] + 10 digits)
            if "isbn" in isbn_value.lower() or re.search(r"97[89]\d{10}", isbn_value):
                return True

    return False


def _is_professional_tool(metadata: dict) -> bool:
    """
    Check if document was created with professional publishing tools.

    Professional publishing software (InDesign, LaTeX, etc.) is typically
    used for book production rather than personal documents.

    Args:
                    metadata: Metadata dictionary from Tika

    Returns:
                    bool: True if professional tool detected, False otherwise
    """
    if not metadata:
        return False

    # List of professional publishing tools
    # These tools are primarily used for professional book/magazine production
    professional_tools = [
        "Adobe InDesign",
        "InDesign",
        "LaTeX",  # Academic publishing system
        "XeTeX",  # LaTeX variant
        "LuaTeX",  # LaTeX variant
        "Calibre",  # E-book management
        "Scribus",  # Open-source desktop publishing
        "QuarkXPress",  # Professional page layout
        "FrameMaker",  # Technical documentation
        "Microsoft Publisher",
        "Sigil",  # EPUB editor
        "Vellum",  # Book formatting
    ]

    # Check creator/producer metadata fields
    # Different standards store creator info in different fields
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

    # Search for professional tool names in creator fields
    for field in creator_fields:
        if field in metadata and metadata[field]:
            # Convert to string and lowercase for comparison
            creator = str(metadata[field])
            for tool in professional_tools:
                # Case-insensitive substring search
                if tool.lower() in creator.lower():
                    return True

    return False


def _has_chapter_structure(text: str) -> bool:
    """
    Check for chapter-based structure in document.

    Books are typically organized into chapters with standardized headings.
    Looks for multiple chapter markers with numbers or roman numerals.

    Args:
                    text: Document text to analyze

    Returns:
                    bool: True if chapter structure found, False otherwise
    """
    # Look for multiple chapter headings
    # (?:^|\n) matches start of string or newline (chapter headings start lines)
    # (?:Chapter|CHAPTER) matches "Chapter" in any case
    # (?:\d+|[IVXLCDM]+|One|Two|...) matches numbers in various formats
    chapter_pattern = (
        r"(?:^|\n)\s*(?:Chapter|CHAPTER)\s+(?:\d+|[IVXLCDM]+|One|Two|Three|Four|Five)"
    )
    # Check first 20000 characters for chapter markers
    matches = re.findall(chapter_pattern, text[:20000])
    # Need at least 3 chapters to confirm structure
    return len(matches) >= 3


def _has_copyright_notice(text: str) -> bool:
    """
    Check for copyright notices typical of published books.

    Professional books include copyright pages with specific legal language.
    Checks first 5000 characters where copyright notices typically appear.

    Args:
                    text: Document text to analyze

    Returns:
                    bool: True if copyright notice found, False otherwise
    """
    # Common copyright notice patterns
    # © is copyright symbol, (19|20)\d{2} matches years 1900-2099
    copyright_patterns = [
        r"(?:©|Copyright|COPYRIGHT)\s*(?:19|20)\d{2}",
        r"All rights reserved",  # Standard copyright phrase
        r"Published by",  # Publisher statement
        r"No part of this (?:book|publication) may be reproduced",  # Rights reservation
    ]

    # Check first 5000 characters (typical location of copyright page)
    sample = text[:5000]

    # Search for any copyright pattern
    for pattern in copyright_patterns:
        if re.search(pattern, sample, re.IGNORECASE):
            return True

    return False


def _get_page_count(metadata: dict) -> int:
    """
    Extract page count from document metadata.

    Different metadata standards store page count in different fields.
    Returns 0 if page count not found or invalid.

    Args:
                    metadata: Metadata dictionary from Tika

    Returns:
                    int: Number of pages, 0 if not found
    """
    if not metadata:
        return 0

    # Check various page count field names
    # Different PDF producers use different field names
    page_fields = ["xmpTPg:NPages", "meta:page-count", "Page-Count", "pageCount"]

    for field in page_fields:
        if field in metadata:
            try:
                # Convert to integer, may raise ValueError
                return int(metadata[field])
            except (ValueError, TypeError):
                # Skip invalid values and try next field
                continue

    return 0


def _has_table_of_contents(text: str) -> bool:
    """
    Check for table of contents in document.

    Books typically include table of contents near beginning with chapter
    listings and page numbers. Uses pattern matching to identify TOC markers.

    Args:
                    text: Document text to analyze

    Returns:
                    bool: True if TOC found, False otherwise
    """
    # TOC pattern markers
    toc_patterns = [
        # Look for "Table of Contents" heading
        r"(?:^|\n)\s*(?:Table of Contents|CONTENTS|TABLE OF CONTENTS)",
        # Look for TOC entry pattern: "Chapter 1 ... 5" (chapter title with page number)
        r"(?:^|\n)\s*(?:Chapter|Part)\s+\d+.*?\d+\s*$",
    ]

    # TOC usually appears near beginning of book
    sample = text[:10000]

    # Search for TOC patterns
    for pattern in toc_patterns:
        # re.MULTILINE treats string as multiple lines for ^ and $
        if re.search(pattern, sample, re.MULTILINE | re.IGNORECASE):
            return True

    return False


def _has_bibliography(text: str) -> bool:
    """
    Check for bibliography or references section in document.

    Academic and non-fiction books include citations and references.
    Looks for bibliography heading and citation patterns at end of document.

    Args:
                    text: Document text to analyze

    Returns:
                    bool: True if bibliography found, False otherwise
    """
    # Bibliography section markers
    biblio_patterns = [
        # Look for bibliography/references heading
        r"(?:^|\n)\s*(?:Bibliography|BIBLIOGRAPHY|References|REFERENCES|Works Cited)",
        # Look for numbered citation pattern: [1] Author et al. 2020
        r"\[\d+\]\s+\w+.*?(?:19|20)\d{2}",
    ]

    # Check last 20000 characters (bibliography usually at end)
    sample = text[-20000:]

    # Search for bibliography patterns
    for pattern in biblio_patterns:
        # re.findall() returns list of all matches
        matches = re.findall(pattern, sample, re.IGNORECASE)
        # Need multiple citations/references to confirm bibliography
        if len(matches) >= 2:
            return True

    return False


def is_code(file_path: Path) -> bool:
    """
    Check if file is source code based on extension.

    Simple extension-based detection using CODE_EXTENSIONS list from config.
    Matches file extension against known programming language extensions.

    Args:
                    file_path: Path object to check

    Returns:
                    bool: True if file is code, False otherwise
    """
    # Check if file extension is in CODE_EXTENSIONS list
    # CODE_EXTENSIONS contains extensions like .py, .js, .java, etc.
    if file_path.suffix in CODE_EXTENSIONS:
        return True

    return False
