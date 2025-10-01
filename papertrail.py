"""
PaperTrail Main Processing Pipeline

Complete document processing pipeline with session tracking, conversion, sanitization,
metadata extraction, semantic analysis, and database tabulation with encryption.

This is the main entry point for the PaperTrail system, orchestrating a multi-stage
document processing workflow that takes raw artifacts and processes them through
various stages until they are ready for storage and retrieval.

Author: Ashiq Gazi
"""

import logging
from datetime import datetime

# These define the folder structure for the document processing pipeline
from config import (
    ARCHIVAL_DIR,
    COMPLETED_ARTIFACTS_DIR,
    CONVERTED_ARTIFACT_DIR,
    EMBELLISHED_ARTIFACTS_DIR,
    FAILED_ARTIFACTS_DIR,
    LOG_DIR,
    METADATA_EXTRACTED_DIR,
    PROTECTED_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    SCANNED_ARTIFACTS_DIR,
    SEMANTICS_EXTRACTED_DIR,
    SESSION_LOG_FILE_PREFIX,
    SYSTEM_DIRECTORIES,
    TRANSLATED_ARTIFACTS_DIR,
    UNPROCESSED_ARTIFACTS_DIR,
)

# Import all processor modules that handle each stage of the pipeline
from src.processors import (
    convert,
    embellish,
    extract_metadata,
    extract_semantics,
    password_protect,
    sanitize,
    scan,
    tabulate,
    translate_multilingual,
)

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

# Ensure all required directories exist before processing begins
# This prevents errors during processing if any directory is missing
# parents=True creates parent directories if needed, exist_ok=True prevents errors if already exists
for directory in SYSTEM_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Generate a unique log file name for this session using timestamp
# Format: prefix-YYYY-MM-DD_HH-MM-SS.log
# This ensures each run has its own log file for auditing and debugging
log_file_name = (
    f"{SESSION_LOG_FILE_PREFIX}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# Setup dual logging: console output for real-time monitoring and file output for persistence
# Console handler allows operators to monitor progress in real-time
# File handler creates a permanent record of all processing activities
handlers = [
    logging.StreamHandler(),  # Outputs log messages to console/terminal
    logging.FileHandler(
        (LOG_DIR / log_file_name),  # Outputs log messages to timestamped file
        encoding="utf-8",  # Ensures proper handling of international characters
    ),
]

# Configure the root logger with INFO level and custom format
# INFO level captures important events without excessive debug details
# Format includes timestamp, logger name, severity level, and the actual message
logging.basicConfig(
    level=logging.INFO,  # Set minimum logging level (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Structured log format
    handlers=handlers,  # Apply both console and file handlers
)

# Create a named logger for this application
# This allows filtering and identifying PaperTrail-specific log entries
logger = logging.getLogger("PAPERTRAIL")

# Log the session startup banner
# This marks the beginning of a new processing session in the logs
logger.info(
    "WELCOME TO PAPERTRAIL! AN AUTOMATED ARTIFACT REGISTRY AND ORGANISATION SYSTEM"
)

# ============================================================================
# STAGE 1: SANITIZATION
# ============================================================================
sanitize(
    logger=logger,  # Pass logger for tracking sanitization activities
    source_dir=UNPROCESSED_ARTIFACTS_DIR,  # Input: raw, unprocessed documents
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents that fail sanitization
    success_dir=SANITIZED_ARTIFACTS_DIR,  # Output: clean, validated documents
)
# ============================================================================
# STAGE 2: IMAGE SCANNING TO PDF
# ============================================================================
scan(
    logger=logger,  # Pass logger for tracking embellishment activities
    source_dir=SANITIZED_ARTIFACTS_DIR,  # Input: converted documents in standard format
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where embellishment fails
    success_dir=SCANNED_ARTIFACTS_DIR,  # Output: enhanced documents with improved presentation
)

# ============================================================================
# STAGE 3: METADATA EXTRACTION
# ============================================================================
extract_metadata(
    logger=logger,  # Pass logger for tracking metadata extraction activities
    source_dir=SCANNED_ARTIFACTS_DIR,  # Input: sanitized documents from previous stage
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where metadata extraction fails
    success_dir=METADATA_EXTRACTED_DIR,  # Output: documents with extracted metadata
)

# ============================================================================
# STAGE 4: FILE TYPE CONVERSION
# ============================================================================
# Third processing stage: convert documents to standardized formats
# This stage normalizes different file formats (DOCX, XLSX, images, etc.) into
# consistent formats for downstream processing (typically PDF and/or plain text)
# Original files are archived, converted versions proceed to next stage
convert(
    logger=logger,  # Pass logger for tracking conversion activities
    source_dir=METADATA_EXTRACTED_DIR,  # Input: documents with extracted metadata
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents that fail conversion
    archive_dir=ARCHIVAL_DIR,  # Output: original files archived for preservation
    success_dir=CONVERTED_ARTIFACT_DIR,  # Output: standardized format documents
)

# ============================================================================
# STAGE 5: DOCUMENT EMBELLISHMENT
# ============================================================================
embellish(
    logger=logger,  # Pass logger for tracking embellishment activities
    source_dir=CONVERTED_ARTIFACT_DIR,  # Input: converted documents in standard format
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where embellishment fails
    success_dir=EMBELLISHED_ARTIFACTS_DIR,  # Output: enhanced documents with improved presentation
)

# ============================================================================
# STAGE 6: SEMANTIC EXTRACTION
# ============================================================================
extract_semantics(
    logger=logger,  # Pass logger for tracking semantic analysis activities
    source_dir=EMBELLISHED_ARTIFACTS_DIR,  # Input: embellished documents with improved formatting
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where semantic analysis fails
    success_dir=SEMANTICS_EXTRACTED_DIR,  # Output: documents with semantic annotations
)

# ============================================================================
# STAGE 7: TRANSLATION
# ============================================================================
translate_multilingual(
    logger=logger,  # Pass logger for tracking translation activities
    source_dir=SEMANTICS_EXTRACTED_DIR,  # Input: documents with semantic annotations
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where translation fails
    success_dir=TRANSLATED_ARTIFACTS_DIR,  # Output: documents in target languages
)

# ============================================================================
# STAGE 8: PASSWORD PROTECTION
# ============================================================================
password_protect(
    logger=logger,  # Pass logger for tracking password protection activities
    source_dir=TRANSLATED_ARTIFACTS_DIR,  # Input: translated documents
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents where protection fails
    success_dir=PROTECTED_ARTIFACTS_DIR,  # Output: password-protected and encrypted documents
)

# ============================================================================
# STAGE 9: DATABASE TABULATION
# ============================================================================

tabulate(
    logger=logger,  # Pass logger for tracking database insertion activities
    source_dir=PROTECTED_ARTIFACTS_DIR,  # Input: documents with all processing complete
    failure_dir=FAILED_ARTIFACTS_DIR,  # Output: documents that fail database insertion
    success_dir=COMPLETED_ARTIFACTS_DIR,  # Output: fully processed documents ready for use
)

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================
# Log successful completion of entire processing pipeline
# All documents have been processed through all seven stages and are ready for use
logger.info("All processing stages completed successfully!")
