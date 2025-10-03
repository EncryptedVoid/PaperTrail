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
    ANALYSED_ARTIFACTS_DIR,
    ARCHIVAL_DIR,
    ARCHIVE_EMAILS,
    COMPLETED_ARTIFACTS_DIR,
    CONVERTED_ARTIFACT_DIR,
    ENCRYPTED_ARTIFACTS_DIR,
    FAILED_ARTIFACTS_DIR,
    LOG_DIR,
    REVIEWED_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    SESSION_LOG_FILE_PREFIX,
    SYSTEM_DIRECTORIES,
    TABULATED_DIR,
    UNPROCESSED_ARTIFACTS_DIR,
)

# Import all processor modules that handle each stage of the pipeline
from src.processors import (
    archive_emails,
    convert,
    encrypt,
    extract_artifact_data,
    manual_backup,
    manual_review,
    sanitize,
    tabulate,
    transform,
    translate_multilingual,
)
from utilities.email_archival_client import (
    archive_gmail_starred,
    archive_outlook_flagged,
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

if ARCHIVE_EMAILS:
    archive_gmail_starred(logger=logger, target_dir=UNPROCESSED_ARTIFACTS_DIR)
    archive_outlook_flagged(logger=logger, target_dir=UNPROCESSED_ARTIFACTS_DIR)

# SANITIZATION STAGE (AUTOMATIC) - STAYS THE SAME
sanitize(
    logger=logger,
    source_dir=UNPROCESSED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=SANITIZED_ARTIFACTS_DIR,
)

# REVIEW STAGE (MANUAL) - COMBINE SCAN, EMBELLISH, AND COMBINE/SPLICE (IMGs, DOCs) STAGE TO PREPARE ALL DOCUMENTS AND IMAGES
# 	QUICKLY REVIEW ITEMS (VIEW ITEMS, TEXT/JSONS/ETC, AUTOPLAY VIDEOS W/MUTED AUDIO)
# 	RIGHT TO KEEP, LEFT TO DESTROY, DOWN FOR FURTHER REVIEW, UP TO SCAN (DETECT HANDWRITING AND DOCUMENT) AND FURTHER REVIEW
manual_review(
    logger=logger,
    source_dir=SANITIZED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    archive_dir=ARCHIVAL_DIR,
    success_dir=REVIEWED_ARTIFACTS_DIR,
)

# CONVERSION (AUTOMATIC)
# 	CONVERT TO COMMON FILE TYPES
convert(
    logger=logger,
    source_dir=REVIEWED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    archive_dir=ARCHIVAL_DIR,
    success_dir=CONVERTED_ARTIFACT_DIR,
)

# DATA EXTRACTION & EMBEDDING  (AUTOMATIC)
# 	EXTRACT METADATA WITH APACHE TIKA
# 	EXTRACT SEMANTICS DATA
# 		EXECUTIVE TITLE, QUICK SUMMARY, EXECUTIVE SUMMARY, LANGUAGE OF DOCUMENT, UTILITY SUMMARY, CONFIDENTIALITY LEVEL, ISSUING BODY
extract_artifact_data(
    logger=logger,
    source_dir=CONVERTED_ARTIFACT_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=ANALYSED_ARTIFACTS_DIR,
)

# 	TRANSLATE ALL DOCUMENTS AND THUMBNAILS TO ENG, FRA, DEU, BEN, POL AND LINK THESE TO THE MAIN DOCUMENT
translate_multilingual(
    logger=logger,
    source_dir=ANALYSED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=ANALYSED_ARTIFACTS_DIR,
)

# ENCRYPTION STAGE  (AUTOMATIC)
# 	PASSWORD PROTECTION W/PASSWORD STRENGTH RELATIVE TO CONFIDENTIALITY LEVEL AND PASSWORD VAULT
# 	EMBED AND PROTECT METADATA WITH PRIVATE/PUBLIC LEVELS
encrypt(
    logger=logger,
    source_dir=ANALYSED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=ENCRYPTED_ARTIFACTS_DIR,
)

# TABULATION
# 	SQL DATABASE WITH PASSWORD PROTECTION
# 	GUI CLIENT W/FUZZY FINDING AND ITEM PREVIEW WITH ITS THUMBNAIL AND THEN UNLOCKING IF FURTHER VIEWING NEEDED
# 	AUTO-COPY TEXT FOR SHARING
# 		ITEM NAME, ITEM TITLE, ITEM QUICK SUMMARY, ITEM PASSWORD, ITEM CONFIDENTIALITY LEVEL, WATERMARK OPTION
tabulate(
    logger=logger,
    source_dir=ENCRYPTED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=TABULATED_DIR,
)

manual_backup(
    logger=logger,
    source_dir=TABULATED_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=COMPLETED_ARTIFACTS_DIR,
)

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================
# Log successful completion of entire processing pipeline
# All documents have been processed through all seven stages and are ready for use
logger.info("All processing stages completed successfully!")
