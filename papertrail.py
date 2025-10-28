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

from config import (
	LOG_DIR ,
	LOG_FORMAT ,
	LOG_LEVEL ,
	SESSION_LOG_FILE_PREFIX ,
	SYSTEM_DIRECTORIES ,
	UNPROCESSED_ARTIFACTS_DIR ,
)
from stages.archive_emails import archiving_emails
from stages.auto_sort import automatically_sorting
from stages.file_conversion import converting_files
from stages.identify_duplicates import DuplicateReviewer
from stages.manual_review import FileReviewUI
from stages.sanitize import sanitizing
from utilities.metadata_extraction import extracting_metadata
from utilities.semantics_extraction import extracting_semantics
from utilities.visual_processor import VisualProcessor

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
    logging.FileHandler(
        (LOG_DIR / log_file_name),  # Outputs log messages to timestamped file
        encoding="utf-8",  # Ensures proper handling of international characters
    ),
]

# Configure the root logger with INFO level and custom format
# INFO level captures important events without excessive debug details
# Format includes timestamp, logger name, severity level, and the actual message
logging.basicConfig(
    level=LOG_LEVEL,  # Set minimum logging level (INFO, WARNING, ERROR, CRITICAL)
    format=LOG_FORMAT,  # Structured log format
    handlers=handlers,  # Apply both console and file handlers
)

# Create a named logger for this application
# This allows filtering and identifying PaperTrail-specific log entries
logger = logging.getLogger("PAPERTRAIL")

# ============================================================================
# PIPELINE STARTING
# ============================================================================

# Log the session startup banner
# This marks the beginning of a new processing session in the logs
logger.info("WELCOME TO PAPERTRAIL! AN AUTOMATED ARTIFACT ORGANISATION SYSTEM")

visual_processor = VisualProcessor(logger=logger)

archiving_emails(logger=logger)
sanitizing(logger=logger, source_dir=UNPROCESSED_ARTIFACTS_DIR)

app = DuplicateReviewer(UNPROCESSED_ARTIFACTS_DIR)
app.run()

automatically_sorting(
    logger=logger,
    visual_processor=visual_processor,
    source_dir=UNPROCESSED_ARTIFACTS_DIR,
)

extracting_metadata(logger=logger, source_dir=UNPROCESSED_ARTIFACTS_DIR)

converting_files(logger=logger, source_dir=UNPROCESSED_ARTIFACTS_DIR)

app = FileReviewUI(UNPROCESSED_ARTIFACTS_DIR)
app.run()

extracting_semantics(logger=logger, source_dir=UNPROCESSED_ARTIFACTS_DIR)

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================
# Log successful completion of entire processing pipeline
# All documents have been processed through all seven stages and are ready for use
logger.info("All processing stages completed successfully!")
