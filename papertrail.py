"""
PaperTrail Main Processing Pipeline

Complete document processing pipeline with session tracking, conversion, sanitization,
metadata extraction, semantic analysis, and database tabulation with encryption.

Author: Ashiq Gazi
"""

import logging
from datetime import datetime

from config import (
    ARCHIVAL_DIR,
    COMPLETED_ARTIFACTS_DIR,
    CONVERTED_ARTIFACT_DIR,
    FAILED_ARTIFACTS_DIR,
    LOG_DIR,
    METADATA_EXTRACTED_DIR,
    SANITIZED_ARTIFACTS_DIR,
    SEMANTICS_EXTRACTED_DIR,
    SESSION_LOG_FILE_PREFIX,
    SYSTEM_DIRECTORIES,
    UNPROCESSED_ARTIFACTS_DIR,
)
from src.processors import (
    convert,
    extract_metadata,
    extract_semantics,
    sanitize,
    tabulate,
)

# Ensure all directories exist
for directory in SYSTEM_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

log_file_name = (
    f"{SESSION_LOG_FILE_PREFIX}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log"
)

# Setup logging handlers
handlers = [
    logging.StreamHandler(),  # Console output
    logging.FileHandler(
        (LOG_DIR / log_file_name),
        encoding="utf-8",
    ),  # File output
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger("PAPERTRAIL")

logger.info(
    "WELCOME TO PAPERTRAIL! AN AUTOMATED ARTIFACT REGISTRY AND ORGANISATION SYSTEM"
)

sanitize(
    logger=logger,
    source_dir=UNPROCESSED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=SANITIZED_ARTIFACTS_DIR,
)

extract_metadata(
    logger=logger,
    source_dir=SANITIZED_ARTIFACTS_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=METADATA_EXTRACTED_DIR,
)

convert(
    logger=logger,
    source_dir=METADATA_EXTRACTED_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    archive_dir=ARCHIVAL_DIR,
    success_dir=CONVERTED_ARTIFACT_DIR,
)

extract_semantics(
    logger=logger,
    source_dir=CONVERTED_ARTIFACT_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=SEMANTICS_EXTRACTED_DIR,
)

tabulate(
    logger=logger,
    source_dir=CONVERTED_ARTIFACT_DIR,
    failure_dir=FAILED_ARTIFACTS_DIR,
    success_dir=COMPLETED_ARTIFACTS_DIR,
)

logger.info("All processing stages completed successfully!")
