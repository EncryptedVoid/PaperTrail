"""
PaperTrail Main Processing Pipeline

Complete document processing pipeline with session tracking, conversion, sanitization,
metadata extraction, semantic analysis, and database tabulation with encryption.

Author: Ashiq Gazi
"""

from datetime import datetime
import logging
from pathlib import Path

# Pipeline imports
from pipelines.sanitizer_pipeline import SanitizerPipeline, SanitizationReport
from pipelines.metadata_pipeline import MetadataPipeline, MetadataReport
from pipelines.semantics_pipeline import SemanticsPipeline, SemanticExtractionReport

# from pipelines.database_pipeline import DatabasePipeline, TabulationReport

# Configuration imports
from config import (
    SESSION_LOG_FILE_PREFIX,
    LOG_DIR,
    SYSTEM_DIRECTORIES,
    UNPROCESSED_ARTIFACTS_DIR,
    FOR_REVIEW_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    ARCHIVE_DIR,
    METADATA_EXTRACTED_DIR,
    SEMANTICS_EXTRACTED_DIR,
    ENCRYPTED_DIR,
    PROCESSING_COMPLETED_DIR,
)


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging for the session."""
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create session-specific log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{SESSION_LOG_FILE_PREFIX}-{timestamp}.log"
    log_filepath = LOG_DIR / log_filename

    # Setup logging handlers
    handlers = [
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_filepath, encoding="utf-8"),  # File output
    ]

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Session log: {log_filepath}")

    return logger


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for directory in SYSTEM_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Main PaperTrail processing pipeline execution."""

    # Initialize logging
    logger = setup_logging()

    # Ensure all directories exist
    ensure_directories()

    try:
        logger.info("Starting sanitization stage...")
        sanitizer_agent: SanitizerPipeline = SanitizerPipeline(logger=logger)
        sanitization_report: SanitizationReport = sanitizer_agent.sanitize(
            source_path=UNPROCESSED_ARTIFACTS_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR,
            success_dir=SANITIZED_ARTIFACTS_DIR,
        )
        print(sanitization_report)

        logger.info("Starting metadata extraction stage...")
        metadata_agent: MetadataPipeline = MetadataPipeline(logger=logger)
        metadata_report: MetadataReport = metadata_agent.extract_metadata(
            source_dir=SANITIZED_ARTIFACTS_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR,
            success_dir=METADATA_EXTRACTED_DIR,
        )
        print(metadata_report)

        logger.info("Starting semantic extraction stage...")
        semantics_agent: SemanticsPipeline = SemanticsPipeline(logger=logger)
        semantics_report: SemanticExtractionReport = semantics_agent.extract_semantics(
            source_dir=METADATA_EXTRACTED_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR / "semantic_failures",
            success_dir=SEMANTICS_EXTRACTED_DIR,
        )
        print(semantics_report)

        # logger.info("Starting database tabulation stage...")
        # database_agent: DatabasePipeline = DatabasePipeline(logger=logger)
        # database_report: TabulationReport = database_agent.tabulate(
        #     source_dir=SEMANTICS_EXTRACTED_DIR,
        #     review_dir=FOR_REVIEW_ARTIFACTS_DIR / "tabulation_failures",
        #     success_dir=PROCESSING_COMPLETED_DIR,
        #     use_passphrase=True,  # Use memorable passphrases vs random passwords
        #     export_spreadsheets=True,  # Generate Excel/CSV exports
        #     encrypt_files=True,  # Enable file encryption
        # )

        logger.info("All processing stages completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")

    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
