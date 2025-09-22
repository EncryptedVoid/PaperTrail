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
from pipelines.database_pipeline import DatabasePipeline, TabulationReport

# Processor imports
from processors.language_processor import LanguageProcessor
from processors.visual_processor import VisualProcessor
from processors.conversion_processor import ConversionProcessor

# Utility imports
from utilities.session_tracking_agent import SessionTracker

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

    # Initialize session tracking
    session_agent = SessionTracker(logger=logger)
    session_agent.start()

    try:
        # ========================================
        # STAGE 1: FILE CONVERSION
        # ========================================
        logger.info("Starting file conversion stage...")
        session_agent.start_stage("conversion")

        conversion_agent = ConversionProcessor(
            logger=logger, archive_dir=ARCHIVE_DIR, review_dir=FOR_REVIEW_ARTIFACTS_DIR
        )

        # Note: Conversion is handled individually within sanitization stage
        # This stage setup is for tracking purposes

        # ========================================
        # STAGE 2: SANITIZATION AND DUPLICATE DETECTION
        # ========================================
        logger.info("Starting sanitization stage...")
        session_agent.start_stage("sanitization")

        sanitizer_agent = SanitizerPipeline(logger=logger)
        sanitization_report = sanitizer_agent.sanitize(
            conversion_agent=conversion_agent,
            source_path=UNPROCESSED_ARTIFACTS_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR,
            success_dir=SANITIZED_ARTIFACTS_DIR,
        )

        session_agent.update(sanitization_report)
        session_agent.display_update()

        # ========================================
        # STAGE 3: METADATA EXTRACTION
        # ========================================
        logger.info("Starting metadata extraction stage...")
        session_agent.start_stage("metadata_extraction")

        metadata_agent = MetadataPipeline(logger=logger)
        metadata_report = metadata_agent.extract_metadata(
            source_dir=SANITIZED_ARTIFACTS_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR,
            success_dir=METADATA_EXTRACTED_DIR,
        )

        session_agent.update(metadata_report)
        session_agent.display_update()

        # ========================================
        # STAGE 4: SEMANTIC EXTRACTION AND ANALYSIS
        # ========================================
        logger.info("Starting semantic extraction stage...")
        session_agent.start_stage("semantic_extraction")

        # Initialize AI processors
        visual_processor = VisualProcessor(logger=logger)
        language_processor = LanguageProcessor(logger=logger)

        semantics_agent = SemanticsPipeline(
            logger=logger,
            session_agent=session_agent,
            visual_processor=visual_processor,
            field_extractor=language_processor,
        )

        semantics_report = semantics_agent.extract_semantics(
            source_dir=METADATA_EXTRACTED_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR / "semantic_failures",
            success_dir=SEMANTICS_EXTRACTED_DIR,
        )

        session_agent.update(semantics_report)
        session_agent.display_update()

        # ========================================
        # STAGE 5: DATABASE TABULATION AND ENCRYPTION
        # ========================================
        logger.info("Starting database tabulation stage...")
        session_agent.start_stage("tabulation_encryption")

        database_agent = DatabasePipeline(logger=logger)
        database_report = database_agent.tabulate(
            source_dir=SEMANTICS_EXTRACTED_DIR,
            review_dir=FOR_REVIEW_ARTIFACTS_DIR / "tabulation_failures",
            success_dir=PROCESSING_COMPLETED_DIR,
            use_passphrase=True,  # Use memorable passphrases vs random passwords
            export_spreadsheets=True,  # Generate Excel/CSV exports
            encrypt_files=True,  # Enable file encryption
        )

        session_agent.update(database_report)
        session_agent.display_update()

        # ========================================
        # SESSION COMPLETION
        # ========================================
        logger.info("All processing stages completed successfully!")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        session_agent.add_error("Processing interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        session_agent.add_error(f"Fatal error: {str(e)}")
        raise

    finally:
        # Always end session tracking and display final report
        session_agent.end()
        session_agent.display_session_report()


if __name__ == "__main__":
    main()
