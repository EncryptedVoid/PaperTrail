"""
PaperTrail Main Processing Pipeline

Complete document processing pipeline with session tracking, conversion, sanitization,
metadata extraction, semantic analysis, and database tabulation with encryption.

Author: Ashiq Gazi
"""

import logging
from datetime import datetime

# Configuration imports
from config import (
	ARCHIVE_DIR ,
	COMPLETED_ARTIFACTS_DIR ,
	CONVERTED_ARTIFACT_DIR ,
	FOR_REVIEW_ARTIFACTS_DIR ,
	LOG_DIR ,
	METADATA_EXTRACTED_DIR ,
	SANITIZED_ARTIFACTS_DIR ,
	SEMANTICS_EXTRACTED_DIR ,
	SESSION_LOG_FILE_PREFIX ,
	SYSTEM_DIRECTORIES ,
	UNPROCESSED_ARTIFACTS_DIR ,
)
# Pipeline imports
from src.pipelines import (
	ConversionPipeline ,
	DatabasePipeline ,
	MetadataPipeline ,
	MetadataReport ,
	SanitizationReport ,
	SanitizerPipeline ,
	SemanticExtractionReport ,
	SemanticsPipeline ,
	TabulationReport ,
)

# subprocess.check_call(
#     [
#         sys.executable,
#         "-m",
#         "pip",
#         "install",
#         "--extra-index-url",
#         "https://download.pytorch.org/whl/cu128",
#         "-e",
#         ".",
#     ]
# )

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

try:
    logger.info("Starting sanitization stage...")
    sanitizer_agent: SanitizerPipeline = SanitizerPipeline(logger=logger)
    sanitization_report: SanitizationReport = sanitizer_agent.sanitize(
        source_dir=UNPROCESSED_ARTIFACTS_DIR,
        failure_dir=FOR_REVIEW_ARTIFACTS_DIR,
        success_dir=SANITIZED_ARTIFACTS_DIR,
    )
    print(sanitization_report)

    logger.info("Starting metadata extraction stage...")
    metadata_agent: MetadataPipeline = MetadataPipeline(logger=logger)
    metadata_report: MetadataReport = metadata_agent.extract_metadata(
        source_dir=SANITIZED_ARTIFACTS_DIR,
        failure_dir=FOR_REVIEW_ARTIFACTS_DIR,
        success_dir=METADATA_EXTRACTED_DIR,
    )
    print(metadata_report)

    logger.info("Starting semantic extraction stage...")
    semantics_agent: SemanticsPipeline = SemanticsPipeline(logger=logger)
    semantics_report: SemanticExtractionReport = semantics_agent.extract_semantics(
        source_dir=METADATA_EXTRACTED_DIR,
        failure_dir=FOR_REVIEW_ARTIFACTS_DIR,
        success_dir=SEMANTICS_EXTRACTED_DIR,
    )
    print(semantics_report)

    logger.info("Starting artifact file type conversion stage...")
    conversion_agent: ConversionPipeline = ConversionPipeline(
        logger=logger, archive_dir=ARCHIVE_DIR
    )
    conversion_report = conversion_agent.convert(
        source_dir=SEMANTICS_EXTRACTED_DIR,
        failure_dir=FOR_REVIEW_ARTIFACTS_DIR,
        success_dir=CONVERTED_ARTIFACT_DIR,
    )
    print(conversion_report)

    logger.info("Starting database tabulation stage...")
    database_agent: DatabasePipeline = DatabasePipeline(logger=logger)
    database_report: TabulationReport = database_agent.tabulate(
        source_dir=CONVERTED_ARTIFACT_DIR,
        failure_dir=FOR_REVIEW_ARTIFACTS_DIR / "tabulation_failures",
        success_dir=COMPLETED_ARTIFACTS_DIR,
    )
    print(database_report)

    logger.info("All processing stages completed successfully!")

except KeyboardInterrupt:
    logger.warning("Processing interrupted by user")

except Exception as e:
    logger.error(f"Fatal error during processing: {e}")
    raise
