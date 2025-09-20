from datetime import datetime
import logging
from pipelines.sanitizer_pipeline import SanitizerPipeline, SanitizationReport
from pipelines.metadata_pipeline import MetadataPipeline, MetadataReport
from pipelines.semantics_pipeline import SemanticsPipeline, SemanticsReport
from pipelines.database_pipeline import DatabasePipeline, TabulationReport
from processors.language_processor import LanguageProcessor
from processors.visual_processor import VisualProcessor
from processors.conversion_processor import ConversionProcessor
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


for path in SYSTEM_DIRECTORIES:
    path.mkdir(parents=True, exist_ok=True)

# Setup comprehensive logging (both console and file)
handlers = [
    logging.FileHandler(
        f"{LOG_DIR}/{f"{SESSION_LOG_FILE_PREFIX}-{datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")}.log"}",
        encoding="utf-8",
    ),
]
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)

session_agent: SessionTracker = SessionTracker(logger=logger)

session_agent.start()

conversion_agent: ConversionProcessor = ConversionProcessor(
    logger=logger, archive_dir=ARCHIVE_DIR, review_dir=FOR_REVIEW_ARTIFACTS_DIR
)
sanitizer_agent: SanitizerPipeline = SanitizerPipeline(logger=logger)
sanitization_report: SanitizationReport = sanitizer_agent.sanitize(
    conversion_agent=conversion_agent,
    source_path=UNPROCESSED_ARTIFACTS_DIR,
    review_dir=FOR_REVIEW_ARTIFACTS_DIR,
    success_dir=SANITIZED_ARTIFACTS_DIR,
)
session_agent.update(sanitization_report)
session_agent.display_update()

metadata_agent: MetadataPipeline = MetadataPipeline(logger=logger)
metadata_report: MetadataReport = metadata_agent.extract_metadata(
    source_dir=SANITIZED_ARTIFACTS_DIR,
    review_dir=FOR_REVIEW_ARTIFACTS_DIR,
    success_dir=METADATA_EXTRACTED_DIR,
)
session_agent.update(metadata_report)
session_agent.display_update()

visual_processor: VisualProcessor = VisualProcessor(logger=logger)
language_processor: LanguageProcessor = LanguageProcessor(logger=logger)
semantics_agent: SemanticsPipeline = SemanticsPipeline(
    logger=logger,
    session_agent=session_agent,
    visual_processor=visual_processor,
    language_processor=language_processor,
)
semantics_report: SemanticsReport = semantics_agent.extract_semantics(
    source_dir=METADATA_EXTRACTED_DIR,
    review_dir=METADATA_EXTRACTED_DIR / "failed",
    success_dir=SEMANTICS_EXTRACTED_DIR,
)
session_agent.update(semantics_report)
session_agent.display_update()

database_agent: DatabasePipeline = DatabasePipeline(
    logger=logger, session_agent=session_agent
)
database_report: TabulationReport = database_agent.tabulate(
    source_dir=ENCRYPTED_DIR,
    review_dir=ENCRYPTED_DIR / "failed",
    success_dir=PROCESSING_COMPLETED_DIR,
    use_passphrase=True,  # Use memorable passphrases vs random passwords
    export_spreadsheets=True,  # Generate Excel/CSV exports
    encrypt_files=True,  # Enable file encryption
)
session_agent.update(database_report)
session_agent.display_update()

session_agent.end()
session_agent.display_session_report()
