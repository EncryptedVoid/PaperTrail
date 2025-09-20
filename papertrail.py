import os
from datetime import datetime
import logging
from pipelines.sanitizer_pipeline import SanitizerPipeline, SanitizationReport
from pipelines.metadata_pipeline import MetadataPipeline, MetadataReport
from config import LOG_DIR, SYSTEM_DIRECTORIES, UNPROCESSED_ARTIFACTS_DIR, FOR_REVIEW_ARTIFACTS_DIR, SANITIZED_ARTIFACTS_DIR


for path in SYSTEM_DIRECTORIES:
    path.mkdir(parents=True, exist_ok=True)

# Setup comprehensive logging (both console and file)
session_timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")
log_filename = f"PAPERTRAIL-SESSION-{session_timestamp}.log"
handlers = [
    logging.FileHandler(f"{LOG_DIR/{log_filename}", encoding="utf-8"),
]
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)

security_agent: SecurityAgent = SecurityAgent()
session_agent: SessionTracker = SessionTracker()
sanitizer_agent: SanitizerPipeline = SanitizerPipeline(logger=logger, session_agent=session_agent)
metadata_agent: MetadataPipeline = MetadataPipeline(logger=logger,  session_agent=session_agent)

sanitization_report: SanitizationReport = sanitizer_agent.sanitize(source_path=UNPROCESSED_ARTIFACTS_DIR, review_dir=FOR_REVIEW_ARTIFACTS_DIR, success_dir=SANITIZED_ARTIFACTS_DIR)
session_agent.update(sanitization_report)

metadata_report: MetadataReport = metadata_agent.extract_metadata(source_dir=SANITIZED_ARTIFACTS_DIR, review_dir=FOR_REVIEW_ARTIFACTS_DIR, success_dir=METADATA_EXTRACTED_DIR)
session_agent.update(metadata_report)