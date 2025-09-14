#!C:/Users/UserX/AppData/Local/Programs/Python/Python313/python.exe

from tqdm import tqdm
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from collections import defaultdict
from checksum_utils import HashAlgorithm, generate_checksum
from uuid_utils import generate_uuid4plus
from metadata_processor import MetadataExtractor
from visual_processor import QwenDocumentProcessor
from language_processor import LanguageProcessor

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================

# Model settings
GPU_MEMORY_LIMIT_PERCENT: float = 75.0
MODEL_REFRESH_INTERVAL: int = 20
MAX_MEMORY_THRESHOLD_PERCENT: float = 75.0

# Checksum algorithm for duplicate detection and integrity verification
CHECKSUM_ALGORITHM: HashAlgorithm = HashAlgorithm.SHA3_512

# Enhanced supported extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    # Text & Documents
    ".txt",
    ".md",
    ".rtf",
    ".doc",
    ".docx",
    ".pdf",
    ".odt",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".ods",
    ".odp",
    ".odg",
    # Data formats
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
    ".heif",
    ".raw",
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".tga",
    ".psd",
    # Email & Communication
    ".eml",
    ".msg",
    ".mbox",
    ".ics",
    ".vcs",
}

# Enhanced unsupported extensions
UNSUPPORTED_EXTENSIONS: Set[str] = {
    # Audio
    "mp3",
    "aac",
    "ogg",
    "wma",
    "m4a",
    "wav",
    "flac",
    "aiff",
    # Video
    "mp4",
    "avi",
    "mov",
    "mkv",
    "wmv",
    "flv",
    "webm",
    "ogv",
    "m4v",
    # 3D & CAD
    "obj",
    "fbx",
    "dae",
    "3ds",
    "blend",
    "dwg",
    "dxf",
    "step",
    # Executables & System
    "exe",
    "msi",
    "dmg",
    "pkg",
    "deb",
    "rpm",
    "dll",
    "so",
    "dylib",
    # Databases
    "db",
    "sqlite",
    "mdb",
    "accdb",
    # Proprietary
    "indd",
    "fla",
    "swf",
    "sav",
    "dta",
    "sas7bdat",
    "mat",
    "hdf5",
    # Archives (if extraction added)
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

# File names for permanent storage
CHECKSUM_HISTORY_FILE = "checksum_history.txt"

# Base directory for all processing
BASE_DIR: Path = Path(r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run")

# Directory structure - each directory represents a processing stage
PATHS: Dict[str, Path] = {
    "base_dir": BASE_DIR,
    "unprocessed_dir": BASE_DIR / "unprocessed_artifacts",
    "rename_dir": BASE_DIR / "identified_artifacts",
    "metadata_dir": BASE_DIR / "metadata_extracted",
    "semantics_dir": BASE_DIR / "visually_processed",
    "logs_dir": BASE_DIR / "session_logs",
    "completed_dir": BASE_DIR / "completed_artifacts",
    "temp_dir": BASE_DIR / "temp",
    "review_dir": BASE_DIR
    / "review_required",  # For zero-byte files and problematic files
    "profiles_dir": BASE_DIR
    / "artifact_profiles",  # JSON profile files for each artifact
}

# =====================================================================
# INITIAL SETUP - DIRECTORIES, LOGGING, SESSION TRACKING
# =====================================================================

# Create all required directories - but preserve existing ones
try:
    for name, path in PATHS.items():
        if name.endswith("_dir"):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    # Setup comprehensive logging (both console and file)
    session_timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")
    log_filename = f"PAPERTRAIL-SESSION-{session_timestamp}.log"

    handlers = [
        logging.FileHandler(
            f"{str(PATHS['logs_dir'])}/{log_filename}", encoding="utf-8"
        ),
        logging.StreamHandler(),  # Also log to console for real-time monitoring
    ]

    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
except Exception as e:
    raise e

# =====================================================================
# SESSION INITIALIZATION AND STATE TRACKING
# =====================================================================

# Initialize session tracking
session_start_time = datetime.now()
session_data = {
    "session_id": session_timestamp,
    "start_time": session_start_time.isoformat(),
    "status": "running",
    "total_files": 0,
    "processed_files": 0,
    "file_types": defaultdict(int),
    "stage_counts": {
        "unprocessed": 0,
        "rename": 0,
        "metadata": 0,
        "semantics": 0,
        "completed": 0,
        "failed": 0,
        "review": 0,
    },
    "performance": {"files_per_minute": 0.0, "total_runtime_seconds": 0},
    "errors": [],
}

# SESSION JSON file path
session_json_path = PATHS["logs_dir"] / f"SESSION-{session_timestamp}.json"

logger.info("=" * 80)
logger.info("PAPERTRAIL DOCUMENT PROCESSING PIPELINE STARTED")
logger.info("=" * 80)
logger.info(f"Session ID: {session_timestamp}")
logger.info(f"Logging initialized. Log file: {log_filename}")
logger.info(f"Session JSON: {session_json_path.name}")
logger.info(f"Base directory: {PATHS['base_dir']}")
logger.info(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
logger.info(f"Unsupported extensions: {UNSUPPORTED_EXTENSIONS}")

# Load permanent checksum history (one checksum per line)
checksum_history: Set[str] = set()
checksum_history_path = PATHS["base_dir"] / CHECKSUM_HISTORY_FILE

if checksum_history_path.exists():
    try:
        with open(checksum_history_path, "r", encoding="utf-8") as f:
            checksum_history = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(checksum_history)} permanent checksums from history")
    except Exception as e:
        logger.warning(f"Failed to load permanent checksums: {e}")
        raise e
else:
    logger.info("No permanent checksum history found - creating new file")
    checksum_history_path.touch()


def update_session_json():
    """Update the SESSION JSON file with current progress data"""
    try:
        with open(session_json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to update SESSION JSON: {e}")


def log_detailed_progress(file_name: str, file_size: int, stage: str):
    """Log detailed multi-line progress information after each file"""
    elapsed_seconds = (datetime.now() - session_start_time).total_seconds()
    files_per_minute = (
        (session_data["processed_files"] / elapsed_seconds * 60)
        if elapsed_seconds > 0
        else 0
    )

    # Update performance metrics
    session_data["performance"]["files_per_minute"] = round(files_per_minute, 2)
    session_data["performance"]["total_runtime_seconds"] = int(elapsed_seconds)

    # Format file size for human readability
    if file_size < 1024:
        size_str = f"{file_size}B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size/1024:.1f}KB"
    else:
        size_str = f"{file_size/(1024*1024):.1f}MB"

    # Create file type summary
    file_type_summary = ", ".join(
        [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
    )

    # Create stage status summary
    stage_summary = ", ".join(
        [
            f"{stage}({count})"
            for stage, count in session_data["stage_counts"].items()
            if count > 0
        ]
    )

    logger.info("=" * 60)
    logger.info(
        f"[File {session_data['processed_files']}/{session_data['total_files']}] Completed: {file_name} ({size_str})"
    )
    logger.info(f"Stage: {stage}")
    logger.info(
        f"Session Runtime: {elapsed_seconds//3600:02.0f}:{(elapsed_seconds%3600)//60:02.0f}:{elapsed_seconds%60:02.0f} | Speed: {files_per_minute:.1f} files/min"
    )
    logger.info(f"File Types: {file_type_summary}")
    logger.info(f"Stage Status: {stage_summary}")
    logger.info("=" * 60)


def update_stage_counts(from_stage: str, to_stage: str, session_data: Dict[str, Any]):
    """
    Properly update stage counts when moving files between stages
    Always decrements source stage and increments destination stage
    """
    if from_stage in session_data["stage_counts"]:
        session_data["stage_counts"][from_stage] -= 1
        # Ensure count doesn't go negative
        if session_data["stage_counts"][from_stage] < 0:
            session_data["stage_counts"][from_stage] = 0

    if to_stage in session_data["stage_counts"]:
        session_data["stage_counts"][to_stage] += 1
    else:
        session_data["stage_counts"][to_stage] = 1


# =====================================================================
# STAGE 1: DUPLICATE DETECTION AND CHECKSUM VERIFICATION
# =====================================================================

# Get all artifacts in unprocessed directory and sort by size (smallest first)
unprocessed_artifacts: List[Path] = [
    item for item in PATHS["unprocessed_dir"].iterdir()
]

if len(unprocessed_artifacts) == 0:

    logger.info("\n" + "=" * 80)
    logger.info(
        "STAGE 1: DUPLICATE DETECTION, UNSUPPORTED FILES, AND CHECKSUM VERIFICATION"
    )
    logger.info("=" * 80)
    logger.info("Scanning unprocessed artifacts for duplicates and zero-byte files...")

    # Sort by file size - process smallest files first for faster initial feedback
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)

    session_data["total_files"] = len(unprocessed_artifacts)
    logger.info(
        f"Found {len(unprocessed_artifacts)} artifacts to process (sorted smallest to largest)"
    )

    # Count file types for initial summary
    for artifact in unprocessed_artifacts:
        ext = artifact.suffix.lower()
        session_data["file_types"][ext] += 1

    initial_file_summary = ", ".join(
        [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
    )
    logger.info(f"File type breakdown: {initial_file_summary}")

    # Update session JSON with initial state
    update_session_json()

    # Process each artifact for duplicate detection
    skipped_duplicates = 0
    skipped_zero_byte = 0
    unsupported_artifacts = 0
    remaining_artifacts: List[Path] = []

    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Checking for duplicates, zeros, and unsupported artifacts",
        unit="artifacts",
    ):
        try:
            review_location = PATHS["review_dir"] / artifact.name

            if artifact.suffix in UNSUPPORTED_EXTENSIONS:
                logger.debug(
                    f"Found artifact to an unsupported file type. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                unsupported_artifacts += 1
                continue

            # Check for zero-byte files (corrupted/incomplete downloads)
            file_size = artifact.stat().st_size
            if file_size == 0:
                # Handle duplicate names in review folder
                # counter = 1
                # while review_location.exists():
                #     name_part = artifact.stem
                #     ext_part = artifact.suffix
                #     review_location = PATHS["review_dir"] / f"{name_part}_{counter}{ext_part}"
                #     counter += 1

                logger.debug(
                    f"Found artifact to be a zero-size item. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                skipped_zero_byte += 1
                continue

            # Generate checksum for duplicate detection
            checksum = generate_checksum(artifact, algorithm=CHECKSUM_ALGORITHM)
            logger.debug(f"Generated checksum for {artifact.name}: {checksum[:16]}...")

            # Check if this file has been processed before (permanent history)
            if checksum in checksum_history:
                logger.info(
                    f"Duplicate detected. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                skipped_duplicates += 1
                continue

            # Add to permanent checksums and save immediately
            checksum_history.add(checksum)
            with open(checksum_history_path, "a", encoding="utf-8") as f:
                f.write(f"{checksum}\n")

            # This is a new file - keep for processing
            remaining_artifacts.append(artifact)
            session_data["stage_counts"]["unprocessed"] += 1

        except Exception as e:
            logger.error(
                f"Failed to process {artifact.name} in duplicate detection: {e}"
            )
            session_data["errors"].append(
                {"file": artifact.name, "stage": "duplicate_detection", "error": str(e)}
            )

    logger.info(f"Duplicate detection complete:")
    logger.info(f"  - {len(remaining_artifacts)} new files to process")
    logger.info(f"  - {skipped_duplicates} duplicates skipped")
    logger.info(f"  - {skipped_zero_byte} zero-byte files moved to review")
    logger.info(f"  - {unsupported_artifacts} unsupported files moved to review")

    # Update session data and JSON
    session_data["total_files"] = len(remaining_artifacts)
    update_session_json()

    # =====================================================================
    # STAGE 2: UUID RENAMING AND PROFILE CREATION
    # =====================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: UUID RENAMING AND PROFILE CREATION")
    logger.info("=" * 80)
    logger.info("Renaming artifacts with unique UUIDs and creating profile files...")

    remaining_artifacts: List[Path] = [
        item for item in PATHS["unprocessed_dir"].iterdir()
    ]

    for artifact in tqdm(
        remaining_artifacts, desc="Preparing artifact profiles", unit="artifact"
    ):
        try:
            # Generate unique identifier for this artifact
            artifact_id: str = generate_uuid4plus()
            original_name = artifact.name
            file_size = artifact.stat().st_size

            # Create initial profile data
            profile_data = {
                "uuid": artifact_id,
                "original_filename": original_name,
                "file_size": file_size,
                "file_extension": artifact.suffix.lower(),
                "stages": {
                    "renamed": {
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                    }
                },
            }

            # Rename artifact with UUID (preserving extension)
            new_artifact_name = f"ARTIFACT-{artifact_id}{artifact.suffix}"
            renamed_artifact = artifact.rename(PATHS["rename_dir"] / new_artifact_name)

            # Create corresponding profile file
            profile_filename = f"PROFILE-{artifact_id}.json"
            profile_path = PATHS["profiles_dir"] / profile_filename

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Update session tracking
            session_data["processed_files"] += 1
            session_data["stage_counts"]["unprocessed"] -= 1
            session_data["stage_counts"]["rename"] += 1

            # Log detailed progress
            log_detailed_progress(new_artifact_name, file_size, "renamed")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Renamed: {original_name} -> {new_artifact_name}")
            logger.debug(f"Created profile: {profile_filename}")

        except Exception as e:
            logger.error(f"Failed to rename {artifact.name}: {e}")
            session_data["errors"].append(
                {"file": artifact.name, "stage": "rename", "error": str(e)}
            )

    logger.info(
        f"Renaming stage complete - {session_data['stage_counts']['rename']} files renamed"
    )


# =====================================================================
# STAGE 3: METADATA EXTRACTION
# =====================================================================

# Get all artifacts from rename directory, sorted by size
renamed_artifacts: List[Path] = [item for item in PATHS["rename_dir"].iterdir()]

if len(renamed_artifacts) == 0:
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: METADATA EXTRACTION")
    logger.info("=" * 80)
    logger.info("Extracting technical metadata from renamed artifacts...")

    renamed_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(renamed_artifacts)} artifacts ready for metadata extraction"
    )

    # Initialize metadata extractor
    extractor = MetadataExtractor(logger=logger)
    logger.info("Document Metadata Extractor initialized")

    for artifact in tqdm(
        renamed_artifacts, desc="Extracting metadata", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Extract metadata using the extractor
            logger.info(f"Extracting metadata for: {artifact.name}")
            extracted_metadata: Dict[str, Any] = extractor.extract(artifact)

            # Update profile with metadata and stage completion
            profile_data["metadata"] = extracted_metadata
            profile_data["stages"]["metadata_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(PATHS["metadata_dir"] / artifact.name)

            # Update session tracking
            update_stage_counts("rename", "metadata", session_data)

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "metadata_extraction")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Metadata extracted and saved for: {artifact.name}")

            STAGE_3 = True

        except Exception as e:
            logger.error(f"Failed to extract metadata for {artifact.name}: {e}")

            # Create failed subdirectory if needed and move file there
            failed_dir = PATHS["rename_dir"] / "failed"
            failed_dir.mkdir(exist_ok=True)

            try:
                artifact.rename(failed_dir / artifact.name)
                update_stage_counts("rename", "failed", session_data)
                logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
            except Exception as move_error:
                logger.error(
                    f"Failed to move {artifact.name} to failed directory: {move_error}"
                )

            session_data["errors"].append(
                {"file": artifact.name, "stage": "metadata_extraction", "error": str(e)}
            )

    logger.info(
        f"Metadata extraction complete - {session_data['stage_counts']['metadata']} files processed"
    )

# =====================================================================
# STAGE 4: SEMANTIC EXTRACTION (VISUAL PROCESSING)
# =====================================================================

# Get all artifacts from metadata directory, sorted by size
metadata_artifacts: List[Path] = [item for item in PATHS["metadata_dir"].iterdir()]

if len(metadata_artifacts) == 0:

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4: SEMANTIC EXTRACTION (VISUAL PROCESSING)")
    logger.info("=" * 80)
    logger.info("Extracting semantic data using visual processor...")

    metadata_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(metadata_artifacts)} artifacts ready for semantic extraction"
    )

    # Initialize visual processor
    processor = QwenDocumentProcessor(logger=logger)
    logger.info("LLM-based Document Semantics and Text Extractor initialized")

    for artifact in tqdm(
        metadata_artifacts, desc="Extracting semantic descriptions", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Extract semantic data using visual processor
            logger.info(f"Extracting semantics for: {artifact.name}")
            extracted_semantics: Dict[str, str] = processor.extract_article_semantics(
                document=artifact
            )

            # Update profile with semantics and stage completion
            profile_data["semantics"] = extracted_semantics
            profile_data["stages"]["semantic_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(PATHS["semantics_dir"] / artifact.name)

            # Update session tracking
            update_stage_counts("metadata", "semantics", session_data)

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "semantic_extraction")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Semantics extracted and saved for: {artifact.name}")

        except Exception as e:
            logger.error(f"Failed to extract semantics for {artifact.name}: {e}")

            # Create failed subdirectory if needed and move file there
            failed_dir = PATHS["metadata_dir"] / "failed"
            failed_dir.mkdir(exist_ok=True)

            try:
                artifact.rename(failed_dir / artifact.name)
                update_stage_counts("metadata", "failed", session_data)
                logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
            except Exception as move_error:
                logger.error(
                    f"Failed to move {artifact.name} to failed directory: {move_error}"
                )

            session_data["errors"].append(
                {"file": artifact.name, "stage": "semantic_extraction", "error": str(e)}
            )

    logger.info(
        f"Semantic extraction complete - {session_data['stage_counts']['semantics']} files processed"
    )

# =====================================================================
# STAGE 5: LLM FIELD EXTRACTION (SEMANTIC METADATA)
# =====================================================================


# Get all artifacts from semantics directory, sorted by size
semantic_artifacts: List[Path] = [item for item in PATHS["semantics_dir"].iterdir()]

if len(semantic_artifacts) == 0:

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5: LLM FIELD EXTRACTION (SEMANTIC METADATA)")
    logger.info("=" * 80)
    logger.info("Extracting structured document fields using LLM...")

    semantic_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(semantic_artifacts)} artifacts ready for LLM field extraction"
    )

    # Initialize LLM field extractor
    field_extractor = LanguageProcessor(logger=logger)
    logger.info("LLM Document Field Extractor initialized")

    # Create new directory for LLM-processed artifacts
    PATHS["llm_processed_dir"] = BASE_DIR / "llm_processed"
    PATHS["llm_processed_dir"].mkdir(parents=True, exist_ok=True)

    for artifact in tqdm(
        semantic_artifacts,
        desc="Extracting semantic metadata with LLM processing",
        unit="artifact",
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Get the data we need for LLM processing
            ocr_text = profile_data.get("semantics", {}).get("all_text", "")
            visual_description = profile_data.get("semantics", {}).get(
                "all_imagery", ""
            )
            metadata = profile_data.get("metadata", {})

            # Check if we have enough content to process

            # Check if OCR completely failed
            if (
                not ocr_text or ocr_text.strip() in ["No text found in document.", ""]
            ) and (
                not visual_description
                or visual_description.strip() in ["No visual content described.", ""]
            ):

                logger.error(
                    f"Both OCR text extraction and visual processing failed for {artifact.name} - moving to review"
                )

                # Move to review folder instead of processing through LLM
                review_dir = PATHS["semantics_dir"] / "ocr_failed"
                review_dir.mkdir(exist_ok=True)
                artifact.rename(review_dir / artifact.name)

                # Update profile to mark as failed
                profile_data["stages"]["llm_field_extraction"] = {
                    "status": "failed",
                    "reason": "OCR_and_visual_processing_failed",
                }
                continue  # Skip LLM processing

            logger.info(f"Extracting structured fields for: {artifact.name}")
            logger.debug(
                f"OCR text length: {len(ocr_text)} chars, Visual desc length: {len(visual_description)} chars"
            )

            # Extract structured fields using LLM
            extraction_result = field_extractor.extract_fields(
                ocr_text=ocr_text,
                visual_description=visual_description,
                metadata=metadata,
                uuid=artifact_id,
            )

            # Update profile with LLM-extracted fields and stage completion
            profile_data["llm_extraction"] = extraction_result
            profile_data["stages"]["llm_field_extraction"] = {
                "status": "completed" if extraction_result["success"] else "failed",
                "timestamp": datetime.now().isoformat(),
                "model_used": field_extractor.model,
                "fields_extracted": len(extraction_result.get("extracted_fields", {})),
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(PATHS["llm_processed_dir"] / artifact.name)

            # Update session tracking
            if extraction_result["success"]:
                update_stage_counts("semantics", "llm_processed", session_data)
            else:
                update_stage_counts("semantics", "failed", session_data)

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            extraction_status = (
                "successful" if extraction_result["success"] else "failed"
            )
            log_detailed_progress(
                artifact.name, file_size, f"llm_extraction_{extraction_status}"
            )

            # Update session JSON after each file
            update_session_json()

            # Log extraction results
            if extraction_result["success"]:
                extracted_fields = extraction_result["extracted_fields"]
                non_unknown_fields = sum(
                    1 for v in extracted_fields.values() if v != "UNKNOWN"
                )
                total_fields = len(extracted_fields)
                logger.info(
                    f"LLM extraction successful for {artifact.name}: {non_unknown_fields}/{total_fields} fields extracted"
                )

                # Log some key extracted fields for verification
                key_fields = ["title", "document_type", "issuer_name", "date_of_issue"]
                extracted_sample = {
                    k: extracted_fields.get(k, "UNKNOWN") for k in key_fields
                }
                logger.debug(f"Key fields extracted: {extracted_sample}")
            else:
                logger.warning(
                    f"LLM extraction failed for {artifact.name}: {extraction_result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Failed to process LLM extraction for {artifact.name}: {e}")

            # Create failed subdirectory if needed and move file there
            failed_dir = PATHS["semantics_dir"] / "failed"
            failed_dir.mkdir(exist_ok=True)

            try:
                artifact.rename(failed_dir / artifact.name)
                update_stage_counts("semantics", "failed", session_data)
                logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
            except Exception as move_error:
                logger.error(
                    f"Failed to move {artifact.name} to failed directory: {move_error}"
                )

            session_data["errors"].append(
                {
                    "file": artifact.name,
                    "stage": "llm_field_extraction",
                    "error": str(e),
                }
            )

    # Log extractor statistics
    extractor_stats = field_extractor.get_stats()
    logger.info(
        f"LLM field extraction complete - {session_data['stage_counts'].get('llm_processed', 0)} files processed successfully"
    )
    logger.info(
        f"LLM model used: {extractor_stats['model']} on {extractor_stats['host']}"
    )
    logger.info(f"Total LLM API calls made: {extractor_stats['total_processed']}")

# =====================================================================
# STAGE 6: COMPLETION AND FINAL PROCESSING
# =====================================================================


# Get all artifacts from LLM processed directory, sorted by size
llm_processed_artifacts: List[Path] = [
    item for item in PATHS["llm_processed_dir"].iterdir()
]

if len(llm_processed_artifacts) == 0:

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 6: COMPLETION AND FINAL PROCESSING")
    logger.info("=" * 80)
    logger.info("Moving fully processed artifacts to completion directory...")

    llm_processed_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(f"Found {len(llm_processed_artifacts)} artifacts ready for completion")

    for artifact in tqdm(
        llm_processed_artifacts, desc="Completing processing", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Mark as completed in profile
            profile_data["stages"]["completed"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "total_stages_completed": len(
                    [
                        s
                        for s in profile_data["stages"].values()
                        if s.get("status") == "completed"
                    ]
                ),
            }

            # Add final processing summary
            profile_data["processing_summary"] = {
                "total_file_size_mb": profile_data.get("file_size", 0) / (1024 * 1024),
                "stages_completed": list(profile_data["stages"].keys()),
                "has_ocr_text": bool(
                    profile_data.get("semantics", {}).get("all_text", "").strip()
                ),
                "has_visual_description": bool(
                    profile_data.get("semantics", {}).get("all_imagery", "").strip()
                ),
                "llm_extraction_success": profile_data.get("llm_extraction", {}).get(
                    "success", False
                ),
                "fields_with_data": (
                    len(
                        [
                            v
                            for v in profile_data.get("llm_extraction", {})
                            .get("extracted_fields", {})
                            .values()
                            if v != "UNKNOWN"
                        ]
                    )
                    if profile_data.get("llm_extraction", {}).get("success")
                    else 0
                ),
            }

            # Save final profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to completed directory
            final_artifact = artifact.rename(PATHS["completed_dir"] / artifact.name)

            # Update session tracking
            update_stage_counts("llm_processed", "completed", session_data)

            # Log detailed progress
            file_size = final_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "completed")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Processing completed for: {artifact.name}")

        except Exception as e:
            logger.error(f"Failed to complete processing for {artifact.name}: {e}")
            session_data["errors"].append(
                {"file": artifact.name, "stage": "completion", "error": str(e)}
            )

# =====================================================================
# STAGE 7: DATABASE FORMATION
# =====================================================================

profiles: List[Path] = [item for item in PATHS["profiles_dir"].iterdir()]

if len(profiles) == 0:

    logger.info(f"Found {len(profiles)} artifacts ready for completion")

    for artifact in tqdm(profiles, desc="Forming Database", unit="profile"):
        logger.info("TO BE IMPLEMENTED - FORMING DATABASES")


# =====================================================================
# FINAL SESSION SUMMARY AND COMPLETION
# =====================================================================

# Mark session as completed
session_data["status"] = "completed"
session_data["end_time"] = datetime.now().isoformat()
update_session_json()

# Calculate final statistics
total_elapsed = (datetime.now() - session_start_time).total_seconds()
successful_files = session_data["stage_counts"]["completed"]
failed_files = session_data["stage_counts"]["failed"]

logger.info("\n" + "=" * 80)
logger.info("PAPERTRAIL SESSION PROCESSING COMPLETED!")
logger.info("=" * 80)
logger.info(f"Session ID: {session_timestamp}")
logger.info(
    f"Total Runtime: {total_elapsed//3600:02.0f}:{(total_elapsed%3600)//60:02.0f}:{total_elapsed%60:02.0f}"
)
logger.info(f"Successfully Processed: {successful_files} files")
logger.info(f"Failed Processing: {failed_files} files")
logger.info(f"Zero-byte Files: {skipped_zero_byte} files")
logger.info(f"Duplicates Skipped: {skipped_duplicates} files")
logger.info(f"Unsupported files: {unsupported_artifacts} files")
logger.info(
    f"Average Speed: {session_data['performance']['files_per_minute']:.1f} files/min"
)

# Final file type summary
final_file_summary = ", ".join(
    [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
)
logger.info(f"File Types Processed: {final_file_summary}")

# Directory locations summary
logger.info("\nOutput Locations:")
logger.info(f"  Completed Files: {PATHS['completed_dir']}")
logger.info(f"  Profile Data: {PATHS['profiles_dir']}")
logger.info(f"  Session Logs: {PATHS['logs_dir']}")
logger.info(f"  Files for Review: {PATHS['review_dir']}")

# Error summary if any
if session_data["errors"]:
    logger.info(f"\nErrors Encountered ({len(session_data['errors'])}):")
    for error in session_data["errors"]:
        logger.info(f"  {error['file']} ({error['stage']}): {error['error']}")

logger.info("=" * 80)
