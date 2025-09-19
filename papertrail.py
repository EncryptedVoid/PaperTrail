#!C:/Users/UserX/AppData/Local/Programs/Python/Python313/python.exe

from tqdm import tqdm
import logging
import json
import subprocess
import requests
import time
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from collections import defaultdict
from checksum_utils import HashAlgorithm, generate_checksum
from uuid_utils import generate_uuid4plus
from encryption_utils import (
    generate_passphrase,
    generate_password,
    encrypt_file,
    decrypt_file,
)
from metadata_processor import MetadataExtractor
from visual_processor import (
    VisualProcessor,
    ProcessingMode,
    VisionModelSpec,
    HardwareConstraints,
    ProcessingStats,
)
from language_processor import LanguageProcessor
from database_processor import create_final_spreadsheet


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
    "stl",
    "gcode"
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
    "process_completed_dir": BASE_DIR / "processed_artifacts",
    "encrypted_dir": BASE_DIR / "encrypted_artifacts",
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

if len(unprocessed_artifacts) > 0:

    logger.info("=" * 80)
    logger.info(
        "DUPLICATE DETECTION, UNSUPPORTED FILES, AND CHECKSUM VERIFICATION STAGE"
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

    logger.info("=" * 80)
    logger.info("UUID RENAMING AND PROFILE CREATION STAGE")
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
                "checksum": generate_checksum(artifact, algorithm=CHECKSUM_ALGORITHM),
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

if len(renamed_artifacts) > 0:
    logger.info("=" * 80)
    logger.info("AUTOMATED TECHNICAL METADATA EXTRACTION STAGE")
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

if len(metadata_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("ARTIFACT VISUAL PROCESSING STAGE")
    logger.info("=" * 80)
    logger.info("Extracting semantic data using enhanced visual processor...")

    metadata_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(metadata_artifacts)} artifacts ready for semantic extraction"
    )

    # Enhanced visual processor configuration
    try:
        # Configure visual processing parameters (can be made configurable via config file)
        visual_config = {
            "max_gpu_vram_gb": 12.0,  # Use most of your 9.6GB GPU
            "max_ram_gb": 48.0,  # Limit to 48GB as requested (was auto-detecting 44.8GB)
            "force_cpu": False,  # Make sure GPU is used
            "processing_mode": ProcessingMode.FAST,  # Changed from HIGH_QUALITY
            "refresh_interval": 5,  # More frequent refresh due to memory pressure
            "memory_threshold": 70.0,  # Lower threshold for cleanup
            "auto_model_selection": True,
            "preferred_model": "Qwen/Qwen2-VL-2B-Instruct",  # Force smaller, faster model
        }

        # logger.info("=== PERFORMANCE DEBUG INFO ===")
        # logger.info(f"GPU Available: {torch.cuda.is_available()}")
        # logger.info(f"GPU Device Count: {torch.cuda.device_count()}")
        # if torch.cuda.is_available():
        #     logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        #     logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        # logger.info(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        # logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        # logger.info("=" * 30)

        # Initialize enhanced visual processor
        processor = VisualProcessor(
            logger=logger,
            max_gpu_vram_gb=visual_config["max_gpu_vram_gb"],
            max_ram_gb=visual_config["max_ram_gb"],
            force_cpu=visual_config["force_cpu"],
            processing_mode=visual_config["processing_mode"],
            refresh_interval=visual_config["refresh_interval"],
            memory_threshold=visual_config["memory_threshold"],
            auto_model_selection=visual_config["auto_model_selection"],
            preferred_model=visual_config["preferred_model"],
        )

        # Log initialization details
        logger.info("Enhanced Visual Processor initialized successfully")
        initial_stats = processor.get_processing_stats()
        logger.info(f"Selected model: {initial_stats['current_model']['name']}")
        logger.info(f"Processing mode: {initial_stats['processing_mode']}")
        logger.info(f"Device: {initial_stats['device']}")

        # Log hardware constraints
        hw_constraints = initial_stats["hardware_constraints"]
        logger.info(
            f"Hardware constraints: GPU VRAM={hw_constraints['max_gpu_vram_gb']:.1f}GB, "
            f"RAM={hw_constraints['max_ram_gb']:.1f}GB, Force CPU={hw_constraints['force_cpu']}"
        )

        # Show available models
        available_models = processor.get_available_models()
        suitable_models = [m for m in available_models if m["fits_constraints"]]
        logger.info(
            f"Available models for current hardware: {[m['name'] for m in suitable_models]}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize enhanced visual processor: {e}")
        logger.error("Falling back to basic configuration...")

        # Fallback to basic configuration if enhanced setup fails
        processor = VisualProcessor(
            logger=logger,
            auto_model_selection=False,
            force_cpu=True,  # Use CPU as safest fallback
        )
        logger.warning("Using fallback visual processor configuration")

    # Track processing statistics
    processing_stats = {
        "total_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "model_refreshes": 0,
        "model_switches": 0,
        "start_time": datetime.now(),
        "quality_scores": [],
    }

    # Create semantics directory
    PATHS["semantics_dir"] = BASE_DIR / "semantics"
    PATHS["semantics_dir"].mkdir(parents=True, exist_ok=True)

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

            # Check if file is too corrupted to process
            file_size = artifact.stat().st_size
            if file_size < 1024:  # Less than 1KB might be corrupted
                logger.warning(
                    f"File {artifact.name} is very small ({file_size} bytes), may be corrupted"
                )

            # Track pre-processing stats to detect model refreshes/switches
            pre_processing_stats = processor.get_processing_stats()

            # Extract semantic data using enhanced visual processor
            logger.info(f"Extracting semantics for: {artifact.name}")
            logger.debug(f"File size: {file_size / (1024*1024):.2f}MB")

            try:
                extracted_semantics: Dict[str, str] = (
                    processor.extract_article_semantics(document=artifact)
                )

                # Track post-processing stats
                post_processing_stats = processor.get_processing_stats()
                processing_stats["total_processed"] += 1
                processing_stats["successful_extractions"] += 1

                # Check if model was refreshed or switched during processing
                if (
                    pre_processing_stats["memory_refreshes"]
                    < post_processing_stats["memory_refreshes"]
                ):
                    processing_stats["model_refreshes"] += 1
                    logger.info(
                        f"Model was refreshed during processing of {artifact.name}"
                    )

                if (
                    pre_processing_stats["model_switches"]
                    < post_processing_stats["model_switches"]
                ):
                    processing_stats["model_switches"] += 1
                    logger.info(
                        f"Model was switched during processing of {artifact.name}"
                    )

                # Calculate quality score for this extraction
                text_length = len(extracted_semantics.get("all_text", ""))
                imagery_length = len(extracted_semantics.get("all_imagery", ""))

                # Simple quality heuristic
                has_meaningful_text = (
                    text_length > 50
                    and "No text found" not in extracted_semantics.get("all_text", "")
                )
                has_meaningful_imagery = (
                    imagery_length > 100
                    and "No visual content"
                    not in extracted_semantics.get("all_imagery", "")
                )

                quality_score = 0
                if has_meaningful_text:
                    quality_score += 50
                if has_meaningful_imagery:
                    quality_score += 50

                processing_stats["quality_scores"].append(quality_score)

                # Enhanced stage completion data
                stage_completion_data = {
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "model_used": post_processing_stats["current_model"]["name"],
                    "device": post_processing_stats["device"],
                    "processing_mode": post_processing_stats["processing_mode"],
                    "text_length": text_length,
                    "imagery_length": imagery_length,
                    "quality_score": quality_score,
                    "model_refreshed": pre_processing_stats["memory_refreshes"]
                    < post_processing_stats["memory_refreshes"],
                }

                # Update profile with semantics and enhanced stage completion
                profile_data["semantics"] = extracted_semantics
                profile_data["stages"]["semantic_extraction"] = stage_completion_data

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

                # Enhanced logging
                logger.info(f"Semantics extracted successfully for {artifact.name}")
                logger.debug(
                    f"Text extracted: {text_length} chars, Imagery: {imagery_length} chars, Quality: {quality_score}%"
                )

                # Log memory usage occasionally
                if processing_stats["total_processed"] % 5 == 0:
                    current_stats = processor.get_processing_stats()
                    memory_info = current_stats.get("memory_usage", {})
                    if "gpu_memory_percent" in memory_info:
                        logger.debug(
                            f"GPU memory: {memory_info['gpu_memory_percent']:.1f}%, "
                            f"RAM: {memory_info['system_ram_percent']:.1f}%"
                        )

            except Exception as extraction_error:
                logger.error(
                    f"Semantic extraction failed for {artifact.name}: {extraction_error}"
                )
                processing_stats["failed_extractions"] += 1

                # Check if we should try a different model for persistent failures
                if (
                    processing_stats["failed_extractions"] > 3
                    and processing_stats["failed_extractions"] % 5 == 0
                ):

                    available_models = processor.get_available_models()
                    current_model = processor.current_model_spec.model_id

                    # Try switching to a different model
                    other_models = [
                        m
                        for m in available_models
                        if m["fits_constraints"] and m["model_id"] != current_model
                    ]
                    if other_models:
                        new_model = other_models[0]["model_id"]
                        logger.info(
                            f"High failure rate detected. Attempting to switch from {current_model} to {new_model}"
                        )

                        if processor.switch_model(new_model):
                            processing_stats["model_switches"] += 1
                            logger.info(f"Successfully switched to model: {new_model}")
                        else:
                            logger.warning(f"Failed to switch to model: {new_model}")

                # Create failed entry in profile
                profile_data["stages"]["semantic_extraction"] = {
                    "status": "failed",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(extraction_error),
                }

                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=2)

                # Move to failed directory
                failed_dir = PATHS["metadata_dir"] / "failed"
                failed_dir.mkdir(exist_ok=True)
                artifact.rename(failed_dir / artifact.name)
                update_stage_counts("metadata", "failed", session_data)

                session_data["errors"].append(
                    {
                        "file": artifact.name,
                        "stage": "semantic_extraction",
                        "error": str(extraction_error),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                continue

            # Progress reporting every 20 files
            if processing_stats["total_processed"] % 20 == 0:
                current_stats = processor.get_processing_stats()
                elapsed_time = datetime.now() - processing_stats["start_time"]
                avg_quality = (
                    sum(processing_stats["quality_scores"])
                    / len(processing_stats["quality_scores"])
                    if processing_stats["quality_scores"]
                    else 0
                )

                logger.info(f"=== Processing Progress Update ===")
                logger.info(f"Files processed: {processing_stats['total_processed']}")
                logger.info(
                    f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
                )
                logger.info(f"Average quality score: {avg_quality:.1f}%")
                logger.info(f"Model refreshes: {processing_stats['model_refreshes']}")
                logger.info(f"Model switches: {processing_stats['model_switches']}")
                logger.info(f"Current model: {current_stats['current_model']['name']}")
                logger.info(f"Elapsed time: {elapsed_time}")
                logger.info(
                    f"Avg time per document: {current_stats.get('avg_processing_time_per_doc', 0):.2f}s"
                )
                logger.info("=" * 35)

        except Exception as e:
            logger.error(f"Failed to process {artifact.name}: {e}")
            processing_stats["failed_extractions"] += 1

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
                {
                    "file": artifact.name,
                    "stage": "semantic_extraction",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Enhanced final statistics and optimization report
    final_stats = processor.get_processing_stats()
    processing_time = datetime.now() - processing_stats["start_time"]
    avg_quality = (
        sum(processing_stats["quality_scores"])
        / len(processing_stats["quality_scores"])
        if processing_stats["quality_scores"]
        else 0
    )

    logger.info("=" * 80)
    logger.info("SEMANTIC EXTRACTION COMPLETE - FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {processing_stats['total_processed']}")
    logger.info(f"Successful extractions: {processing_stats['successful_extractions']}")
    logger.info(f"Failed extractions: {processing_stats['failed_extractions']}")
    logger.info(
        f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
    )
    logger.info(f"Average quality score: {avg_quality:.1f}%")
    logger.info(f"Total processing time: {processing_time}")
    logger.info(
        f"Average time per file: {processing_time / max(processing_stats['total_processed'], 1)}"
    )
    logger.info("=== VISUAL PROCESSOR STATISTICS: ===")
    logger.info(f"Final model used: {final_stats['current_model']['name']}")
    logger.info(f"Device: {final_stats['device']}")
    logger.info(f"Processing mode: {final_stats['processing_mode']}")
    logger.info(f"Pages processed: {final_stats['pages_processed']}")
    logger.info(
        f"Text extraction success rate: {final_stats.get('text_extraction_success_rate', 0):.1%}"
    )
    logger.info(
        f"Description success rate: {final_stats.get('description_success_rate', 0):.1%}"
    )
    logger.info(f"Model refreshes performed: {processing_stats['model_refreshes']}")
    logger.info(f"Model switches performed: {processing_stats['model_switches']}")

    # Memory usage summary
    memory_usage = final_stats.get("memory_usage", {})
    if "gpu_memory_percent" in memory_usage:
        logger.info(
            f"Final GPU memory usage: {memory_usage['gpu_memory_percent']:.1f}%"
        )
    logger.info(f"Final RAM usage: {memory_usage.get('system_ram_percent', 0):.1f}%")

    # Get and log optimization suggestions
    # try:
    #     optimization_report = processor.optimize_performance()
    #     suggestions = optimization_report.get("optimization_suggestions", [])

    #     if suggestions:
    #         logger.info("")
    #         logger.info("PERFORMANCE OPTIMIZATION SUGGESTIONS:")
    #         for i, suggestion in enumerate(suggestions, 1):
    #             logger.info(f"{i}. {suggestion}")
    #     else:
    #         logger.info(
    #             "No performance optimization suggestions - system is running optimally"
    #         )

    # except Exception as e:
    #     logger.warning(f"Could not generate optimization report: {e}")

    # Save processing statistics to session data
    session_data["semantic_extraction_stats"] = {
        "processing_stats": processing_stats,
        "final_processor_stats": final_stats,
        "processing_time_seconds": processing_time.total_seconds(),
        "average_quality_score": avg_quality,
    }

# =====================================================================
# STAGE 5: LLM FIELD EXTRACTION (SEMANTIC METADATA)
# =====================================================================


# Get all artifacts from semantics directory, sorted by size
semantic_artifacts: List[Path] = [item for item in PATHS["semantics_dir"].iterdir()]

if len(semantic_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("LLM-BASED SEMANTIC METADATA EXTRACTION STAGE")
    logger.info("=" * 80)
    logger.info("Extracting structured document fields using LLM...")

    semantic_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(semantic_artifacts)} artifacts ready for LLM field extraction"
    )

    field_extractor = LanguageProcessor(
        logger=logger,
        max_ram_gb=48.0,  # Auto-detect (or set specific limit like 16.0)
        max_gpu_vram_gb=12.0,  # Auto-detect (or set specific limit like 8.0)
        max_cpu_cores=None,  # Auto-detect (or set specific limit like 8)
        auto_model_selection=False,  # Automatically select best model for hardware
    )

    # Track processing statistics
    processing_stats = {
        "total_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "context_refreshes": 0,
        "model_switches": 0,
        "start_time": datetime.now(),
    }

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
                processing_stats["failed_extractions"] += 1
                continue  # Skip LLM processing

            logger.info(f"Extracting structured fields for: {artifact.name}")
            logger.debug(
                f"OCR text length: {len(ocr_text)} chars, Visual desc length: {len(visual_description)} chars"
            )

            # Extract structured fields using enhanced LLM
            extraction_result = field_extractor.extract_fields(
                ocr_text=ocr_text,
                visual_description=visual_description,
                metadata=metadata,
                uuid=artifact_id,
            )

            processing_stats["total_processed"] += 1

            # Update profile with LLM-extracted fields and stage completion
            profile_data["llm_extraction"] = extraction_result

            # Enhanced stage completion tracking
            stage_completion_data = {
                "status": "completed" if extraction_result["success"] else "failed",
                "timestamp": datetime.now().isoformat(),
                "model_used": extraction_result.get("model_used", "unknown"),
                "extraction_mode": extraction_result.get("extraction_mode", "unknown"),
                "fields_extracted": len(extraction_result.get("extracted_fields", {})),
                "processing_time_ms": extraction_result.get("processing_time_ms", 0),
            }

            if not extraction_result["success"]:
                stage_completion_data["error"] = extraction_result.get(
                    "error", "Unknown error"
                )

            profile_data["stages"]["llm_field_extraction"] = stage_completion_data

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(
                PATHS["process_completed_dir"] / artifact.name
            )

            # Update session tracking
            if extraction_result["success"]:
                update_stage_counts("semantics", "llm_processed", session_data)
                processing_stats["successful_extractions"] += 1
            else:
                update_stage_counts("semantics", "failed", session_data)
                processing_stats["failed_extractions"] += 1

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

            # Enhanced logging of extraction results
            if extraction_result["success"]:
                extracted_fields = extraction_result["extracted_fields"]
                non_unknown_fields = sum(
                    1 for v in extracted_fields.values() if v != "UNKNOWN"
                )
                total_fields = len(extracted_fields)

                logger.info(
                    f"LLM extraction successful for {artifact.name}: {non_unknown_fields}/{total_fields} fields extracted "
                    f"using {extraction_result.get('extraction_mode', 'unknown')} mode"
                )

                # Log some key extracted fields for verification
                key_fields = ["title", "document_type", "issuer_name", "date_of_issue"]
                extracted_sample = {
                    k: extracted_fields.get(k, "UNKNOWN") for k in key_fields
                }
                logger.debug(f"Key fields extracted: {extracted_sample}")

                # Log field extraction quality score
                quality_score = (non_unknown_fields / total_fields) * 100
                logger.debug(f"Field extraction quality: {quality_score:.1f}%")

            else:
                logger.warning(
                    f"LLM extraction failed for {artifact.name}: {extraction_result.get('error', 'Unknown error')}"
                )

            # Log progress every 50 files
            if processing_stats["total_processed"] % 50 == 0:
                current_stats = field_extractor.get_stats()
                elapsed_time = datetime.now() - processing_stats["start_time"]

                logger.info(f"=== Processing Progress Update ===")
                logger.info(f"Files processed: {processing_stats['total_processed']}")
                logger.info(
                    f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
                )
                logger.info(
                    f"Context refreshes: {processing_stats['context_refreshes']}"
                )
                logger.info(f"Model switches: {processing_stats['model_switches']}")
                logger.info(f"Current model: {current_stats['model']}")
                logger.info(f"Elapsed time: {elapsed_time}")
                logger.info(
                    f"Prompts since last refresh: {current_stats['prompts_since_refresh']}"
                )
                logger.info("=" * 35)

        except Exception as e:
            logger.error(f"Failed to process LLM extraction for {artifact.name}: {e}")
            processing_stats["failed_extractions"] += 1

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
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Enhanced final statistics logging
    final_stats = field_extractor.get_stats()
    processing_time = datetime.now() - processing_stats["start_time"]

    logger.info("=" * 80)
    logger.info("LLM FIELD EXTRACTION COMPLETE - FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {processing_stats['total_processed']}")
    logger.info(f"Successful extractions: {processing_stats['successful_extractions']}")
    logger.info(f"Failed extractions: {processing_stats['failed_extractions']}")
    logger.info(
        f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
    )
    logger.info(f"Total processing time: {processing_time}")
    logger.info(
        f"Average time per file: {processing_time / max(processing_stats['total_processed'], 1)}"
    )
    # logger.info("")
    logger.info("LLM MODEL STATISTICS:")
    logger.info(f"Final model used: {final_stats['model']}")
    logger.info(f"Total LLM API calls made: {final_stats['total_processed']}")
    logger.info(f"Context refreshes performed: {processing_stats['context_refreshes']}")
    logger.info(f"Model switches performed: {processing_stats['model_switches']}")
    logger.info(
        f"Hardware utilized: RAM={final_stats['hardware_constraints']['max_ram_gb']:.1f}GB, "
        f"GPU={final_stats['hardware_constraints']['max_gpu_vram_gb']:.1f}GB, "
        f"CPU={final_stats['hardware_constraints']['max_cpu_cores']} cores"
    )
    logger.info(f"Extraction mode: {final_stats['extraction_mode']}")

    # Save final processing statistics to session data
    session_data["llm_extraction_stats"] = {
        "processing_stats": processing_stats,
        "final_extractor_stats": final_stats,
        "processing_time_seconds": processing_time.total_seconds(),
    }

    logger.info("=" * 80)


# =====================================================================
# STAGE 6: COMPLETION AND FINAL PROCESSING
# =====================================================================

# Get all artifacts from LLM processed directory, sorted by size
llm_processed_artifacts: List[Path] = [
    item for item in PATHS["process_completed_dir"].iterdir()
]

if len(llm_processed_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("FINAL PROCESSING STAGE")
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
            final_artifact = artifact.rename(
                PATHS["process_completed_dir"] / artifact.name
            )

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

if len(profiles) > 0:  # Fixed condition - process when we HAVE profiles

    logger.info("=" * 80)
    logger.info("DATABASE FORMATION STAGE")
    logger.info("=" * 80)
    logger.info("Creating final spreadsheet databases from processed profiles...")

    logger.info(f"Found {len(profiles)} profiles ready for database formation")

    try:
        # Create the final spreadsheet using the database processor
        output_files = create_final_spreadsheet(
            profiles_dir=PATHS["profiles_dir"],
            output_dir=PATHS["base_dir"],
            logger=logger,
        )

        if output_files:
            logger.info("Database formation completed successfully!")

            # Update session data with database creation info
            session_data["database_files"] = {}

            for file_type, file_path in output_files.items():
                logger.info(f"  {file_type.upper()}: {file_path}")
                session_data["database_files"][file_type] = str(file_path)

                # Update stage counts for completed database formation
                session_data["stage_counts"]["database_formed"] = len(profiles)

            # Log database statistics
            logger.info(f"Database contains {len(profiles)} document profiles")

            # Calculate some summary statistics
            completed_profiles = 0
            successful_llm_extractions = 0

            for profile in profiles:
                try:
                    with open(profile, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)

                    # Count completed profiles
                    if "completed" in profile_data.get("stages", {}):
                        completed_profiles += 1

                    # Count successful LLM extractions
                    if profile_data.get("llm_extraction", {}).get("success", False):
                        successful_llm_extractions += 1

                except Exception as e:
                    logger.debug(
                        f"Failed to read profile {profile.name} for statistics: {e}"
                    )

            logger.info(f"  - {completed_profiles} fully completed documents")
            logger.info(
                f"  - {successful_llm_extractions} successful LLM field extractions"
            )
            logger.info(
                f"  - {len(profiles) - completed_profiles} documents with partial processing"
            )

            # Add database formation timestamp
            # session_data["stages"]["database_formation"] = {
            #     "status": "completed",
            #     "timestamp": datetime.now().isoformat(),
            #     "files_created": list(output_files.keys()),
            #     "total_profiles_exported": len(profiles),
            # }

        else:
            logger.warning("No database files were created")
            # session_data["stages"]["database_formation"] = {
            #     "status": "failed",
            #     "timestamp": datetime.now().isoformat(),
            #     "reason": "No output files generated",
            # }

    except Exception as e:
        logger.error(f"Database formation failed: {e}")
        # session_data["errors"].append(
        #     {
        #         "file": "database_formation",
        #         "stage": "database_formation",
        #         "error": str(e),
        #         "timestamp": datetime.now().isoformat(),
        #     }
        # )

        # session_data["stages"]["database_formation"] = {
        #     "status": "failed",
        #     "timestamp": datetime.now().isoformat(),
        #     "error": str(e),
        # }

    # Update session JSON with final database information
    update_session_json()

    logger.info("Database formation stage complete")

else:
    logger.warning("No profiles found for database formation")
    # session_data["stages"]["database_formation"] = {
    #     "status": "skipped",
    #     "timestamp": datetime.now().isoformat(),
    #     "reason": "No profiles found",
    # }

# =====================================================================
# STAGE 8: ARTIFACT ENCRYPTION AND PASSWORD PROTECTING
# =====================================================================

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

logger.info("=" * 80)
logger.info("PAPERTRAIL SESSION PROCESSING COMPLETED!")
logger.info("=" * 80)
logger.info(f"Session ID: {session_timestamp}")
logger.info(
    f"Total Runtime: {total_elapsed//3600:02.0f}:{(total_elapsed%3600)//60:02.0f}:{total_elapsed%60:02.0f}"
)
logger.info(f"Successfully Processed: {successful_files} files")
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
logger.info(f"  Completed Files: {PATHS['process_completed_dir']}")
logger.info(f"  Profile Data: {PATHS['profiles_dir']}")
logger.info(f"  Session Logs: {PATHS['logs_dir']}")
logger.info(f"  Files for Review: {PATHS['review_dir']}")

# Error summary if any
if session_data["errors"]:
    logger.info(f"\nErrors Encountered ({len(session_data['errors'])}):")
    for error in session_data["errors"]:
        logger.info(f"  {error['file']} ({error['stage']}): {error['error']}")

logger.info("=" * 80)
