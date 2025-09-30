"""
Metadata Pipeline Module

A robust file processing pipeline that handles comprehensive metadata extraction
from various file types including images, file_paths, PDFs, office files, audio,
video, archives, and more.

This module provides functionality to extract metadata by:
- Detecting file types and routing to appropriate extractors
- Extracting filesystem metadata (size, dates, permissions)
- Processing image EXIF, IPTC, XMP data, color profiles, and technical details
- Extracting file_path properties, structure information, and content analysis
- Processing audio/video metadata including codecs, duration, and technical specs
- Analyzing archive contents and compression details
- Extracting code and text file metrics and language detection
- Handling extraction failures gracefully with fallback methods
- Maintaining detailed operation logs and error tracking
- Updating artifact profiles with extracted metadata

Author: Ashiq Gazi
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any , Dict , List

import pypdf
from torch.onnx.errors import UnsupportedOperatorError
from tqdm import tqdm

from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	AUDIO_TYPES ,
	DOCUMENT_TYPES ,
	IMAGE_TYPES ,
	MAX_PDF_SIZE_BEFORE_SUBSETTING ,
	PROFILE_PREFIX ,
	TEMP_DIR ,
	VIDEO_TYPES ,
)
from utilities import (
	AudioProcessor ,
	LanguageProcessor ,
	VisualProcessor ,
	compile_doc_subset ,
	compile_video_snapshot_subset ,
)


def extract_semantics(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:
    """
    Extract metadata and text from all files in a directory and update their profiles.

    This method performs comprehensive metadata extraction by validating dependencies,
    discovering artifact files, loading existing profiles, extracting metadata and text
    using Apache Tika, updating profiles with extraction results, and moving processed
    files to appropriate directories based on success or failure.

    The extraction process includes:
    - Java runtime validation (version 11+)
    - Apache Tika JAR validation
    - Filesystem metadata extraction
    - Technical metadata extraction (EXIF, IPTC, file_path properties, etc.)
    - Full text content extraction
    - Profile updates with extraction timestamps and results
    - Graceful error handling with detailed logging

    Args:
            logger: Logger instance for tracking operations and errors
            source_dir: Directory containing artifact files to process. Files must follow
                    the ARTIFACT-{uuid}.ext naming convention
            failure_dir: Directory to move files that fail processing. Files are moved here
                    when profile loading, metadata extraction, or text extraction fails
            success_dir: Directory to move files after successful metadata extraction and
                    profile updates

    Returns:
            Tuple containing (metadata, text) from the last successfully processed file,
            or None if no files were processed successfully. Both elements may be None
            if no extraction occurred.

    Raises:
            FileNotFoundError: If Tika JAR is not found at the configured path or if
                    source directory does not exist
            RuntimeError: If Java is not installed, not in PATH, or version is below
                    the minimum required version (Java 11+)
            EnvironmentError: If Java version cannot be determined or other environment
                    issues prevent execution

    Note:
            - Files are processed in order of size (smallest first) for faster feedback
            - Requires corresponding PROFILE-{uuid}.json files in ARTIFACT_PROFILES_DIR
            - Tika commands timeout after 120 seconds to prevent hanging on large files
            - All profile updates include timestamps for audit trail
            - Extraction failures are logged with full exception details (exc_info=True)
    """

    # Log metadata extraction stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("SEMANTICS EXTRACTION AND ARTIFACT DESCRIPTION STAGE")
    logger.info("=" * 80)

    # Discover all artifact files in the source directory
    # Filter for files that start with the ARTIFACT_PREFIX to ensure we only process valid artifacts
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]

    # Handle empty directory case - exit early if no artifacts found
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

    # Initialize processor instances for different artifact types
    # VisualProcessor handles images and file_paths with OCR capabilities
    visual_processor: VisualProcessor = VisualProcessor(logger=logger)
    # LanguageProcessor extracts structured fields from text descriptions
    language_processor: LanguageProcessor = LanguageProcessor(logger=logger)
    # AudioProcessor handles audio and video transcription
    audio_processor: AudioProcessor = AudioProcessor(logger=logger)

    # Sort files by size for consistent processing order (smaller files first)
    # This provides faster initial feedback and helps identify issues early
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Process each artifact file with progress tracking
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Extracting semantical data",
        unit="artifacts",
    ):
        try:
            logger.info(f"Processing artifact: {artifact.name}")

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            # We strip the prefix and hyphen to get just the UUID portion
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]

            # Construct the path to the corresponding profile file
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            # Each artifact must have a corresponding profile for tracking
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data from JSON file
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                # Handle corrupted JSON files specifically
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                # Catch any other file reading errors
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Extract metadata using Apache Tika
            logger.debug(f"Extracting metadata for: {artifact.name}")

            # Route to appropriate processor based on file type
            # Documents and images use visual processing with OCR
            if artifact.suffix in DOCUMENT_TYPES or artifact.suffix in IMAGE_TYPES:
                # Extract both OCR text and visual descriptions from the file_path
                if (
                    len(pypdf.PdfReader(str(artifact)).pages)
                ) > MAX_PDF_SIZE_BEFORE_SUBSETTING:
                    subset: Path = compile_doc_subset(
                        input_pdf=artifact,
                        set_size=MAX_PDF_SIZE_BEFORE_SUBSETTING,
                        temp_dir=TEMP_DIR,
                    )
                else:
                    subset: Path = artifact
                ocr_text, visual_description = visual_processor.extract_semantics(
                    file_path=subset
                )

                # Process the extracted text and visual data to create structured descriptors
                artifact_descriptors = language_processor.extract_fields(
                    text_description=ocr_text,
                    visual_description=visual_description,
                )
            # Video and audio files use audio processing with transcription
            elif (artifact.suffix in VIDEO_TYPES) and (
                audio_processor.has_audio(artifact)
            ):
                ocr_text, visual_description = visual_processor.extract_semantics(
                    file_path=compile_video_snapshot_subset(
                        video_path=artifact,
                        set_size=MAX_PDF_SIZE_BEFORE_SUBSETTING,
                        temp_dir=TEMP_DIR,
                    )
                )

                # Extract transcription from audio/video and visual descriptions from video frames
                transcription = audio_processor.transcribe_audio(file_path=artifact)

                # Process the transcription and visual data to create structured descriptors
                artifact_descriptors = language_processor.extract_fields(
                    text_description=transcription,
                    visual_description=visual_description,
                )
            elif artifact.suffix in AUDIO_TYPES:
                if not audio_processor.has_audio(artifact):
                    raise SystemError(
                        "Audio type artifact does not have audio. Malformed audio type."
                    )

                # Extract transcription from audio/video and visual descriptions from video frames
                transcription = audio_processor.transcribe_audio(file_path=artifact)

                # Process the transcription and visual data to create structured descriptors
                artifact_descriptors = language_processor.extract_fields(
                    text_description=transcription,
                    visual_description="THIS FILE IS AN AUDIO FILE. NO VISUAL DESCRIPTION EXISTS.",
                )
            else:
                # Unsupported file types should fail explicitly
                raise UnsupportedOperatorError(
                    "Unsupported file type for semantic extraction"
                )

            # Update profile with extracted metadata
            # Store the structured descriptors in the profile for later retrieval
            artifact_profile_data["extracted_semantics"] = artifact_descriptors

            # Store extracted text if available
            # This preserves the raw extracted text along with metadata about the extraction
            if artifact_descriptors:
                # Track extraction success with character count and timestamp
                artifact_profile_data["text_extraction"] = {
                    "success": True,
                    "character_count": len(artifact_descriptors),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Mark extraction as failed if no descriptors were generated
                artifact_profile_data["text_extraction"] = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            # Update stage tracking
            # Initialize stage progression data if it doesn't exist in the profile
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            # Mark this processing stage as completed with timestamp
            artifact_profile_data["stage_progression_data"]["semantics_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile back to disk
            # This persists all the extracted metadata and processing status
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    # Use indent for readable JSON and ensure_ascii=False for unicode support
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory after successful processing
            success_location: Path = success_dir / artifact.name

            # Handle naming conflicts in success directory
            # If a file with the same name exists, append a counter to make it unique
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                # Keep incrementing counter until we find a unique filename
                while success_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / new_name
                    counter += 1

            # Perform the actual file move operation
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

        except Exception as e:
            # Catch any errors during processing to prevent pipeline failure
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory for later inspection
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            # Same conflict resolution strategy as success directory
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                # Keep incrementing counter until we find a unique filename
                while failure_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / new_name
                    counter += 1

            # Move the failed artifact for later review and debugging
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            # Continue processing remaining artifacts despite this failure
            continue

    # Log completion of the entire extraction stage
    logger.info("Semantics extraction stage completed")
    return None
