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

import datetime
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import exiftool
import ollama
import pypdf
from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    AUDIO_TYPES,
    CODE_TYPES,
    DOCUMENT_TYPES,
    FIELD_PROMPTS,
    IMAGE_TYPES,
    JAVA_PATH,
    MAX_PDF_SIZE_BEFORE_SUBSETTING,
    METADATA_EXTRACTION_TIMEOUT,
    PREFERRED_LANGUAGE_MODEL,
    PROFILE_PREFIX,
    SYSTEM_PROMPT,
    TEMP_DIR,
    TEXT_TYPES,
    TIKA_APP_JAR_PATH,
    VIDEO_TYPES,
)
from utilities import (
    AudioProcessor,
    VisualProcessor,
    compile_doc_subset,
    compile_video_snapshot_subset,
    ensure_apache_tika,
    ensure_java,
    ensure_ollama,
)
from utilities.dependancy_ensurance import ensure_exiftool


class LanguageExtractionReport(TypedDict):
    """
    Structured type definition for field extraction results.

    This TypedDict ensures consistent return format from the extract_fields method
    and provides clear documentation of expected fields in the response.
    """

    success: bool
    processing_timestamp: str
    model_used: str
    processing_time_seconds: float
    extracted_fields: Optional[Dict[str, str]]  # Only present on success
    error: Optional[str]  # Only present on failure
    error_type: Optional[str]  # Only present on failure


def extract_artifact_data(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:

    ensure_ollama()
    ensure_java()
    ensure_apache_tika()
    ensure_exiftool()

    # Log metadata extraction stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("SEMANTICS EXTRACTION STAGE")
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

            # Build Tika command for metadata extraction
            # -m: Extract metadata only
            # -j: Output as JSON
            metadata_cmd: List[str] = [
                JAVA_PATH,
                "-jar",
                str(TIKA_APP_JAR_PATH),
                "-m",
                "-j",
                str(artifact),
            ]

            # Execute Tika with timeout to prevent hanging on large files
            metadata_process: subprocess.CompletedProcess = subprocess.run(
                metadata_cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Check if Tika execution was successful
            if metadata_process.returncode != 0:
                error_msg: str = (
                    f"Tika metadata extraction failed for {artifact.name}: "
                    f"{metadata_process.stderr}"
                )
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

            # Parse JSON output from Tika
            extracted_metadata: Dict[str, Any] = json.loads(metadata_process.stdout)

            # Check if extraction produced valid results
            if not extracted_metadata:
                error_msg: str = f"No metadata extracted for {artifact.name}"
                logger.warning(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"Successfully extracted metadata for: {artifact.name}")

            # Extract text content from the file
            logger.debug(f"Extracting text content for: {artifact.name}")

            # Determine file category and extract content accordingly
            content = None
            file_extension = artifact.suffix.lstrip(
                "."
            )  # Remove leading dot for comparison

            # 1. DOCUMENTS: Tika → QWEN OCR → Visual Description
            if file_extension in DOCUMENT_TYPES:
                logger.debug(f"Processing as document: {artifact.name}")

                # Try Apache Tika text extraction first
                text_cmd = [
                    JAVA_PATH,
                    "-jar",
                    str(TIKA_APP_JAR_PATH),
                    "--text",
                    str(artifact),
                ]

                text_process = subprocess.run(
                    text_cmd,
                    capture_output=True,
                    text=True,
                    timeout=METADATA_EXTRACTION_TIMEOUT,
                )

                if text_process.returncode == 0:
                    content = text_process.stdout.strip()

                # Fallback to QWEN visual extraction if Tika fails or returns empty
                if not content:
                    logger.warning(
                        f"Apache Tika failed for {artifact.name}, trying QWEN OCR"
                    )

                    # Create subset for large PDFs
                    if (
                        file_extension == "pdf"
                        and len(pypdf.PdfReader(str(artifact)).pages)
                        > MAX_PDF_SIZE_BEFORE_SUBSETTING
                    ):
                        subset = compile_doc_subset(
                            input_pdf=artifact,
                            set_size=MAX_PDF_SIZE_BEFORE_SUBSETTING,
                            temp_dir=TEMP_DIR,
                        )
                    else:
                        subset = artifact

                    # Try QWEN OCR
                    content = visual_processor.extract_text(file_path=subset)

                    # If OCR fails, get visual description
                    if not content:
                        logger.warning(
                            f"QWEN OCR failed for {artifact.name}, using visual description"
                        )
                        content = visual_processor.extract_visual_description(
                            file_path=subset
                        )

            # 2. IMAGES: Visual Description directly
            elif file_extension in IMAGE_TYPES:
                logger.debug(f"Processing as image: {artifact.name}")
                content = visual_processor.extract_visual_description(
                    file_path=artifact
                )

            # 3. VIDEO/AUDIO: Transcription → (if video) Visual Description
            elif file_extension in VIDEO_TYPES or file_extension in AUDIO_TYPES:
                logger.debug(f"Processing as audio/video: {artifact.name}")

                # Verify file actually has audio before attempting transcription
                if audio_processor.has_audio(artifact):
                    # Try transcription
                    content = audio_processor.transcribe_audio(file_path=artifact)

                    # Fallback to visual description for videos
                    if not content and file_extension in VIDEO_TYPES:
                        logger.warning(
                            f"Transcription failed for {artifact.name}, extracting visual description from frames"
                        )

                        subset = compile_video_snapshot_subset(
                            video_path=artifact,
                            set_size=6,
                            temp_dir=TEMP_DIR,
                        )

                        content = visual_processor.extract_visual_description(
                            file_path=subset
                        )
                else:
                    logger.warning(
                        f"File {artifact.name} has no audio stream to transcribe"
                    )

                    # If it's a video without audio, try visual description
                    if file_extension in VIDEO_TYPES:
                        logger.debug(
                            f"Extracting visual description for silent video: {artifact.name}"
                        )
                        subset = compile_video_snapshot_subset(
                            video_path=artifact,
                            set_size=6,
                            temp_dir=TEMP_DIR,
                        )
                        content = visual_processor.extract_visual_description(
                            file_path=subset
                        )

            # 4. TEXT FILES: Use text directly
            elif file_extension in TEXT_TYPES or file_extension in CODE_TYPES:
                logger.debug(f"Processing as text file: {artifact.name}")

                try:
                    with open(artifact, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                except UnicodeDecodeError:
                    # Try with latin-1 encoding for non-UTF8 files
                    try:
                        with open(artifact, "r", encoding="latin-1") as f:
                            content = f.read().strip()
                        logger.debug(f"Read {artifact.name} with latin-1 encoding")
                    except Exception as e:
                        logger.warning(f"Failed to read text file {artifact.name}: {e}")
                        content = None

            # 5. UNSUPPORTED: Raise error
            else:
                raise RuntimeError(
                    f"Unsupported file type: {artifact.suffix}. Cannot extract semantical data"
                )

            # Validate that we extracted some content
            if not content:
                raise ValueError(f"No content could be extracted from {artifact.name}")

            logger.debug(
                f"Successfully extracted {len(content)} characters from {artifact.name}"
            )

            # Extract structured semantic fields from the content
            semantical_descriptors = extract_fields(logger=logger, content=content)

            metadata_json = json.dumps(semantical_descriptors)

            with exiftool.ExifTool() as et:
                et.execute(
                    b"-XMP-custom:Metadata=" + metadata_json.encode("utf-8"),
                    str(artifact).encode("utf-8"),
                )

            # Update profile with extracted semantics
            artifact_profile_data["extracted_semantics"] = semantical_descriptors

            # Update stage tracking
            # Initialize stage progression data if it doesn't exist in the profile
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            # Store text extraction metadata
            if semantical_descriptors:
                artifact_profile_data["text_extraction"] = {
                    "success": True,
                    "character_count": len(content),
                    "extraction_method": (
                        "direct_text"
                        if file_extension in TEXT_TYPES or file_extension in CODE_TYPES
                        else "processed"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }

                # Mark this processing stage as completed with timestamp
                artifact_profile_data["stage_progression_data"][
                    "semantics_extraction"
                ] = {
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                artifact_profile_data["text_extraction"] = {
                    "success": False,
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
            success_location: Path = success_dir / artifact

            # Perform the actual file move operation
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

        except Exception as e:
            # Catch any errors during processing to prevent pipeline failure
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory for later inspection
            failure_location: Path = failure_dir / artifact

            # Move the failed artifact for later review and debugging
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            # Continue processing remaining artifacts despite this failure
            continue

    # Log completion of the entire extraction stage
    logger.info("Semantics extraction stage completed")
    return None


"""
LLM-based field extraction with hardware auto-detection and context management

This module provides intelligent document field extraction using local LLM models through OLLAMA.
It automatically detects hardware capabilities, selects optimal models, and manages context to
prevent performance degradation during batch processing.

Key Features:
- Automatic hardware detection and model selection
- Context refresh to prevent model degradation
- Multiple extraction strategies (single/multi-prompt)
- Comprehensive validation and error handling
- Performance monitoring and statistics

Dependencies:
- requests: HTTP communication with OLLAMA API
- psutil: System hardware detection
- GPUtil: GPU memory detection (optional)
"""


def extract_fields(logger: logging.Logger, content: str) -> LanguageExtractionReport:
    """
    Extract all document fields using LLM with comprehensive error handling.

    This method processes document content through the configured LLM model to extract
    structured field data. It iterates through all configured field prompts, handles
    individual field extraction failures gracefully, and provides detailed performance
    metrics and error reporting.

    Returns:
                    ExtractionReport: Structured report containing either:
                                    - On success: extracted_fields dict with field_name -> extracted_value mappings
                                    - On failure: error details and diagnostic information
                                    Both cases include processing metadata and performance metrics

    Note:
                    Individual field extraction failures are logged but don't stop the overall
                    process. Failed fields are marked as "UNKNOWN" in the results.
    """

    # Initialize ALL instance variables FIRST to ensure consistent object state
    logger: logging.Logger = logger

    # Start timing for performance monitoring
    start_time = datetime.now()

    logger.debug("Extracting fields from summaries...")

    try:
        # Attempt to establish connection to OLLAMA service
        client = ollama.Client()

        # Log successful initialization with model details
        logger.info("Language processor initialization completed successfully")
        logger.info(f"Active model: {PREFERRED_LANGUAGE_MODEL}")

        # Main extraction loop - process each configured field independently
        extracted_fields: Dict[str, str] = {}

        # Iterate through all field prompts defined in configuration
        for field_name, field_instruction in FIELD_PROMPTS.items():
            try:
                logger.debug(f"Extracting field: {field_name}")

                # Construct complete prompt combining system instructions with document data
                # This ensures the model has full context for accurate extraction
                complete_prompt: str = f"""
{SYSTEM_PROMPT}

DOCUMENT CONTENTS:
{content}

TASK:
=====
{field_instruction}"""

                # Check if client is properly initialized
                if client is None:
                    raise RuntimeError("OLLAMA client is not initialized")

                # Send prompt to LLM and wait for complete response
                # stream=False ensures we get the full response before proceeding
                response: Dict[str, Any] = client.generate(
                    model=PREFERRED_LANGUAGE_MODEL,
                    prompt=complete_prompt,
                    stream=False,
                )

                # Extract the text response from the LLM output
                response_text: str = response["response"]
                extracted_fields[field_name] = response_text

                # Log successful extraction with response preview
                logger.info(
                    f"Extracted field '{field_name}': {response_text[:100]}..."
                    if len(response_text) > 100
                    else f"Extracted field '{field_name}': {response_text}"
                )

            except Exception as e:
                # Handle individual field extraction failures gracefully
                # Log the error but continue processing other fields
                logger.warning(f"Failed to extract {field_name}: {e}")
                extracted_fields[field_name] = "UNKNOWN"

        # Calculate total processing time for performance monitoring
        processing_time: float = (datetime.now() - start_time).total_seconds()

        logger.info(f"Processing time: {processing_time:.2f}s")

        # Return successful extraction report with all metadata
        return LanguageExtractionReport(
            success=True,
            extracted_fields=extracted_fields,
            processing_timestamp=datetime.now().isoformat(),
            model_used=PREFERRED_LANGUAGE_MODEL,
            processing_time_seconds=processing_time,
            error=None,
            error_type=None,
        )

    except Exception as e:
        # Comprehensive error handling with context preservation
        processing_time = (datetime.now() - start_time).total_seconds()

        # Log detailed error information for debugging
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Processing time before failure: {processing_time:.2f}s")

        # Log full traceback for debugging if available
        if hasattr(e, "__traceback__"):
            import traceback

            logger.debug("Full traceback:", exc_info=True)

        # Return failure report with diagnostic information
        return LanguageExtractionReport(
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            processing_timestamp=datetime.now().isoformat(),
            processing_time_seconds=processing_time,
            model_used=PREFERRED_LANGUAGE_MODEL,
            extracted_fields=None,
        )
