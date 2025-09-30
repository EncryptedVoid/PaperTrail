"""
Metadata Pipeline Module

A robust file processing pipeline that handles comprehensive metadata extraction
from various file types including images, documents, PDFs, office files, audio,
video, archives, and more.

This module provides functionality to extract metadata by:
- Detecting file types and routing to appropriate extractors
- Extracting filesystem metadata (size, dates, permissions)
- Processing image EXIF, IPTC, XMP data, color profiles, and technical details
- Extracting document properties, structure information, and content analysis
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
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    JAVA_PATH,
    MIN_JAVA_VERSION,
    PROFILE_PREFIX,
    TIKA_APP_JAR_PATH,
)


def extract_metadata(
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
    - Technical metadata extraction (EXIF, IPTC, document properties, etc.)
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

    # Validate that Tika JAR exists before processing
    tika_jar_path: Path = Path(TIKA_APP_JAR_PATH)
    if not tika_jar_path.exists():
        error_msg: str = f"Tika JAR not found at: {TIKA_APP_JAR_PATH}"
        logger.error(error_msg, exc_info=True)
        raise FileNotFoundError(error_msg)

    # Validate Java runtime is available
    try:
        subprocess.run(
            [JAVA_PATH, "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        error_msg: str = (
            "Java not found. Please install Java 11+ and ensure it's in your PATH"
        )
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # Validate Java version meets minimum requirements
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            [JAVA_PATH, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # Java version information is in stderr, first line
        version_output: str = result.stderr.split("\n")[0]

        # Parse version number (handles both old and new formats)
        # Old format: java version "1.8.0_292"
        # New format: openjdk version "11.0.12"
        match: Optional[re.Match] = re.search(r'version "(\d+)\.(\d+)', version_output)
        if not match:
            # Try Java 8 format
            match = re.search(r'version "1\.(\d+)', version_output)
            if match:
                major_version: int = int(match.group(1))
            else:
                error_msg: str = "Could not parse Java version from output"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)
        else:
            major_version: int = int(match.group(1))

        if major_version < MIN_JAVA_VERSION:
            error_msg: str = (
                f"Java {major_version} found, but Java {MIN_JAVA_VERSION}+ is required. "
                f"Please upgrade your Java installation."
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

        logger.debug(f"Java version validated: {major_version}")

    except FileNotFoundError as e:
        error_msg: str = (
            f"Java not found at {JAVA_PATH}. Please install Java {MIN_JAVA_VERSION}+ "
            "and ensure it's in your PATH."
        )
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    except subprocess.TimeoutExpired as e:
        error_msg: str = "Java version check timed out"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg: str = (
            f"Error checking Java version: {e}. Please ensure Java {MIN_JAVA_VERSION}+ is installed."
        )
        logger.error(error_msg, exc_info=True)
        raise EnvironmentError(error_msg) from e

    # Log metadata extraction stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("METADATA EXTRACTION AND PROFILE UPDATE STAGE")
    logger.info("=" * 80)

    # Discover all artifact files in the source directory
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]

    # Handle empty directory case
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

    # Sort files by size for consistent processing order (smaller files first)
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Track last successful extraction for return value
    last_metadata: Optional[Dict[str, Any]] = None
    last_text: Optional[str] = None

    # Process each artifact file
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Extracting technical metadata",
        unit="artifacts",
    ):
        try:
            logger.info(f"Processing artifact: {artifact.name}")

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
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
                str(tika_jar_path),
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

            # Build Tika command for text extraction
            # --text: Extract full text content
            text_cmd: List[str] = [
                JAVA_PATH,
                "-jar",
                str(tika_jar_path),
                "--text",
                str(artifact),
            ]

            # Execute Tika with timeout
            text_process: subprocess.CompletedProcess = subprocess.run(
                text_cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Check if text extraction was successful
            if text_process.returncode != 0:
                error_msg: str = (
                    f"Tika text extraction failed for {artifact.name}: "
                    f"{text_process.stderr}"
                )
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

            extracted_text: str = text_process.stdout.strip()

            # Check if text extraction produced results
            if not extracted_text:
                logger.warning(f"No text content extracted for {artifact.name}")

            logger.debug(f"Successfully extracted text for: {artifact.name}")

            # Update profile with extracted metadata
            artifact_profile_data["extracted_metadata"] = extracted_metadata

            # Store extracted text if available
            if extracted_text:
                artifact_profile_data["extracted_text"] = extracted_text
                artifact_profile_data["text_extraction"] = {
                    "success": True,
                    "character_count": len(extracted_text),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                artifact_profile_data["text_extraction"] = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            # Update stage tracking
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            artifact_profile_data["stage_progression_data"]["metadata_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile back to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory
            success_location: Path = success_dir / artifact.name

            # Handle naming conflicts
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                while success_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

            # Track last successful extraction
            last_metadata = extracted_metadata
            last_text = extracted_text

        except Exception as e:
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                while failure_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue

    logger.info("Metadata extraction stage completed")
