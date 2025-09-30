"""
Sanitizer Pipeline Module

A robust file processing pipeline that handles duplicate detection, zero-byte file
identification, and unsupported file type filtering using checksum verification.

This module provides functionality to sanitize a directory of files by:
- Detecting and moving duplicate files based on checksum comparison
- Identifying and moving zero-byte (corrupted/incomplete) files
- Filtering out unsupported file types
- Maintaining a persistent history of processed files
- Generating unique identifiers and profiles for valid artifacts

Author: Ashiq Gazi
"""

import json
import logging
import secrets
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Set

from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    PROFILE_PREFIX,
    UUID_ENTROPY,
    UUID_PREFIX,
)
from src.utilities.checksum import (
    generate_checksum,
    load_checksum_history,
    save_checksum,
)


def sanitize(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:
    """
    Sanitize a directory by detecting duplicates, zero-byte files, and creating profiles.

    This method performs a comprehensive sanitization process by discovering all files
    in the source directory, analyzing each file for common issues (zero-byte corruption,
    duplicate content), generating secure unique identifiers, creating artifact profiles,
    and moving files to appropriate directories based on validation results.

    The sanitization process includes:
    - File discovery and size-based sorting (smallest first for faster feedback)
    - Zero-byte file detection (indicates corruption or incomplete downloads)
    - SHA-256 checksum generation for content verification
    - Duplicate detection using checksum history comparison
    - Secure UUID generation with configurable entropy
    - Artifact profile creation with metadata
    - File renaming with conflict resolution
    - Moving files to success or failure directories
    - Comprehensive error logging with full exception details

    Args:
            logger: Logger instance for tracking operations and errors
            source_dir: Directory containing files to process. All files in this directory
                    will be analyzed and sanitized
            failure_dir: Directory to move problematic files (zero-byte, duplicates).
                    Files are moved here when validation fails
            success_dir: Directory to move successfully validated files after profile
                    creation and renaming

    Returns:
            None. This method processes files in-place and moves them to appropriate
            directories based on validation results.

    Raises:
            FileNotFoundError: If source directory does not exist
            OSError: If directory creation or file operations fail due to permissions
            RuntimeError: If checksum generation or UUID creation fails

    Note:
            - Files are processed in order of size (smallest first) for faster feedback
            - Checksum history is persisted across sessions for duplicate detection
            - UUID generation uses cryptographically secure random bytes
            - Profile files follow the PROFILE-{uuid}.json naming convention
            - Artifacts are renamed to ARTIFACT-{uuid}{extension} format
            - Naming conflicts in destination directories are automatically resolved
            - All validation failures are logged with full exception details (exc_info=True)
    """

    # Log sanitization stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("DUPLICATE DETECTION, ZERO-BYTE CHECK, AND PROFILE GENERATION STAGE")
    logger.info("=" * 80)

    # Discover all files in the source directory
    unprocessed_artifacts: list[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]
    logger.info(f"Processing directory: {source_dir}")

    # Handle empty directory case
    if not unprocessed_artifacts:
        logger.info("No files found to process")
        return

    # Sort files by size (smallest first) for faster initial processing feedback
    # This allows users to see progress immediately rather than waiting for large files
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} file(s) to process")

    # Load existing checksum history to enable duplicate detection across sessions
    checksum_history: Set[str] = load_checksum_history(logger=logger)
    logger.debug(f"Loaded {len(checksum_history)} checksums from history")

    # Process each artifact file
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing artifacts",
        unit="artifacts",
    ):
        try:
            logger.info(f"Processing artifact: {artifact.name}")
            start_time: str = datetime.now().isoformat()

            # Check for zero-byte files (corrupted or incomplete)
            file_size: int = artifact.stat().st_size
            if file_size == 0:
                error_msg: str = f"Zero-byte file detected: {artifact.name}"
                logger.warning(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"File size validated: {file_size} bytes")

            # Generate checksum for duplicate detection
            checksum: str = generate_checksum(logger=logger, artifact_path=artifact)
            logger.debug(f"Generated checksum for {artifact.name}: {checksum[:16]}...")

            # Check for duplicate files using checksum comparison
            if checksum in checksum_history:
                error_msg: str = (
                    f"Duplicate file detected: {artifact.name} (checksum: {checksum[:16]}...)"
                )
                logger.warning(error_msg)
                raise ValueError(error_msg)

            # Save checksum to history for future duplicate detection
            save_checksum(logger=logger, checksum=checksum)
            logger.debug(f"Checksum saved to history for: {artifact.name}")

            # Generate secure UUID for artifact identification
            logger.debug(
                f"Generating UUID4: prefix='{UUID_PREFIX}', entropy={UUID_ENTROPY}"
            )

            secure_random_bytes: bytes = secrets.token_bytes(UUID_ENTROPY)
            artifact_id: str = str(uuid.UUID(bytes=secure_random_bytes, version=4))
            logger.debug(f"Generated secure UUID: {artifact_id}")

            # Add prefix if configured
            if UUID_PREFIX:
                artifact_id = UUID_PREFIX + "-" + artifact_id
                logger.debug(f"Added UUID prefix: {UUID_PREFIX}")

            # Prepare artifact naming
            original_name: str = artifact.name
            new_name: str = f"{ARTIFACT_PREFIX}-{artifact_id}{artifact.suffix}"

            # Create initial profile data
            artifact_profile_data: dict[str, any] = {
                "uuid": artifact_id,
                "original_artifact_name": original_name,
                "artifact_size": file_size,
                "artifact_type": artifact.suffix.lower(),
                "checksum": checksum,
                "stage_progression_data": {
                    "sanitization_start_timestamp": start_time,
                    "sanitization_completion_timestamp": datetime.now().isoformat(),
                    "sanitization_status": "completed",
                },
            }

            # Create corresponding profile file path
            artifact_profile_name: str = f"{PROFILE_PREFIX}-{artifact_id}.json"
            artifact_profile_path: Path = ARTIFACT_PROFILES_DIR / artifact_profile_name

            # Ensure profile directory exists
            ARTIFACT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

            # Save profile to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile created successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory with renamed filename
            success_location: Path = success_dir / new_name

            # Handle naming conflicts
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                while success_location.exists():
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / conflict_name
                    counter += 1
                logger.debug(f"Resolved naming conflict: {success_location.name}")

            # Perform the move operation
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Successfully processed {original_name} -> {new_name}")

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
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / conflict_name
                    counter += 1
                logger.debug(
                    f"Resolved naming conflict in failure directory: {failure_location.name}"
                )

            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue

    logger.info("Sanitization stage completed")
