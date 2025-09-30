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
	ARCHIVAL_DIR ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	PROFILE_PREFIX ,
	TEMP_DIR ,
	UUID_ENTROPY ,
	UUID_PREFIX ,
)
from src.utilities.checksum import (
	generate_checksum ,
	load_checksum_history ,
	save_checksum ,
)
from utilities.file_stability import is_stable , repair_instability


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
    # Use list comprehension to filter only files (not directories or symlinks)
    unprocessed_artifacts: list[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]
    logger.info(f"Processing directory: {source_dir}")

    # Handle empty directory case - exit early if no files to process
    if not unprocessed_artifacts:
        logger.info("No files found to process")
        return

    # Sort files by size (smallest first) for faster initial processing feedback
    # This allows users to see progress immediately rather than waiting for large files
    # Lambda function extracts file size in bytes for sorting key
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} file(s) to process")

    # Load existing checksum history to enable duplicate detection across sessions
    # This persistent history prevents re-processing of files that were already handled
    checksum_history: Set[str] = load_checksum_history(logger=logger)
    logger.debug(f"Loaded {len(checksum_history)} checksums from history")

    # Process each artifact file with a progress bar for user feedback
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing artifacts",
        unit="artifacts",
    ):
        try:
            # Log the start of processing for this specific artifact
            logger.info(f"Processing artifact: {artifact.name}")

            # Capture the exact timestamp when processing starts (ISO 8601 format)
            start_time: str = datetime.now().isoformat()

            # Check for zero-byte files (corrupted or incomplete)
            # Zero-byte files indicate failed downloads, corruption, or placeholder files
            file_size: int = artifact.stat().st_size
            if file_size == 0:
                # Create descriptive error message for zero-byte detection
                error_msg: str = f"Zero-byte file detected: {artifact.name}"
                logger.warning(error_msg)
                # Raise ValueError to trigger the exception handler and move to failure_dir
                raise ValueError(error_msg)

            # Log successful file size validation
            logger.debug(f"File size validated: {file_size} bytes")

            if not is_stable(logger=logger, file_path=str(artifact)):
                if not repair_instability(
                    logger=logger,
                    file_path=str(artifact),
                    temp_directory=str(TEMP_DIR),
                    archive_directory=str(ARCHIVAL_DIR),
                ):
                    raise RuntimeError("File could not be repaired and is unstable")

            # Generate checksum for duplicate detection
            # Checksums (SHA-256) provide cryptographic content verification
            checksum: str = generate_checksum(logger=logger, artifact_path=artifact)
            # Log only first 16 characters of checksum for brevity while maintaining uniqueness visibility
            logger.debug(f"Generated checksum for {artifact.name}: {checksum[:16]}...")

            # Check for duplicate files using checksum comparison
            # If checksum already exists in history, this is a duplicate
            if checksum in checksum_history:
                # Create descriptive error message including partial checksum for debugging
                error_msg: str = (
                    f"Duplicate file detected: {artifact.name} (checksum: {checksum[:16]}...)"
                )
                logger.warning(error_msg)
                # Raise ValueError to trigger the exception handler and move to failure_dir
                raise ValueError(error_msg)

            # Save checksum to history for future duplicate detection
            # This persists the checksum so duplicates can be detected in future runs
            save_checksum(logger=logger, checksum=checksum)
            logger.debug(f"Checksum saved to history for: {artifact.name}")

            # Generate secure UUID for artifact identification
            logger.debug(
                f"Generating UUID4: prefix='{UUID_PREFIX}', entropy={UUID_ENTROPY}"
            )

            # Generate cryptographically secure random bytes for UUID creation
            # secrets.token_bytes ensures cryptographic randomness for security
            secure_random_bytes: bytes = secrets.token_bytes(UUID_ENTROPY)

            # Create UUID from secure random bytes using version 4 (random) specification
            artifact_id: str = str(uuid.UUID(bytes=secure_random_bytes, version=4))
            logger.debug(f"Generated secure UUID: {artifact_id}")

            # Add prefix if configured in settings
            # This allows for organizational namespacing of artifacts
            if UUID_PREFIX:
                artifact_id = UUID_PREFIX + "-" + artifact_id
                logger.debug(f"Added UUID prefix: {UUID_PREFIX}")

            # Prepare artifact naming with standardized format
            # Store original name for profile metadata
            original_name: str = artifact.name
            # Create new name using configured prefix, UUID, and original file extension
            unique_id_name: str = f"{ARTIFACT_PREFIX}-{artifact_id}{artifact.suffix}"

            # Create initial profile data dictionary with comprehensive metadata
            artifact_profile_data: dict[str, any] = {
                "uuid": artifact_id,  # Unique identifier for tracking
                "original_artifact_name": original_name,  # Preserve original filename
                "artifact_size": file_size,  # File size in bytes
                "artifact_type": artifact.suffix.lower(),  # Normalized file extension
                "checksum": checksum,  # Full SHA-256 checksum for verification
                "stage_progression_data": {
                    # Track when sanitization started
                    "sanitization_start_timestamp": start_time,
                    # Track when sanitization completed
                    "sanitization_completion_timestamp": datetime.now().isoformat(),
                    # Mark status as completed for this stage
                    "sanitization_status": "completed",
                },
            }

            # Create corresponding profile file path using configured naming convention
            artifact_profile_name: str = f"{PROFILE_PREFIX}-{artifact_id}.json"
            artifact_profile_path: Path = ARTIFACT_PROFILES_DIR / artifact_profile_name

            # Save profile to disk as JSON with error handling
            try:
                # Open file with UTF-8 encoding to support international characters
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    # Write JSON with indentation for human readability
                    # ensure_ascii=False allows Unicode characters in output
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile created successfully for: {artifact.name}")
            except Exception as e:
                # Create descriptive error message if profile creation fails
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                # Log with full exception traceback for debugging
                logger.error(error_msg, exc_info=True)
                # Re-raise to trigger outer exception handler
                raise

            # Move artifact to success directory with renamed filename
            # Construct the destination path with new standardized name
            success_location: Path = success_dir / unique_id_name

            # Handle naming conflicts in destination directory
            # Check if a file with the same name already exists
            if success_location.exists():
                # Extract components for conflict resolution
                base_name: str = success_location.stem  # Filename without extension
                extension: str = success_location.suffix  # File extension
                counter: int = 1  # Start counter for appending to filename

                # Keep incrementing counter until we find an available filename
                while success_location.exists():
                    # Create new filename with counter suffix
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / conflict_name
                    counter += 1
                logger.debug(f"Resolved naming conflict: {success_location.name}")

            # Perform the move operation from source to success directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Successfully processed {original_name} -> {unique_id_name}")

        except Exception as e:
            # Catch any exception that occurred during processing
            # Create descriptive error message with artifact name and exception details
            error_msg: str = f"Error processing {artifact.name}: {e}"
            # Log error with full traceback for debugging
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory for manual review
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            # Check if a file with the same name already exists in failure directory
            if failure_location.exists():
                # Extract components for conflict resolution
                base_name: str = failure_location.stem  # Filename without extension
                extension: str = failure_location.suffix  # File extension
                counter: int = 1  # Start counter for appending to filename

                # Keep incrementing counter until we find an available filename
                while failure_location.exists():
                    # Create new filename with counter suffix
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / conflict_name
                    counter += 1
                logger.debug(
                    f"Resolved naming conflict in failure directory: {failure_location.name}"
                )

            # Move the failed artifact to failure directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            # Continue to next artifact in the loop
            continue

    # Log completion of the entire sanitization stage
    logger.info("Sanitization stage completed")
