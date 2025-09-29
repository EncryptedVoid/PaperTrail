"""
Sanitizer Pipeline Module

A robust file processing pipeline that handles duplicate detection, zero-byte file
identification, and unsupported file type filtering using checksum verification.

This module provides functionality to sanitize a directory of files or a single file by:
- Detecting and moving duplicate files based on checksum comparison
- Identifying and moving zero-byte (corrupted/incomplete) files
- Filtering out unsupported file types
- Maintaining a persistent history of processed files

Author: Ashiq Gazi
"""

import json
import logging
import secrets
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, TypedDict

from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    PROFILE_PREFIX,
    UNSUPPORTED_EXTENSIONS,
    UUID_ENTROPY,
    UUID_PREFIX,
)
from src.utilities.common import (
    generate_checksum,
    load_checksum_history,
    move,
    save_checksum,
)


class SanitizationReport(TypedDict):
    """
    Type definition for the sanitization report returned by the sanitize method.

    Attributes:
        processed_artifacts: Number of files that successfully passed all sanitization checks
        total_artifacts: Total number of files discovered in the source directory/file
        duplicates_moved: Count of duplicate files moved to review directory
        zero_byte_moved: Count of zero-byte files moved to review directory
        unsupported_moved: Count of unsupported file types moved to review directory
        errors: List of error messages encountered during processing
        file_types: Dictionary mapping file extensions to their occurrence counts
        remaining_artifacts: List of file paths that passed sanitization and remain for processing
    """

    processed_artifacts: int
    total_artifacts: int
    duplicates_moved: int
    zero_byte_moved: int
    unsupported_moved: int
    errors: List[str]
    file_types: Dict[str, int]
    remaining_artifacts: List[str]


class SanitizerPipeline:
    """
    A file sanitization pipeline for processing directories of artifacts or single files.

    This class handles the detection and removal of duplicate files, zero-byte files,
    and unsupported file types from a source directory or single file, moving problematic files to
    a review directory while maintaining a persistent checksum history.

    The pipeline works in the following stages:
    1. Path validation and file discovery
    2. File type analysis and initial categorization
    3. Individual file processing (unsupported -> zero-byte -> duplicate checks)
    4. Checksum generation and history management
    5. Safe file movement with conflict resolution
    """

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """
        Initialize the SanitizerPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
        """
        self.logger = logger

    def sanitize(
        self,
        source_path: Path,
        failure_dir: Path,
        success_dir: Path,
    ) -> SanitizationReport:
        """
        Sanitize a directory or single file by detecting duplicates, zero-byte files, and unsupported files.

        This method performs a comprehensive sanitization process:
        1. Validates input paths exist and are accessible
        2. Discovers all files (either from directory or single file)
        3. Analyzes file types and generates initial statistics
        4. Processes each file through multiple filters:
            - Unsupported file type check (based on extension)
            - Zero-byte file detection (corrupted/incomplete files)
            - Duplicate detection (using checksum comparison)
        5. Moves problematic files to review directory
        6. Updates checksum history for future duplicate detection
        7. Generates comprehensive report of all operations

        Args:
            source_path: Directory containing files to process OR a single file to process
            failure_dir: Directory to move problematic files to for manual review
            success_dir: Directory to move successfully processed files to

        Returns:
            SanitizationReport containing detailed results of the sanitization process

        Note:
            Files are processed in order of size (smallest first) to provide faster
            initial feedback and optimize processing time for large directories.
        """

        if not source_path.exists():
            error_msg = f"Source path does not exist: {source_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not failure_dir.exists() or not failure_dir.is_dir():
            error_msg = (
                f"Review directory does not exist or is not a directory: {failure_dir}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not success_dir.exists() or not success_dir.is_dir():
            error_msg = (
                f"Success directory does not exist or is not a directory: {success_dir}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Initialize report structure with default values
        report: SanitizationReport = {
            "processed_artifacts": 0,
            "total_artifacts": 0,
            "duplicates_moved": 0,
            "zero_byte_moved": 0,
            "unsupported_moved": 0,
            "errors": [],
            "file_types": defaultdict(int),
            "remaining_artifacts": [],
        }

        # Determine if source_path is a directory or a single file and get file list
        try:
            if source_path.is_dir():
                # Source is a directory - get all files in it
                unprocessed_artifacts = [
                    item for item in source_path.iterdir() if item.is_file()
                ]
                self.logger.info(f"Processing directory: {source_path}")
            elif source_path.is_file():
                # Source is a single file
                unprocessed_artifacts = [source_path]
                self.logger.info(f"Processing single file: {source_path}")
            else:
                error_msg = (
                    f"Source path is neither a file nor a directory: {source_path}"
                )
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                return report
        except Exception as e:
            error_msg = f"Failed to scan source path: {e}"
            self.logger.error(error_msg)
            report["errors"].append(error_msg)
            return report

        # Handle empty directory case
        if not unprocessed_artifacts:
            self.logger.info("No files found to process")
            return report

        # Sort files by size (smallest first) for faster initial processing feedback
        # This allows users to see progress immediately rather than waiting for large files
        unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
        report["total_artifacts"] = len(unprocessed_artifacts)

        # Log sanitization stage header for clear progress tracking
        self.logger.info("=" * 80)
        self.logger.info(
            "DUPLICATE DETECTION, UNSUPPORTED FILES, AND CHECKSUM VERIFICATION STAGE"
        )
        self.logger.info("=" * 80)
        self.logger.info(f"Found {len(unprocessed_artifacts)} file(s) to process")

        # Analyze file types for initial summary and planning purposes
        for artifact in unprocessed_artifacts:
            ext = artifact.suffix.lower()  # Normalize extension to lowercase
            report["file_types"][ext] += 1

        # Generate and log file type breakdown for user awareness
        file_summary = ", ".join(
            [f"{count} {ext}" for ext, count in report["file_types"].items()]
        )
        self.logger.info(f"File type breakdown: {file_summary}")

        # Load existing checksum history to enable duplicate detection across sessions
        checksum_history: Set[str] = load_checksum_history(logger=self.logger)

        # Process each file through the sanitization pipeline
        # Use progress bar only for multiple files
        if len(unprocessed_artifacts) > 1:
            artifact_iterator: Any = tqdm(
                unprocessed_artifacts,
                desc="Sanitizing artifact list for duplicates, zeros, and unsupported files",
                unit="artifacts",
            )
        else:
            artifact_iterator = unprocessed_artifacts

        for artifact in artifact_iterator:
            try:
                # STAGE 1: Check for unsupported file types (fastest check first)
                # This check is performed first as it only requires extension comparison
                if artifact.suffix.lower() in UNSUPPORTED_EXTENSIONS:
                    review_location = failure_dir / artifact.name
                    if move(artifact, review_location):
                        report["unsupported_moved"] += 1
                        self.logger.debug(
                            f"Moved unsupported file {artifact.name} to review"
                        )
                    continue  # Skip to next file, no further processing needed

                # STAGE 2: Check for zero-byte files (corrupted or incomplete)
                # This check is performed second as it only requires file stat operation
                file_size = artifact.stat().st_size
                if file_size == 0:
                    review_location = failure_dir / artifact.name
                    if move(artifact, review_location):
                        report["zero_byte_moved"] += 1
                        self.logger.debug(
                            f"Moved zero-byte file {artifact.name} to review"
                        )
                    continue  # Skip to next file, no checksum needed for zero-byte files

                # STAGE 3: Generate checksum for duplicate detection
                # This is the most expensive operation, so it's performed last
                checksum = generate_checksum(logger=self.logger, artifact_path=artifact)
                self.logger.debug(
                    f"Generated checksum for {artifact.name}: {checksum[:16]}..."
                )

                # STAGE 4: Check for duplicate files using checksum comparison
                # If checksum exists in history, this file is a duplicate
                if checksum in checksum_history:
                    review_location = failure_dir / artifact.name
                    if move(artifact, review_location):
                        report["duplicates_moved"] += 1
                        self.logger.info(
                            f"Moved duplicate file {artifact.name} to review"
                        )
                    continue  # Skip to next file, duplicate handled
                else:
                    save_checksum(logger=self.logger, checksum=checksum)

                # STAGE 5: Generate profile and move to success directory
                # Log UUID generation request for audit trail
                self.logger.debug(
                    f"Generating UUID4: prefix='{UUID_PREFIX}', entropy={UUID_ENTROPY}"
                )

                secure_random_bytes: bytes = secrets.token_bytes(UUID_ENTROPY)
                artifact_id: str = str(uuid.UUID(bytes=secure_random_bytes, version=4))

                self.logger.debug(f"Generated secure UUID: {artifact_id}")

                # Add prefix if provided
                if UUID_PREFIX:
                    artifact_id = UUID_PREFIX + "-" + artifact_id
                    self.logger.debug(f"Added prefix: {UUID_PREFIX}")

                original_name = artifact.name
                new_name = f"{ARTIFACT_PREFIX}-{artifact_id}{artifact.suffix}"
                file_size = artifact.stat().st_size

                # Create initial profile data
                profile_data = {
                    "uuid": artifact_id,
                    "original_artifact_name": original_name,
                    "artifact_size": file_size,
                    "artifact_type": artifact.suffix.lower(),
                    "timestamp": datetime.now().isoformat(),
                }

                # Create corresponding profile file
                artifact_profile_name = f"{PROFILE_PREFIX}-{artifact_id}.json"
                artifact_profile = ARTIFACT_PROFILES_DIR / artifact_profile_name

                # Ensure profile directory exists
                ARTIFACT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

                # Save final profile
                with open(artifact_profile, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=2)

                # Move artifact to success directory with error handling
                success_path = success_dir / new_name
                if success_path.exists():
                    # Handle naming conflict
                    base_name = success_path.stem
                    extension = success_path.suffix
                    counter = 1
                    while success_path.exists():
                        new_name = f"{base_name}_{counter}{extension}"
                        success_path = success_dir / new_name
                        counter += 1

                # Perform the move
                moved_artifact = artifact.rename(success_path)

                # Track this file as successfully processed
                report["remaining_artifacts"].append(str(moved_artifact))
                report["processed_artifacts"] += 1

                self.logger.info(
                    f"Successfully processed {original_name} -> {new_name}"
                )

            except Exception as e:
                # Capture and log any unexpected errors during file processing
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                # Continue processing remaining files despite individual failures

        # Generate final summary report for user review
        self.logger.info("Sanitization complete:")
        self.logger.info(
            f"  - {report['processed_artifacts']} file(s) passed sanitization"
        )
        self.logger.info(f"  - {report['duplicates_moved']} duplicates moved to review")
        self.logger.info(
            f"  - {report['zero_byte_moved']} zero-byte files moved to review"
        )
        self.logger.info(
            f"  - {report['unsupported_moved']} unsupported files moved to review"
        )

        # Warn about any errors encountered during processing
        if report["errors"]:
            self.logger.warning(f"  - {len(report['errors'])} errors encountered")

        return report
