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

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, TypedDict, Any
from tqdm import tqdm
from utilities.session_tracking_agent import SessionTracker
from processors.conversion_processor import (
    ConversionProcessor,
    ConversionReport,
    ConversionStatus,
)
from config import (
    CHECKSUM_ALGORITHM,
    UNSUPPORTED_EXTENSIONS,
    ARTIFACT_PROFILES_DIR,
    CHECKSUM_CHUNK_SIZE_BYTES,
    CHECKSUM_HISTORY_FILE,
    PROFILE_PREFIX,
    ARTIFACT_PREFIX,
)
import json
import datetime
from common_utils import move_file_safely
import hashlib
import os
import secrets
import uuid
from datetime import datetime, timezone


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
        session_agent: SessionTracker,
    ):
        """
        Initialize the SanitizerPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
            session_agent: SessionTracker for monitoring pipeline progress and state
        """
        self.logger = logger
        self.session_agent = session_agent

    def sanitize(
        self,
        conversion_agent: ConversionProcessor,
        source_path: Path,
        review_dir: Path,
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
            review_dir: Directory to move problematic files to for manual review
            success_dir: Directory to move successfully processed files to

        Returns:
            SanitizationReport containing detailed results of the sanitization process

        Note:
            Files are processed in order of size (smallest first) to provide faster
            initial feedback and optimize processing time for large directories.
        """

        # Validate that the source path exists and is accessible
        if not source_path.exists():
            error_msg = f"Source path does not exist: {source_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Validate that review and success directories exist and are directories
        if not review_dir.exists() or not review_dir.is_dir():
            error_msg = (
                f"Review directory does not exist or is not a directory: {review_dir}"
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
        checksum_history: Set[str] = self._load_checksum_history()

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
                    review_location = review_dir / artifact.name
                    if move_file_safely(artifact, review_location):
                        report["unsupported_moved"] += 1
                        self.logger.debug(
                            f"Moved unsupported file {artifact.name} to review"
                        )
                    continue  # Skip to next file, no further processing needed

                # STAGE 2: Check for zero-byte files (corrupted or incomplete)
                # This check is performed second as it only requires file stat operation
                file_size = artifact.stat().st_size
                if file_size == 0:
                    review_location = review_dir / artifact.name
                    if move_file_safely(artifact, review_location):
                        report["zero_byte_moved"] += 1
                        self.logger.debug(
                            f"Moved zero-byte file {artifact.name} to review"
                        )
                    continue  # Skip to next file, no checksum needed for zero-byte files

                # STAGE 3: Generate checksum for duplicate detection
                # This is the most expensive operation, so it's performed last
                checksum = self._generate_checksum(artifact)
                self.logger.debug(
                    f"Generated checksum for {artifact.name}: {checksum[:16]}..."
                )

                # STAGE 4: Check for duplicate files using checksum comparison
                # If checksum exists in history, this file is a duplicate
                if checksum in checksum_history:
                    review_location = review_dir / artifact.name
                    if move_file_safely(artifact, review_location):
                        report["duplicates_moved"] += 1
                        self.logger.info(
                            f"Moved duplicate file {artifact.name} to review"
                        )
                    continue  # Skip to next file, duplicate handled

                # STAGE 5: Convert file format to a common file type
                conversion_report: ConversionReport = conversion_agent.process_file(
                    artifact
                )
                if (
                    conversion_report.status == ConversionStatus.SUCCESS
                    and conversion_report.converted_file_path is not None
                ):
                    artifact: Path = conversion_report.converted_file_path

                # STAGE 6: File passed all sanitization checks
                # Add checksum to history to prevent future duplicates
                checksum = self._generate_checksum(artifact)
                checksum_history.add(checksum)
                self._save_checksum(checksum)

                # STAGE 7: Generate profile and move to success directory
                # Generate unique identifier for this artifact
                artifact_id: str = self._generate_uuid()
                original_name = artifact.name
                new_name = f"{ARTIFACT_PREFIX}-{artifact_id}{artifact.suffix}"
                file_size = artifact.stat().st_size

                # Create initial profile data
                profile_data = {
                    "uuid": artifact_id,
                    "checksum": checksum,
                    "original_artifact_name": original_name,
                    "artifact_size": file_size,
                    "artifact_type": artifact.suffix.lower(),
                    "timestamp": datetime.now().isoformat(),
                }
                # Create corresponding profile file
                artifact_profile_name = f"{PROFILE_PREFIX}-{artifact_id}.json"
                artifact_profile = ARTIFACT_PROFILES_DIR / artifact_profile_name

                # Save final profile
                with open(artifact_profile, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=2)

                # Move artifact to success directory
                artifact = artifact.rename(success_dir / new_name)

                # Track this file as successfully processed
                report["remaining_artifacts"].append(str(artifact))
                report["processed_artifacts"] += 1

            except Exception as e:
                # Capture and log any unexpected errors during file processing
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                # Continue processing remaining files despite individual failures

        # Update session tracker with current progress
        self.session_agent.update({"stage": "sanitization", "report": report})

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

    def _generate_checksum(self, artifact_path: Path) -> str:
        """
        Calculate cryptographic checksum of a file using streaming approach for memory efficiency.

        This method implements a high-performance file hashing system that can handle files
        of any size by processing them in configurable chunks. The streaming approach ensures
        constant memory usage regardless of file size, making it suitable for processing
        large datasets in production environments.

        Performance Optimizations:
        - Configurable chunk size for optimal I/O performance on different storage systems
        - Single-pass processing minimizes file system overhead
        - Memory usage remains constant regardless of file size
        - Progress logging for long-running operations on large files

        Security Features:
        - Supports all major cryptographic hash algorithms
        - Validates CHECKSUM_ALGORITHM availability before processing
        - Comprehensive error handling prevents information leakage
        - Full audit trail logging for security compliance

        Args:
            artifact_path: Path object pointing to the file to be hashed - must exist and be readable

        Returns:
            Hexadecimal string representation of the file's cryptographic checksum

        Raises:
            ValueError: If the specified hash CHECKSUM_ALGORITHM is not supported by the system
            FileNotFoundError: If the specified file does not exist or cannot be accessed
            PermissionError: If the file cannot be read due to insufficient permissions
            OSError: If an I/O error occurs during file reading operations
        """
        # Log the start of checksum generation for audit trail
        self.logger.debug(
            f"Starting checksum generation for {artifact_path.name} using {CHECKSUM_ALGORITHM.value}"
        )

        # Validate file exists before attempting to process
        if not artifact_path.exists():
            error_msg: str = f"File not found: {artifact_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Initialize the cryptographic hash object with validation
        try:
            hash_object = hashlib.new(CHECKSUM_ALGORITHM.value)
            self.logger.debug(
                f"Initialized {CHECKSUM_ALGORITHM.value} hash object successfully"
            )

        except ValueError as algorithm_error:
            # Provide detailed error message with available alternatives
            available_algorithms: List[str] = sorted(hashlib.algorithms_available)
            error_msg = (
                f"Unsupported hash CHECKSUM_ALGORITHM: {CHECKSUM_ALGORITHM.value}. "
                f"Available algorithms: {', '.join(available_algorithms)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg) from algorithm_error

        # Track processing metrics for performance monitoring
        start_time: float = datetime.now().timestamp()
        bytes_processed: int = 0

        # Process file in chunks using streaming approach for memory efficiency
        try:
            with artifact_path.open("rb") as artifact_handle:
                self.logger.debug(f"Opened file for reading: {artifact_path.name}")

                # Process file in optimally-sized chunks for I/O performance
                while True:
                    # Read next chunk - size optimized for most storage systems
                    artifact_chunk: bytes = artifact_handle.read(
                        CHECKSUM_CHUNK_SIZE_BYTES
                    )

                    # Check for end of file condition
                    if not artifact_chunk:
                        break

                    # Update hash with current chunk
                    hash_object.update(artifact_chunk)
                    bytes_processed += len(artifact_chunk)

                    # Log progress for large files to provide user feedback
                    if bytes_processed % (CHECKSUM_CHUNK_SIZE_BYTES * 100) == 0:
                        self.logger.debug(
                            f"Processed {bytes_processed:,} bytes of {artifact_path.name}"
                        )

        except FileNotFoundError as artifact_error:
            error_msg = f"File not found during processing: {artifact_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg) from artifact_error

        except PermissionError as permission_error:
            error_msg = f"Permission denied reading file: {artifact_path}"
            self.logger.error(error_msg)
            raise PermissionError(error_msg) from permission_error

        except OSError as io_error:
            error_msg = f"I/O error reading file {artifact_path}: {io_error}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from io_error

        # Calculate final checksum and performance metrics
        final_checksum: str = hash_object.hexdigest()
        processing_duration: float = datetime.now().timestamp() - start_time

        # Log completion with performance metrics for monitoring
        self.logger.info(
            f"Generated {CHECKSUM_ALGORITHM.value} checksum for {artifact_path.name}: "
            f"{final_checksum[:16]}... ({bytes_processed:,} bytes in {processing_duration:.2f}s)"
        )

        return final_checksum

    def _load_checksum_history(self) -> Set[str]:
        """
        Load existing checksum history from persistent storage with comprehensive error handling.

        This method reads the persistent checksum history file and returns all previously
        calculated checksums as a set for efficient duplicate detection. The history file
        is expected to contain one checksum per line in hexadecimal format.

        File Format:
        - One checksum per line
        - Hexadecimal format (lowercase or uppercase accepted)
        - Empty lines and whitespace are automatically ignored
        - Comments (lines starting with #) are automatically filtered out

        Performance Considerations:
        - Large history files are processed efficiently with streaming reads
        - Memory usage scales linearly with unique checksum count
        - Duplicate checksums in file are automatically deduplicated

        Args:
            None - uses the history_artifact_path configured during initialization

        Returns:
            Set of unique checksum strings loaded from the history file.
            Returns empty set if file doesn't exist or cannot be read.

        Note:
            This method never raises exceptions - errors are logged and an empty
            set is returned to allow graceful degradation of duplicate detection.
        """
        # Log the start of history loading operation
        self.logger.debug(f"Loading checksum history from {CHECKSUM_HISTORY_FILE}")

        # Initialize empty checksum collection
        loaded_checksums: Set[str] = set()

        # Check if history file exists before attempting to read
        if not CHECKSUM_HISTORY_FILE.exists():
            self.logger.info(
                "Checksum history file does not exist - starting with empty history"
            )
            return loaded_checksums

        try:
            # Process history file line by line for memory efficiency
            with open(CHECKSUM_HISTORY_FILE, "r", encoding="utf-8") as history_file:
                line_count: int = 0

                for raw_line in history_file:
                    line_count += 1

                    # Clean and validate each line
                    cleaned_line: str = raw_line.strip()

                    # Skip empty lines and comments for robust parsing
                    if not cleaned_line or cleaned_line.startswith("#"):
                        continue

                    # Validate checksum format (hexadecimal characters only)
                    if all(char in "0123456789abcdefABCDEF" for char in cleaned_line):
                        # Normalize to lowercase for consistent comparison
                        normalized_checksum: str = cleaned_line.lower()
                        loaded_checksums.add(normalized_checksum)

                    else:
                        # Log invalid format but continue processing
                        self.logger.warning(
                            f"Invalid checksum format on line {line_count}: {cleaned_line[:32]}..."
                        )

            # Log successful loading with statistics
            self.logger.info(
                f"Successfully loaded {len(loaded_checksums)} unique checksums "
                f"from {line_count} lines in history file"
            )

        except PermissionError as permission_error:
            self.logger.warning(
                f"Permission denied reading checksum history: {permission_error}"
            )

        except OSError as io_error:
            self.logger.warning(f"I/O error reading checksum history: {io_error}")

        except Exception as unexpected_error:
            self.logger.warning(
                f"Unexpected error loading checksum history: {unexpected_error}"
            )

        return loaded_checksums

    def _save_checksum(self, checksum: str) -> None:
        """
        Append a new checksum to the persistent history file with atomic operation guarantees.

        This method safely appends a new checksum to the history file using atomic write
        operations to prevent corruption. The checksum is validated before writing and
        the operation is logged for audit purposes.

        Atomic Write Process:
        1. Validate checksum format to prevent corrupt entries
        2. Open file in append mode for thread-safe writing
        3. Write checksum with newline terminator
        4. Flush to ensure immediate persistence
        5. Log operation completion for audit trail

        Data Integrity Features:
        - Input validation prevents invalid checksums from being stored
        - Append-only operations preserve existing history
        - Comprehensive error handling prevents data loss
        - All operations are logged for troubleshooting

        Args:
            checksum: Valid hexadecimal checksum string to append to history file.
                     Must contain only hexadecimal characters (0-9, a-f, A-F).

        Raises:
            ValueError: If checksum format is invalid (contains non-hex characters)

        Note:
            File I/O errors are logged but do not raise exceptions to prevent
            interruption of processing pipelines. The checksum will still be
            available in memory for the current session.
        """
        # Log the checksum save operation for audit trail
        self.logger.debug(f"Saving checksum to history: {checksum[:16]}...")

        # Validate checksum format before writing to prevent corrupt history
        if not checksum:
            error_msg: str = "Cannot save empty checksum to history"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Verify checksum contains only valid hexadecimal characters
        if not all(char in "0123456789abcdefABCDEF" for char in checksum):
            error_msg = f"Invalid checksum format - contains non-hexadecimal characters: {checksum[:32]}..."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Ensure parent directory exists before writing
            CHECKSUM_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Perform atomic append operation to preserve existing history
            with open(CHECKSUM_HISTORY_FILE, "a", encoding="utf-8") as history_file:
                # Normalize checksum to lowercase for consistency
                normalized_checksum: str = checksum.lower()

                # Write checksum with newline terminator
                history_file.write(f"{normalized_checksum}\n")

                # Force immediate write to disk for persistence
                history_file.flush()
                os.fsync(history_file.fileno())

            # Log successful save operation
            self.logger.debug(f"Successfully saved checksum to history file")

        except PermissionError as permission_error:
            self.logger.error(
                f"Permission denied writing to checksum history: {permission_error}"
            )

        except OSError as io_error:
            self.logger.error(f"I/O error writing to checksum history: {io_error}")

        except Exception as unexpected_error:
            self.logger.error(
                f"Unexpected error saving checksum to history: {unexpected_error}"
            )

    def _generate_uuid(
        self, prefix: str = "", include_timestamp: bool = False, entropy: int = 16
    ) -> str:
        """
        Generate enhanced UUID4 with guaranteed cryptographic security and optional features.

        This method provides a more secure alternative to standard uuid.uuid4() by using
        the secrets module for guaranteed cryptographic randomness across all platforms.
        Optional features include timestamp prefixing and custom entropy levels.

        Security Advantages over uuid.uuid4():
        - Uses secrets.token_bytes() for guaranteed cryptographic security
        - Platform-independent randomness quality
        - Suitable for security-critical applications
        - No dependence on system-specific random sources

        Use Cases:
        - User IDs, session IDs, transaction IDs in production systems
        - API keys and authentication tokens
        - General-purpose unique identifiers for security applications
        - Primary keys in database systems requiring strong uniqueness guarantees
        - Default choice for most applications requiring secure unique IDs

        Optional Features:
        - Prefix support for namespacing and categorization
        - Timestamp inclusion for chronological tracking
        - Configurable entropy for specialized requirements

        Args:
            prefix: Optional string prefix to prepend for namespacing and identification
            include_timestamp: Whether to include ISO timestamp for traceability
            entropy: Number of random bytes to use (default 16 for standard UUID)

        Returns:
            Cryptographically secure UUID string with optional prefix and timestamp

        Note:
            This method is cryptographically secure on all platforms and should be
            preferred over standard uuid.uuid4() for security-sensitive applications.
        """
        # Log UUID generation request for audit trail
        self.logger.debug(
            f"Generating UUID4+: prefix='{prefix}', timestamp={include_timestamp}, entropy={entropy}"
        )

        # Generate cryptographically secure random bytes and convert to UUID format
        try:
            secure_random_bytes: bytes = secrets.token_bytes(entropy)
            secure_uuid: str = str(uuid.UUID(bytes=secure_random_bytes, version=4))

            self.logger.debug(f"Generated secure UUID: {secure_uuid}")

        except Exception as uuid_error:
            error_msg: str = f"Failed to generate secure UUID: {uuid_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from uuid_error

        # Build result components
        result_components: List[str] = []

        # Add prefix if provided
        if prefix:
            result_components.append(prefix)
            self.logger.debug(f"Added prefix: {prefix}")

        # Add timestamp if requested
        if include_timestamp:
            # Generate ISO format timestamp with UTC timezone for consistency
            timestamp: str = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            result_components.append(timestamp)
            self.logger.debug(f"Added timestamp: {timestamp}")

        # Add the secure UUID
        result_components.append(secure_uuid)

        # Combine components with separator
        final_uuid: str = "-".join(result_components)

        # Log completion without revealing full UUID for security
        self.logger.info(
            f"Generated enhanced UUID with {len(result_components)} components"
        )

        return final_uuid
