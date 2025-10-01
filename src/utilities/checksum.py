import hashlib
import os
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Set

# Import all security-related constants from centralized configuration
from config import (
    CHECKSUM_ALGORITHM,
    CHECKSUM_CHUNK_SIZE_BYTES,
    CHECKSUM_HISTORY_FILE,
)


def generate_checksum(logger: Logger, artifact_path: Path) -> str:
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
    logger.debug(
        f"Starting checksum generation for {artifact_path.name} using {CHECKSUM_ALGORITHM.value}"
    )

    # Validate file exists before attempting to process
    if not artifact_path.exists():
        error_msg: str = f"File not found: {artifact_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Initialize the cryptographic hash object with validation
    try:
        hash_object = hashlib.new(CHECKSUM_ALGORITHM.value)
        logger.debug(f"Initialized {CHECKSUM_ALGORITHM.value} hash object successfully")

    except ValueError as algorithm_error:
        # Provide detailed error message with available alternatives
        available_algorithms: List[str] = sorted(hashlib.algorithms_available)
        error_msg = (
            f"Unsupported hash CHECKSUM_ALGORITHM: {CHECKSUM_ALGORITHM.value}. "
            f"Available algorithms: {', '.join(available_algorithms)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from algorithm_error

    # Track processing metrics for performance monitoring
    start_time: float = datetime.now().timestamp()
    bytes_processed: int = 0

    # Process file in chunks using streaming approach for memory efficiency
    try:
        with artifact_path.open("rb") as artifact_handle:
            logger.debug(f"Opened file for reading: {artifact_path.name}")

            # Process file in optimally-sized chunks for I/O performance
            while True:
                # Read next chunk - size optimized for most storage systems
                artifact_chunk: bytes = artifact_handle.read(CHECKSUM_CHUNK_SIZE_BYTES)

                # Check for end of file condition
                if not artifact_chunk:
                    break

                # Update hash with current chunk
                hash_object.update(artifact_chunk)
                bytes_processed += len(artifact_chunk)

                # Log progress for large files to provide user feedback
                if bytes_processed % (CHECKSUM_CHUNK_SIZE_BYTES * 100) == 0:
                    logger.debug(
                        f"Processed {bytes_processed:,} bytes of {artifact_path.name}"
                    )

    except FileNotFoundError as artifact_error:
        error_msg = f"File not found during processing: {artifact_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from artifact_error

    except PermissionError as permission_error:
        error_msg = f"Permission denied reading file: {artifact_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as io_error:
        error_msg = f"I/O error reading file {artifact_path}: {io_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from io_error

    # Calculate final checksum and performance metrics
    final_checksum: str = hash_object.hexdigest()
    processing_duration: float = datetime.now().timestamp() - start_time

    # Log completion with performance metrics for monitoring
    logger.info(
        f"Generated {CHECKSUM_ALGORITHM.value} checksum for {artifact_path.name}: "
        f"{final_checksum[:16]}... ({bytes_processed:,} bytes in {processing_duration:.2f}s)"
    )

    return final_checksum


def load_checksum_history(logger: Logger) -> Set[str]:
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
    logger.debug(f"Loading checksum history from {CHECKSUM_HISTORY_FILE}")

    # Initialize empty checksum collection
    loaded_checksums: Set[str] = set()

    # Check if history file exists before attempting to read
    if not CHECKSUM_HISTORY_FILE.exists():
        logger.info(
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
                    logger.warning(
                        f"Invalid checksum format on line {line_count}: {cleaned_line[:32]}..."
                    )

        # Log successful loading with statistics
        logger.info(
            f"Successfully loaded {len(loaded_checksums)} unique checksums "
            f"from {line_count} lines in history file"
        )

    except PermissionError as permission_error:
        logger.warning(
            f"Permission denied reading checksum history: {permission_error}"
        )

    except OSError as io_error:
        logger.warning(f"I/O error reading checksum history: {io_error}")

    except Exception as unexpected_error:
        logger.warning(f"Unexpected error loading checksum history: {unexpected_error}")

    return loaded_checksums


def save_checksum(logger: Logger, checksum: str) -> None:
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
        interruption of processing processors. The checksum will still be
        available in memory for the current session.
    """
    # Log the checksum save operation for audit trail
    logger.debug(f"Saving checksum to history: {checksum[:16]}...")

    # Validate checksum format before writing to prevent corrupt history
    if not checksum:
        error_msg: str = "Cannot save empty checksum to history"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Verify checksum contains only valid hexadecimal characters
    if not all(char in "0123456789abcdefABCDEF" for char in checksum):
        error_msg = f"Invalid checksum format - contains non-hexadecimal characters: {checksum[:32]}..."
        logger.error(error_msg)
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
        logger.debug("Successfully saved checksum to history file")

    except PermissionError as permission_error:
        logger.error(
            f"Permission denied writing to checksum history: {permission_error}"
        )

    except OSError as io_error:
        logger.error(f"I/O error writing to checksum history: {io_error}")

    except Exception as unexpected_error:
        logger.error(f"Unexpected error saving checksum to history: {unexpected_error}")
