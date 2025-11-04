"""
Sanitizer Pipeline Module

A robust file processing pipeline that handles duplicate detection, zero-byte file
identification, and unsupported file type filtering using checksum verification.

This module provides functionality to sanitize a directory of files by:
- Detecting and moving duplicate files based on checksum comparison
- Identifying and moving zero-byte (corrupted/incomplete) files
- Filtering out unsupported file types and password-protected files
- Maintaining a persistent history of processed files
- Generating unique identifiers and profiles for valid artifacts

The sanitization process validates each file's integrity, checks for duplicates using
SHA-256 checksums, and organizes files into appropriate directories based on their
status. Processing statistics and timing information are logged for monitoring.
"""

import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict , List

from tqdm import tqdm

from config import (
	CORRUPTED_ARTIFACTS_DIR ,
	DUPLICATE_ARTIFACTS_DIR ,
	PASSWORD_PROTECTED_ARTIFACTS_DIR ,
	UNSUPPORTED_ARTIFACTS_DIR ,
)
from utilities.checksum import generate_checksum , save_checksum
from utilities.sanitization import (
	is_file_corrupted ,
	is_file_empty ,
	is_password_protected ,
	is_supported_file ,
)


def sanitizing(logger: logging.Logger, source_dir: Path) -> None:
    """
    Sanitize a directory of files by detecting and moving duplicates, corrupted files,
    unsupported file types, and password-protected files.

    This function processes all files in the source directory, checking each file for:
    1. Duplicate content (via checksum comparison)
    2. Corruption or empty files
    3. Unsupported file types
    4. Password protection

    Files that fail any check are moved to appropriate quarantine directories.
    Processing statistics and timing information are logged.

    Args:
        logger: Logger instance for recording processing events and statistics
        source_dir: Path object pointing to the directory containing files to process

    Returns:
        None

    Side Effects:
        - Moves files to quarantine directories (duplicates, corrupted, unsupported, password-protected)
        - Saves checksums of processed files to persistent storage
        - Logs detailed processing information and statistics
    """
    # Record the start time to calculate total processing duration
    start_time = time.time()
    logger.info(f"Starting sanitization process for directory: {source_dir}")

    # Use Path.iterdir() to get all items in directory, filter to only regular files
    # This excludes subdirectories, symlinks, and other non-file items
    unprocessed_artifacts: List[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]

    # Handle empty directory case - exit early if no files to process
    # This prevents unnecessary processing and provides clear feedback
    if not unprocessed_artifacts:
        logger.info("No files found in source directory, sanitization complete")
        return None

    # Sort files by size (smallest first) for faster initial processing feedback
    # Smaller files process faster, giving users immediate progress indication
    # The lambda function retrieves file size in bytes using Path.stat().st_size
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)

    total_files = len(unprocessed_artifacts)
    logger.info(f"Found {total_files} file(s) to process")
    logger.info(f"Files sorted by size for optimal processing order")

    # Initialize statistics tracking using simple counters
    # These track the outcomes of file processing for reporting
    stats = {
        "processed": 0,  # Total files successfully processed
        "duplicates": 0,  # Files identified as duplicates
        "corrupted": 0,  # Files that are corrupted or empty
        "unsupported": 0,  # Files with unsupported types
        "password_protected": 0,  # Files that are password-protected
        "valid": 0,  # Files that passed all checks
    }

    # Track file types for breakdown statistics using defaultdict
    # defaultdict automatically initializes missing keys with int() = 0
    file_type_counts: Dict[str, int] = defaultdict(int)

    # List to store checksums of successfully processed files
    # Used to detect duplicates during current processing session
    processed_checksums: List[str] = []

    logger.info("Beginning file-by-file sanitization checks")

    # Process each artifact file with a progress bar for user feedback
    # tqdm provides a visual progress bar in the console
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing files",
        unit="files",
    ):
        try:
            # Track the file extension for statistics
            # suffix property returns the file extension including the dot (e.g., '.pdf')
            file_ext = artifact.suffix.lower() if artifact.suffix else "no_extension"
            file_type_counts[file_ext] += 1

            logger.debug(
                f"Processing file: {artifact.name} (size: {artifact.stat().st_size} bytes)"
            )

            # Generate SHA-256 checksum for the file to detect duplicates
            # Checksum is a unique fingerprint based on file content
            artifact_checksum = generate_checksum(logger=logger, artifact_path=artifact)
            logger.debug(
                f"Generated checksum for {artifact.name}: {artifact_checksum[:16]}..."
            )

            # Check if this checksum has been seen before (duplicate detection)
            # Duplicates are files with identical content regardless of filename
            if artifact_checksum in processed_checksums:
                logger.info(f"Duplicate detected: {artifact.name}")
                # shutil.move() relocates the file to the duplicates directory
                shutil.move(src=artifact, dst=DUPLICATE_ARTIFACTS_DIR / artifact.name)
                stats["duplicates"] += 1
                logger.debug(
                    f"Moved duplicate file to: {DUPLICATE_ARTIFACTS_DIR / artifact.name}"
                )
                continue  # Skip further processing of duplicate files

            # File is not a duplicate, add checksum to processed list
            processed_checksums.append(artifact_checksum)

            # Check if file is empty (0 bytes) or corrupted (unreadable/invalid format)
            # Empty files may indicate failed downloads or interrupted transfers
            # Corrupted files cannot be reliably processed
            if is_file_empty(file_path=artifact):
                logger.info(f"Empty file detected: {artifact.name}")
                shutil.move(src=artifact, dst=CORRUPTED_ARTIFACTS_DIR / artifact.name)
                stats["corrupted"] += 1
                logger.debug(
                    f"Moved empty file to: {CORRUPTED_ARTIFACTS_DIR / artifact.name}"
                )
                continue

            if is_file_corrupted(file_path=artifact):
                logger.info(f"Corrupted file detected: {artifact.name}")
                shutil.move(src=artifact, dst=CORRUPTED_ARTIFACTS_DIR / artifact.name)
                stats["corrupted"] += 1
                logger.debug(
                    f"Moved corrupted file to: {CORRUPTED_ARTIFACTS_DIR / artifact.name}"
                )
                continue

            # Check if file type is supported using Apache Tika content detection
            # Tika analyzes file content rather than just extension to prevent spoofing
            if not is_supported_file(file_path=artifact):
                logger.info(f"Unsupported file type detected: {artifact.name}")
                shutil.move(src=artifact, dst=UNSUPPORTED_ARTIFACTS_DIR / artifact.name)
                stats["unsupported"] += 1
                logger.debug(
                    f"Moved unsupported file to: {UNSUPPORTED_ARTIFACTS_DIR / artifact.name}"
                )
                continue

            # Check if file is password-protected (encrypted)
            # Password-protected files cannot be automatically processed
            if is_password_protected(file_path=artifact):
                logger.info(f"Password-protected file detected: {artifact.name}")
                shutil.move(
                    src=artifact, dst=PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name
                )
                stats["password_protected"] += 1
                logger.debug(
                    f"Moved password-protected file to: {PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name}"
                )
                continue

            # File passed all validation checks - save its checksum for future reference
            # Persistent checksum storage prevents reprocessing the same files
            save_checksum(logger=logger, checksum=artifact_checksum)
            stats["valid"] += 1
            logger.debug(f"File validated successfully: {artifact.name}")

            stats["processed"] += 1

        except Exception as e:
            # Catch any unexpected errors during file processing
            # Log the error but continue processing remaining files
            logger.error(f"Error processing file {artifact.name}: {str(e)}")
            continue

    # Calculate total processing time by subtracting start time from current time
    elapsed_time = time.time() - start_time

    # Log comprehensive statistics about the sanitization process
    logger.info("Sanitization process completed")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Total files processed: {stats['processed']}/{total_files}")
    logger.info(f"Valid files: {stats['valid']}")
    logger.info(f"Duplicates removed: {stats['duplicates']}")
    logger.info(f"Corrupted/empty files removed: {stats['corrupted']}")
    logger.info(f"Unsupported files removed: {stats['unsupported']}")
    logger.info(f"Password-protected files removed: {stats['password_protected']}")

    # Log file type breakdown for analysis
    # This helps identify patterns in the processed files
    if file_type_counts:
        logger.info("File type breakdown:")
        # Sort file types by count (descending) for easier reading
        for file_type, count in sorted(
            file_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"  {file_type}: {count} file(s)")

    return None
