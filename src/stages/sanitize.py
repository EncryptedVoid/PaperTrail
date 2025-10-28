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

import logging
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
    CORRUPTED_ARTIFACTS_DIR,
    DUPLICATE_ARTIFACTS_DIR,
    PASSWORD_PROTECTED_ARTIFACTS_DIR,
    UNSUPPORTED_ARTIFACTS_DIR,
)
from utilities.checksum import generate_checksum, save_checksum
from utilities.sanitization import (
    is_file_corrupted,
    is_file_empty,
    is_password_protected,
    is_supported_file,
)


def sanitizing(logger: logging.Logger, source_dir: Path) -> None:

    # Discover all files in the source directory
    # Use list comprehension to filter only files (not directories or symlinks)
    unprocessed_artifacts: list[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]
    logger.info(f"Processing directory: {source_dir}")

    # Handle empty directory case - exit early if no files to process
    if not unprocessed_artifacts:
        logger.info("No files found to process")
        return None

    # Sort files by size (smallest first) for faster initial processing feedback
    # This allows users to see progress immediately rather than waiting for large files
    # Lambda function extracts file size in bytes for sorting key
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len( unprocessed_artifacts )} file(s) to process")

    processed_checksums: List[str] = []

    # Process each artifact file with a progress bar for user feedback
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing from duplicates, corruption, and instability",
        unit="artifacts",
    ):
        artifact_checksum = generate_checksum(artifact)
        if artifact_checksum not in processed_checksums:
            processed_checksums.append(
                generate_checksum(logger=logger, artifact_path=artifact)
            )
        else:
            shutil.move(src=artifact, dst=DUPLICATE_ARTIFACTS_DIR)

        if is_file_empty(file_path=artifact) or is_file_corrupted(file_path=artifact):
            shutil.move(src=artifact, dst=CORRUPTED_ARTIFACTS_DIR)
        elif not is_supported_file(file_path=artifact):
            shutil.move(src=artifact, dst=UNSUPPORTED_ARTIFACTS_DIR)
        elif is_password_protected(file_path=artifact):
            shutil.move(src=artifact, dst=PASSWORD_PROTECTED_ARTIFACTS_DIR)

        save_checksum(logger=logger, checksum=artifact_checksum)

    return None
