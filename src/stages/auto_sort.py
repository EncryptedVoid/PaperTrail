"""
Auto-sorting module for categorizing and organizing artifact files.

This module provides automated file sorting functionality that analyzes artifacts
and moves them to appropriate destination directories based on file type, content,
and metadata. Supports various file types including bookmarks, code, Anki decks,
backup codes, books, and financial documents.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Dict , List

from tqdm import tqdm

from config import (
	ANKI_DIR ,
	BITWARDEN_DIR ,
	CALIBRE_LIBRARY_DIR ,
	DIGITAL_ASSET_MANAGEMENT_DIR ,
	FIREFLYIII_DIR ,
	GITLAB_DIR ,
	LINKWARDEN_DIR ,
	PERFORMANCE_PORTFOLIO_DIR ,
	UNSUPPORTED_ARTIFACTS_DIR ,
)
from utilities.automatic_sorting import (
	is_anki_deck ,
	is_backup_codes_file ,
	is_book ,
	is_bookmark_file ,
	is_code ,
	is_financial_document ,
	is_supported ,
)
from utilities.visual_processor import VisualProcessor


def automatically_sorting(
    logger: logging.Logger, visual_processor: VisualProcessor, source_dir: Path
):
    """
    Automatically sort and organize artifact files from a source directory.

    This function discovers all files in the source directory, analyzes each one
    to determine its type, and moves it to the appropriate destination directory.
    Provides detailed logging and statistics about the sorting process.

    Args:
                    logger: Logger instance for recording process information and errors
                    visual_processor: VisualProcessor instance for analyzing image/PDF content
                    source_dir: Path object pointing to the directory containing artifacts to sort

    Returns:
                    None: Function completes silently after processing all artifacts

    Processing Order:
                    Files are sorted by size (smallest first) to provide faster initial feedback
                    and help identify processing issues early in the workflow.
    """
    # Record start time for performance tracking
    start_time = time.time()
    logger.info(f"Starting automatic sorting process for directory: {source_dir}")

    # Initialize statistics dictionary to track sorting results
    # Keys represent destination categories, values are file counts
    stats: Dict[str, int] = {
        "bookmarks": 0,
        "code": 0,
        "anki_decks": 0,
        "backup_codes": 0,
        "books": 0,
        "financial_docs": 0,
        "unsupported": 0,
        "errors": 0,
    }

    # Discover all artifact files in the source directory
    # Using list() on iterdir() materializes the generator into a list for processing
    try:
        unprocessed_artifacts: List[Path] = [
            item for item in source_dir.iterdir() if item.is_file()
        ]
    except Exception as e:
        logger.error(f"Failed to read source directory {source_dir}: {e}")
        return None

    # Handle empty directory case - exit early if no artifacts found
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        elapsed_time = time.time() - start_time
        logger.info(f"Sorting process completed in {elapsed_time:.2f} seconds")
        return None

    # Sort files by size for consistent processing order (smaller files first)
    # stat().st_size returns file size in bytes
    # This provides faster initial feedback and helps identify issues early
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Calculate total size of all artifacts for reporting
    total_size_bytes = sum(item.stat().st_size for item in unprocessed_artifacts)
    total_size_mb = total_size_bytes / (1024 * 1024)
    logger.info(f"Total data to process: {total_size_mb:.2f} MB")

    # Process each artifact file with progress tracking
    # tqdm provides console progress bar, desc sets bar label, unit customizes counter
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Auto-sorting artifacts",
        unit="artifacts",
    ):
        try:
            # Log current file being processed with size information
            file_size_kb = artifact.stat().st_size / 1024
            logger.info(f"Processing file: {artifact.name} ({file_size_kb:.2f} KB)")

            # Check if file is an HTML bookmark export
            # is_bookmark_file() analyzes HTML structure for Netscape bookmark format
            if artifact.suffix == ".html" and is_bookmark_file(file_path=artifact):
                logger.info(f"Detected bookmark file: {artifact.name}")
                # shutil.move() performs atomic move operation to destination
                shutil.move(src=artifact, dst=LINKWARDEN_DIR)
                stats["bookmarks"] += 1
                logger.info(f"Moved bookmark file to: {LINKWARDEN_DIR}")

            # Check if file is source code based on file extension
            # is_code() matches extension against CODE_EXTENSIONS from config
            elif is_code(file_path=artifact):
                logger.info(f"Detected code file: {artifact.name}")
                shutil.move(src=artifact, dst=GITLAB_DIR)
                stats["code"] += 1
                logger.info(f"Moved code file to: {GITLAB_DIR}")

            # Check if file is an Anki flashcard deck
            # is_anki_deck() supports .apkg, .colpkg, .anki2, .anki21, and text formats
            elif is_anki_deck(file_path=artifact):
                logger.info(f"Detected Anki deck: {artifact.name}")
                shutil.move(src=artifact, dst=ANKI_DIR)
                stats["anki_decks"] += 1
                logger.info(f"Moved Anki deck to: {ANKI_DIR}")

            # Check if file contains 2FA backup/recovery codes
            # is_backup_codes_file() analyzes content for code patterns
            elif is_backup_codes_file(file_path=artifact):
                logger.info(f"Detected backup codes file: {artifact.name}")
                shutil.move(src=artifact, dst=BITWARDEN_DIR)
                stats["backup_codes"] += 1
                logger.info(f"Moved backup codes to: {BITWARDEN_DIR}")

            # Check if file is a book (EPUB, PDF with ISBN, etc.)
            # is_book() uses metadata analysis and content inspection
            elif is_book(file_path=artifact):
                logger.info(f"Detected book: {artifact.name}")
                shutil.move(src=artifact, dst=CALIBRE_LIBRARY_DIR)
                stats["books"] += 1
                logger.info(f"Moved book to: {CALIBRE_LIBRARY_DIR}")

            # Check if file is a financial document (invoice, receipt, statement)
            # is_financial_document() uses visual_processor for OCR and text analysis
            elif is_financial_document(
                file_path=artifact, visual_processor=visual_processor, logger=logger
            ):
                logger.info(f"Detected financial document: {artifact.name}")

                # Financial documents need to be copied to multiple locations
                # shutil.copy2() preserves metadata (timestamps, permissions)
                logger.info(f"Copying financial document to: {FIREFLYIII_DIR}")
                shutil.copy2(src=artifact, dst=FIREFLYIII_DIR)

                logger.info(
                    f"Copying financial document to: {PERFORMANCE_PORTFOLIO_DIR}"
                )
                shutil.copy2(src=artifact, dst=PERFORMANCE_PORTFOLIO_DIR)

                # Final move to primary storage location
                logger.info(
                    f"Moving financial document to: {DIGITAL_ASSET_MANAGEMENT_DIR}"
                )
                shutil.move(src=artifact, dst=DIGITAL_ASSET_MANAGEMENT_DIR)

                stats["financial_docs"] += 1
                logger.info(
                    f"Completed financial document processing for: {artifact.name}"
                )

            # Check if file type is supported but didn't match specific categories
            # is_supported() checks against known file type lists (EMAIL_TYPES, etc.)
            elif not is_supported(file_path=artifact):
                logger.warning(f"Unsupported file type: {artifact.name}")
                shutil.move(src=artifact, dst=UNSUPPORTED_ARTIFACTS_DIR)
                stats["unsupported"] += 1
                logger.info(f"Moved unsupported file to: {UNSUPPORTED_ARTIFACTS_DIR}")

            else:
                # File is supported but didn't match any category
                # Leave in source directory for manual review
                logger.warning(
                    f"Supported file type but no category match: {artifact.name}"
                )

        except Exception as e:
            # Catch and log any errors during file processing
            # Continue processing remaining files despite errors
            stats["errors"] += 1
            logger.error(f"Error processing {artifact.name}: {e}", exc_info=True)

    # Calculate elapsed time for performance reporting
    elapsed_time = time.time() - start_time

    # Log comprehensive sorting statistics
    logger.info("Sorting process completed")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Total files processed: {len(unprocessed_artifacts)}")

    # Log breakdown of sorted files by category
    logger.info("Sorting statistics by category:")
    logger.info(f"  Bookmarks: {stats['bookmarks']}")
    logger.info(f"  Code files: {stats['code']}")
    logger.info(f"  Anki decks: {stats['anki_decks']}")
    logger.info(f"  Backup codes: {stats['backup_codes']}")
    logger.info(f"  Books: {stats['books']}")
    logger.info(f"  Financial documents: {stats['financial_docs']}")
    logger.info(f"  Unsupported files: {stats['unsupported']}")
    logger.info(f"  Errors encountered: {stats['errors']}")

    # Calculate and log average processing time per file
    if len(unprocessed_artifacts) > 0:
        avg_time = elapsed_time / len(unprocessed_artifacts)
        logger.info(f"Average time per file: {avg_time:.3f} seconds")

    return None
