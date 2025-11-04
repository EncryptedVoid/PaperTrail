"""
Metadata Extraction Pipeline Module

A robust pipeline that extracts metadata from files using Apache Tika and saves
the results as JSON files with UUID-based naming.

This module provides functionality to process a directory of files by:
- Extracting comprehensive metadata from each file using Apache Tika
- Saving metadata to JSON files with UUID-based naming (UUID.json)
- Processing files in size-sorted order for optimal feedback
- Tracking detailed statistics about extraction success and failures
- Logging timing information and file type breakdowns

The extraction process validates each file, extracts all available metadata fields,
and saves them to JSON files using the same UUID as the source file name. Processing
statistics and timing information are logged for monitoring and analysis.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utilities.extract_metadata import extract_metadata


def extracting_metadata(
    logger: logging.Logger, source_dir: Path, output_dir: Path
) -> None:
    """
    Extract metadata from all files in a directory using Apache Tika.

    This function processes all files in the source directory, extracting metadata
    from each file and saving the results to JSON files. Files are expected to have
    UUID-based names, and the output JSON files will use the same UUID with a .json
    extension (e.g., file 'abc-123' produces 'abc-123.json').

    The function processes files in size order (smallest first) to provide faster
    initial feedback. Comprehensive statistics and timing information are logged
    throughout the process.

    Args:
        logger: Logger instance for recording processing events and statistics
        source_dir: Path object pointing to the directory containing files to process
        output_dir: Path object pointing to the directory where JSON files will be saved

    Returns:
        None

    Side Effects:
        - Creates JSON files in the output directory
        - Logs detailed processing information and statistics
    """
    # Record the start time to calculate total processing duration
    start_time = time.time()
    logger.info(f"Starting metadata extraction process for directory: {source_dir}")
    logger.info(f"Output directory for JSON files: {output_dir}")

    # Ensure output directory exists, create if necessary
    # exist_ok=True prevents error if directory already exists
    # parents=True creates parent directories if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory validated and ready")

    # Use Path.iterdir() to get all items in directory, filter to only regular files
    # This excludes subdirectories, symlinks, and other non-file items
    files_to_process: List[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]

    # Handle empty directory case - exit early if no files to process
    # This prevents unnecessary processing and provides clear feedback
    if not files_to_process:
        logger.info("No files found in source directory, extraction complete")
        return None

    # Sort files by size (smallest first) for faster initial processing feedback
    # Smaller files process faster, giving users immediate progress indication
    # The lambda function retrieves file size in bytes using Path.stat().st_size
    files_to_process.sort(key=lambda p: p.stat().st_size)

    total_files = len(files_to_process)
    logger.info(f"Found {total_files} file(s) to process")
    logger.info(f"Files sorted by size for optimal processing order")

    # Initialize statistics tracking using simple counters
    # These track the outcomes of metadata extraction for reporting
    stats = {
        "processed": 0,  # Total files attempted
        "successful": 0,  # Files with successful metadata extraction
        "failed": 0,  # Files that failed metadata extraction
        "total_fields_extracted": 0,  # Sum of all metadata fields across files
    }

    # Track file types for breakdown statistics using defaultdict
    # defaultdict automatically initializes missing keys with int() = 0
    file_type_counts: Dict[str, int] = defaultdict(int)

    # Track cumulative size of processed files for statistics
    total_bytes_processed = 0

    logger.info("Beginning file-by-file metadata extraction")

    # Process each file with a progress bar for user feedback
    # tqdm provides a visual progress bar in the console
    for file_path in tqdm(
        files_to_process,
        desc="Extracting metadata",
        unit="files",
    ):
        try:
            # Track the file extension for statistics
            # suffix property returns the file extension including the dot (e.g., '.pdf')
            file_ext = file_path.suffix.lower() if file_path.suffix else "no_extension"
            file_type_counts[file_ext] += 1

            # Get file size for statistics tracking
            # stat() returns file statistics, st_size is the size in bytes
            file_size = file_path.stat().st_size
            total_bytes_processed += file_size

            logger.debug(
                f"Processing file: {file_path.name} (size: {file_size:,} bytes, type: {file_ext})"
            )

            # Construct output JSON path using the same UUID name as the input file
            # file_path.stem gets the filename without extension (the UUID)
            # We append .json to create the output filename
            output_json_path = output_dir / f"{file_path.stem}.json"

            logger.debug(f"Output JSON will be saved to: {output_json_path}")

            # Call the Tika metadata extraction function
            # This returns a dictionary of all extracted metadata fields
            metadata = extract_metadata(
                file_path=file_path, output_json_path=output_json_path, logger=logger
            )

            # Count the number of metadata fields extracted for statistics
            # len() on dictionary returns the number of key-value pairs
            fields_extracted = len(metadata)
            stats["total_fields_extracted"] += fields_extracted

            logger.debug(
                f"Extracted {fields_extracted} metadata fields from {file_path.name}"
            )

            # Mark this file as successfully processed
            stats["successful"] += 1
            stats["processed"] += 1

        except FileNotFoundError as e:
            # File or Tika JAR not found during processing
            logger.error(f"File not found error for {file_path.name}: {str(e)}")
            stats["failed"] += 1
            stats["processed"] += 1
            continue

        except RuntimeError as e:
            # Tika extraction or JSON parsing failed
            logger.error(f"Metadata extraction failed for {file_path.name}: {str(e)}")
            stats["failed"] += 1
            stats["processed"] += 1
            continue

        except Exception as e:
            # Catch any unexpected errors during file processing
            # Log the error but continue processing remaining files
            logger.error(f"Unexpected error processing file {file_path.name}: {str(e)}")
            stats["failed"] += 1
            stats["processed"] += 1
            continue

    # Calculate total processing time by subtracting start time from current time
    elapsed_time = time.time() - start_time

    # Calculate average metadata fields per file for additional insight
    # Avoid division by zero if no files were successfully processed
    avg_fields_per_file = (
        stats["total_fields_extracted"] / stats["successful"]
        if stats["successful"] > 0
        else 0
    )

    # Calculate processing speed in files per second
    files_per_second = total_files / elapsed_time if elapsed_time > 0 else 0

    # Log comprehensive statistics about the extraction process
    logger.info("Metadata extraction process completed")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Processing speed: {files_per_second:.2f} files/second")
    logger.info(f"Total files processed: {stats['processed']}/{total_files}")
    logger.info(f"Successful extractions: {stats['successful']}")
    logger.info(f"Failed extractions: {stats['failed']}")
    logger.info(
        f"Total data processed: {total_bytes_processed:,} bytes ({total_bytes_processed / 1024 / 1024:.2f} MB)"
    )
    logger.info(f"Total metadata fields extracted: {stats['total_fields_extracted']}")
    logger.info(f"Average fields per file: {avg_fields_per_file:.1f}")

    # Log file type breakdown for analysis
    # This helps identify patterns in the processed files
    if file_type_counts:
        logger.info("File type breakdown:")
        # Sort file types by count (descending) for easier reading
        # The lambda function extracts the count (x[1]) for sorting
        for file_type, count in sorted(
            file_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            # Calculate percentage of total files for each type
            percentage = (count / total_files) * 100
            logger.info(f"  {file_type}: {count} file(s) ({percentage:.1f}%)")

    return None
