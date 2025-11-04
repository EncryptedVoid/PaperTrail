"""
Apache Tika Metadata Extractor Module

This module provides functionality to extract metadata from files using Apache Tika.
It processes a file, retrieves all available metadata fields, and saves them to a JSON file.

Usage:
    from pathlib import Path
    import logging
    from tika_metadata_extractor import extract_metadata

    logger = logging.getLogger(__name__)
    file_path = Path("document.pdf")
    output_json = Path("metadata.json")

    extract_metadata(file_path, output_json, logger)
"""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

# Constant: Path to Apache Tika JAR file
# Update this path to match your Tika installation location
TIKA_JAR_PATH = "/path/to/tika-app.jar"


def extract_metadata(file_path: Path, output_json_path: Path, logger) -> Dict[str, Any]:
    """
    Extract metadata from a file using Apache Tika and save to JSON.

    This function calls Apache Tika via subprocess to extract all available metadata
    from the specified file, then saves the results as formatted JSON.

    Args:
                    file_path: Path object pointing to the file to extract metadata from
                    output_json_path: Path object specifying where to save the JSON output
                    logger: Logger instance for recording operation details and progress

    Returns:
                    Dictionary containing all extracted metadata fields

    Raises:
                    FileNotFoundError: If the input file or Tika JAR doesn't exist
                    RuntimeError: If Tika extraction fails or output cannot be parsed
    """
    # Record the start time to calculate total processing duration
    start_time = time.time()

    logger.info(f"Starting metadata extraction for file: {file_path}")

    # Convert TIKA_JAR_PATH string to Path object for consistent path handling
    tika_jar = Path(TIKA_JAR_PATH)

    # Validate that the input file exists on the filesystem
    if not file_path.exists():
        logger.error(f"Input file does not exist: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Log basic file information
    file_size = file_path.stat().st_size
    logger.info(f"Input file size: {file_size:,} bytes")

    # Validate that the Tika JAR file exists at the specified location
    if not tika_jar.exists():
        logger.error(f"Tika JAR file does not exist: {tika_jar}")
        raise FileNotFoundError(f"Tika JAR not found: {tika_jar}")

    logger.info(f"Using Tika JAR: {tika_jar}")

    # Construct the command to run Tika
    # java: Invokes the Java runtime
    # -jar: Specifies we're running a JAR file
    # --json: Tells Tika to output metadata in JSON format
    cmd = ["java", "-jar", str(tika_jar), "--json", str(file_path)]

    logger.info("Executing Tika extraction process")

    # Record timing for the Tika subprocess execution
    tika_start = time.time()

    try:
        # Execute the Tika command as a subprocess
        # capture_output=True: Captures stdout and stderr for processing
        # text=True: Returns output as string rather than bytes
        # check=True: Raises CalledProcessError if command returns non-zero exit code
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Calculate how long the Tika extraction took
        tika_duration = time.time() - tika_start
        logger.info(f"Tika extraction completed in {tika_duration:.2f} seconds")

        # Parse the JSON string returned by Tika into a Python dictionary
        # json.loads() converts the JSON string to a dict object
        metadata = json.loads(result.stdout)

        # Count total number of metadata fields extracted
        total_fields = len(metadata)
        logger.info(f"Successfully extracted {total_fields} metadata fields")

        # Analyze the types of values in the metadata for statistics
        # Counter creates a frequency count of value types
        type_counts = Counter()
        for key, value in metadata.items():
            # Get the type name (e.g., 'str', 'int', 'list', 'dict')
            type_name = type(value).__name__
            type_counts[type_name] += 1

        # Log breakdown of metadata field types
        logger.info(f"Metadata type breakdown: {dict(type_counts)}")

        # Log sample of metadata keys for verification (first 5 keys)
        sample_keys = list(metadata.keys())[:5]
        logger.info(f"Sample metadata fields: {', '.join(sample_keys)}")

        logger.info(f"Writing metadata to output file: {output_json_path}")

        # Write the metadata dictionary to a JSON file
        # 'w': Open file in write mode
        # encoding='utf-8': Use UTF-8 encoding to support international characters
        # indent=2: Pretty-print JSON with 2-space indentation
        # ensure_ascii=False: Allow non-ASCII characters in output
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Get the size of the output JSON file for logging
        output_size = output_json_path.stat().st_size
        logger.info(f"Output JSON file size: {output_size:,} bytes")

        # Calculate total operation duration
        total_duration = time.time() - start_time
        logger.info(
            f"Metadata extraction completed successfully in {total_duration:.2f} seconds"
        )

        # Return the metadata dictionary for potential further processing
        return metadata

    except subprocess.CalledProcessError as e:
        # Tika process failed (non-zero exit code)
        logger.error(f"Tika extraction process failed with exit code {e.returncode}")
        logger.error(f"Tika error output: {e.stderr}")
        raise RuntimeError(f"Tika extraction failed: {e.stderr}")

    except json.JSONDecodeError as e:
        # Tika output was not valid JSON
        logger.error(f"Failed to parse Tika output as JSON: {e}")
        logger.error(f"Tika raw output: {result.stdout[:500]}")  # Log first 500 chars
        raise RuntimeError(f"Failed to parse Tika output as JSON: {e}")

    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during metadata extraction: {e}")
        raise
