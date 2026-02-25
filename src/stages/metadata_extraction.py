"""
Metadata Extraction Pipeline Module

A robust pipeline that extracts metadata from artifacts using Apache Tika and saves
the results as JSON artifacts with UUID-based naming.

This module provides functionality to process a directory of artifacts by:
- Extracting comprehensive metadata from each artifact using Apache Tika
- Saving metadata to JSON artifacts with UUID-based naming (UUID.json)
- Processing artifacts in size-sorted order for optimal feedback
- Tracking detailed statistics about extraction success and failures
- Logging timing information and artifact type breakdowns

The extraction process validates each artifact, extracts all available metadata fields,
and saves them to JSON artifacts using the same UUID as the source artifact name. Processing
statistics and timing information are logged for monitoring and analysis.
"""

import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    AUDIO_TYPES,
    PROFILE_PREFIX,
    TIKA_APP_JAR_PATH,
    VIDEO_TYPES,
)
from utilities.dependancy_ensurance import ensure_apache_tika, ensure_ffmpeg


# Good news — the path fix worked! The .pptx extracted successfully. This is a different error.
# The key line is: Cannot run program "env": CreateProcess error=2
# Tika is trying to use an ExternalParser for the .mp4 file, which invokes the Unix env command to shell out to an
# external tool (likely ffmpeg). That command doesn't exist on Windows, so it crashes.
# This is a known Tika limitation on Windows for media files. You have two options:
# Option A: Install ffmpeg and handle gracefully. Install ffmpeg and add it to PATH — but Tika may still fail because
# it uses the Unix env command internally. So you'd also want to catch these failures gracefully rather than treating
# them as hard errors.


def extracting_metadata(logger: logging.Logger, source_dir: Path) -> None:
    """
    Extract metadata from all artifacts in a directory using Apache Tika.

    This function processes all artifacts in the source directory, extracting metadata
    from each artifact and saving the results to JSON artifacts. Files are expected to have
    UUID-based names, and the output JSON artifacts will use the same UUID with a .json
    extension (e.g., artifact 'abc-123' produces 'abc-123.json').

    The function processes artifacts in size order (smallest first) to provide faster
    initial feedback. Comprehensive statistics and timing information are logged
    throughout the process.

    Args:
                    logger: Logger instance for recording processing events and statistics
                    source_dir: Path object pointing to the directory containing artifacts to process

    Returns:
                    None

    Side Effects:
                    - Creates JSON artifacts in the output directory
                    - Logs detailed processing information and statistics
    """

    # ensure_java( )
    ensure_apache_tika()
    ensure_ffmpeg()

    # Log conversion stage header for clear progress tracking
    # This helps distinguish conversion logs from other pipeline stages
    logger.info("=" * 80)
    logger.info("ARTIFACT METADATA EXTRACTION STAGE")
    logger.info("=" * 80)

    logger.info(f"Starting metadata extraction process for directory: {source_dir}")

    # Ensure output directory exists, create if necessary
    # exist_ok=True prevents error if directory already exists
    # parents=True creates parent directories if needed
    ARTIFACT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifact Profile directory validated and ready")

    # Use Path.iterdir() to get all items in directory, filter to only regular artifacts
    # This excludes subdirectories, symlinks, and other non-artifact items
    unprocessed_artifacts: List[Path] = [
        item for item in source_dir.iterdir() if item.is_file()
    ]

    # Handle empty directory case - exit early if no artifacts to process
    # This prevents unnecessary processing and provides clear feedback
    if not unprocessed_artifacts:
        logger.info("No artifacts found in source directory, sanitization skipped")
        return None

    # Sort artifacts by size (smallest first) for faster initial processing feedback
    # Smaller artifacts process faster, giving users immediate progress indication
    # The lambda function retrieves artifact size in bytes using Path.stat().st_size
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)

    total_artifacts = len(unprocessed_artifacts)
    logger.info(f"Found {total_artifacts} artifact(s) to process")
    logger.info(f"Files sorted by size for optimal processing order")

    # Initialize statistics tracking using simple counters
    # These track the outcomes of metadata extraction for reporting
    stats = {
        "successful": 0,  # Files with successful metadata extraction
        "failed": 0,  # Files that failed metadata extraction
    }

    logger.info("Beginning artifact-by-artifact sanitization process")

    # Process each artifact with a progress bar for user feedback
    # tqdm provides a visual progress bar in the console
    for raw_artifact in tqdm(
        unprocessed_artifacts,
        desc="Extracting metadata",
        unit="artifacts",
    ):
        try:
            start_time = time.time()

            logger.info(f"Processing artifact: {raw_artifact.name}	")

            unique_id = uuid.uuid4()

            artifact = raw_artifact.rename(
                raw_artifact.parent
                / f"{ARTIFACT_PREFIX}-{unique_id}{raw_artifact.suffix}"
            )
            logger.debug(
                f"Renaming artifact with UUID4 to avoid collisions: {artifact}	"
            )

            artifact_profile_json = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{unique_id}.json"
            )

            logger.debug(f"Output JSON will be saved to: {artifact_profile_json}")

            if (
                artifact.suffix.strip().strip(".").lower() in AUDIO_TYPES
                or artifact.suffix.strip().strip(".").lower() in VIDEO_TYPES
            ):
                logger.info(
                    f"Starting metadata extraction for {artifact} using Tika JAR: {TIKA_APP_JAR_PATH}"
                )

                artifact_metadata_extraction_cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(artifact),
                ]

            else:
                # Call the Tika metadata extraction function
                # This returns a dictionary of all extracted metadata fields
                # Record the start time to calculate total processing duration

                logger.info(
                    f"Starting metadata extraction for {artifact} using Tika JAR: {TIKA_APP_JAR_PATH}"
                )

                # Construct the command to run Tika
                # java: Invokes the Java runtime
                # -jar: Specifies we're running a JAR artifact
                # --json: Tells Tika to output metadata in JSON format
                artifact_metadata_extraction_cmd = [
                    "java",
                    "-jar",
                    str(TIKA_APP_JAR_PATH),
                    "--json",
                    artifact.as_uri(),
                ]

            try:
                # Execute the Tika command as a subprocess
                # capture_output=True: Captures stdout and stderr for processing
                # text=True: Returns output as string rather than bytes
                # check=True: Raises CalledProcessError if command returns non-zero exit code
                result = subprocess.run(
                    artifact_metadata_extraction_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse the JSON string returned by Tika into a Python dictionary
                # json.loads() converts the JSON string to a dict object
                metadata = json.loads(result.stdout)

                # Count total number of metadata fields extracted
                total_fields = len(metadata)
                logger.info(f"Successfully extracted {total_fields} metadata fields")

                logger.info(
                    f"Writing metadata to output artifact: {artifact_profile_json}"
                )

                # Write the metadata dictionary to a JSON artifact
                # 'w': Open artifact in write mode
                # encoding='utf-8': Use UTF-8 encoding to support international characters
                # indent=2: Pretty-print JSON with 2-space indentation
                # ensure_ascii=False: Allow non-ASCII characters in output
                with open(artifact_profile_json, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Get the size of the output JSON artifact for logging
                logger.info(
                    f"Output JSON artifact size: {artifact_profile_json.stat( ).st_size:,} bytes"
                )

                # Calculate total operation duration
                total_duration = time.time() - start_time
                logger.info(
                    f"Metadata extraction completed successfully in {total_duration:.2f} seconds"
                )

            except subprocess.CalledProcessError as e:
                # Tika process failed (non-zero exit code)
                logger.error(
                    f"Tika extraction process failed with exit code {e.returncode}"
                )
                logger.error(f"Tika error output: {e.stderr}")
                raise RuntimeError(f"Tika extraction failed: {e.stderr}")

            except json.JSONDecodeError as e:
                # Tika output was not valid JSON
                logger.error(f"Failed to parse Tika output as JSON: {e}")
                raise RuntimeError(f"Failed to parse Tika output as JSON: {e}")

            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f"Unexpected error during metadata extraction: {e}")
                raise

            # Mark this artifact as successfully processed
            stats["successful"] += 1

        except FileNotFoundError as e:
            # File or Tika JAR not found during processing
            logger.error(f"File not found error for {artifact.name}: {str( e )}")
            stats["failed"] += 1
            continue

        except RuntimeError as e:
            # Tika extraction or JSON parsing failed
            logger.error(f"Metadata extraction failed for {artifact.name}: {str( e )}")
            stats["failed"] += 1
            continue

        except Exception as e:
            # Catch any unexpected errors during artifact processing
            # Log the error but continue processing remaining artifacts
            logger.error(
                f"Unexpected error processing artifact {artifact.name}: {str( e )}"
            )
            stats["failed"] += 1
            continue

    # Log comprehensive statistics about the extraction process
    logger.info("Metadata extraction process completed")
    logger.info(f"Total Successful Extractions: {stats[ 'successful' ]}")
    logger.info(f"Total Failed Extractions: {stats[ 'failed' ]}")

    return None
