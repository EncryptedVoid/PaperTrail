import json
import logging
import shutil
import subprocess
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    PDF_ARRANGER_EXE_PATH,
    PROFILE_PREFIX,
)
from utilities.dependancy_ensurance import ensure_pdfarranger, ensure_unpaper
from utilities.document_scanning import professional_scan
from utilities.initial_review_gui import FilePreviewApp


def manual_review(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    archive_dir: Path,
    success_dir: Path,
) -> None:

    ensure_pdfarranger()
    ensure_unpaper()
    # Log sanitization stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("MANUAL REVIEW STAGE")
    logger.info("=" * 80)

    root = tk.Tk()

    destinations = [
        success_dir,
        failure_dir,
        source_dir / "scan",
        source_dir / "embellish",
    ]
    labels = ["[1] KEEP", "[2] DELETE", "[3] SCAN", "[4] PDF EMBELLISHMENT"]

    app = FilePreviewApp(root, source_dir, destinations, labels)

    root.mainloop()

    # Discover all files in the source directory
    # Use list comprehension to filter only files (not directories or symlinks)
    unprocessed_artifacts: list[Path] = [
        item for item in (source_dir / "embellish").iterdir() if item.is_file()
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
    logger.info(f"Found {len( unprocessed_artifacts )} file(s) to process")

    # Process each artifact file with a progress bar for user feedback
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing from duplicates, corruption, and instability",
        unit="artifacts",
    ):

        try:
            # Log the start of processing for this specific artifact
            logger.info(f"Processing artifact: {artifact.name}")

            # Capture the exact timestamp when processing starts (ISO 8601 format)
            start_time: str = datetime.now().isoformat()

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Open a specific PDF
            subprocess.Popen([PDF_ARRANGER_EXE_PATH, artifact])

            # Create initial profile data dictionary with comprehensive metadata
            artifact_profile_data: Dict[str, Any] = {
                "stage_progression_data": {
                    # Track when sanitization started
                    "embellishment_start_timestamp": start_time,
                    # Track when sanitization completed
                    "embellishment_completion_timestamp": datetime.now().isoformat(),
                    # Mark status as completed for this stage
                    "embellishment_status": "completed",
                },
            }

            # Save updated profile back to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory with renamed filename
            # Construct the destination path with new standardized name
            success_location: Path = success_dir / artifact

            # Perform the move operation from source to success directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Successfully processed {artifact}")

            # Open file with UTF-8 encoding to support international characters
            with open(artifact_profile_path, "w", encoding="utf-8") as f:
                # Write JSON with indentation for human readability
                # ensure_ascii=False allows Unicode characters in output
                json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Catch any exception that occurred during processing
            # Create descriptive error message with artifact name and exception details
            error_msg: str = f"Error processing {artifact.name}: {e}"
            # Log error with full traceback for debugging
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory for manual review
            failure_location: Path = failure_dir / artifact

            # Move the failed artifact to failure directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            # Continue to next artifact in the loop
            continue

    # Discover all files in the source directory
    # Use list comprehension to filter only files (not directories or symlinks)
    unprocessed_artifacts: list[Path] = [
        item for item in (source_dir / "scan").iterdir() if item.is_file()
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
    logger.info(f"Found {len( unprocessed_artifacts )} file(s) to process")

    # Process each artifact file with a progress bar for user feedback
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Sanitizing from duplicates, corruption, and instability",
        unit="artifacts",
    ):

        try:
            # Log the start of processing for this specific artifact
            logger.info(f"Processing artifact: {artifact.name}")

            # Capture the exact timestamp when processing starts (ISO 8601 format)
            start_time: str = datetime.now().isoformat()

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            output = professional_scan(artifact, archive_dir)

            # Create initial profile data dictionary with comprehensive metadata
            artifact_profile_data: Dict[str, Any] = {
                "stage_progression_data": {
                    # Track when sanitization started
                    "scanning_start_timestamp": start_time,
                    # Track when sanitization completed
                    "scanning_completion_timestamp": datetime.now().isoformat(),
                    # Mark status as completed for this stage
                    "scanning_status": "completed",
                },
            }

            # Save updated profile back to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory with renamed filename
            # Construct the destination path with new standardized name
            success_location: Path = success_dir / artifact

            # Perform the move operation from source to success directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Successfully processed {artifact}")

            # Open file with UTF-8 encoding to support international characters
            with open(artifact_profile_path, "w", encoding="utf-8") as f:
                # Write JSON with indentation for human readability
                # ensure_ascii=False allows Unicode characters in output
                json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Catch any exception that occurred during processing
            # Create descriptive error message with artifact name and exception details
            error_msg: str = f"Error processing {artifact.name}: {e}"
            # Log error with full traceback for debugging
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory for manual review
            failure_location: Path = failure_dir / artifact

            # Move the failed artifact to failure directory
            # Convert Path objects to strings for shutil.move compatibility
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            # Continue to next artifact in the loop
            continue

    # Log completion of the entire sanitization stage
    logger.info("Sanitization stage completed")
