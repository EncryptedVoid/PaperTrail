"""
Metadata Pipeline Module

A robust file processing pipeline that handles comprehensive metadata extraction
from various file types including images, documents, PDFs, office files, audio,
video, archives, and more.

This module provides functionality to extract metadata by:
- Detecting file types and routing to appropriate extractors
- Extracting filesystem metadata (size, dates, permissions)
- Processing image EXIF, IPTC, XMP data, color profiles, and technical details
- Extracting document properties, structure information, and content analysis
- Processing audio/video metadata including codecs, duration, and technical specs
- Analyzing archive contents and compression details
- Extracting code and text file metrics and language detection
- Handling extraction failures gracefully with fallback methods
- Maintaining detailed operation logs and error tracking
- Updating artifact profiles with extracted metadata

Author: Ashiq Gazi
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any , Dict , List , Optional , Tuple

from tqdm import tqdm

from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	PROFILE_PREFIX ,
	TIKA_APP_JAR_PATH ,
)
from utilities.common import move


class MetadataPipeline:
    """
    A metadata extraction pipeline for processing directories of artifacts.

    This class handles the extraction of comprehensive metadata from various file types,
    including filesystem information, image EXIF data, document properties, audio/video
    metadata, archive analysis, and much more. It maintains artifact profiles and
    provides detailed reporting of all operations.

    The pipeline works in the following stages:
    1. Directory validation and file discovery
    2. File type detection and routing
    3. Filesystem metadata extraction
    4. Specialized metadata extraction (images, documents, audio, video, etc.)
    5. Profile data integration and updates
    6. Error handling and fallback processing
    7. Comprehensive reporting and logging
    """

    def __init__(
        self,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the MetadataPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors

        Raises:
            FileNotFoundError: If the Tika JAR file is not found at the specified path
            RuntimeError: If Java is not installed or not accessible
        """
        self.logger: logging.Logger = logger

        # Validate that Tika JAR exists
        if not Path(TIKA_APP_JAR_PATH).exists():
            raise FileNotFoundError(f"Tika JAR not found at: {TIKA_APP_JAR_PATH}")

        # Validate that Java is installed and accessible
        if not self._check_java():
            raise RuntimeError(
                "Java not found. Please install Java 11+ and ensure it's in your PATH"
            )

        # Log Java version for debugging
        java_version = self._get_java_version()
        self.logger.info(f"Java version detected: {java_version}")

    def extract_metadata(
        self, source_dir: Path, failure_dir: Path, success_dir: Path
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Extract metadata and text from all files in a directory and update their profiles.

        This method performs comprehensive metadata extraction:
        1. Validates input directories exist and are accessible
        2. Discovers all artifact files in the source directory
        3. Loads existing artifact profiles for metadata integration
        4. Extracts filesystem and specialized metadata for each file
        5. Extracts text content from each file
        6. Updates artifact profiles with extracted metadata and text
        7. Moves processed files to success or failure directory based on results

        Args:
            source_dir: Directory containing files to extract metadata from
            failure_dir: Directory to move files that fail processing
            success_dir: Directory to move files after successful metadata extraction

        Returns:
            Tuple containing:
                - metadata: Dictionary of metadata from the last successfully processed file,
                           or None if no files were processed
                - text: Extracted text content from the last successfully processed file,
                       or None if no text was extracted

        Raises:
            FileNotFoundError: If source directory does not exist

        Note:
            This method expects files to follow the ARTIFACT-{uuid}.ext naming convention
            and looks for corresponding PROFILE-{uuid}.json files for updates.
        """

        # Discover all artifact files in the source directory
        try:
            unprocessed_artifacts: List[Path] = [
                item
                for item in source_dir.iterdir()
                if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
            ]
        except Exception as e:
            error_msg = f"Failed to scan source directory: {e}"
            self.logger.error(error_msg)
            return None, None

        # Handle empty directory case
        if not unprocessed_artifacts:
            self.logger.info("No artifact files found in source directory")
            return None, None

        # Sort files by size for consistent processing order (smaller files first)
        unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)

        # Log metadata extraction stage header for clear progress tracking
        self.logger.info("=" * 80)
        self.logger.info("METADATA EXTRACTION AND PROFILE UPDATE STAGE")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Found {len(unprocessed_artifacts)} artifact files to process"
        )

        # Track the last successfully processed file's data
        last_metadata: Optional[Dict[str, Any]] = None
        last_text: Optional[str] = None

        # Process each artifact file
        for artifact in tqdm(
            unprocessed_artifacts,
            desc="Extracting technical metadata",
            unit="artifacts",
        ):
            try:
                # Extract UUID from filename for profile lookup
                # Expected format: ARTIFACT-{uuid}.ext
                artifact_id: str = artifact.stem[
                    len(ARTIFACT_PREFIX) + 1 :
                ]  # Remove "ARTIFACT-" prefix
                profile_path: Path = (
                    ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
                )

                # Validate that profile exists for this artifact
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)

                    move(source=artifact, destination=failure_dir)
                    continue

                # Load existing profile data
                profile_data: Dict[str, Any]
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)
                except Exception as e:
                    error_msg = f"Failed to load profile for {artifact.name}: {e}"
                    self.logger.error(error_msg)

                    move(source=artifact, destination=failure_dir)
                    continue

                # Extract metadata using Apache Tika
                self.logger.info(f"Extracting metadata for: {artifact.name}")

                # Extract text content from the file
                self.logger.info(f"Extracting text content for: {artifact.name}")
                extracted_text: Optional[str] = self._extract_text(artifact)

                # Check if extraction was successful
                if not extracted_text:
                    error_msg = f"Metadata extraction failed for {artifact.name}"
                    self.logger.error(error_msg)

                    # Move to failure directory
                    try:
                        artifact.rename(failure_dir / artifact.name)
                        self.logger.info(
                            f"Moved failed file to: {failure_dir / artifact.name}"
                        )
                    except Exception as move_error:
                        error_msg = f"Failed to move {artifact.name}: {move_error}"
                        self.logger.error(error_msg)
                    continue

                # Store extracted text if available
                if extracted_text:
                    profile_data["extracted_text"] = extracted_text
                    profile_data["text_extraction"] = {
                        "success": True,
                        "character_count": len(extracted_text),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    profile_data["text_extraction"] = {
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Update stage tracking
                if "stages" not in profile_data:
                    profile_data["stages"] = {}

                profile_data["stages"]["metadata_extraction"] = {
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }

                # Save updated profile back to disk
                try:
                    with open(profile_path, "w", encoding="utf-8") as f:
                        json.dump(profile_data, f, indent=2, ensure_ascii=False)
                    self.logger.debug(
                        f"Profile updated successfully for: {artifact.name}"
                    )
                except Exception as e:
                    error_msg = f"Failed to save profile for {artifact.name}: {e}"
                    self.logger.error(error_msg)

                # Move successfully processed file to success directory
                try:
                    success_location: Path = success_dir / artifact.name
                    artifact.rename(success_location)
                    self.logger.info(f"Moved processed file to: {success_location}")

                    # Store this file's data as the last successful extraction
                    last_metadata = metadata_result["metadata"]
                    last_text = extracted_text

                except Exception as move_error:
                    error_msg = f"Failed to move {artifact.name} to success directory: {move_error}"
                    self.logger.error(error_msg)

            except Exception as e:
                # Catch-all error handler for unexpected exceptions
                error_msg = f"Unexpected error processing {artifact.name}: {e}"
                self.logger.error(error_msg, exc_info=True)

                # Move failed file to failure directory
                try:
                    artifact.rename(failure_dir / artifact.name)
                    self.logger.info(
                        f"Moved failed file to: {failure_dir / artifact.name}"
                    )
                except Exception as move_error:
                    error_msg = f"Failed to move {artifact.name}: {move_error}"
                    self.logger.error(error_msg)

        # Log completion
        self.logger.info("=" * 80)
        self.logger.info("METADATA EXTRACTION COMPLETE")
        self.logger.info("=" * 80)

        # Return the last successfully processed file's metadata and text
        return last_metadata, last_text

    def _check_java(self) -> bool:
        """
        Check if Java is installed and accessible.

        Returns:
            bool: True if Java is available, False otherwise
        """
        try:
            subprocess.run(
                [self.java_path, "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _get_java_version(self) -> str:
        """
        Get the installed Java version.

        Returns:
            str: Java version string, or "Unknown" if version cannot be determined
        """
        try:
            result = subprocess.run(
                [self.java_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Java version is typically in stderr, first line
            version_line = result.stderr.split("\n")[0]
            return version_line
        except Exception:
            return "Unknown"

    def _extract_metadata(self, file_path: Path) -> MetadataExtractionResult:
        """
        Extract metadata from a file using Apache Tika.

        This method uses Tika's metadata extraction capabilities with JSON output
        to retrieve comprehensive metadata including:
        - File format and MIME type
        - Author, creation date, modification date
        - Title, subject, keywords
        - Application-specific metadata (EXIF for images, ID3 for audio, etc.)

        Args:
            file_path: Path object pointing to the file to extract metadata from

        Returns:
            MetadataExtractionResult containing:
                - success: Whether extraction succeeded
                - metadata: Dictionary of extracted metadata fields
                - file_path: String path to the processed file
                - error: Error message if extraction failed, None otherwise

        Note:
            Uses Tika's -m (metadata) and -j (JSON output) flags for structured extraction
        """
        # Initialize result structure
        result: MetadataExtractionResult = {
            "success": False,
            "metadata": {},
            "file_path": str(file_path),
            "error": None,
        }

        try:
            # Validate file exists
            if not file_path.exists():
                result["error"] = f"File not found: {file_path}"
                return result

            self.logger.debug(f"Extracting metadata from: {file_path}")

            # Build Tika command for metadata extraction
            # -m: Extract metadata only
            # -j: Output as JSON
            cmd: List[str] = [
                JAVA_PATH,
                "-jar",
                TIKA_APP_JAR_PATH,
                "-m",
                "-j",
                str(file_path),
            ]

            # Execute Tika with timeout to prevent hanging on large files
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Check if Tika execution was successful
            if process.returncode != 0:
                result["error"] = (
                    f"Tika error (code {process.returncode}): {process.stderr}"
                )
                return result

            # Parse JSON output from Tika
            output: Any = json.loads(process.stdout)

            # Handle different output formats from Tika
            if isinstance(output, dict):
                # Single document - use directly
                result["metadata"] = output
                result["success"] = True
            elif isinstance(output, list) and len(output) > 0:
                # Multiple documents (e.g., from archive) - use first
                result["metadata"] = output[0]
                result["success"] = True
            else:
                result["error"] = f"Unexpected output type: {type(output)}"

        except json.JSONDecodeError as e:
            result["error"] = f"JSON parse error: {e}"
            self.logger.error(f"Failed to parse Tika JSON output for {file_path}: {e}")
        except subprocess.TimeoutExpired:
            result["error"] = "Timeout (>120 seconds)"
            self.logger.error(f"Tika extraction timeout for {file_path}")
        except Exception as e:
            result["error"] = f"Error: {str(e)}"
            self.logger.error(
                f"Unexpected error during metadata extraction for {file_path}: {e}"
            )

        return result

    def _extract_text(self, file_path: Path) -> Optional[str]:
        """
        Extract text content from a file using Apache Tika.

        This method uses Tika's text extraction capabilities to retrieve
        readable text content from various file formats including:
        - Documents (PDF, Word, Excel, PowerPoint)
        - Images with OCR
        - HTML and XML files
        - Plain text files
        - And many more formats

        Args:
            file_path: Path object pointing to the file to extract text from

        Returns:
            Extracted text as string if successful, None if extraction fails

        Note:
            Uses Tika's --text flag for full text extraction
        """
        try:
            # Validate file exists
            if not file_path.exists():
                self.logger.warning(f"File not found for text extraction: {file_path}")
                return None

            self.logger.debug(f"Extracting text content from: {file_path}")

            # Build Tika command for text extraction
            # --text: Extract full text content
            cmd: List[str] = [
                JAVA_PATH,
                "-jar",
                TIKA_APP_JAR_PATH,
                "--text",
                str(file_path),
            ]

            # Execute Tika with timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            # Return extracted text if successful, None otherwise
            if process.returncode == 0:
                extracted_text: str = process.stdout.strip()
                self.logger.debug(
                    f"Successfully extracted {len(extracted_text)} characters from {file_path}"
                )
                return extracted_text if extracted_text else None
            else:
                self.logger.warning(
                    f"Text extraction failed for {file_path}: {process.stderr}"
                )
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"Text extraction timeout for {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            return None
