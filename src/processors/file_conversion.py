"""
Conversion Pipeline Module

A robust file conversion pipeline that handles file type detection, quality enhancement,
and format conversion for various media types including images, videos, audio, and documents.

This module provides functionality to convert files by:
- Detecting file type through extension and content analysis using Magika
- Converting files to standardized formats with quality enhancement
- Archiving original files before conversion for backup purposes
- Handling unsupported conversions by moving files to failure directory
- Maintaining proper error handling and logging throughout the process
- Updating artifact profiles with conversion metadata and results

Author: Ashiq Gazi
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from magika import Magika
from magika.types import MagikaResult
from tqdm import tqdm

from config import (
    ARCHIVE_TYPES,
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    AUDIO_TYPES,
    CODE_TYPES,
    DOCUMENT_TYPES,
    IMAGE_TYPES,
    MIN_FILE_TYPE_CONF_SCORE,
    PROFILE_PREFIX,
    TEXT_TYPES,
    VIDEO_TYPES,
)


def convert(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    archive_dir: Path,
    success_dir: Path,
) -> None:
    """
    Convert files to standardized formats based on detected file types.

    This method performs comprehensive file conversion by detecting file types using
    Magika content analysis, routing files to appropriate converters based on type,
    archiving original files before conversion, and moving converted files to the
    success directory. Files that fail type detection or conversion are moved to
    the failure directory.

    The conversion process includes:
    - File discovery and size-based sorting (smallest first for faster feedback)
    - Profile loading and validation for each artifact
    - Content-based file type detection using Magika with confidence scoring
    - Original file archiving before conversion for backup purposes
    - Type-specific conversion routing (documents, images, videos, audio, archives)
    - Profile updates with conversion metadata and results
    - File moving to success or failure directories based on conversion results
    - Comprehensive error logging with full exception details

    Supported file types and conversions:
    - Documents: Converted to PDF format
    - Images: Converted to PNG format
    - Videos: Converted to MP4 format
    - Audio: Converted to MP3 format
    - Archives: Converted to 7Z format
    - Text: Passed through without conversion
    - Code: Passed through without conversion

    Args:
            logger: Logger instance for tracking operations and errors
            source_dir: Directory containing artifact files to process. Files must follow
                    the ARTIFACT-{uuid}.ext naming convention
            failure_dir: Directory to move files that fail type detection or conversion.
                    Files are moved here when validation or conversion fails
            archive_dir: Directory to store original files before conversion for backup
                    purposes. All files are copied here before conversion attempts
            success_dir: Directory to move files after successful conversion and profile
                    updates. Converted files are moved here with their new extensions

    Returns:
            None. This method processes files in-place and moves them to appropriate
            directories based on conversion results.

    Raises:
            FileNotFoundError: If source directory or artifact profile does not exist
            ValueError: If profile JSON is corrupted or file type confidence is too low
            TypeError: If file type detection does not meet minimum confidence score

    Note:
            - Files are processed in order of size (smallest first) for faster feedback
            - Requires corresponding PROFILE-{uuid}.json files in ARTIFACT_PROFILES_DIR
            - Original files are archived before conversion for backup purposes
            - File type detection uses Magika with minimum confidence threshold
            - Unsupported file types are passed through without conversion
            - Text and code files are not converted but still processed
            - Naming conflicts in destination directories are automatically resolved
            - All conversion failures are logged with full exception details (exc_info=True)
    """

    # ============================================================================
    # INITIALIZATION AND SETUP
    # ============================================================================

    # Log conversion stage header for clear progress tracking
    # This helps distinguish conversion logs from other pipeline stages
    logger.info("=" * 80)
    logger.info("FILE TYPE DETECTION AND CONVERSION STAGE")
    logger.info("=" * 80)

    # Validate source directory exists before attempting to process files
    # Early validation prevents wasted processing time
    if not source_dir.exists():
        error_msg: str = f"Source directory does not exist: {source_dir}"
        logger.error(error_msg, exc_info=True)
        raise FileNotFoundError(error_msg)

    # ============================================================================
    # FILE DISCOVERY
    # ============================================================================

    # Discover all artifact files in the source directory
    # Only process files (not directories) that match the artifact naming convention
    # Expected format: ARTIFACT-{uuid}.{extension}
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]
    logger.info(f"Processing directory: {source_dir}")

    # Handle empty directory case gracefully
    # No need to initialize Magika or continue processing if no files found
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return

    # ============================================================================
    # MAGIKA INITIALIZATION
    # ============================================================================

    # Initialize Magika for content-based file type detection
    # Magika analyzes file content (not just extensions) for accurate type detection
    # This is more reliable than extension-based detection which can be easily spoofed
    magika: Magika = Magika()
    logger.debug("Initialized Magika file type detector")

    # ============================================================================
    # FILE SORTING AND PREPARATION
    # ============================================================================

    # Sort files by size (smallest first) for faster initial processing feedback
    # This allows users to see progress immediately rather than waiting for large files
    # UX improvement: small files complete quickly, giving immediate feedback
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} file(s) to process")

    # ============================================================================
    # MAIN PROCESSING LOOP
    # ============================================================================

    # Process each artifact file with progress bar
    # tqdm provides visual progress feedback to users
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Converting artifacts",
        unit="artifacts",
    ):
        try:
            # Log the start of processing for this artifact
            logger.info(f"Processing artifact: {artifact.name}")
            # Record start timestamp for performance tracking
            start_time: str = datetime.now().isoformat()

            # ====================================================================
            # PROFILE LOADING AND VALIDATION
            # ====================================================================

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            # We strip the prefix and extension to get just the UUID
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]

            # Construct the path to the corresponding profile JSON file
            # Profile contains metadata about the artifact's processing history
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            # Every artifact must have a corresponding profile
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            # This contains all metadata from previous pipeline stages
            artifact_profile_data: Dict[str, Any]
            try:
                # Read and parse the JSON profile file
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                # Handle corrupted JSON files specifically
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                # Handle any other file reading errors
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # ====================================================================
            # FILE TYPE DETECTION
            # ====================================================================

            # Detect file type using Magika content analysis
            # Magika uses deep learning to analyze file content, not just extension
            logger.debug(f"Detecting file type for: {artifact.name}")
            result: MagikaResult = magika.identify_path(Path(artifact))

            # Validate confidence score meets minimum threshold
            # Low confidence means the file type couldn't be reliably determined
            # This prevents incorrect conversions based on uncertain type detection
            if result.score < MIN_FILE_TYPE_CONF_SCORE:
                error_msg: str = (
                    f"Type detection confidence too low for {artifact.name}: "
                    f"{result.score} < {MIN_FILE_TYPE_CONF_SCORE}"
                )
                logger.warning(error_msg)
                raise TypeError(error_msg)

            # Extract detected file type label
            # Convert to lowercase for consistent comparison with type constants
            file_type: str = result.output.ct_label.lower()
            logger.info(
                f"Detected type for {artifact.name}: {file_type} (confidence: {result.score})"
            )

            # ====================================================================
            # FILE ARCHIVING
            # ====================================================================

            # Archive original file before conversion
            # This provides a backup in case conversion fails or produces unwanted results
            # Users can always recover the original file from the archive
            archive_location: Path = archive_dir / artifact.name
            shutil.copy(str(artifact), str(archive_location))
            logger.debug(f"Archived original file to: {archive_location}")

            # ====================================================================
            # CONVERSION ROUTING
            # ====================================================================

            # Initialize conversion tracking variables
            # These track what conversion (if any) was performed
            converted_extension: str = (
                artifact.suffix
            )  # Default: keep original extension
            conversion_performed: bool = (
                False  # Track if we actually converted the file
            )

            # Route to appropriate converter based on detected file type
            # Each type has its own specialized conversion function

            if file_type in DOCUMENT_TYPES:
                # Convert documents to PDF for standardization
                # PDF is universal and preserves formatting
                logger.debug(f"Converting document: {artifact.name}")
                _convert_document(artifact, "pdf")
                converted_extension = ".pdf"
                conversion_performed = True

            elif file_type in IMAGE_TYPES:
                # Convert images to PNG for lossless quality
                # PNG supports transparency and high quality
                logger.debug(f"Converting image: {artifact.name}")
                _convert_image(artifact, "png")
                converted_extension = ".png"
                conversion_performed = True

            elif file_type in VIDEO_TYPES:
                # Convert videos to MP4 for broad compatibility
                # MP4 is universally supported and efficient
                logger.debug(f"Converting video: {artifact.name}")
                _convert_video(artifact, "mp4")
                converted_extension = ".mp4"
                conversion_performed = True

            elif file_type in AUDIO_TYPES:
                # Convert audio to MP3 for broad compatibility
                # MP3 is universally supported with good quality at high bitrates
                logger.debug(f"Converting audio: {artifact.name}")
                _convert_audio(artifact, "mp3")
                converted_extension = ".mp3"
                conversion_performed = True

            elif file_type in ARCHIVE_TYPES:
                # Convert archives to 7Z for better compression
                # 7Z provides excellent compression ratios
                logger.debug(f"Converting archive: {artifact.name}")
                _convert_archive(artifact, "7z")
                converted_extension = ".7z"
                conversion_performed = True

            elif file_type in EMAIL_TYPES:
                # Convert email files to standard EML format
                # EML is the standard MIME email format
                logger.debug(f"Converting email: {artifact.name}")
                _convert_email(artifact, "eml")
                converted_extension = ".eml"
                conversion_performed = True

            elif file_type in TEXT_TYPES:
                # Text files don't need conversion
                # They're already in a readable, portable format
                logger.debug(f"Text file, no conversion needed: {artifact.name}")

            elif file_type in CODE_TYPES:
                # Code files don't need conversion
                # Source code should remain in its original format
                logger.debug(f"Code file, no conversion needed: {artifact.name}")

            else:
                # Log unsupported file types
                # These files will be passed through without conversion
                logger.warning(
                    f"Unsupported file type: {file_type} for {artifact.name}"
                )

            # ====================================================================
            # PROFILE UPDATE
            # ====================================================================

            # Update profile with conversion metadata
            # This records what happened during the conversion stage

            # Ensure the stage_progression_data structure exists
            # Create it if this is the first stage to update the profile
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            # Add comprehensive conversion stage data
            # This metadata is crucial for tracking and auditing
            artifact_profile_data["stage_progression_data"]["conversion"] = {
                "status": "completed",  # Mark this stage as successfully completed
                "timestamp": datetime.now().isoformat(),  # When conversion finished
                "detected_file_type": file_type,  # What type Magika detected
                "detection_confidence": result.score,  # Confidence level of detection
                "conversion_performed": conversion_performed,  # Whether we converted
                "original_extension": artifact.suffix,  # Original file extension
                "converted_extension": converted_extension,  # New extension after conversion
                "start_timestamp": start_time,  # When processing started
            }

            # Save updated profile back to disk
            # This persists the conversion metadata for future pipeline stages
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    # Use indent=2 for human-readable JSON
                    # ensure_ascii=False allows Unicode characters
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                # Profile update failures are critical - they prevent audit trail
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # ====================================================================
            # FILE RELOCATION
            # ====================================================================

            # Prepare new filename with converted extension if conversion was performed
            if conversion_performed:
                # Use the new extension from the conversion
                new_name: str = artifact.stem + converted_extension
            else:
                # Keep the original filename unchanged
                new_name: str = artifact.name

            # Move artifact to success directory
            # This indicates successful processing
            success_location: Path = success_dir / new_name

            # Handle naming conflicts in the success directory
            # If a file with the same name already exists, append a counter
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                # Keep incrementing counter until we find an available name
                while success_location.exists():
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / conflict_name
                    counter += 1
                logger.debug(f"Resolved naming conflict: {success_location.name}")

            # Perform the move operation to success directory
            # This completes the conversion process for this artifact
            shutil.move(str(artifact), str(success_location))
            logger.info(
                f"Successfully processed {artifact.name} -> {success_location.name}"
            )

        except Exception as e:
            # ====================================================================
            # ERROR HANDLING
            # ====================================================================

            # Log the error with full exception details
            # exc_info=True includes the full stack trace
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory
            # This allows for manual inspection and debugging
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            # Same logic as success directory to prevent overwrites
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                # Keep incrementing counter until we find an available name
                while failure_location.exists():
                    conflict_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / conflict_name
                    counter += 1
                logger.debug(
                    f"Resolved naming conflict in failure directory: {failure_location.name}"
                )

            # Move the failed artifact to failure directory
            # Don't raise the exception - continue processing other files
            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue  # Continue with next artifact

    # Log completion of the entire conversion stage
    logger.info("Conversion stage completed")


# ==============================================================================
# IMAGE CONVERSION FUNCTIONS
# ==============================================================================


def _convert_image(file_path: Path, to_format: str) -> None:
    """
    Convert image files preserving original quality and resolution.
    Uses lossless or high-quality compression settings.

    This function uses ImageMagick (via Wand) to perform high-quality conversions.
    Quality settings are optimized for each output format to minimize loss.

    Args:
            file_path: Path to the image file to convert
            to_format: Target format (e.g., 'png', 'jpg', 'webp', 'tiff')

    Note:
            - Original dimensions are preserved (no resizing)
            - Quality settings are format-specific for optimal results
            - Original file is replaced with converted version in-place
    """
    from wand.image import Image

    # Create output path with new extension
    output_path = file_path.with_suffix(f".{to_format}")

    # Open the image using ImageMagick
    with Image(filename=str(file_path)) as img:
        # Get original dimensions and properties for logging/debugging
        original_width = img.width
        original_height = img.height
        original_format = img.format

        # Set output format
        # This tells ImageMagick what format to save as
        img.format = to_format

        # Quality settings based on format
        # Each format has different quality characteristics
        if to_format.lower() == "png":
            # PNG is lossless, use maximum compression
            # Quality 100 means best compression without quality loss
            img.compression_quality = 100
        elif to_format.lower() in ["jpg", "jpeg"]:
            # Use very high quality for JPEG (95+ to minimize loss)
            # 98 provides near-lossless quality with good file size
            img.compression_quality = 98
        elif to_format.lower() == "webp":
            # WebP with near-lossless quality
            # 95 provides excellent quality with better compression than JPEG
            img.compression_quality = 95
        elif to_format.lower() == "tiff":
            # TIFF lossless
            # 100 means no compression (or lossless if compression is used)
            img.compression_quality = 100

        # Preserve original resolution - no upscaling or downscaling
        # We don't call any resize methods, so dimensions remain unchanged
        # This ensures we maintain the original image quality

        # Save the converted image to the output path
        img.save(filename=str(output_path))

    # Replace original with converted
    # This is a two-step process to ensure atomicity
    file_path.unlink()  # Delete the original file
    output_path.rename(
        file_path.with_suffix(f".{to_format}")
    )  # Rename converted to original name


# ==============================================================================
# VIDEO CONVERSION FUNCTIONS
# ==============================================================================


def _convert_video(file_path: Path, to_format: str) -> None:
    """
    Convert video files preserving original quality and resolution.
    Uses high-quality codecs with settings that maintain source quality.

    This function uses FFmpeg to perform high-quality video conversions.
    The CRF (Constant Rate Factor) setting of 18 provides visually lossless quality.

    Args:
            file_path: Path to the video file to convert
            to_format: Target format (e.g., 'mp4', 'mkv', 'webm')

    Note:
            - Uses H.264 codec for broad compatibility
            - CRF 18 provides visually lossless quality
            - Audio is encoded to AAC at high bitrate
            - Original file is replaced with converted version in-place
    """
    import ffmpeg

    # Probe the input file to get its properties
    # This helps us preserve source quality settings
    try:
        # Get detailed information about the video file
        probe = ffmpeg.probe(str(file_path))

        # Extract video stream information (resolution, bitrate, etc.)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

        # Extract audio stream information (may not exist for silent videos)
        audio_info = next(
            (s for s in probe["streams"] if s["codec_type"] == "audio"), None
        )

        # Get source properties for quality preservation
        width = int(video_info["width"])
        height = int(video_info["height"])

        # Get source bitrate if available
        # This helps us maintain or exceed the source quality
        source_bitrate = video_info.get("bit_rate")
        if source_bitrate:
            source_bitrate = int(source_bitrate)

    except Exception as e:
        # If probe fails, use conservative defaults
        # This ensures conversion still works even if we can't read properties
        source_bitrate = None

    # Create output path with new extension
    output_path = file_path.with_suffix(f".{to_format}")

    # Build ffmpeg command with quality preservation
    stream = ffmpeg.input(str(file_path))

    # Video encoding settings - preserve quality
    video_params = {
        "vcodec": "libx264",  # H.264 codec - widely compatible
        "preset": "slow",  # Slower = better quality at same file size
        "crf": 18,  # 18 = visually lossless (0-51 scale, 18-23 is high quality)
        # Lower CRF = better quality, 18 is near-perfect
    }

    # If source has high bitrate, preserve it
    # This ensures we don't downgrade high-quality sources
    if source_bitrate and source_bitrate > 5000000:  # > 5 Mbps
        video_params["video_bitrate"] = source_bitrate

    # Audio encoding settings - preserve quality
    audio_params = {
        "acodec": "aac",  # AAC codec - widely compatible
        # Use 320k for videos with audio, 192k for others (conservative)
        "audio_bitrate": "320k" if audio_info else "192k",  # High quality audio
    }

    # Apply settings to the stream
    stream = ffmpeg.output(
        stream,
        str(output_path),
        **video_params,  # Unpack video parameters
        **audio_params,  # Unpack audio parameters
    )

    # Run conversion
    # quiet=True suppresses verbose output
    # overwrite_output=True allows overwriting existing files
    ffmpeg.run(stream, quiet=True, overwrite_output=True)

    # Replace original with converted
    file_path.unlink()  # Delete original
    output_path.rename(file_path.with_suffix(f".{to_format}"))  # Rename converted


# ==============================================================================
# AUDIO CONVERSION FUNCTIONS
# ==============================================================================


def _convert_audio(file_path: Path, to_format: str) -> None:
    """
    Convert audio files preserving original quality.
    Uses high bitrate or lossless compression based on source.

    This function uses FFmpeg to perform high-quality audio conversions.
    Bitrate is set to 320kbps for lossy formats (maximum quality for MP3).

    Args:
            file_path: Path to the audio file to convert
            to_format: Target format (e.g., 'mp3', 'flac', 'aac', 'wav', 'ogg')

    Note:
            - Sample rate is preserved from source
            - Uses maximum bitrate for lossy formats (320k)
            - FLAC and WAV provide lossless quality
            - Original file is replaced with converted version in-place
    """
    import ffmpeg

    # Probe the input file to get audio properties
    try:
        # Get detailed information about the audio file
        probe = ffmpeg.probe(str(file_path))

        # Extract audio stream information
        audio_info = next(s for s in probe["streams"] if s["codec_type"] == "audio")

        # Get source properties for quality preservation
        # Sample rate affects audio quality and should be preserved
        sample_rate = int(audio_info.get("sample_rate", 44100))

        # Get source bitrate if available
        source_bitrate = audio_info.get("bit_rate")
        if source_bitrate:
            source_bitrate = int(source_bitrate)

    except Exception:
        # Use high-quality defaults if probe fails
        # 48kHz is professional audio quality
        sample_rate = 48000
        source_bitrate = None

    # Create output path with new extension
    output_path = file_path.with_suffix(f".{to_format}")
    stream = ffmpeg.input(str(file_path))

    # Audio encoding settings based on format
    # Each format has different characteristics and use cases
    if to_format.lower() == "mp3":
        # High quality MP3
        audio_params = {
            "acodec": "libmp3lame",  # Best MP3 encoder
            "audio_bitrate": "320k",  # Maximum MP3 bitrate (highest quality)
            "ar": sample_rate,  # Preserve sample rate from source
        }
    elif to_format.lower() == "flac":
        # FLAC is lossless - perfect reproduction of source
        audio_params = {
            "acodec": "flac",
            "compression_level": 8,  # Maximum compression (still lossless)
            # Higher compression = smaller file, same quality
            "ar": sample_rate,
        }
    elif to_format.lower() in ["aac", "m4a"]:
        # High quality AAC - better than MP3 at same bitrate
        audio_params = {
            "acodec": "aac",
            "audio_bitrate": "320k",  # Very high quality AAC
            "ar": sample_rate,
        }
    elif to_format.lower() == "wav":
        # WAV is uncompressed - largest file size, no loss
        audio_params = {
            "acodec": "pcm_s16le",  # 16-bit PCM (standard CD quality)
            "ar": sample_rate,
        }
    elif to_format.lower() == "ogg":
        # High quality Ogg Vorbis - open source, good quality
        audio_params = {
            "acodec": "libvorbis",
            "audio_bitrate": "320k",
            "ar": sample_rate,
        }
    else:
        # Default high quality settings for unknown formats
        audio_params = {
            "audio_bitrate": "320k",
            "ar": sample_rate,
        }

    # Apply audio parameters to the stream
    stream = ffmpeg.output(stream, str(output_path), **audio_params)

    # Run the conversion
    ffmpeg.run(stream, quiet=True, overwrite_output=True)

    # Replace original with converted
    file_path.unlink()  # Delete original
    output_path.rename(file_path.with_suffix(f".{to_format}"))  # Rename converted


# ==============================================================================
# DOCUMENT CONVERSION FUNCTIONS
# ==============================================================================


def _convert_document(file_path: Path, to_format: str) -> None:
    """
    Convert document files preserving formatting and embedded images.
    Uses high-quality PDF rendering settings.

    This function uses Pandoc to convert various document formats to PDF.
    XeLaTeX engine provides better Unicode and font support than pdflatex.

    Args:
            file_path: Path to the document file to convert
            to_format: Target format (typically 'pdf')

    Note:
            - Uses XeLaTeX engine for better Unicode support
            - DPI set to 300 for high-quality image rendering
            - Preserves formatting, styles, and embedded media
            - Original file is replaced with converted version in-place
    """
    import pypandoc

    # Create output path with new extension
    output_path = file_path.with_suffix(f".{to_format}")

    # Extra args for quality preservation
    extra_args = []

    if to_format == "pdf":
        # Configure Pandoc for high-quality PDF output
        extra_args = [
            "--pdf-engine=xelatex",  # Better Unicode and font support than pdflatex
            # Handles international characters and complex scripts
            "--dpi=300",  # High DPI for images (300 is print quality)
            # Lower DPI would result in blurry images
        ]

    # Perform the conversion using Pandoc
    # Pandoc is a universal document converter supporting many formats
    pypandoc.convert_file(
        str(file_path), to_format, outputfile=str(output_path), extra_args=extra_args
    )

    # Replace original with converted
    file_path.unlink()  # Delete original
    output_path.rename(file_path.with_suffix(f".{to_format}"))  # Rename converted


# ==============================================================================
# ARCHIVE CONVERSION FUNCTIONS
# ==============================================================================


def _convert_archive(file_path: Path, to_format: str) -> None:
    """
    Convert archive files to the specified format using py7zr.

    This function extracts archives to a temporary directory, then re-archives
    them in the target format. Supports ZIP, TAR, 7Z formats.

    Args:
        file_path: Path to the archive file to convert
        to_format: Target format (e.g., '7z', 'zip')

    Note:
        - Uses temporary directory for extraction
        - Preserves directory structure within archives
        - Original file is replaced with converted version in-place
    """
    import py7zr
    import tempfile

    # Create temporary extraction directory
    # This is automatically cleaned up when the context exits
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract based on source format
        # Different archive formats require different extraction methods
        if file_path.suffix in [".zip"]:
            # Handle ZIP archives using standard zipfile module
            import zipfile

            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Extract all contents to temporary directory
                zip_ref.extractall(temp_path)

        elif file_path.suffix in [".tar", ".tar.gz", ".tgz", ".tar.bz2"]:
            # Handle TAR archives (including compressed variants)
            import tarfile

            # 'r:*' automatically detects compression type (gz, bz2, xz, etc.)
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(temp_path)

        elif file_path.suffix in [".7z"]:
            # Handle 7Z archives
            with py7zr.SevenZipFile(file_path, "r") as archive:
                archive.extractall(temp_path)
        else:
            # Unsupported archive format - exit without conversion
            return

        # Create new archive in target format
        output_path = file_path.with_suffix(f".{to_format}")

        if to_format == "7z":
            # Create 7Z archive
            with py7zr.SevenZipFile(output_path, "w") as archive:
                # Recursively add all files from temp directory
                # rglob("*") finds all files recursively
                for item in temp_path.rglob("*"):
                    if item.is_file():
                        # Store with relative path to preserve directory structure
                        archive.write(item, item.relative_to(temp_path))

        elif to_format == "zip":
            # Create ZIP archive
            import zipfile

            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
                # ZIP_DEFLATED provides compression
                for item in temp_path.rglob("*"):
                    if item.is_file():
                        # Store with relative path to preserve directory structure
                        zip_ref.write(item, item.relative_to(temp_path))

        # Replace original with converted
        file_path.unlink()  # Delete original
        output_path.rename(file_path.with_suffix(f".{to_format}"))  # Rename converted


# ==============================================================================
# PLACEHOLDER CONVERSION FUNCTIONS
# ==============================================================================


def _convert_text(file_path: Path, to_format: str) -> None:
    """
    Convert text files to the specified format using direct manipulation or Pandoc.

    Args:
            file_path: Path to the text file to convert
            to_format: Target format (e.g., 'md', 'html', 'txt')

    Note:
            This is a placeholder function. Implementation should handle text encoding
            conversion and format transformation.
    """
    # TODO: Implement text conversion logic
    # Could use Pandoc for markdown/HTML conversion
    # Could handle encoding conversions (UTF-8, ASCII, etc.)
    pass


def _convert_code(file_path: Path, to_format: str) -> None:
    """
    Convert source code files with formatting or syntax highlighting.

    Args:
            file_path: Path to the code file to convert
            to_format: Target format (e.g., 'html', 'pdf' with highlighting)

    Note:
            This is a placeholder function. Implementation should handle syntax
            highlighting, formatting, and code beautification.
    """
    # TODO: Implement code conversion logic
    # Could use Pygments for syntax highlighting
    # Could use Black/autopep8 for Python formatting
    # Could convert to HTML with syntax highlighting
    pass


# ==============================================================================
# EMAIL CONVERSION FUNCTIONS
# ==============================================================================


def _convert_email(file_path: Path, to_format: str = "eml") -> None:
    """
    Convert email files to EML format (standard MIME format).

    Supports:
    - MSG (Outlook) -> EML
    - MBOX (Unix mailbox) -> EML (extracts all emails)
    - PST (Outlook data file) -> EML (extracts all emails)
    - EML -> EML (validation)

    For multi-message formats (MBOX, PST), creates individual EML files
    with naming pattern: {original_stem}_email_{index:04d}.eml

    Args:
            file_path: Path to the email file to convert
            to_format: Target format (default: 'eml')

    Note:
            - Multi-message formats create multiple EML files
            - First email is renamed to match original artifact name
            - Preserves email headers, body (HTML and plain text), and attachments
            - Original file is deleted after successful conversion
    """
    import email
    from email.message import EmailMessage

    # Get file extension to determine source format
    suffix = file_path.suffix.lower()

    # ============================================================================
    # MSG FORMAT (Outlook single message)
    # ============================================================================
    if suffix == ".msg":
        import extract_msg

        # Create output path for converted EML file
        output_path = file_path.with_suffix(f".{to_format}")

        # Open and parse the MSG file
        msg = extract_msg.Message(str(file_path))

        # Create EML format email
        # EmailMessage is the modern Python email format
        eml = EmailMessage()

        # Add standard email headers
        # Use empty string as fallback if header is missing
        eml["Subject"] = msg.subject or ""
        eml["From"] = msg.sender or ""
        eml["To"] = msg.to or ""
        eml["Date"] = msg.date or ""
        eml["Cc"] = msg.cc or ""

        # Set body (prefer HTML, fallback to plain text)
        # Many emails have both HTML and plain text versions
        if msg.htmlBody:
            # Add plain text version first (required)
            eml.set_content(msg.body or "", subtype="plain")
            # Add HTML version as alternative (preferred by email clients)
            eml.add_alternative(msg.htmlBody, subtype="html")
        elif msg.body:
            # Only plain text available
            eml.set_content(msg.body)

        # Handle attachments
        # MSG files can contain multiple attachments
        for attachment in msg.attachments:
            # Get attachment filename (prefer long name, fallback to short name)
            filename = (
                attachment.longFilename or attachment.shortFilename or "attachment"
            )

            # Add attachment to email
            eml.add_attachment(
                attachment.data,  # Binary attachment data
                maintype="application",  # Generic application type
                subtype="octet-stream",  # Generic binary data
                filename=filename,
            )

        # Write to EML file
        # EML files are just text representations of MIME messages
        with open(output_path, "wb") as f:
            f.write(eml.as_bytes())

        # Clean up
        msg.close()

        # Replace original with converted
        file_path.unlink()  # Delete original MSG file
        output_path.rename(file_path.with_suffix(f".{to_format}"))

    # ============================================================================
    # MBOX FORMAT (Unix mailbox with multiple emails)
    # ============================================================================
    elif suffix == ".mbox":
        import mailbox

        # Open MBOX file
        # MBOX stores multiple emails in a single file
        mbox = mailbox.mbox(str(file_path))
        email_count = len(mbox)

        # Validate that MBOX contains at least one email
        if email_count == 0:
            raise ValueError(f"MBOX file is empty: {file_path}")

        # Extract all emails to individual EML files
        # This makes each email independently accessible
        base_name = file_path.stem
        parent_dir = file_path.parent

        # Iterate through all messages in the MBOX
        for idx, msg in enumerate(mbox):
            # Create unique filename for each email
            # Use zero-padded index for proper sorting
            eml_name = f"{base_name}_email_{idx:04d}.eml"
            eml_path = parent_dir / eml_name

            # Write message to EML file
            # Messages are already in MIME format
            with open(eml_path, "wb") as f:
                f.write(msg.as_bytes())

        # Clean up
        mbox.close()

        # Remove original MBOX file
        file_path.unlink()

        # Rename the first extracted email to match original name
        # This maintains the original artifact naming convention
        # Other emails keep their numbered names
        first_email = parent_dir / f"{base_name}_email_0000.eml"
        if first_email.exists():
            first_email.rename(file_path.with_suffix(".eml"))

    # ============================================================================
    # PST FORMAT (Outlook data file with folders and multiple emails)
    # ============================================================================
    elif suffix == ".pst":
        import pypff

        # Open PST file
        # PST files contain entire Outlook mailboxes with folder structure
        pst = pypff.file()
        pst.open(str(file_path))

        # Get root folder of the PST
        # PST files are organized in a folder hierarchy
        root = pst.get_root_folder()

        base_name = file_path.stem
        parent_dir = file_path.parent

        # Use list to maintain counter across recursive calls
        # This ensures unique numbering across all folders
        email_counter = [0]  # List allows modification in nested function

        def extract_emails_from_folder(folder, folder_path=""):
            """
            Recursively extract emails from PST folders.

            PST files have a tree structure of folders (like Outlook's folder pane).
            We need to traverse this tree to find all emails.

            Args:
                    folder: Current folder object to process
                    folder_path: String path showing folder hierarchy (for logging)
            """
            # Process all messages in current folder
            for item in folder.sub_messages:
                try:
                    # Create EML message object
                    eml = EmailMessage()

                    # Add headers
                    # PST items may have incomplete header information
                    if item.subject:
                        eml["Subject"] = item.subject
                    if item.sender_name:
                        eml["From"] = item.sender_name
                    if item.get_transport_headers():
                        # Parse transport headers if available
                        # These contain To, Cc, Bcc, etc.
                        headers = item.get_transport_headers()
                        if headers:
                            eml["To"] = headers.get("To", "")
                            eml["Cc"] = headers.get("Cc", "")

                    # Add body (both plain text and HTML if available)
                    body = item.plain_text_body
                    html = item.html_body

                    if body:
                        # Decode body handling potential encoding issues
                        eml.set_content(body.decode("utf-8", errors="ignore"))
                    if html:
                        # Add HTML alternative
                        eml.add_alternative(
                            html.decode("utf-8", errors="ignore"), subtype="html"
                        )

                    # Add attachments
                    # PST items can have multiple attachments
                    for attachment_idx in range(item.number_of_attachments):
                        attachment = item.get_attachment(attachment_idx)
                        # Get attachment name or create default
                        att_name = attachment.name or f"attachment_{attachment_idx}"
                        # Read attachment binary data
                        att_data = attachment.read_buffer(attachment.size)

                        # Add to email
                        eml.add_attachment(
                            att_data,
                            maintype="application",
                            subtype="octet-stream",
                            filename=att_name,
                        )

                    # Save EML file with unique numbered name
                    eml_name = f"{base_name}_email_{email_counter[0]:04d}.eml"
                    eml_path = parent_dir / eml_name

                    with open(eml_path, "wb") as f:
                        f.write(eml.as_bytes())

                    # Increment counter for next email
                    email_counter[0] += 1

                except Exception as e:
                    # Log but continue processing other emails
                    # Some emails may be corrupted or have missing data
                    print(f"Warning: Failed to extract email {email_counter[0]}: {e}")
                    email_counter[0] += 1

            # Recursively process subfolders
            # PST folders can be nested arbitrarily deep
            for subfolder_idx in range(folder.number_of_sub_folders):
                subfolder = folder.get_sub_folder(subfolder_idx)
                # Get folder name or create default
                subfolder_name = subfolder.name or f"folder_{subfolder_idx}"
                # Build hierarchical path for logging
                new_path = (
                    f"{folder_path}/{subfolder_name}" if folder_path else subfolder_name
                )
                # Recursive call to process subfolder
                extract_emails_from_folder(subfolder, new_path)

        # Start extraction from root folder
        # This initiates the recursive traversal of the entire PST structure
        extract_emails_from_folder(root)

        # Clean up PST file handle
        pst.close()

        # Verify that we extracted at least one email
        if email_counter[0] == 0:
            raise ValueError(f"No emails found in PST file: {file_path}")

        # Remove original PST file
        file_path.unlink()

        # Rename the first extracted email to match original name
        # This maintains the artifact naming convention
        first_email = parent_dir / f"{base_name}_email_0000.eml"
        if first_email.exists():
            first_email.rename(file_path.with_suffix(".eml"))

    # ============================================================================
    # EML FORMAT (already in target format, just validate)
    # ============================================================================
    elif suffix == ".eml":
        # Validate EML format by attempting to parse it
        # This ensures the file is valid MIME format
        with open(file_path, "rb") as f:
            try:
                email.message_from_binary_file(f)
            except Exception as e:
                # Invalid EML format
                raise ValueError(f"Invalid EML file: {e}")

        # Already in EML format, no conversion needed
        pass

    else:
        # Unsupported email format
        raise ValueError(f"Unsupported email format: {suffix}")
