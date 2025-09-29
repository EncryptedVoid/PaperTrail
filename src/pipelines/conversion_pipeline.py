"""
Conversion Pipeline Module

A robust file conversion pipeline that handles file type detection, quality enhancement,
and format conversion for various media types including images, videos, audio, and documents.

This module provides functionality to convert files by:
- Detecting file type through extension and content analysis
- Converting files to standardized formats with quality enhancement
- Handling unsupported conversions by moving files to review directory
- Maintaining proper error handling and logging throughout the process

Author: Ashiq Gazi
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import ffmpeg
import filetype
import rawpy
from PIL import Image , ImageEnhance
from pillow_heif import register_heif_opener
from tqdm import tqdm

from config import (
	AUDIO_BITRATE ,
	AUDIO_CODEC ,
	AUDIO_SAMPLE_RATE ,
	ENABLE_IMAGE_ENHANCEMENT ,
	ENABLE_UPSCALING ,
	ENABLE_VIDEO_ENHANCEMENT ,
	EXTENSION_MAPPING ,
	IMAGE_CONTRAST_FACTOR ,
	IMAGE_SHARPENING_FACTOR ,
	IMAGE_TARGET_SIZE ,
	LIBREOFFICE_EXECUTABLE_LOCATION ,
	LIBREOFFICE_TIMEOUT ,
	PNG_COMPRESS_LEVEL ,
	REQUIRE_DETECTION_AGREEMENT ,
	USE_CONTENT_DETECTION ,
	VIDEO_4K_THRESHOLD ,
	VIDEO_CODEC ,
	VIDEO_CRF ,
	VIDEO_PIXEL_FORMAT ,
	VIDEO_PRESET ,
	VIDEO_TARGET_1080P ,
	VIDEO_TARGET_4K ,
	VIDEO_UPSCALE_THRESHOLD ,
)

register_heif_opener()


class ConversionPipeline:
    """
    A file conversion pipeline for processing directories of files or single files.

    This class handles the detection and conversion of various file types including
    images, videos, audio files, and documents. Files are converted to standardized
    formats with quality enhancement options while maintaining proper error handling
    and file organization.

    The pipeline works in the following stages:
    1. Path validation and file discovery
    2. Archive management (move original, create working copy)
    3. File type detection (extension and/or content analysis)
    4. Format conversion with quality enhancement
    5. Success/failure handling and cleanup
    """

    def __init__(self, logger: logging.Logger, archive_dir: Path):
        """
        Initialize the ConversionPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
            archive_dir: Directory to store original files and working copies
        """
        self.logger = logger
        self.archive_dir = archive_dir

        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def convert(
        self,
        source_dir: Path,
        failure_dir: Path,
        success_dir: Path,
    ) -> int:
        """
        Convert files in a directory or single file to standardized formats with quality enhancement.

        This method performs a comprehensive conversion process:
        1. Validates input paths exist and are accessible
        2. Discovers all files (either from directory or single file)
        3. Processes each file through conversion pipeline:
            - Moves original to archive for safety
            - Creates working copy for conversion
            - Detects file type using extension and/or content analysis
            - Converts to target format with quality enhancements
            - Handles success/failure cases appropriately
        4. Manages file organization and cleanup

        Args:
            source_dir: Directory containing files to process OR a single file to process
            failure_dir: Directory to move files that cannot be converted
            success_dir: Directory to move successfully converted files

        Returns:
            Number of files successfully converted

        Raises:
            FileNotFoundError: If source_path, failure_dir, or success_dir don't exist
        """

        if not source_dir.exists():
            error_msg = f"Source path does not exist: {source_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not failure_dir.exists() or not failure_dir.is_dir():
            error_msg = (
                f"Failure directory does not exist or is not a directory: {failure_dir}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not success_dir.exists() or not success_dir.is_dir():
            error_msg = (
                f"Success directory does not exist or is not a directory: {success_dir}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Determine if source_path is a directory or single file and get file list
        try:
            if source_dir.is_dir():
                unprocessed_files = [
                    item for item in source_dir.iterdir() if item.is_file()
                ]
                self.logger.info(f"Processing directory: {source_dir}")
            elif source_dir.is_file():
                unprocessed_files = [source_dir]
                self.logger.info(f"Processing single file: {source_dir}")
            else:
                error_msg = (
                    f"Source path is neither a file nor a directory: {source_dir}"
                )
                self.logger.error(error_msg)
                return 0
        except Exception as e:
            error_msg = f"Failed to scan source path: {e}"
            self.logger.error(error_msg)
            return 0

        # Handle empty directory case
        if not unprocessed_files:
            self.logger.info("No files found to process")
            return 0

        # Sort files by size for better progress feedback
        unprocessed_files.sort(key=lambda p: p.stat().st_size)

        # Log conversion stage header
        self.logger.info("=" * 80)
        self.logger.info("FILE CONVERSION AND QUALITY ENHANCEMENT STAGE")
        self.logger.info("=" * 80)
        self.logger.info(f"Found {len(unprocessed_files)} file(s) to convert")

        successful_conversions = 0

        # Process each file through the conversion pipeline
        if len(unprocessed_files) > 1:
            file_iterator = tqdm(
                unprocessed_files,
                desc="Converting files to standardized formats",
                unit="files",
            )
        else:
            file_iterator = unprocessed_files

        for file_path in file_iterator:
            try:
                if self._process_single_file(file_path, failure_dir, success_dir):
                    successful_conversions += 1

            except Exception as e:
                error_msg = f"Unexpected error processing {file_path.name}: {e}"
                self.logger.error(error_msg)
                # Move problematic file to failure directory for manual review
                try:
                    failure_path = failure_dir / file_path.name
                    if file_path.exists():
                        file_path.rename(failure_path)
                        self.logger.info(f"Moved failed file to review: {failure_path}")
                except Exception as move_error:
                    self.logger.error(f"Failed to move problematic file: {move_error}")

        # Generate final summary
        self.logger.info("Conversion process complete:")
        self.logger.info(f"  - {successful_conversions} file(s) successfully converted")
        self.logger.info(
            f"  - {len(unprocessed_files) - successful_conversions} file(s) moved to review"
        )

        return successful_conversions

    def _process_single_file(
        self,
        file_path: Path,
        failure_dir: Path,
        success_dir: Path,
    ) -> bool:
        """
        Process a single file through the complete conversion workflow.

        Workflow stages:
        1. Move original file to archive for safety
        2. Create working copy for conversion
        3. Detect file type using configured detection method
        4. Convert file to target format with quality enhancement
        5. Handle success/failure and cleanup

        Args:
            file_path: Path to the file to be processed
            failure_dir: Directory for files that cannot be converted
            success_dir: Directory for successfully converted files

        Returns:
            True if conversion was successful, False otherwise
        """

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        self.logger.info(f"Starting conversion of: {file_path.name}")

        try:
            # Stage 1: Move original file to archive for safety
            archive_original = self.archive_dir / file_path.name
            self.logger.debug(
                f"Moving file to archive: {file_path} -> {archive_original}"
            )

            # Handle naming conflicts in archive
            if archive_original.exists():
                counter = 1
                base_name = archive_original.stem
                extension = archive_original.suffix
                while archive_original.exists():
                    archive_original = (
                        self.archive_dir / f"{base_name}_{counter}{extension}"
                    )
                    counter += 1

            shutil.move(str(file_path), str(archive_original))

            # Stage 2: Create working copy for conversion
            archive_copy = self.archive_dir / f"temp_{archive_original.name}"
            self.logger.debug(f"Creating working copy: {archive_copy}")
            shutil.copy2(str(archive_original), str(archive_copy))

            # Stage 3: Detect file type using configured method
            file_type = self._detect_file_type(archive_copy)

            if file_type is None:
                self.logger.warning(
                    f"Cannot determine file type for: {archive_copy.name}"
                )
                self._handle_conversion_failure(
                    archive_original, archive_copy, failure_dir
                )
                return False

            # Stage 4: Convert the working copy
            self.logger.debug(
                f"Converting file of type '{file_type}': {archive_copy.name}"
            )
            converted_path = self._convert_file_by_type(file_type, archive_copy)

            if converted_path and converted_path.exists():
                # Stage 5a: Success - move converted file to success directory
                success_path = success_dir / converted_path.name

                # Handle naming conflicts in success directory
                if success_path.exists():
                    counter = 1
                    base_name = success_path.stem
                    extension = success_path.suffix
                    while success_path.exists():
                        success_path = success_dir / f"{base_name}_{counter}{extension}"
                        counter += 1

                shutil.move(str(converted_path), str(success_path))

                # Clean up working copy
                if archive_copy.exists():
                    archive_copy.unlink()

                self.logger.info(
                    f"Successfully converted: {file_path.name} -> {success_path.name}"
                )
                return True
            else:
                # Stage 5b: Conversion failed
                self.logger.warning(f"Conversion failed for: {archive_copy.name}")
                self._handle_conversion_failure(
                    archive_original, archive_copy, failure_dir
                )
                return False

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            # Try to clean up and move to failure directory
            try:
                if file_path.exists():
                    failure_path = failure_dir / file_path.name
                    shutil.move(str(file_path), str(failure_path))
                    self.logger.info(f"Moved failed file to review: {failure_path}")
            except Exception as cleanup_error:
                self.logger.error(f"Failed to clean up after error: {cleanup_error}")
            return False

    def _detect_file_type(self, file_path: Path) -> Optional[str]:
        """
        Detect file type using extension and/or content analysis based on configuration.

        Detection methods:
        - Extension-only: Fast detection based on file extension
        - Content-only: Slower but more accurate detection using file content
        - Both required: Extension and content must agree (most secure)
        - Content preferred: Use content detection with extension fallback

        Args:
            file_path: Path to file for type detection

        Returns:
            Detected file type string or None if detection fails/disagrees
        """

        extension_type = self._detect_by_extension(file_path)

        if not USE_CONTENT_DETECTION:
            self.logger.debug(f"Using extension-only detection: {extension_type}")
            return extension_type

        content_type = self._detect_by_content(file_path)

        if REQUIRE_DETECTION_AGREEMENT:
            # Both extension and content must agree
            if extension_type == content_type and extension_type is not None:
                self.logger.debug(f"Detection agreement: {extension_type}")
                return extension_type
            else:
                self.logger.warning(
                    f"Detection mismatch - extension: {extension_type}, content: {content_type}"
                )
                return None
        else:
            # Prefer content detection, fallback to extension
            detected_type = content_type or extension_type
            detection_method = "content" if content_type else "extension"
            self.logger.debug(f"Using {detection_method} detection: {detected_type}")
            return detected_type

    def _detect_by_extension(self, file_path: Path) -> Optional[str]:
        """
        Detect file type based on file extension.

        Args:
            file_path: Path to file for extension-based detection

        Returns:
            File type string or None if extension is not supported
        """

        extension = file_path.suffix.lower()
        return EXTENSION_MAPPING.get(extension)

    def _detect_by_content(self, file_path: Path) -> Optional[str]:
        """Detect file type by content using magic numbers"""
        try:
            kind = filetype.guess(str(file_path))

            if kind is None:
                return None

            mime_type = kind.mime

            if mime_type.startswith("image/"):
                return "image"
            elif mime_type.startswith("video/"):
                return "video"
            elif mime_type.startswith("audio/"):
                return "audio"
            elif mime_type in [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ]:
                return "document"
            else:
                return None
        except Exception:
            return None

    def _convert_file_by_type(self, file_type: str, file_path: Path) -> Optional[Path]:
        """
        Convert file to target format based on detected type using appropriate converter.

        Args:
            file_type: Detected file type string
            file_path: Path to file to be converted

        Returns:
            Path to converted file or None if conversion failed
        """

        try:
            if file_type == "image":
                return self._convert_image_to_png(file_path)
            elif file_type == "video":
                return self._convert_video_to_mp4(file_path)
            elif file_type == "audio":
                return self._convert_audio_to_mp3(file_path)
            elif file_type == "document":
                return self._convert_document_to_pdf(file_path)
            elif file_type == "video_audio":  # Special case for 3GP files
                return self._convert_3gp_file(file_path)
            else:
                self.logger.error(f"Unsupported file type for conversion: {file_type}")
                return None

        except Exception as e:
            self.logger.error(f"Conversion failed for {file_path.name}: {e}")
            return None

    def _convert_image_to_png(self, file_path: Path) -> Optional[Path]:
        """
        Convert image files to PNG format with optional quality enhancement.

        Supports various image formats including RAW files (CR2, ARW, NEF).
        Applies upscaling, sharpening, and contrast enhancement based on configuration.

        Args:
            file_path: Path to image file to be converted

        Returns:
            Path to converted PNG file or None if conversion failed
        """

        try:
            output_path = file_path.with_suffix(".png")
            self.logger.debug(f"Converting image to PNG: {file_path.name}")

            # Handle RAW files with special processing
            if file_path.suffix.lower() in [".cr2", ".arw", ".nef"]:
                self.logger.debug("Processing RAW image file")
                with rawpy.imread(str(file_path)) as raw:
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=True,
                        output_bps=16,
                    )
                image = Image.fromarray(rgb_array)
            else:
                # Handle standard image formats
                self.logger.debug("Processing standard image file")
                image = Image.open(file_path)

            # Convert to RGB color space if needed
            if image.mode in ["RGBA", "LA"]:
                self.logger.debug(
                    f"Converting {image.mode} to RGB with white background"
                )
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != "RGB":
                self.logger.debug(f"Converting {image.mode} to RGB")
                image = image.convert("RGB")

            # Apply upscaling if image is smaller than target size
            if ENABLE_UPSCALING and ENABLE_IMAGE_ENHANCEMENT:
                width, height = image.size
                if max(width, height) < IMAGE_TARGET_SIZE:
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = IMAGE_TARGET_SIZE
                        new_height = int((height * IMAGE_TARGET_SIZE) / width)
                    else:
                        new_height = IMAGE_TARGET_SIZE
                        new_width = int((width * IMAGE_TARGET_SIZE) / height)

                    self.logger.debug(f"Upscaling image to {new_width}x{new_height}")
                    image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

            # Apply quality enhancements if enabled
            if ENABLE_IMAGE_ENHANCEMENT:
                self.logger.debug("Applying image quality enhancements")

                # Sharpening enhancement
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(IMAGE_SHARPENING_FACTOR)

                # Contrast enhancement
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(IMAGE_CONTRAST_FACTOR)

            # Save as PNG with configured compression
            self.logger.debug(f"Saving PNG with compression level {PNG_COMPRESS_LEVEL}")
            image.save(
                output_path, "PNG", optimize=False, compress_level=PNG_COMPRESS_LEVEL
            )

            return output_path

        except Exception as e:
            self.logger.error(f"Image conversion failed for {file_path.name}: {e}")
            return None

    def _convert_video_to_mp4(self, file_path: Path) -> Optional[Path]:
        """
        Convert video files to MP4 format with optional quality enhancement.

        Applies upscaling and quality settings based on configuration.
        Uses FFmpeg for conversion with optimized settings.

        Args:
            file_path: Path to video file to be converted

        Returns:
            Path to converted MP4 file or None if conversion failed
        """

        try:
            output_path = file_path.with_suffix(".mp4")
            self.logger.debug(f"Converting video to MP4: {file_path.name}")

            # Analyze video properties
            probe = ffmpeg.probe(str(file_path))
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
            width = int(video_info["width"])
            height = int(video_info["height"])

            # Build FFmpeg conversion stream
            stream = ffmpeg.input(str(file_path))

            # Apply upscaling if enabled and video is below threshold
            if ENABLE_UPSCALING and ENABLE_VIDEO_ENHANCEMENT:
                max_dimension = max(width, height)
                if max_dimension < VIDEO_UPSCALE_THRESHOLD:
                    target_height = VIDEO_TARGET_1080P
                    self.logger.debug(f"Upscaling video to {target_height}p")
                    stream = ffmpeg.filter(
                        stream, "scale", -2, target_height, flags="lanczos"
                    )
                elif max_dimension < VIDEO_4K_THRESHOLD:
                    target_height = min(VIDEO_TARGET_4K, height * 2)
                    self.logger.debug(f"Upscaling video to {target_height}p")
                    stream = ffmpeg.filter(
                        stream, "scale", -2, target_height, flags="lanczos"
                    )

            # Configure output with quality settings
            stream = ffmpeg.output(
                stream,
                str(output_path),
                vcodec=VIDEO_CODEC,
                crf=VIDEO_CRF,
                preset=VIDEO_PRESET,
                acodec="aac",
                audio_bitrate=AUDIO_BITRATE,
                **{"pix_fmt": VIDEO_PIXEL_FORMAT},
            )

            # Execute conversion
            self.logger.debug("Starting FFmpeg video conversion")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            return output_path

        except Exception as e:
            self.logger.error(f"Video conversion failed for {file_path.name}: {e}")
            return None

    def _convert_audio_to_mp3(self, file_path: Path) -> Optional[Path]:
        """
        Convert audio files to MP3 format with high quality settings.

        Uses FFmpeg for conversion with configured bitrate and sample rate.

        Args:
            file_path: Path to audio file to be converted

        Returns:
            Path to converted MP3 file or None if conversion failed
        """

        try:
            output_path = file_path.with_suffix(".mp3")
            self.logger.debug(f"Converting audio to MP3: {file_path.name}")

            # Build FFmpeg conversion stream
            stream = ffmpeg.input(str(file_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec=AUDIO_CODEC,
                audio_bitrate=AUDIO_BITRATE,
                ar=AUDIO_SAMPLE_RATE,
            )

            # Execute conversion
            self.logger.debug("Starting FFmpeg audio conversion")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            return output_path

        except Exception as e:
            self.logger.error(f"Audio conversion failed for {file_path.name}: {e}")
            return None

    def _convert_3gp_file(self, file_path: Path) -> Optional[Path]:
        """
        Convert 3GP files by determining if they contain video or audio content.

        3GP files can contain video or audio only. This method analyzes the content
        and routes to appropriate conversion method.

        Args:
            file_path: Path to 3GP file to be converted

        Returns:
            Path to converted file or None if conversion failed
        """

        try:
            self.logger.debug(f"Analyzing 3GP file content: {file_path.name}")
            probe = ffmpeg.probe(str(file_path))

            # Check for video streams to determine conversion path
            has_video = any(s["codec_type"] == "video" for s in probe["streams"])

            if has_video:
                self.logger.debug("3GP contains video content, converting to MP4")
                return self._convert_video_to_mp4(file_path)
            else:
                self.logger.debug("3GP contains audio-only content, converting to MP3")
                return self._convert_audio_to_mp3(file_path)

        except Exception as e:
            self.logger.error(f"3GP analysis failed for {file_path.name}: {e}")
            return None

    def _convert_document_to_pdf(self, file_path: Path) -> Optional[Path]:
        """
        Convert document files to PDF format using LibreOffice.

        Supports various document formats including DOC, DOCX, PPT, PPTX, XLS, XLSX.
        Returns original file if already in PDF format.

        Args:
            file_path: Path to document file to be converted

        Returns:
            Path to converted/original PDF file or None if conversion failed
        """

        # Return original file if already PDF
        if file_path.suffix.lower() == ".pdf":
            self.logger.debug(f"File already in PDF format: {file_path.name}")
            return file_path

        try:
            output_path = file_path.with_suffix(".pdf")
            self.logger.debug(f"Converting document to PDF: {file_path.name}")

            # Build LibreOffice conversion command
            conversion_command = [
                LIBREOFFICE_EXECUTABLE_LOCATION,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(file_path.parent),
                str(file_path),
            ]

            # Execute conversion with timeout
            self.logger.debug("Starting LibreOffice document conversion")
            result = subprocess.run(
                conversion_command,
                capture_output=True,
                text=True,
                timeout=LIBREOFFICE_TIMEOUT,
                check=False,
            )

            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                self.logger.error(f"LibreOffice conversion failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"Document conversion timed out for {file_path.name}")
            return None
        except Exception as e:
            self.logger.error(f"Document conversion failed for {file_path.name}: {e}")
            return None

    def _handle_conversion_failure(
        self,
        original_path: Path,
        copy_path: Path,
        failure_dir: Path,
    ) -> None:
        """
        Handle conversion failure by cleaning up working copy and moving original to failure directory.

        This method ensures proper cleanup when conversion fails, moving the original
        file to the failure directory for manual review and removing temporary files.

        Args:
            original_path: Path to original file in archive
            copy_path: Path to working copy used for conversion
            failure_dir: Directory to move failed files to
        """

        try:
            # Clean up working copy
            if copy_path.exists():
                copy_path.unlink()
                self.logger.debug(f"Removed working copy: {copy_path.name}")

            # Move original file to failure directory for manual review
            failure_path = failure_dir / original_path.name

            # Handle naming conflicts in failure directory
            if failure_path.exists():
                counter = 1
                base_name = failure_path.stem
                extension = failure_path.suffix
                while failure_path.exists():
                    failure_path = failure_dir / f"{base_name}_{counter}{extension}"
                    counter += 1

            shutil.move(str(original_path), str(failure_path))
            self.logger.info(f"Moved failed conversion to review: {failure_path.name}")

        except Exception as e:
            self.logger.error(f"Error handling conversion failure: {e}")
