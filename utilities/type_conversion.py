import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance
import ffmpeg
import rawpy
from pillow_heif import register_heif_opener
from dataclasses import dataclass, field
import magic
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from config import (
    USE_CONTENT_DETECTION,
    REQUIRE_DETECTION_AGREEMENT,
    ENABLE_UPSCALING,
    ENABLE_IMAGE_ENHANCEMENT,
    IMAGE_TARGET_SIZE,
    IMAGE_SHARPENING_FACTOR,
    IMAGE_CONTRAST_FACTOR,
    PNG_COMPRESS_LEVEL,
    ENABLE_VIDEO_ENHANCEMENT,
    VIDEO_UPSCALE_THRESHOLD,
    VIDEO_TARGET_1080P,
    VIDEO_TARGET_4K,
    VIDEO_4K_THRESHOLD,
    VIDEO_CRF,
    VIDEO_PRESET,
    VIDEO_CODEC,
    VIDEO_PIXEL_FORMAT,
    AUDIO_BITRATE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CODEC,
    LIBREOFFICE_TIMEOUT,
)

# Register HEIF opener for HEIC files
register_heif_opener()


class ConversionStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DETECTION_MISMATCH = "detection_mismatch"


@dataclass
class ConversionReport:
    original_file_type: str
    target_file_type: str
    original_extension: str
    target_extension: str
    conversion_time_seconds: float
    status: ConversionStatus
    original_size_bytes: int
    converted_size_bytes: Optional[int]
    original_file_path: Optional[Path] = None  # Final location of original file
    converted_file_path: Optional[Path] = None  # Final location of converted file
    error_message: Optional[str] = None
    quality_enhancements_applied: List[str] = field(default_factory=list)
    detection_method: str = "extension"

    def __post_init__(self):
        if self.quality_enhancements_applied is None:
            self.quality_enhancements_applied = []


class ConversionProcessor:

    # File type mappings
    EXTENSION_MAPPING = {
        # Images
        ".jpeg": "image",
        ".jpg": "image",
        ".png": "image",
        ".heic": "image",
        ".cr2": "image",
        ".arw": "image",
        ".nef": "image",
        ".webp": "image",
        # Videos
        ".mov": "video",
        ".mp4": "video",
        ".webm": "video",
        ".amv": "video",
        ".3gp": "video_audio",  # Special case - need to probe
        # Audio
        ".wav": "audio",
        ".mp3": "audio",
        ".m4a": "audio",
        ".ogg": "audio",
        # Documents
        ".pptx": "document",
        ".doc": "document",
        ".docx": "document",
        ".rtf": "document",
        ".epub": "document",
        ".pub": "document",
        ".djvu": "document",
        ".pdf": "document",
    }

    # Target formats for each type
    TARGET_FORMATS = {
        "image": ".png",
        "video": ".mp4",
        "audio": ".mp3",
        "document": ".pdf",
    }

    def __init__(self, logger, archive_dir: Path, review_dir: Path):
        self.logger = logger
        self.archive_dir = archive_dir
        self.review_dir = review_dir

        self.logger.info(
            f"Initialized converter with archive_dir: {self.archive_dir}, review_dir: {self.review_dir}"
        )

    def process_file(self, file_path: Path) -> ConversionReport:
        """
        Main function to process a file according to the workflow:
        1. Move to archive
        2. Copy in archive
        3. Detect file type (extension + content)
        4. Convert copy with quality enhancement
        5. Handle success/failure

        Returns ConversionReport with all details
        """
        start_time = time.time()
        original_size = file_path.stat().st_size if file_path.exists() else 0

        self.logger.info(f"Starting processing of file: {file_path}")

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return ConversionReport(
                original_file_type="unknown",
                target_file_type="unknown",
                original_extension=file_path.suffix,
                target_extension="",
                conversion_time_seconds=time.time() - start_time,
                status=ConversionStatus.FAILURE,
                original_size_bytes=0,
                converted_size_bytes=None,
                original_file_path=None,  # File not found
                converted_file_path=None,
                error_message=f"File not found: {file_path}",
            )

        try:
            # Step 1: Move file to archive
            archive_original = self.archive_dir / file_path.name
            self.logger.info(
                f"Moving file to archive: {file_path} -> {archive_original}"
            )
            shutil.move(str(file_path), str(archive_original))

            # Step 2: Create copy for conversion
            archive_copy = self.archive_dir / f"copy_{file_path.name}"
            self.logger.info(f"Creating copy for conversion: {archive_copy}")
            shutil.copy2(str(archive_original), str(archive_copy))

            # Step 3: Detect file type
            file_type, detection_method = self._detect_file_type(archive_copy)

            if file_type is None:
                # Detection mismatch - move to review
                self.logger.warning(f"File type detection mismatch for: {archive_copy}")
                review_path = self._handle_conversion_failure(
                    archive_original, archive_copy
                )
                return ConversionReport(
                    original_file_type="unknown",
                    target_file_type="unknown",
                    original_extension=file_path.suffix,
                    target_extension="",
                    conversion_time_seconds=time.time() - start_time,
                    status=ConversionStatus.DETECTION_MISMATCH,
                    original_size_bytes=original_size,
                    converted_size_bytes=None,
                    original_file_path=review_path,  # File moved to review
                    converted_file_path=None,
                    error_message="File extension and content type don't match",
                    detection_method=detection_method,
                )

            # Step 4: Convert the copy
            self.logger.info(
                f"Starting conversion of: {archive_copy} (type: {file_type})"
            )
            converted_path, enhancements = self._convert_file_by_type(
                file_type, archive_copy
            )

            if converted_path and converted_path.exists():
                # Success: Remove the copy, keep converted file
                converted_size = converted_path.stat().st_size
                self.logger.info(f"Conversion successful: {converted_path}")
                if archive_copy.exists():
                    archive_copy.unlink()
                    self.logger.debug(f"Removed temporary copy: {archive_copy}")

                return ConversionReport(
                    original_file_type=file_type,
                    target_file_type=file_type,  # Same logical type, different format
                    original_extension=file_path.suffix,
                    target_extension=self.TARGET_FORMATS.get(
                        file_type, file_path.suffix
                    ),
                    conversion_time_seconds=time.time() - start_time,
                    status=ConversionStatus.SUCCESS,
                    original_size_bytes=original_size,
                    converted_size_bytes=converted_size,
                    original_file_path=archive_original,  # Original file in archive
                    converted_file_path=converted_path,  # Converted file in archive
                    quality_enhancements_applied=enhancements,
                    detection_method=detection_method,
                )
            else:
                # Conversion failed
                self.logger.warning(f"Conversion failed for: {archive_copy}")
                review_path = self._handle_conversion_failure(
                    archive_original, archive_copy
                )
                return ConversionReport(
                    original_file_type=file_type,
                    target_file_type=file_type,
                    original_extension=file_path.suffix,
                    target_extension=self.TARGET_FORMATS.get(
                        file_type, file_path.suffix
                    ),
                    conversion_time_seconds=time.time() - start_time,
                    status=ConversionStatus.FAILURE,
                    original_size_bytes=original_size,
                    converted_size_bytes=None,
                    original_file_path=review_path,  # File moved to review
                    converted_file_path=None,
                    error_message="Conversion process failed",
                    detection_method=detection_method,
                )

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            # Move to review if something went wrong
            review_path = None
            if file_path.exists():
                review_path = self.review_dir / file_path.name
                shutil.move(str(file_path), str(review_path))
                self.logger.info(f"Moved failed file to review: {review_path}")

            return ConversionReport(
                original_file_type="unknown",
                target_file_type="unknown",
                original_extension=file_path.suffix,
                target_extension="",
                conversion_time_seconds=time.time() - start_time,
                status=ConversionStatus.FAILURE,
                original_size_bytes=original_size,
                converted_size_bytes=None,
                original_file_path=review_path,  # File moved to review
                converted_file_path=None,
                error_message=str(e),
            )

    def _detect_file_type(self, file_path: Path) -> tuple[Optional[str], str]:
        """Detect file type using both extension and content"""
        ext_type = self._detect_by_extension(file_path)

        if not USE_CONTENT_DETECTION:
            return ext_type, "extension"

        content_type = self._detect_by_content(file_path)

        if REQUIRE_DETECTION_AGREEMENT:
            # Both must agree
            if ext_type == content_type and ext_type is not None:
                return ext_type, "both"
            else:
                return None, "mismatch"  # Disagreement - move to review
        else:
            # Prefer content detection, fallback to extension
            return content_type or ext_type, "content" if content_type else "extension"

    def _detect_by_extension(self, file_path: Path) -> Optional[str]:
        """Detect file type by extension"""
        ext = file_path.suffix.lower()
        return self.EXTENSION_MAPPING.get(ext)

    def _detect_by_content(self, file_path: Path) -> Optional[str]:
        """Detect file type by content using magic numbers"""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)

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

    def _convert_file_by_type(
        self, file_type: str, file_path: Path
    ) -> tuple[Optional[Path], List[str]]:
        """Convert file based on detected type using strategy pattern"""

        if file_type == "image":
            return self._convert_image_to_png(file_path)
        elif file_type == "video":
            return self._convert_video_to_mp4(file_path)
        elif file_type == "audio":
            return self._convert_audio_to_mp3(file_path)
        elif file_type == "document":
            return self._convert_document_to_pdf(file_path)
        elif file_type == "video_audio":  # Special case for 3GP
            return self._convert_3gp(file_path)
        else:
            self.logger.error(f"Unsupported file type: {file_type}")
            return None, []

    def _convert_image_to_png(
        self, file_path: Path
    ) -> tuple[Optional[Path], List[str]]:
        """Convert images to PNG with quality enhancement"""
        enhancements = []

        try:
            output_path = file_path.with_suffix(".png")
            self.logger.info(f"Converting image to PNG: {file_path} -> {output_path}")

            if file_path.suffix.lower() in [".cr2", ".arw", ".nef"]:
                # Handle RAW files
                self.logger.debug(f"Processing RAW file: {file_path}")
                with rawpy.imread(str(file_path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=True,
                        output_bps=16,
                    )
                image = Image.fromarray(rgb)
                enhancements.append("RAW processing")
            else:
                # Handle regular image files
                self.logger.debug(f"Processing regular image file: {file_path}")
                image = Image.open(file_path)

            # Convert to RGB if necessary
            if image.mode in ["RGBA", "LA"]:
                self.logger.debug(f"Converting {image.mode} to RGB")
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
                enhancements.append("Alpha channel removal")
            elif image.mode != "RGB":
                self.logger.debug(f"Converting {image.mode} to RGB")
                image = image.convert("RGB")
                enhancements.append(f"{image.mode} to RGB conversion")

            # Quality enhancement: Upscale if image is small and enhancement enabled
            if ENABLE_UPSCALING and ENABLE_IMAGE_ENHANCEMENT:
                width, height = image.size
                target_size = IMAGE_TARGET_SIZE
                self.logger.debug(f"Original image size: {width}x{height}")

                if max(width, height) < target_size:
                    # Calculate new size maintaining aspect ratio
                    if width > height:
                        new_width = target_size
                        new_height = int((height * target_size) / width)
                    else:
                        new_height = target_size
                        new_width = int((width * target_size) / height)

                    self.logger.info(
                        f"Upscaling image from {width}x{height} to {new_width}x{new_height}"
                    )
                    image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )
                    enhancements.append(f"Upscaled to {new_width}x{new_height}")

            # Enhance image quality if enabled
            if ENABLE_IMAGE_ENHANCEMENT:
                self.logger.debug("Applying image enhancements")

                # Sharpening
                enhancer = ImageEnhance.Sharpness(image)
                sharp_factor = IMAGE_SHARPENING_FACTOR
                image = enhancer.enhance(sharp_factor)
                enhancements.append(f"Sharpening ({sharp_factor})")

                # Contrast
                enhancer = ImageEnhance.Contrast(image)
                contrast_factor = IMAGE_CONTRAST_FACTOR
                image = enhancer.enhance(contrast_factor)
                enhancements.append(f"Contrast boost ({contrast_factor})")

            # Save with configured compression
            self.logger.debug(f"Saving PNG to: {output_path}")
            compress_level = PNG_COMPRESS_LEVEL
            image.save(
                output_path, "PNG", optimize=False, compress_level=compress_level
            )
            self.logger.info(f"Successfully converted image to PNG: {output_path}")
            return output_path, enhancements

        except Exception as e:
            self.logger.error(f"Error converting image {file_path}: {e}")
            return None, []

    def _convert_video_to_mp4(
        self, file_path: Path
    ) -> tuple[Optional[Path], List[str]]:
        """Convert videos to MP4 with quality enhancement"""
        enhancements = []

        try:
            output_path = file_path.with_suffix(".mp4")
            self.logger.info(f"Converting video to MP4: {file_path} -> {output_path}")

            # Get video info
            self.logger.debug(f"Probing video file: {file_path}")
            probe = ffmpeg.probe(str(file_path))
            video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

            width = int(video_info["width"])
            height = int(video_info["height"])
            self.logger.debug(f"Original video resolution: {width}x{height}")

            # Determine target resolution if upscaling enabled
            scale_filter = None
            if ENABLE_UPSCALING and ENABLE_VIDEO_ENHANCEMENT:
                upscale_threshold = VIDEO_UPSCALE_THRESHOLD
                target_1080p = VIDEO_TARGET_1080P
                target_4k = VIDEO_TARGET_4K

                if max(width, height) < upscale_threshold:
                    target_height = target_1080p
                    scale_filter = f"scale=-2:{target_height}:flags=lanczos"
                    enhancements.append(f"Upscaled to {target_height}p")
                    self.logger.info(f"Upscaling video to {target_height}p")
                elif max(width, height) < VIDEO_4K_THRESHOLD:
                    target_height = min(target_4k, height * 2)
                    scale_filter = f"scale=-2:{target_height}:flags=lanczos"
                    enhancements.append(f"Upscaled to {target_height}p")
                    self.logger.info(f"Upscaling video to {target_height}p")

            # Build FFmpeg command
            self.logger.debug("Building FFmpeg conversion command")
            stream = ffmpeg.input(str(file_path))

            if scale_filter:
                stream = ffmpeg.filter(
                    stream,
                    "scale",
                    -2,
                    int(scale_filter.split(":")[1]),
                    flags="lanczos",
                )

            crf = VIDEO_CRF
            preset = VIDEO_PRESET
            codec = VIDEO_CODEC
            pixel_fmt = VIDEO_PIXEL_FORMAT
            audio_bitrate = AUDIO_BITRATE

            stream = ffmpeg.output(
                stream,
                str(output_path),
                vcodec=codec,
                crf=crf,
                preset=preset,
                acodec="aac",
                audio_bitrate=audio_bitrate,
                **{"pix_fmt": pixel_fmt},
            )

            enhancements.extend(
                [f"CRF {crf}", f"Preset {preset}", f"Audio {audio_bitrate}"]
            )

            self.logger.info("Starting video conversion with FFmpeg")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            self.logger.info(f"Successfully converted video to MP4: {output_path}")
            return output_path, enhancements

        except Exception as e:
            self.logger.error(f"Error converting video {file_path}: {e}")
            return None, []

    def _convert_audio_to_mp3(
        self, file_path: Path
    ) -> tuple[Optional[Path], List[str]]:
        """Convert audio to MP3 with high quality"""
        enhancements = []

        try:
            output_path = file_path.with_suffix(".mp3")
            self.logger.info(f"Converting audio to MP3: {file_path} -> {output_path}")

            bitrate = AUDIO_BITRATE
            sample_rate = AUDIO_SAMPLE_RATE
            codec = AUDIO_CODEC

            self.logger.debug("Building FFmpeg audio conversion command")
            stream = ffmpeg.input(str(file_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec=codec,
                audio_bitrate=bitrate,
                ar=sample_rate,
            )

            enhancements.extend([f"Bitrate {bitrate}", f"Sample rate {sample_rate}Hz"])

            self.logger.info("Starting audio conversion with FFmpeg")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            self.logger.info(f"Successfully converted audio to MP3: {output_path}")
            return output_path, enhancements

        except Exception as e:
            self.logger.error(f"Error converting audio {file_path}: {e}")
            return None, []

    def _convert_3gp(self, file_path: Path) -> tuple[Optional[Path], List[str]]:
        """Convert 3GP (check if video or audio)"""
        try:
            self.logger.info(f"Processing 3GP file: {file_path}")
            self.logger.debug("Probing 3GP file to determine if video or audio")
            probe = ffmpeg.probe(str(file_path))

            # Check if it has video stream
            has_video = any(s["codec_type"] == "video" for s in probe["streams"])

            if has_video:
                self.logger.info("3GP file contains video, converting to MP4")
                return self._convert_video_to_mp4(file_path)
            else:
                self.logger.info("3GP file is audio-only, converting to MP3")
                return self._convert_audio_to_mp3(file_path)

        except Exception as e:
            self.logger.error(f"Error processing 3GP {file_path}: {e}")
            return None, []

    def _convert_document_to_pdf(
        self, file_path: Path
    ) -> tuple[Optional[Path], List[str]]:
        """Convert documents to PDF using LibreOffice"""
        enhancements = []

        # If already PDF, return as-is
        if file_path.suffix.lower() == ".pdf":
            self.logger.info(f"File already PDF, skipping conversion: {file_path}")
            enhancements.append("already_pdf_format")
            return file_path, enhancements

        try:
            output_path = file_path.with_suffix(".pdf")
            self.logger.info(
                f"Converting document to PDF: {file_path} -> {output_path}"
            )

            # Use LibreOffice for conversion
            cmd = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(file_path.parent),
                str(file_path),
            ]

            timeout = LIBREOFFICE_TIMEOUT
            enhancements.append(f"LibreOffice conversion (timeout: {timeout}s)")

            self.logger.debug(f"Running LibreOffice command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode == 0 and output_path.exists():
                self.logger.info(
                    f"Successfully converted document to PDF: {output_path}"
                )
                return output_path, enhancements
            else:
                self.logger.error(f"LibreOffice conversion failed: {result.stderr}")
                return None, []

        except Exception as e:
            self.logger.error(f"Error converting document {file_path}: {e}")
            return None, []

    def _handle_conversion_failure(self, original_path: Path, copy_path: Path) -> Path:
        """Handle conversion failure: delete copy, move original to review"""
        try:
            self.logger.warning(f"Handling conversion failure for: {original_path}")

            # Delete the copy
            if copy_path.exists():
                copy_path.unlink()
                self.logger.debug(f"Deleted temporary copy: {copy_path}")

            # Move original to review directory
            review_path = self.review_dir / original_path.name
            shutil.move(str(original_path), str(review_path))

            self.logger.warning(
                f"Conversion failed. File moved to review: {review_path}"
            )
            return review_path

        except Exception as e:
            self.logger.error(f"Error handling conversion failure: {e}")
            return original_path
