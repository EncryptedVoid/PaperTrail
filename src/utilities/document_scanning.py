"""
Document Scanner Module - Optimized with unpaper
"""

import logging
import shutil
import subprocess
import tempfile
from io import UnsupportedOperation
from pathlib import Path

import cv2
import numpy as np

from processors.file_conversion import convert_document


def _dewarp_page(image_path: Path, logger: logging.Logger) -> Path:
    """Dewarp curved pages using page-dewarp library."""
    try:
        result = subprocess.run(
            ["page-dewarp", "--help"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            logger.warning("page-dewarp not available, skipping dewarping")
            return image_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("page-dewarp not installed, skipping dewarping")
        return image_path

    try:
        output_path = (
            image_path.parent / f"{image_path.stem}_dewarped{image_path.suffix}"
        )

        result = subprocess.run(
            ["page-dewarp", "-o", "file", str(image_path)],
            capture_output=True,
            timeout=30,
        )

        expected_output = image_path.parent / f"{image_path.stem}_thresh.png"

        if result.returncode == 0 and expected_output.exists():
            logger.info("Dewarping successful")
            expected_output.rename(output_path)
            return output_path
        else:
            logger.warning("Dewarping failed, skipping")
            return image_path

    except subprocess.TimeoutExpired:
        logger.warning("Dewarping timed out")
        return image_path
    except Exception as e:
        logger.warning(f"Dewarping error: {e}")
        return image_path


def _enhance_output(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Enhance with contrast and sharpening."""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        logger.info("Image enhancement complete")
        return denoised
    except Exception as e:
        logger.warning(f"Enhancement failed: {e}")
        return image


def professional_scan(
    file_path: Path, archive_dir: Path, logger: logging.Logger
) -> Path:
    """
    Professionally scan and clean up a document using unpaper + enhancements.

    Processing pipeline:
    1. Archive original file
    2. Run unpaper (deskew, split pages, remove margins, cleanup)
    3. Dewarp (if needed for curved pages)
    4. Enhance contrast and sharpen

    Args:
        file_path: Path to the input document
        archive_dir: Path to directory where original will be archived
        logger: Logger instance for output

    Returns:
        Path to the processed document

    Raises:
        RuntimeError: If scanning fails
    """
    if not isinstance(file_path, Path):
        raise TypeError(f"file_path must be Path object, got {type(file_path)}")

    if not isinstance(archive_dir, Path):
        raise TypeError(f"archive_dir must be Path object, got {type(archive_dir)}")

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        subprocess.run(["unpaper", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "unpaper not installed. Install with:\n"
            "  Linux: sudo apt-get install unpaper\n"
            "  Mac: brew install unpaper"
        )

    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / file_path.name
    logger.info(f"Archiving original file to: {archive_path}")
    shutil.copy2(file_path, archive_path)

    file_extension = file_path.suffix.lower()

    try:
        if file_extension in {
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".tif",
            ".bmp",
            ".ppm",
            ".pgm",
            ".pbm",
        }:
            logger.info(f"Processing image: {file_path}")

            image = cv2.imread(str(file_path))
            if image is None:
                raise RuntimeError(f"Failed to read image: {file_path}")

            with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
                temp_input = Path(tmp.name)

            cv2.imwrite(str(temp_input), image)

            logger.info(
                "Step 1/3: Running unpaper (deskew, cleanup, remove margins)..."
            )
            temp_output = temp_input.parent / f"{temp_input.stem}_unpaper.ppm"

            unpaper_result = subprocess.run(
                [
                    "unpaper",
                    "--layout",
                    "single",
                    "--no-blackfilter",
                    "--no-grayfilter",
                    "--deskew-scan-size",
                    "50",
                    str(temp_input),
                    str(temp_output),
                ],
                capture_output=True,
                timeout=60,
            )

            if unpaper_result.returncode != 0 or not temp_output.exists():
                logger.warning(f"unpaper failed: {unpaper_result.stderr.decode()}")
                temp_output = temp_input

            logger.info("Step 2/3: Dewarping (if needed)...")
            dewarped_path = _dewarp_page(temp_output, logger)

            logger.info("Step 3/3: Enhancing output...")
            processed = cv2.imread(str(dewarped_path))
            if processed is None:
                raise RuntimeError("Failed to read processed image")

            final_image = _enhance_output(processed, logger)

            output_path = file_path.parent / file_path
            cv2.imwrite(str(output_path), final_image)

            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
            if dewarped_path != temp_output:
                dewarped_path.unlink(missing_ok=True)

            logger.info(f"✅ Scanning complete: {output_path}")
            convert_document(file_path=output_path, to_format="pdf")
            return output_path

        elif file_extension == ".pdf":
            logger.info(f"Processing PDF: {file_path}")

            try:
                output_path = file_path.parent / file_path

                result = subprocess.run(
                    [
                        "ocrmypdf",
                        "--deskew",
                        "--rotate-pages",
                        "--clean",
                        "--unpaper-args",
                        "--layout single",
                        "--skip-text",
                        "--jobs",
                        "4",
                        str(file_path),
                        str(output_path),
                    ],
                    capture_output=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    logger.info(f"✅ PDF scanning complete: {output_path}")

                else:
                    raise RuntimeError(f"OCRmyPDF failed: {result.stderr.decode()}")
                    return output_path
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                raise RuntimeError(
                    "OCRmyPDF not available. Install with: pip install ocrmypdf"
                ) from e
        else:
            raise UnsupportedOperation(f"Unsupported file format: {file_extension}")

    except Exception as e:
        logger.error(f"❌ Scanning failed: {e}")
        logger.info(f"Original file preserved at: {archive_path}")
        raise RuntimeError(f"Failed to scan document: {e}") from e


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if len(sys.argv) != 3:
        print("Usage: python document_scanner.py <input_file> <archive_dir>")
        sys.exit(1)

    try:
        output = professional_scan(Path(sys.argv[1]), Path(sys.argv[2]), logger)
        print(f"✅ Success! Scanned document: {output}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
