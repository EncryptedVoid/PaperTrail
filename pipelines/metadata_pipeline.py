"""
Metadata Pipeline Module

A robust file processing pipeline that handles comprehensive metadata extraction
from various file types including images, documents, PDFs, and office files.

This module provides functionality to extract metadata by:
- Detecting file types and routing to appropriate extractors
- Extracting filesystem metadata (size, dates, permissions)
- Processing image EXIF, IPTC, and XMP data
- Extracting document properties and structure information
- Handling extraction failures gracefully with fallback methods
- Maintaining detailed operation logs and error tracking
- Updating artifact profiles with extracted metadata

Author: Ashiq Gazi
"""

import logging
import json
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, TypedDict
from tqdm import tqdm
from utilities.security_agent import SecurityAgent
from utilities.session_tracking_agent import SessionTracker
from config import ARTIFACT_PROFILES_DIR


class MetadataReport(TypedDict):
    """
    Type definition for the metadata extraction report returned by the extract_metadata method.

    Attributes:
        processed_files: Number of files that had metadata successfully extracted
        total_files: Total number of files discovered in the source directory
        failed_extractions: Count of files where metadata extraction failed
        skipped_files: Count of files skipped due to validation issues
        image_files_processed: Count of image files that were processed
        document_files_processed: Count of document files that were processed
        errors: List of error messages encountered during processing
        extraction_summary: Dictionary summarizing extraction results by file type
        profile_updates: Number of artifact profiles successfully updated
    """

    processed_files: int
    total_files: int
    failed_extractions: int
    skipped_files: int
    image_files_processed: int
    document_files_processed: int
    errors: List[str]
    extraction_summary: Dict[str, int]
    profile_updates: int


class MetadataPipeline:
    """
    A metadata extraction pipeline for processing directories of artifacts.

    This class handles the extraction of comprehensive metadata from various file types,
    including filesystem information, image EXIF data, document properties, and more.
    It maintains artifact profiles and provides detailed reporting of all operations.

    The pipeline works in the following stages:
    1. Directory validation and file discovery
    2. File type detection and routing
    3. Filesystem metadata extraction
    4. Specialized metadata extraction (images, documents, etc.)
    5. Profile data integration and updates
    6. Error handling and fallback processing
    7. Comprehensive reporting and logging
    """

    def __init__(
        self,
        logger: logging.Logger,
        session_agent: SessionTracker,
    ):
        """
        Initialize the MetadataPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
            session_agent: SessionTracker for monitoring pipeline progress and state
        """
        self.logger = logger
        self.session_agent = session_agent

    def extract_metadata(
        self, source_dir: Path, review_dir: Path, success_dir: Path
    ) -> MetadataReport:
        """
        Extract metadata from all files in a directory and update their profiles.

        This method performs comprehensive metadata extraction:
        1. Validates input directories exist and are accessible
        2. Discovers all artifact files in the source directory
        3. Loads existing artifact profiles for metadata integration
        4. Extracts filesystem and specialized metadata for each file
        5. Updates artifact profiles with extracted metadata
        6. Moves processed files to target directory
        7. Generates comprehensive report of all operations

        Args:
            source_dir: Directory containing files to extract metadata from
            success_dir: Directory to move files to after metadata extraction

        Returns:
            MetadataReport containing detailed results of the metadata extraction process

        Note:
            This method expects files to follow the ARTIFACT-{uuid}.ext naming convention
            and looks for corresponding PROFILE-{uuid}.json files for updates.
        """

        # Validate that input directories exist and are accessible
        if not source_dir.exists():
            error_msg = f"Source directory does not exist: {source_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not success_dir.exists():
            success_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created target directory: {success_dir}")

        # Initialize report structure with default values
        report: MetadataReport = {
            "processed_files": 0,
            "total_files": 0,
            "failed_extractions": 0,
            "skipped_files": 0,
            "image_files_processed": 0,
            "document_files_processed": 0,
            "errors": [],
            "extraction_summary": {},
            "profile_updates": 0,
        }

        # Discover all artifact files in the source directory
        try:
            artifacts = [
                item
                for item in source_dir.iterdir()
                if item.is_file() and item.name.startswith("ARTIFACT-")
            ]
        except Exception as e:
            error_msg = f"Failed to scan source directory: {e}"
            self.logger.error(error_msg)
            report["errors"].append(error_msg)
            return report

        # Handle empty directory case
        if not artifacts:
            self.logger.info("No artifact files found in source directory")
            return report

        # Sort files by size for consistent processing order
        artifacts.sort(key=lambda p: p.stat().st_size)
        report["total_files"] = len(artifacts)

        # Log metadata extraction stage header for clear progress tracking
        self.logger.info("=" * 80)
        self.logger.info("METADATA EXTRACTION AND PROFILE UPDATE STAGE")
        self.logger.info("=" * 80)
        self.logger.info(f"Found {len(artifacts)} artifact files to process")

        # Process each artifact through the metadata extraction pipeline
        for artifact in tqdm(artifacts, desc="Extracting metadata", unit="artifact"):
            try:
                # STAGE 1: Extract UUID from filename for profile lookup
                artifact_id = artifact.stem[9:]  # Remove "ARTIFACT-" prefix
                profile_path = ARTIFACT_PROFILES_DIR / f"PROFILE-{artifact_id}.json"

                # STAGE 2: Load existing profile
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    report["skipped_files"] += 1

                    # Move file to review directory for manual inspection
                    try:
                        review_location = review_dir / artifact.name
                        artifact.rename(review_location)
                        self.logger.info(
                            f"Moved file with missing profile to review: {review_location}"
                        )
                    except Exception as move_error:
                        self.logger.error(
                            f"Failed to move {artifact.name} to review directory: {move_error}"
                        )
                    continue

                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)

                # STAGE 3: Extract metadata using the extraction engine
                self.logger.info(f"Extracting metadata for: {artifact.name}")
                extracted_metadata: Dict[str, Any] = self._extract_file_metadata(
                    artifact
                )

                # STAGE 4: Check extraction success and update counters
                if "error" in extracted_metadata:
                    report["failed_extractions"] += 1
                    report["errors"].append(
                        f"Metadata extraction failed for {artifact.name}: {extracted_metadata['error']}"
                    )
                else:
                    # Update type-specific counters
                    if extracted_metadata.get("type") == "image":
                        report["image_files_processed"] += 1
                    elif extracted_metadata.get("type") == "document":
                        report["document_files_processed"] += 1

                    # Update extraction summary by file extension
                    ext = artifact.suffix.lower()
                    report["extraction_summary"][ext] = (
                        report["extraction_summary"].get(ext, 0) + 1
                    )

                # STAGE 5: Update profile with metadata and stage completion
                profile_data["metadata"] = extracted_metadata
                profile_data["stages"] = profile_data.get("stages", {})
                profile_data["stages"]["metadata_extraction"] = {
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }

                # Save updated profile
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=2)

                report["profile_updates"] += 1

                # STAGE 6: Move artifact to target directory
                moved_artifact = artifact.rename(success_dir / artifact.name)
                report["processed_files"] += 1

                self.logger.debug(f"Metadata extracted and saved for: {artifact.name}")

            except Exception as e:
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                report["failed_extractions"] += 1

                # Move failed file to failed subdirectory
                failed_dir = source_dir / "failed"
                failed_dir.mkdir(exist_ok=True)

                try:
                    artifact.rename(failed_dir / artifact.name)
                    self.logger.info(
                        f"Moved failed file to: {failed_dir / artifact.name}"
                    )
                except Exception as move_error:
                    self.logger.error(
                        f"Failed to move {artifact.name} to failed directory: {move_error}"
                    )

        # Update session tracker with current progress
        self.session_agent.update({"stage": "metadata_extraction", "report": report})

        # Generate final summary report for user review
        self.logger.info("Metadata extraction complete:")
        self.logger.info(
            f"  - {report['processed_files']} files successfully processed"
        )
        self.logger.info(f"  - {report['image_files_processed']} image files processed")
        self.logger.info(
            f"  - {report['document_files_processed']} document files processed"
        )
        self.logger.info(f"  - {report['profile_updates']} profiles updated")
        self.logger.info(f"  - {report['failed_extractions']} extraction failures")
        self.logger.info(f"  - {report['skipped_files']} files skipped")

        # Warn about any errors encountered during processing
        if report["errors"]:
            self.logger.warning(f"  - {len(report['errors'])} errors encountered")

        return report

    def _extract_file_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Main entry point for metadata extraction"""
        filepath = Path(filepath)
        self.logger.info(f"Starting metadata extraction for: {filepath}")

        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            return {"error": "File not found", "path": str(filepath)}

        # Always get basic info
        self.logger.debug("Extracting basic filesystem metadata")
        result = self._get_basic_info(filepath)

        if "error" in result:
            self.logger.error(
                f"Failed to extract basic info for {filepath}: {result['error']}"
            )
            return result

        # Route to specific extractors
        ext = filepath.suffix.lower()
        self.logger.debug(f"File extension detected: {ext}")

        if ext in {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
            ".heic",
            ".heif",
            ".raw",
            ".cr2",
            ".nef",
            ".arw",
        }:
            self.logger.info(f"Processing as image file: {ext}")
            result.update(self._extract_image_metadata(filepath))
        elif ext in {".pdf", ".docx", ".xlsx", ".pptx"}:
            self.logger.info(f"Processing as document file: {ext}")
            result.update(self._extract_document_metadata(filepath))
        else:
            self.logger.warning(
                f"No specialized extractor available for extension: {ext}"
            )
            result["warning"] = f"No specialized extractor for {ext}"

        self.logger.info(f"Metadata extraction completed for: {filepath}")
        return result

    def _get_basic_info(self, filepath: Path) -> Dict[str, Any]:
        """Extract filesystem metadata"""
        try:
            self.logger.debug(f"Reading file stats for: {filepath}")
            stat = filepath.stat()
            mime_type, _ = mimetypes.guess_type(str(filepath))

            result = {
                "filename": filepath.name,
                "path": str(filepath.absolute()),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": filepath.suffix.lower(),
                "mime_type": mime_type,
            }

            self.logger.debug(
                f"Basic info extracted: size={result['size_mb']}MB, mime_type={mime_type}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to read basic info for {filepath}: {e}")
            return {"error": f"Failed to read basic info: {e}"}

    def _extract_image_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract image metadata with lazy imports"""
        result = {"type": "image"}
        self.logger.debug("Starting image metadata extraction")

        # PIL metadata
        self.logger.debug("Attempting PIL metadata extraction")
        pil_data = self._get_pil_metadata(filepath)
        if "error" not in pil_data:
            self.logger.info("PIL metadata extraction successful")
            result.update(pil_data)
        else:
            self.logger.warning(
                f"PIL metadata extraction failed: {pil_data.get('error')}"
            )

        # Advanced metadata (optional)
        self.logger.debug("Attempting advanced metadata extraction")
        advanced_data = self._get_advanced_image_metadata(filepath)
        if "error" not in advanced_data:
            self.logger.info("Advanced metadata extraction successful")
            result.update(advanced_data)
        else:
            self.logger.debug(
                f"Advanced metadata extraction failed: {advanced_data.get('error')}"
            )

        return result

    def _get_pil_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract PIL-based image metadata"""
        try:
            self.logger.debug("Importing PIL dependencies")
            from PIL import Image
            from PIL.ExifTags import TAGS
            import pillow_heif

            pillow_heif.register_heif_opener()

            self.logger.debug(f"Opening image with PIL: {filepath}")
            with Image.open(filepath) as img:
                # Basic info
                image_info = {
                    "dimensions": img.size,
                    "width": img.size[0],
                    "height": img.size[1],
                    "mode": img.mode,
                    "format": img.format,
                    "has_transparency": img.mode in ("RGBA", "LA")
                    or "transparency" in img.info,
                }

                self.logger.debug(
                    f"Image dimensions: {img.size}, mode: {img.mode}, format: {img.format}"
                )

                # EXIF data
                exif_data = {}
                exif = img.getexif()  # Modern method, not deprecated _getexif
                if exif:
                    exif_data = {TAGS.get(k, f"tag_{k}"): v for k, v in exif.items()}
                    self.logger.debug(f"EXIF data extracted: {len(exif_data)} tags")
                else:
                    self.logger.debug("No EXIF data found")

                return {"image_info": image_info, "exif": exif_data}

        except ImportError as e:
            self.logger.error(f"PIL dependencies not available: {e}")
            return {"error": f"PIL not available: {e}"}
        except Exception as e:
            self.logger.error(f"PIL extraction failed for {filepath}: {e}")
            return {"error": f"PIL extraction failed: {e}"}

    def _get_advanced_image_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract advanced metadata using pyexiv2 if available"""
        try:
            self.logger.debug("Attempting pyexiv2 import")
            import pyexiv2

            self.logger.debug(f"Opening image with pyexiv2: {filepath}")
            with pyexiv2.Image(str(filepath)) as img:
                result = {
                    "advanced_metadata": {
                        "exif": img.read_exif(),
                        "iptc": img.read_iptc(),
                        "xmp": img.read_xmp(),
                    }
                }
                self.logger.info(
                    "Advanced metadata (EXIF/IPTC/XMP) extracted successfully"
                )
                return result

        except ImportError:
            self.logger.debug("pyexiv2 not available - using basic EXIF only")
            return {"info": "pyexiv2 not available - using basic EXIF only"}
        except Exception as e:
            self.logger.error(
                f"Advanced metadata extraction failed for {filepath}: {e}"
            )
            return {"error": f"Advanced metadata extraction failed: {e}"}

    def _extract_document_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Route document extraction based on file type"""
        result = {"type": "document"}
        ext = filepath.suffix.lower()

        extractors = {
            ".pdf": self._extract_pdf_metadata,
            ".docx": self._extract_docx_metadata,
            ".xlsx": self._extract_xlsx_metadata,
            ".pptx": self._extract_pptx_metadata,
        }

        if ext in extractors:
            self.logger.info(f"Using {ext} extractor")
            result.update(extractors[ext](filepath))
        else:
            self.logger.error(f"No extractor available for document type: {ext}")
            result["error"] = f"No extractor for {ext}"

        return result

    def _extract_pdf_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract PDF metadata - prefer PyMuPDF for comprehensive data"""
        self.logger.debug("Starting PDF metadata extraction")

        # Try PyMuPDF first (best overall)
        self.logger.debug("Attempting PyMuPDF extraction")
        pymupdf_result = self._try_pymupdf(filepath)
        if "error" not in pymupdf_result:
            self.logger.info("PDF metadata extracted successfully with PyMuPDF")
            return pymupdf_result

        # Fallback to pdfplumber for basic info
        self.logger.info("PyMuPDF failed, falling back to pdfplumber")
        result = self._try_pdfplumber(filepath)
        if "error" not in result:
            self.logger.info("PDF metadata extracted successfully with pdfplumber")
        else:
            self.logger.error("All PDF extraction methods failed")

        return result

    def _try_pymupdf(self, filepath: Path) -> Dict[str, Any]:
        """PyMuPDF extraction with proper resource management"""
        try:
            self.logger.debug("Importing PyMuPDF (fitz)")
            import fitz

            self.logger.debug(f"Opening PDF with PyMuPDF: {filepath}")
            doc = fitz.open(str(filepath))
            try:
                result = {
                    "pdf_info": {
                        "page_count": doc.page_count,
                        "is_encrypted": doc.needs_pass,
                        "metadata": doc.metadata,
                        "toc": doc.get_toc(),
                        "has_text": (
                            bool(doc[0].get_text_words())
                            if doc.page_count > 0
                            else False
                        ),
                    }
                }
                self.logger.debug(
                    f"PDF info: {doc.page_count} pages, encrypted: {doc.needs_pass}"
                )
                return result
            finally:
                doc.close()  # Ensure cleanup
                self.logger.debug("PyMuPDF document closed")

        except ImportError:
            self.logger.warning("PyMuPDF not available")
            return {"error": "PyMuPDF not available"}
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed for {filepath}: {e}")
            return {"error": f"PyMuPDF extraction failed: {e}"}

    def _try_pdfplumber(self, filepath: Path) -> Dict[str, Any]:
        """Fallback PDF extraction"""
        try:
            self.logger.debug("Importing pdfplumber")
            import pdfplumber

            self.logger.debug(f"Opening PDF with pdfplumber: {filepath}")
            with pdfplumber.open(filepath) as pdf:
                result = {
                    "pdf_info": {
                        "page_count": len(pdf.pages),
                        "has_text": (
                            bool(pdf.pages[0].extract_text()) if pdf.pages else False
                        ),
                    }
                }
                self.logger.debug(f"PDF info: {len(pdf.pages)} pages")
                return result

        except ImportError:
            self.logger.error(
                "No PDF libraries available (neither PyMuPDF nor pdfplumber)"
            )
            return {"error": "No PDF libraries available"}
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed for {filepath}: {e}")
            return {"error": f"PDF extraction failed: {e}"}

    def _extract_docx_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract Word document metadata"""
        try:
            self.logger.debug("Importing python-docx")
            from docx import Document

            self.logger.debug(f"Opening DOCX document: {filepath}")
            doc = Document(str(filepath))
            props = doc.core_properties

            result = {
                "document_info": {
                    "title": props.title,
                    "author": props.author,
                    "created": props.created.isoformat() if props.created else None,
                    "modified": props.modified.isoformat() if props.modified else None,
                    "last_modified_by": props.last_modified_by,
                    "paragraph_count": len(doc.paragraphs),
                    "table_count": len(doc.tables),
                }
            }

            self.logger.info(
                f"DOCX metadata extracted: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables"
            )
            return result

        except ImportError:
            self.logger.error("python-docx library not available")
            return {"error": "python-docx not available"}
        except Exception as e:
            self.logger.error(f"DOCX extraction failed for {filepath}: {e}")
            return {"error": f"DOCX extraction failed: {e}"}

    def _extract_xlsx_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract Excel metadata"""
        try:
            self.logger.debug("Importing openpyxl")
            import openpyxl

            self.logger.debug(f"Opening XLSX workbook: {filepath}")
            wb = openpyxl.load_workbook(str(filepath), data_only=True)
            props = wb.properties

            result = {
                "spreadsheet_info": {
                    "title": props.title,
                    "creator": props.creator,
                    "created": props.created.isoformat() if props.created else None,
                    "modified": props.modified.isoformat() if props.modified else None,
                    "sheet_names": wb.sheetnames,
                    "sheet_count": len(wb.worksheets),
                }
            }

            self.logger.info(
                f"XLSX metadata extracted: {len(wb.worksheets)} sheets - {wb.sheetnames}"
            )
            return result

        except ImportError:
            self.logger.error("openpyxl library not available")
            return {"error": "openpyxl not available"}
        except Exception as e:
            self.logger.error(f"Excel extraction failed for {filepath}: {e}")
            return {"error": f"Excel extraction failed: {e}"}

    def _extract_pptx_metadata(self, filepath: Path) -> Dict[str, Any]:
        """Extract PowerPoint metadata"""
        try:
            self.logger.debug("Importing python-pptx")
            from pptx import Presentation

            self.logger.debug(f"Opening PPTX presentation: {filepath}")
            prs = Presentation(str(filepath))
            props = prs.core_properties

            result = {
                "presentation_info": {
                    "title": props.title,
                    "author": props.author,
                    "created": props.created.isoformat() if props.created else None,
                    "modified": props.modified.isoformat() if props.modified else None,
                    "slide_count": len(prs.slides),
                }
            }

            self.logger.info(f"PPTX metadata extracted: {len(prs.slides)} slides")
            return result

        except ImportError:
            self.logger.error("python-pptx library not available")
            return {"error": "python-pptx not available"}
        except Exception as e:
            self.logger.error(f"PowerPoint extraction failed for {filepath}: {e}")
            return {"error": f"PowerPoint extraction failed: {e}"}
