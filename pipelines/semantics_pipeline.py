"""
Semantics Pipeline Module

A robust file processing pipeline that handles comprehensive semantic data extraction
from various document types including PDFs, images, and office files using advanced
visual processing and LLM-based field extraction techniques.

Author: Ashiq Gazi
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional
from tqdm import tqdm
from processors.session_tracking_agent import SessionTracker
from processors.language_processor import LanguageProcessor, LanguageExtractionReport
from processors.visual_processor import VisualProcessor
from common_utils import move_file_safely
from config import ARTIFACT_PROFILES_DIR, ARTIFACT_PREFIX, PROFILE_PREFIX


class SemanticExtractionReport(TypedDict):
    """
    Type definition for the semantic extraction report returned by the extract_semantics method.
    """

    processed_files: int
    total_files: int
    failed_extractions: int
    skipped_files: int
    ocr_failures: int
    visual_processing_failures: int
    llm_extraction_failures: int
    successful_text_extractions: int
    successful_visual_extractions: int
    successful_field_extractions: int
    errors: List[str]
    quality_metrics: Dict[str, Any]
    processing_times: Dict[str, float]
    model_statistics: Dict[str, Any]
    profile_updates: int
    extracted_semantics: Dict[
        str, Dict[str, Any]
    ]  # artifact_id -> extracted semantic data


class SemanticsPipeline:
    """
    A semantic data extraction pipeline for processing directories of document artifacts.

    This class handles the extraction of comprehensive semantic information from various
    document types using advanced visual processing, OCR technology, and large language
    model-based field extraction. It maintains artifact profiles and provides detailed
    reporting of all operations with quality metrics and performance analytics.
    """

    def __init__(
        self,
        logger: logging.Logger,
        session_agent: SessionTracker,
        visual_processor: VisualProcessor,
        field_extractor: LanguageProcessor,
    ) -> None:
        """
        Initialize the SemanticsPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
            session_agent: SessionTracker for monitoring pipeline progress and state
            visual_processor: Visual processing engine for text and imagery extraction
            field_extractor: LLM-based field extraction engine for structured data
        """
        self.logger: logging.Logger = logger
        self.session_agent: SessionTracker = session_agent
        self.visual_processor: VisualProcessor = visual_processor
        self.field_extractor: LanguageProcessor = field_extractor

    def extract_semantics(
        self,
        source_dir: Path,
        review_dir: Path,
        success_dir: Path,
    ) -> SemanticExtractionReport:
        """
        Extract semantic data from all artifacts in a directory and update their profiles.

        Args:
            source_dir: Directory containing artifacts to extract semantics from
            review_dir: Directory to move problematic files for manual review
            success_dir: Directory to move files to after successful extraction

        Returns:
            SemanticExtractionReport containing detailed results of the extraction process
        """

        # Validate that input directories exist and are accessible
        if not source_dir.exists():
            error_msg: str = f"Source directory does not exist: {source_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not success_dir.exists():
            success_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created target directory: {success_dir}")

        if not review_dir.exists():
            review_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created review directory: {review_dir}")

        # Initialize report structure with default values
        report: SemanticExtractionReport = {
            "processed_files": 0,
            "total_files": 0,
            "failed_extractions": 0,
            "skipped_files": 0,
            "ocr_failures": 0,
            "visual_processing_failures": 0,
            "llm_extraction_failures": 0,
            "successful_text_extractions": 0,
            "successful_visual_extractions": 0,
            "successful_field_extractions": 0,
            "errors": [],
            "quality_metrics": {
                "average_text_length": 0,
                "average_visual_description_length": 0,
                "average_fields_extracted": 0,
                "quality_scores": [],
            },
            "processing_times": {
                "total_time": 0,
                "average_per_file": 0,
                "visual_processing_time": 0,
                "llm_processing_time": 0,
            },
            "model_statistics": {
                "context_refreshes": 0,
                "model_switches": 0,
                "current_model": "unknown",
            },
            "profile_updates": 0,
            "extracted_semantics": {},
        }

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
            report["errors"].append(error_msg)
            return report

        # Handle empty directory case
        if not unprocessed_artifacts:
            self.logger.info("No artifact files found in source directory")
            return report

        # Sort files by size for consistent processing order
        unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
        report["total_files"] = len(unprocessed_artifacts)

        # Log semantic extraction stage header for clear progress tracking
        self.logger.info("=" * 80)
        self.logger.info("SEMANTIC DATA EXTRACTION AND ANALYSIS STAGE")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Found {len(unprocessed_artifacts)} artifact files to process"
        )

        # Initialize processing statistics
        start_time: datetime = datetime.now()
        quality_scores: List[float] = []

        # Process each file through the semantic extraction pipeline
        # Use progress bar only for multiple files
        artifact_iterator: Any
        if len(unprocessed_artifacts) > 1:
            artifact_iterator = tqdm(
                unprocessed_artifacts,
                desc="Extracting semantic data",
                unit="artifacts",
            )
        else:
            artifact_iterator = unprocessed_artifacts

        # Process each artifact through the semantic extraction pipeline
        for artifact in artifact_iterator:
            try:
                # STAGE 1: Extract UUID from filename for profile lookup
                artifact_id: str = artifact.stem[
                    len(ARTIFACT_PREFIX) + 1 :  # Remove "ARTIFACT-" prefix
                ]
                profile_path: Path = (
                    ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
                )

                # STAGE 2: Load existing profile
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    report["skipped_files"] += 1

                    # Move file to review directory for manual inspection
                    review_location: Path = (
                        review_dir / "missing_profile" / artifact.name
                    )
                    review_location.parent.mkdir(parents=True, exist_ok=True)
                    if move_file_safely(artifact, review_location):
                        self.logger.info(
                            f"Moved {artifact.name} to review: missing_profile"
                        )
                    continue

                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_data: Dict[str, Any] = json.load(f)

                # Get original filename from profile for LLM context
                original_filename: str = profile_data.get(
                    "original_artifact_name", artifact.name
                )

                # STAGE 3: Extract semantic data using visual processor
                self.logger.info(f"Extracting visual semantics for: {artifact.name}")

                visual_extraction_start: float = time.time()
                extracted_semantics: Dict[str, str] = (
                    self.visual_processor.extract_semantics(document=artifact)
                )
                visual_processing_time: float = time.time() - visual_extraction_start

                report["processing_times"][
                    "visual_processing_time"
                ] += visual_processing_time

                # STAGE 4: Validate extraction results
                ocr_text: str = extracted_semantics.get("all_text", "")
                visual_description: str = extracted_semantics.get("all_imagery", "")

                # Check if extraction completely failed
                if self._is_extraction_failed(ocr_text, visual_description):
                    self.logger.error(
                        f"Both OCR text extraction and visual processing failed for {artifact.name}"
                    )
                    report["failed_extractions"] += 1
                    report["ocr_failures"] += 1
                    report["visual_processing_failures"] += 1

                    # Move to review folder
                    review_location = review_dir / "extraction_failed" / artifact.name
                    review_location.parent.mkdir(parents=True, exist_ok=True)
                    if move_file_safely(artifact, review_location):
                        self.logger.info(
                            f"Moved {artifact.name} to review: extraction_failed"
                        )

                    self._update_profile_failed(
                        profile_data, profile_path, "OCR_and_visual_processing_failed"
                    )
                    continue

                # STAGE 5: Track successful extractions
                if ocr_text and not self._is_empty_extraction(ocr_text):
                    report["successful_text_extractions"] += 1

                if visual_description and not self._is_empty_extraction(
                    visual_description
                ):
                    report["successful_visual_extractions"] += 1

                # STAGE 6: Extract structured fields using LLM
                self.logger.info(f"Extracting structured fields for: {artifact.name}")

                llm_extraction_start: float = time.time()
                field_extraction_result: LanguageExtractionReport = (
                    self.field_extractor.extract_fields(
                        ocr_text=ocr_text,
                        visual_description=visual_description,
                    )
                )
                llm_processing_time: float = time.time() - llm_extraction_start

                report["processing_times"]["llm_processing_time"] += llm_processing_time

                # STAGE 7: Process extraction results
                if field_extraction_result["success"]:
                    report["successful_field_extractions"] += 1

                    # Calculate quality metrics
                    quality_score: float = self._calculate_quality_score(
                        ocr_text, visual_description, field_extraction_result
                    )
                    quality_scores.append(quality_score)

                    # Store extracted semantics in report
                    report["extracted_semantics"][artifact_id] = {
                        "ocr_text": ocr_text,
                        "visual_description": visual_description,
                        "extracted_fields": field_extraction_result["extracted_fields"],
                        "original_filename": original_filename,
                        "quality_score": quality_score,
                        "processing_timestamp": field_extraction_result[
                            "processing_timestamp"
                        ],
                        "model_used": field_extraction_result["model_used"],
                    }

                    # Update profile with successful extraction
                    self._update_profile_success(
                        profile_data,
                        profile_path,
                        extracted_semantics,
                        field_extraction_result,
                        quality_score,
                    )

                    # Move artifact to success directory
                    success_location: Path = success_dir / artifact.name
                    if move_file_safely(artifact, success_location):
                        report["processed_files"] += 1
                        report["profile_updates"] += 1
                        self.logger.info(f"Successfully processed: {artifact.name}")

                else:
                    report["llm_extraction_failures"] += 1

                    # Move to review directory
                    review_location = (
                        review_dir / "llm_extraction_failed" / artifact.name
                    )
                    review_location.parent.mkdir(parents=True, exist_ok=True)
                    if move_file_safely(artifact, review_location):
                        self.logger.info(
                            f"Moved {artifact.name} to review: llm_extraction_failed"
                        )

                    self._update_profile_failed(
                        profile_data,
                        profile_path,
                        field_extraction_result.get("error", "LLM extraction failed"),
                    )

            except Exception as e:
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                report["failed_extractions"] += 1

                # Move to review directory
                review_location = review_dir / "processing_error" / artifact.name
                review_location.parent.mkdir(parents=True, exist_ok=True)
                if move_file_safely(artifact, review_location):
                    self.logger.info(
                        f"Moved {artifact.name} to review: processing_error"
                    )

        # STAGE 8: Calculate final metrics and statistics
        total_time: float = (datetime.now() - start_time).total_seconds()
        report["processing_times"]["total_time"] = total_time

        if report["processed_files"] > 0:
            report["processing_times"]["average_per_file"] = (
                total_time / report["processed_files"]
            )

        if quality_scores:
            report["quality_metrics"]["quality_scores"] = quality_scores
            report["quality_metrics"]["average_quality_score"] = sum(
                quality_scores
            ) / len(quality_scores)

        # Update model statistics if available
        if self.field_extractor:
            model_stats: Dict[str, Any] = self._get_model_statistics()
            report["model_statistics"].update(model_stats)

        # Generate final summary report for user review
        self._log_final_summary(report)

        return report

    def _is_extraction_failed(self, ocr_text: str, visual_description: str) -> bool:
        """
        Check if both OCR and visual extraction completely failed.

        Args:
            ocr_text: Extracted text content
            visual_description: Visual description

        Returns:
            True if both extractions failed, False otherwise
        """
        ocr_failed: bool = (
            not ocr_text
            or ocr_text.strip() in ["No text found in document.", ""]
            or self._is_empty_extraction(ocr_text)
        )

        visual_failed: bool = (
            not visual_description
            or visual_description.strip() in ["No visual content described.", ""]
            or self._is_empty_extraction(visual_description)
        )

        return ocr_failed and visual_failed

    def _is_empty_extraction(self, content: str) -> bool:
        """
        Check if extracted content is effectively empty.

        Args:
            content: Content to check

        Returns:
            True if content is empty or contains only error messages
        """
        if not content or not content.strip():
            return True

        # Check for common error patterns
        error_patterns: List[str] = [
            "no text found",
            "extraction failed",
            "processing failed",
            "error occurred",
            "unable to extract",
        ]

        content_lower: str = content.lower().strip()
        return any(pattern in content_lower for pattern in error_patterns)

    def _calculate_quality_score(
        self,
        ocr_text: str,
        visual_description: str,
        field_extraction_result: LanguageExtractionReport,
    ) -> float:
        """
        Calculate a quality score for the extraction.

        Args:
            ocr_text: Extracted text content
            visual_description: Visual description
            field_extraction_result: LLM extraction results

        Returns:
            Quality score between 0 and 100
        """
        score: float = 0.0

        # Text quality (40% of score)
        if ocr_text and len(ocr_text.strip()) > 50:
            score += 40.0
        elif ocr_text and len(ocr_text.strip()) > 10:
            score += 20.0

        # Visual description quality (30% of score)
        if visual_description and len(visual_description.strip()) > 100:
            score += 30.0
        elif visual_description and len(visual_description.strip()) > 25:
            score += 15.0

        # Field extraction quality (30% of score)
        extracted_fields: Optional[Dict[str, str]] = field_extraction_result.get(
            "extracted_fields"
        )
        if extracted_fields:
            non_unknown_fields: int = sum(
                1
                for v in extracted_fields.values()
                if v and str(v).upper() != "UNKNOWN"
            )
            total_fields: int = len(extracted_fields)
            if total_fields > 0:
                field_ratio: float = non_unknown_fields / total_fields
                score += 30.0 * field_ratio

        return min(100.0, score)

    def _update_profile_success(
        self,
        profile_data: Dict[str, Any],
        profile_path: Path,
        semantics: Dict[str, str],
        field_results: LanguageExtractionReport,
        quality_score: float,
    ) -> None:
        """
        Update profile with successful extraction results.

        Args:
            profile_data: Profile data dictionary
            profile_path: Path to profile file
            semantics: Extracted semantic data
            field_results: LLM field extraction results
            quality_score: Calculated quality score
        """
        try:
            # Update profile with extraction results
            profile_data["semantics"] = semantics
            profile_data["llm_extraction"] = field_results

            # Update stages tracking
            profile_data.setdefault("stages", {})
            profile_data["stages"]["semantic_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "text_length": len(semantics.get("all_text", "")),
                "visual_length": len(semantics.get("all_imagery", "")),
                "fields_extracted": len(field_results.get("extracted_fields", {})),
            }

            profile_data["stages"]["llm_field_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "model_used": field_results.get("model_used", "unknown"),
                "processing_time_seconds": field_results.get(
                    "processing_time_seconds", 0
                ),
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to update profile {profile_path}: {e}")

    def _update_profile_failed(
        self,
        profile_data: Dict[str, Any],
        profile_path: Path,
        error_reason: str,
    ) -> None:
        """
        Update profile with failed extraction information.

        Args:
            profile_data: Profile data dictionary
            profile_path: Path to profile file
            error_reason: Reason for failure
        """
        try:
            # Update stages tracking with failure information
            profile_data.setdefault("stages", {})
            profile_data["stages"]["semantic_extraction"] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": error_reason,
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to update failed profile {profile_path}: {e}")

    def _get_model_statistics(self) -> Dict[str, Any]:
        """
        Get current model usage statistics.

        Returns:
            Dictionary containing model statistics
        """
        try:
            if hasattr(self.field_extractor, "get_stats"):
                return self.field_extractor.get_stats()
            else:
                return {
                    "context_refreshes": 0,
                    "model_switches": 0,
                    "current_model": "unknown",
                }
        except Exception as e:
            self.logger.debug(f"Failed to get model statistics: {e}")
            return {}

    def _log_final_summary(self, report: SemanticExtractionReport) -> None:
        """
        Log the final summary of semantic extraction results.

        Args:
            report: The completed semantic extraction report
        """
        self.logger.info("Semantic extraction complete:")
        self.logger.info(
            f"  - {report['processed_files']} files successfully processed"
        )
        self.logger.info(
            f"  - {report['successful_text_extractions']} successful text extractions"
        )
        self.logger.info(
            f"  - {report['successful_visual_extractions']} successful visual extractions"
        )
        self.logger.info(
            f"  - {report['successful_field_extractions']} successful field extractions"
        )
        self.logger.info(f"  - {report['profile_updates']} profiles updated")
        self.logger.info(f"  - {report['failed_extractions']} extraction failures")
        self.logger.info(f"  - {report['skipped_files']} files skipped")

        # Processing time summary
        total_time: float = report["processing_times"]["total_time"]
        avg_time: float = report["processing_times"]["average_per_file"]
        self.logger.info(f"  - Total processing time: {total_time:.2f} seconds")
        if avg_time > 0:
            self.logger.info(f"  - Average time per file: {avg_time:.2f} seconds")

        # Quality metrics
        if "average_quality_score" in report["quality_metrics"]:
            avg_quality: float = report["quality_metrics"]["average_quality_score"]
            self.logger.info(f"  - Average quality score: {avg_quality:.1f}/100")

        # Extracted semantics count
        self.logger.info(
            f"  - {len(report['extracted_semantics'])} complete semantic extractions stored in report"
        )

        # Warn about any errors encountered during processing
        if report["errors"]:
            self.logger.warning(f"  - {len(report['errors'])} errors encountered")
