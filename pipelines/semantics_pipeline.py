"""
Semantics Pipeline Module

A robust file processing pipeline that handles comprehensive semantic data extraction
from various document types including PDFs, images, and office files using advanced
visual processing and LLM-based field extraction techniques.

This module provides functionality to extract semantic data by:
- Processing documents through visual recognition systems for text and imagery extraction
- Utilizing large language models for structured field extraction
- Analyzing document content, layout, and metadata
- Extracting domain-specific information based on document type
- Handling extraction failures gracefully with fallback methods
- Maintaining detailed operation logs and quality metrics
- Updating artifact profiles with extracted semantic data
- Moving processed files through appropriate pipeline stages

Author: Ashiq Gazi
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional
from tqdm import tqdm
from utilities.session_tracking_agent import SessionTracker
from processors.language_processor import LanguageProcessor, LanguageExtractionReport
from processors.visual_processor import VisualProcessor, OCRReport
from config import ARTIFACT_PROFILES_DIR, ARTIFACT_PREFIX, PROFILE_PREFIX


class SemanticExtractionReport(TypedDict):
    """
    Type definition for the semantic extraction report returned by the extract_semantics method.

    Attributes:
        processed_files: Number of files that had semantic data successfully extracted
        total_files: Total number of files discovered in the source directory
        failed_extractions: Count of files where semantic extraction failed
        skipped_files: Count of files skipped due to validation issues
        ocr_failures: Count of files where OCR text extraction failed
        visual_processing_failures: Count of files where visual processing failed
        llm_extraction_failures: Count of files where LLM field extraction failed
        successful_text_extractions: Count of files with successful text extraction
        successful_visual_extractions: Count of files with successful visual extraction
        successful_field_extractions: Count of files with successful structured field extraction
        errors: List of error messages encountered during processing
        quality_metrics: Dictionary containing quality assessment metrics
        processing_times: Dictionary containing performance timing metrics
        model_statistics: Dictionary containing model usage and performance statistics
        profile_updates: Number of artifact profiles successfully updated
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


class SemanticsPipeline:
    """
    A semantic data extraction pipeline for processing directories of document artifacts.

    This class handles the extraction of comprehensive semantic information from various
    document types using advanced visual processing, OCR technology, and large language
    model-based field extraction. It maintains artifact profiles and provides detailed
    reporting of all operations with quality metrics and performance analytics.

    The pipeline works in the following stages:
    1. Directory validation and artifact discovery
    2. Artifact profile loading and validation
    3. Visual processing for text and imagery extraction
    4. OCR text extraction with quality assessment
    5. LLM-based structured field extraction
    6. Content analysis and quality scoring
    7. Profile data integration and updates
    8. Error handling and fallback processing
    9. File movement through pipeline stages
    10. Comprehensive reporting and performance metrics

    Processing Methods:
    - Visual processing using advanced computer vision models
    - OCR text extraction with multiple fallback engines
    - LLM field extraction with context-aware prompting
    - Quality assessment and validation of extracted data
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        logger: logging.Logger,
        session_agent: SessionTracker,
        visual_processor: VisualProcessor,
        field_extractor: LanguageProcessor,
    ):
        """
        Initialize the SemanticsPipeline.

        Args:
            logger: Logger instance for recording pipeline operations and errors
            session_agent: SessionTracker for monitoring pipeline progress and state
            visual_processor: Visual processing engine for text and imagery extraction
            field_extractor: LLM-based field extraction engine for structured data
        """
        self.logger = logger
        self.session_agent = session_agent
        self.visual_processor = visual_processor
        self.field_extractor = field_extractor

    def extract_semantics(
        self,
        source_dir: Path,
        review_dir: Path,
        success_dir: Path,
    ) -> SemanticExtractionReport:
        """
        Extract semantic data from all artifacts in a directory and update their profiles.

        This method performs comprehensive semantic extraction:
        1. Validates input directories exist and are accessible
        2. Discovers all artifact files in the source directory
        3. Loads existing artifact profiles for data integration
        4. Processes each artifact through visual and LLM extraction
        5. Extracts text content, visual descriptions, and structured fields
        6. Performs quality assessment and validation
        7. Updates artifact profiles with extracted semantic data
        8. Moves processed files to appropriate target directories
        9. Generates comprehensive report with quality metrics

        Args:
            source_dir: Directory containing artifacts to extract semantics from
            review_dir: Directory to move problematic files for manual review
            success_dir: Directory to move files to after successful extraction

        Returns:
            SemanticExtractionReport containing detailed results of the extraction process

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
        }

        # Discover all artifact files in the source directory
        try:
            unprocessed_artifacts = [
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
        start_time = datetime.now()
        quality_scores = []

        # Process each file through the sanitization pipeline
        # Use progress bar only for multiple files
        if len(unprocessed_artifacts) > 1:
            artifact_iterator: Any = tqdm(
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
                artifact_id = artifact.stem[
                    len(ARTIFACT_PREFIX) :
                ]  # Remove "ARTIFACT-" prefix
                profile_path = (
                    ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
                )

                # STAGE 2: Load existing profile
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    report["skipped_files"] += 1

                    # Move file to review directory for manual inspection
                    self._move_to_review(artifact, review_dir, "missing_profile")
                    continue

                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)

                # STAGE 3: Extract semantic data using visual processor
                self.logger.info(f"Extracting semantics for: {artifact.name}")

                visual_extraction_start = time.time()
                extracted_semantics = self.visual_processor.extract_semantics(
                    document=artifact
                )
                visual_processing_time = time.time() - visual_extraction_start

                report["processing_times"][
                    "visual_processing_time"
                ] += visual_processing_time

                # STAGE 4: Validate extraction results
                ocr_text = extracted_semantics.get("all_text", "")
                visual_description = extracted_semantics.get("all_imagery", "")

                # Check if extraction completely failed
                if self._is_extraction_failed(ocr_text, visual_description):
                    self.logger.error(
                        f"Both OCR text extraction and visual processing failed for {artifact.name}"
                    )
                    report["failed_extractions"] += 1
                    report["ocr_failures"] += 1
                    report["visual_processing_failures"] += 1

                    # Move to review folder instead of processing through LLM
                    self._move_to_review(artifact, review_dir, "extraction_failed")
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
                llm_extraction_start = time.time()
                field_extraction_result = self.field_extractor.extract_fields(
                    ocr_text=ocr_text,
                    visual_description=visual_description,
                )
                llm_processing_time = time.time() - llm_extraction_start

                report["processing_times"]["llm_processing_time"] += llm_processing_time

                # STAGE 7: Process extraction results
                if field_extraction_result.get("success", False):
                    report["successful_field_extractions"] += 1

                    # Calculate quality metrics
                    quality_score = self._calculate_quality_score(
                        ocr_text, visual_description, field_extraction_result
                    )
                    quality_scores.append(quality_score)

                    # Update profile with successful extraction
                    self._update_profile_success(
                        profile_data,
                        profile_path,
                        extracted_semantics,
                        field_extraction_result,
                        quality_score,
                    )

                    # Move artifact to success directory
                    moved_artifact = artifact.rename(success_dir / artifact.name)
                    report["processed_files"] += 1
                    report["profile_updates"] += 1

                    self.logger.info(f"Successfully processed: {artifact.name}")

                else:
                    report["llm_extraction_failures"] += 1
                    self._move_to_review(artifact, review_dir, "llm_extraction_failed")
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
                self._move_to_review(artifact, review_dir, "processing_error")

        # STAGE 8: Calculate final metrics and statistics
        total_time = (datetime.now() - start_time).total_seconds()
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
            model_stats = self._get_model_statistics()
            report["model_statistics"].update(model_stats)

        # Update session tracker with current progress
        self.session_agent.update({"stage": "semantic_extraction", "report": report})

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
        ocr_failed = (
            not ocr_text
            or ocr_text.strip() in ["No text found in document.", ""]
            or self._is_empty_extraction(ocr_text)
        )

        visual_failed = (
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
        error_patterns = [
            "no text found",
            "extraction failed",
            "processing failed",
            "error occurred",
            "unable to extract",
        ]

        content_lower = content.lower().strip()
        return any(pattern in content_lower for pattern in error_patterns)

    def _calculate_quality_score(
        self,
        ocr_text: str,
        visual_description: str,
        field_extraction_result: Dict[str, Any],
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
        score = 0.0

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
        extracted_fields = field_extraction_result.get("extracted_fields", {})
        if extracted_fields:
            non_unknown_fields = sum(
                1
                for v in extracted_fields.values()
                if v and str(v).upper() != "UNKNOWN"
            )
            total_fields = len(extracted_fields)
            if total_fields > 0:
                field_ratio = non_unknown_fields / total_fields
                score += 30.0 * field_ratio

        return min(100.0, score)

    def _update_profile_success(
        self,
        profile_data: Dict[str, Any],
        profile_path: Path,
        semantics: Dict[str, str],
        field_results: Dict[str, Any],
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
                "extraction_mode": field_results.get("extraction_mode", "unknown"),
                "processing_time_ms": field_results.get("processing_time_ms", 0),
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
