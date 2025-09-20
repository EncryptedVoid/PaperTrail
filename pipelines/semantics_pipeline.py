import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz
from PIL import Image
import io
import logging
import gc
import psutil
import time
from enum import Enum
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import requests
import ollama
import json
import re
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import subprocess
import requests
import time
from requests.exceptions import RequestException, ConnectionError, Timeout
import os
import signal
from config import ARTIFACT_PREFIX

for artifact in tqdm(artifacts, desc="Extracting semantic data", unit="artifact"):
    try:
        # Extract UUID from filename for profile lookup
        artifact_id = artifact.stem[len(ARTIFACT_PREFIX) :]  # Remove "ARTIFACT-" prefix
        profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

        # Load existing profile
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # Check if file is too corrupted to process
        file_size = artifact.stat().st_size
        if file_size < 1024:  # Less than 1KB might be corrupted
            logger.warning(
                f"File {artifact.name} is very small ({file_size} bytes), may be corrupted"
            )

        # Track pre-processing stats to detect model refreshes/switches
        pre_processing_stats = processor.get_processing_stats()

        # Extract semantic data using enhanced visual processor
        logger.info(f"Extracting semantics for: {artifact.name}")
        logger.debug(f"File size: {file_size / (1024*1024):.2f}MB")

        # Get the data we need for LLM processing
        ocr_text = profile_data.get("semantics", {}).get("all_text", "")
        visual_description = profile_data.get("semantics", {}).get("all_imagery", "")
        metadata = profile_data.get("metadata", {})

        # Check if we have enough content to process
        # Check if OCR completely failed
        if (
            not ocr_text or ocr_text.strip() in ["No text found in document.", ""]
        ) and (
            not visual_description
            or visual_description.strip() in ["No visual content described.", ""]
        ):

            logger.error(
                f"Both OCR text extraction and visual processing failed for {artifact.name} - moving to review"
            )

            # Move to review folder instead of processing through LLM
            review_dir = PATHS["semantics_dir"] / "ocr_failed"
            review_dir.mkdir(exist_ok=True)
            artifact.rename(review_dir / artifact.name)

            # Update profile to mark as failed
            profile_data["stages"]["llm_field_extraction"] = {
                "status": "failed",
                "reason": "OCR_and_visual_processing_failed",
            }
            processing_stats["failed_extractions"] += 1
            continue  # Skip LLM processing

        logger.info(f"Extracting structured fields for: {artifact.name}")
        logger.debug(
            f"OCR text length: {len(ocr_text)} chars, Visual desc length: {len(visual_description)} chars"
        )

        # Extract structured fields using enhanced LLM
        extraction_result = field_extractor.extract_fields(
            ocr_text=ocr_text,
            visual_description=visual_description,
            metadata=metadata,
            uuid=artifact_id,
        )

        processing_stats["total_processed"] += 1

        # Update profile with LLM-extracted fields and stage completion
        profile_data["llm_extraction"] = extraction_result

        # Enhanced stage completion tracking
        stage_completion_data = {
            "status": "completed" if extraction_result["success"] else "failed",
            "timestamp": datetime.now().isoformat(),
            "model_used": extraction_result.get("model_used", "unknown"),
            "extraction_mode": extraction_result.get("extraction_mode", "unknown"),
            "fields_extracted": len(extraction_result.get("extracted_fields", {})),
            "processing_time_ms": extraction_result.get("processing_time_ms", 0),
        }

        if not extraction_result["success"]:
            stage_completion_data["error"] = extraction_result.get(
                "error", "Unknown error"
            )

        profile_data["stages"]["llm_field_extraction"] = stage_completion_data

        # Save updated profile
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        # Move artifact to next stage directory
        moved_artifact = artifact.rename(PATHS["process_completed_dir"] / artifact.name)

        # Update session tracking
        if extraction_result["success"]:
            update_stage_counts("semantics", "llm_processed", session_data)
            processing_stats["successful_extractions"] += 1
        else:
            update_stage_counts("semantics", "failed", session_data)
            processing_stats["failed_extractions"] += 1

        # Log detailed progress
        file_size = moved_artifact.stat().st_size
        extraction_status = "successful" if extraction_result["success"] else "failed"
        log_detailed_progress(
            artifact.name, file_size, f"llm_extraction_{extraction_status}"
        )

        # Update session JSON after each file
        update_session_json()

        # Enhanced logging of extraction results
        if extraction_result["success"]:
            extracted_fields = extraction_result["extracted_fields"]
            non_unknown_fields = sum(
                1 for v in extracted_fields.values() if v != "UNKNOWN"
            )
            total_fields = len(extracted_fields)

            logger.info(
                f"LLM extraction successful for {artifact.name}: {non_unknown_fields}/{total_fields} fields extracted "
                f"using {extraction_result.get('extraction_mode', 'unknown')} mode"
            )

            # Log some key extracted fields for verification
            key_fields = ["title", "document_type", "issuer_name", "date_of_issue"]
            extracted_sample = {
                k: extracted_fields.get(k, "UNKNOWN") for k in key_fields
            }
            logger.debug(f"Key fields extracted: {extracted_sample}")

            # Log field extraction quality score
            quality_score = (non_unknown_fields / total_fields) * 100
            logger.debug(f"Field extraction quality: {quality_score:.1f}%")

        else:
            logger.warning(
                f"LLM extraction failed for {artifact.name}: {extraction_result.get('error', 'Unknown error')}"
            )

        # Log progress every 50 files
        if processing_stats["total_processed"] % 50 == 0:
            current_stats = field_extractor.get_stats()
            elapsed_time = datetime.now() - processing_stats["start_time"]

            logger.info(f"=== Processing Progress Update ===")
            logger.info(f"Files processed: {processing_stats['total_processed']}")
            logger.info(
                f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
            )
            logger.info(f"Context refreshes: {processing_stats['context_refreshes']}")
            logger.info(f"Model switches: {processing_stats['model_switches']}")
            logger.info(f"Current model: {current_stats['model']}")
            logger.info(f"Elapsed time: {elapsed_time}")
            logger.info(
                f"Prompts since last refresh: {current_stats['prompts_since_refresh']}"
            )
            logger.info("=" * 35)

    except Exception as e:
        logger.error(f"Failed to process LLM extraction for {artifact.name}: {e}")
        processing_stats["failed_extractions"] += 1

        # Create failed subdirectory if needed and move file there
        failed_dir = PATHS["semantics_dir"] / "failed"
        failed_dir.mkdir(exist_ok=True)

        try:
            artifact.rename(failed_dir / artifact.name)
            update_stage_counts("semantics", "failed", session_data)
            logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
        except Exception as move_error:
            logger.error(
                f"Failed to move {artifact.name} to failed directory: {move_error}"
            )

        session_data["errors"].append(
            {
                "file": artifact.name,
                "stage": "llm_field_extraction",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )

        try:
            extracted_semantics: Dict[str, str] = processor.extract_article_semantics(
                document=artifact
            )

            # Track post-processing stats
            post_processing_stats = processor.get_processing_stats()
            processing_stats["total_processed"] += 1
            processing_stats["successful_extractions"] += 1

            # Check if model was refreshed or switched during processing
            if (
                pre_processing_stats["memory_refreshes"]
                < post_processing_stats["memory_refreshes"]
            ):
                processing_stats["model_refreshes"] += 1
                logger.info(f"Model was refreshed during processing of {artifact.name}")

            if (
                pre_processing_stats["model_switches"]
                < post_processing_stats["model_switches"]
            ):
                processing_stats["model_switches"] += 1
                logger.info(f"Model was switched during processing of {artifact.name}")

            # Calculate quality score for this extraction
            text_length = len(extracted_semantics.get("all_text", ""))
            imagery_length = len(extracted_semantics.get("all_imagery", ""))

            # Simple quality heuristic
            has_meaningful_text = (
                text_length > 50
                and "No text found" not in extracted_semantics.get("all_text", "")
            )
            has_meaningful_imagery = (
                imagery_length > 100
                and "No visual content"
                not in extracted_semantics.get("all_imagery", "")
            )

            quality_score = 0
            if has_meaningful_text:
                quality_score += 50
            if has_meaningful_imagery:
                quality_score += 50

            processing_stats["quality_scores"].append(quality_score)

            # Enhanced stage completion data
            stage_completion_data = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "model_used": post_processing_stats["current_model"]["name"],
                "device": post_processing_stats["device"],
                "processing_mode": post_processing_stats["processing_mode"],
                "text_length": text_length,
                "imagery_length": imagery_length,
                "quality_score": quality_score,
                "model_refreshed": pre_processing_stats["memory_refreshes"]
                < post_processing_stats["memory_refreshes"],
            }

            # Update profile with semantics and enhanced stage completion
            profile_data["semantics"] = extracted_semantics
            profile_data["stages"]["semantic_extraction"] = stage_completion_data

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(PATHS["semantics_dir"] / artifact.name)

            # Update session tracking
            update_stage_counts("metadata", "semantics", session_data)

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "semantic_extraction")

            # Update session JSON after each file
            update_session_json()

            # Enhanced logging
            logger.info(f"Semantics extracted successfully for {artifact.name}")
            logger.debug(
                f"Text extracted: {text_length} chars, Imagery: {imagery_length} chars, Quality: {quality_score}%"
            )

            # Log memory usage occasionally
            if processing_stats["total_processed"] % 5 == 0:
                current_stats = processor.get_processing_stats()
                memory_info = current_stats.get("memory_usage", {})
                if "gpu_memory_percent" in memory_info:
                    logger.debug(
                        f"GPU memory: {memory_info['gpu_memory_percent']:.1f}%, "
                        f"RAM: {memory_info['system_ram_percent']:.1f}%"
                    )

        except Exception as extraction_error:
            logger.error(
                f"Semantic extraction failed for {artifact.name}: {extraction_error}"
            )
            processing_stats["failed_extractions"] += 1

            # Check if we should try a different model for persistent failures
            if (
                processing_stats["failed_extractions"] > 3
                and processing_stats["failed_extractions"] % 5 == 0
            ):

                available_models = processor.get_available_models()
                current_model = processor.current_model_spec.model_id

                # Try switching to a different model
                other_models = [
                    m
                    for m in available_models
                    if m["fits_constraints"] and m["model_id"] != current_model
                ]
                if other_models:
                    new_model = other_models[0]["model_id"]
                    logger.info(
                        f"High failure rate detected. Attempting to switch from {current_model} to {new_model}"
                    )

                    if processor.switch_model(new_model):
                        processing_stats["model_switches"] += 1
                        logger.info(f"Successfully switched to model: {new_model}")
                    else:
                        logger.warning(f"Failed to switch to model: {new_model}")

            # Create failed entry in profile
            profile_data["stages"]["semantic_extraction"] = {
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(extraction_error),
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move to failed directory
            failed_dir = PATHS["metadata_dir"] / "failed"
            failed_dir.mkdir(exist_ok=True)
            artifact.rename(failed_dir / artifact.name)
            update_stage_counts("metadata", "failed", session_data)

            session_data["errors"].append(
                {
                    "file": artifact.name,
                    "stage": "semantic_extraction",
                    "error": str(extraction_error),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            continue

        # Progress reporting every 20 files
        if processing_stats["total_processed"] % 20 == 0:
            current_stats = processor.get_processing_stats()
            elapsed_time = datetime.now() - processing_stats["start_time"]
            avg_quality = (
                sum(processing_stats["quality_scores"])
                / len(processing_stats["quality_scores"])
                if processing_stats["quality_scores"]
                else 0
            )

            logger.info(f"=== Processing Progress Update ===")
            logger.info(f"Files processed: {processing_stats['total_processed']}")
            logger.info(
                f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
            )
            logger.info(f"Average quality score: {avg_quality:.1f}%")
            logger.info(f"Model refreshes: {processing_stats['model_refreshes']}")
            logger.info(f"Model switches: {processing_stats['model_switches']}")
            logger.info(f"Current model: {current_stats['current_model']['name']}")
            logger.info(f"Elapsed time: {elapsed_time}")
            logger.info(
                f"Avg time per document: {current_stats.get('avg_processing_time_per_doc', 0):.2f}s"
            )
            logger.info("=" * 35)

    except Exception as e:
        logger.error(f"Failed to process {artifact.name}: {e}")
        processing_stats["failed_extractions"] += 1

        # Create failed subdirectory if needed and move file there
        failed_dir = PATHS["metadata_dir"] / "failed"
        failed_dir.mkdir(exist_ok=True)

        try:
            artifact.rename(failed_dir / artifact.name)
            update_stage_counts("metadata", "failed", session_data)
            logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
        except Exception as move_error:
            logger.error(
                f"Failed to move {artifact.name} to failed directory: {move_error}"
            )

        session_data["errors"].append(
            {
                "file": artifact.name,
                "stage": "semantic_extraction",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )
