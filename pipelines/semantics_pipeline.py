# =====================================================================
# STAGE 4: SEMANTIC EXTRACTION (VISUAL PROCESSING)
# =====================================================================

# Get all artifacts from metadata directory, sorted by size
metadata_artifacts: List[Path] = [item for item in PATHS["metadata_dir"].iterdir()]

if len(metadata_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("ARTIFACT VISUAL PROCESSING STAGE")
    logger.info("=" * 80)
    logger.info("Extracting semantic data using enhanced visual processor...")

    metadata_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(metadata_artifacts)} artifacts ready for semantic extraction"
    )

    # Enhanced visual processor configuration
    try:
        # Configure visual processing parameters (can be made configurable via config file)
        visual_config = {
            "max_gpu_vram_gb": 12.0,  # Use most of your 9.6GB GPU
            "max_ram_gb": 48.0,  # Limit to 48GB as requested (was auto-detecting 44.8GB)
            "force_cpu": False,  # Make sure GPU is used
            "processing_mode": ProcessingMode.FAST,  # Changed from HIGH_QUALITY
            "refresh_interval": 5,  # More frequent refresh due to memory pressure
            "memory_threshold": 70.0,  # Lower threshold for cleanup
            "auto_model_selection": True,
            "preferred_model": "Qwen/Qwen2-VL-2B-Instruct",  # Force smaller, faster model
        }

        # logger.info("=== PERFORMANCE DEBUG INFO ===")
        # logger.info(f"GPU Available: {torch.cuda.is_available()}")
        # logger.info(f"GPU Device Count: {torch.cuda.device_count()}")
        # if torch.cuda.is_available():
        #     logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        #     logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        # logger.info(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        # logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        # logger.info("=" * 30)

        # Initialize enhanced visual processor
        processor = VisualProcessor(
            logger=logger,
            max_gpu_vram_gb=visual_config["max_gpu_vram_gb"],
            max_ram_gb=visual_config["max_ram_gb"],
            force_cpu=visual_config["force_cpu"],
            processing_mode=visual_config["processing_mode"],
            refresh_interval=visual_config["refresh_interval"],
            memory_threshold=visual_config["memory_threshold"],
            auto_model_selection=visual_config["auto_model_selection"],
            preferred_model=visual_config["preferred_model"],
        )

        # Log initialization details
        logger.info("Enhanced Visual Processor initialized successfully")
        initial_stats = processor.get_processing_stats()
        logger.info(f"Selected model: {initial_stats['current_model']['name']}")
        logger.info(f"Processing mode: {initial_stats['processing_mode']}")
        logger.info(f"Device: {initial_stats['device']}")

        # Log hardware constraints
        hw_constraints = initial_stats["hardware_constraints"]
        logger.info(
            f"Hardware constraints: GPU VRAM={hw_constraints['max_gpu_vram_gb']:.1f}GB, "
            f"RAM={hw_constraints['max_ram_gb']:.1f}GB, Force CPU={hw_constraints['force_cpu']}"
        )

        # Show available models
        available_models = processor.get_available_models()
        suitable_models = [m for m in available_models if m["fits_constraints"]]
        logger.info(
            f"Available models for current hardware: {[m['name'] for m in suitable_models]}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize enhanced visual processor: {e}")
        logger.error("Falling back to basic configuration...")

        # Fallback to basic configuration if enhanced setup fails
        processor = VisualProcessor(
            logger=logger,
            auto_model_selection=False,
            force_cpu=True,  # Use CPU as safest fallback
        )
        logger.warning("Using fallback visual processor configuration")

    # Track processing statistics
    processing_stats = {
        "total_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "model_refreshes": 0,
        "model_switches": 0,
        "start_time": datetime.now(),
        "quality_scores": [],
    }

    # Create semantics directory
    PATHS["semantics_dir"] = BASE_DIR / "semantics"
    PATHS["semantics_dir"].mkdir(parents=True, exist_ok=True)

    for artifact in tqdm(
        metadata_artifacts, desc="Extracting semantic descriptions", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
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

            try:
                extracted_semantics: Dict[str, str] = (
                    processor.extract_article_semantics(document=artifact)
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
                    logger.info(
                        f"Model was refreshed during processing of {artifact.name}"
                    )

                if (
                    pre_processing_stats["model_switches"]
                    < post_processing_stats["model_switches"]
                ):
                    processing_stats["model_switches"] += 1
                    logger.info(
                        f"Model was switched during processing of {artifact.name}"
                    )

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

    # Enhanced final statistics and optimization report
    final_stats = processor.get_processing_stats()
    processing_time = datetime.now() - processing_stats["start_time"]
    avg_quality = (
        sum(processing_stats["quality_scores"])
        / len(processing_stats["quality_scores"])
        if processing_stats["quality_scores"]
        else 0
    )

    logger.info("=" * 80)
    logger.info("SEMANTIC EXTRACTION COMPLETE - FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {processing_stats['total_processed']}")
    logger.info(f"Successful extractions: {processing_stats['successful_extractions']}")
    logger.info(f"Failed extractions: {processing_stats['failed_extractions']}")
    logger.info(
        f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
    )
    logger.info(f"Average quality score: {avg_quality:.1f}%")
    logger.info(f"Total processing time: {processing_time}")
    logger.info(
        f"Average time per file: {processing_time / max(processing_stats['total_processed'], 1)}"
    )
    logger.info("=== VISUAL PROCESSOR STATISTICS: ===")
    logger.info(f"Final model used: {final_stats['current_model']['name']}")
    logger.info(f"Device: {final_stats['device']}")
    logger.info(f"Processing mode: {final_stats['processing_mode']}")
    logger.info(f"Pages processed: {final_stats['pages_processed']}")
    logger.info(
        f"Text extraction success rate: {final_stats.get('text_extraction_success_rate', 0):.1%}"
    )
    logger.info(
        f"Description success rate: {final_stats.get('description_success_rate', 0):.1%}"
    )
    logger.info(f"Model refreshes performed: {processing_stats['model_refreshes']}")
    logger.info(f"Model switches performed: {processing_stats['model_switches']}")

    # Memory usage summary
    memory_usage = final_stats.get("memory_usage", {})
    if "gpu_memory_percent" in memory_usage:
        logger.info(
            f"Final GPU memory usage: {memory_usage['gpu_memory_percent']:.1f}%"
        )
    logger.info(f"Final RAM usage: {memory_usage.get('system_ram_percent', 0):.1f}%")

    # Get and log optimization suggestions
    # try:
    #     optimization_report = processor.optimize_performance()
    #     suggestions = optimization_report.get("optimization_suggestions", [])

    #     if suggestions:
    #         logger.info("")
    #         logger.info("PERFORMANCE OPTIMIZATION SUGGESTIONS:")
    #         for i, suggestion in enumerate(suggestions, 1):
    #             logger.info(f"{i}. {suggestion}")
    #     else:
    #         logger.info(
    #             "No performance optimization suggestions - system is running optimally"
    #         )

    # except Exception as e:
    #     logger.warning(f"Could not generate optimization report: {e}")

    # Save processing statistics to session data
    session_data["semantic_extraction_stats"] = {
        "processing_stats": processing_stats,
        "final_processor_stats": final_stats,
        "processing_time_seconds": processing_time.total_seconds(),
        "average_quality_score": avg_quality,
    }


# Get all artifacts from semantics directory, sorted by size
semantic_artifacts: List[Path] = [item for item in PATHS["semantics_dir"].iterdir()]

if len(semantic_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("LLM-BASED SEMANTIC METADATA EXTRACTION STAGE")
    logger.info("=" * 80)
    logger.info("Extracting structured document fields using LLM...")

    semantic_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(semantic_artifacts)} artifacts ready for LLM field extraction"
    )

    field_extractor = LanguageProcessor(
        logger=logger,
        max_ram_gb=48.0,  # Auto-detect (or set specific limit like 16.0)
        max_gpu_vram_gb=12.0,  # Auto-detect (or set specific limit like 8.0)
        max_cpu_cores=None,  # Auto-detect (or set specific limit like 8)
        auto_model_selection=False,  # Automatically select best model for hardware
    )

    # Track processing statistics
    processing_stats = {
        "total_processed": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "context_refreshes": 0,
        "model_switches": 0,
        "start_time": datetime.now(),
    }

    for artifact in tqdm(
        semantic_artifacts,
        desc="Extracting semantic metadata with LLM processing",
        unit="artifact",
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Get the data we need for LLM processing
            ocr_text = profile_data.get("semantics", {}).get("all_text", "")
            visual_description = profile_data.get("semantics", {}).get(
                "all_imagery", ""
            )
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
            moved_artifact = artifact.rename(
                PATHS["process_completed_dir"] / artifact.name
            )

            # Update session tracking
            if extraction_result["success"]:
                update_stage_counts("semantics", "llm_processed", session_data)
                processing_stats["successful_extractions"] += 1
            else:
                update_stage_counts("semantics", "failed", session_data)
                processing_stats["failed_extractions"] += 1

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            extraction_status = (
                "successful" if extraction_result["success"] else "failed"
            )
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
                logger.info(
                    f"Context refreshes: {processing_stats['context_refreshes']}"
                )
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

    # Enhanced final statistics logging
    final_stats = field_extractor.get_stats()
    processing_time = datetime.now() - processing_stats["start_time"]

    logger.info("=" * 80)
    logger.info("LLM FIELD EXTRACTION COMPLETE - FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {processing_stats['total_processed']}")
    logger.info(f"Successful extractions: {processing_stats['successful_extractions']}")
    logger.info(f"Failed extractions: {processing_stats['failed_extractions']}")
    logger.info(
        f"Success rate: {(processing_stats['successful_extractions'] / max(processing_stats['total_processed'], 1)) * 100:.1f}%"
    )
    logger.info(f"Total processing time: {processing_time}")
    logger.info(
        f"Average time per file: {processing_time / max(processing_stats['total_processed'], 1)}"
    )
    # logger.info("")
    logger.info("LLM MODEL STATISTICS:")
    logger.info(f"Final model used: {final_stats['model']}")
    logger.info(f"Total LLM API calls made: {final_stats['total_processed']}")
    logger.info(f"Context refreshes performed: {processing_stats['context_refreshes']}")
    logger.info(f"Model switches performed: {processing_stats['model_switches']}")
    logger.info(
        f"Hardware utilized: RAM={final_stats['hardware_constraints']['max_ram_gb']:.1f}GB, "
        f"GPU={final_stats['hardware_constraints']['max_gpu_vram_gb']:.1f}GB, "
        f"CPU={final_stats['hardware_constraints']['max_cpu_cores']} cores"
    )
    logger.info(f"Extraction mode: {final_stats['extraction_mode']}")

    # Save final processing statistics to session data
    session_data["llm_extraction_stats"] = {
        "processing_stats": processing_stats,
        "final_extractor_stats": final_stats,
        "processing_time_seconds": processing_time.total_seconds(),
    }

    logger.info("=" * 80)


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


class ProcessingMode(Enum):
    """Processing quality modes"""

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass
class VisionModelSpec:
    """Specification for vision models including hardware requirements"""

    name: str
    model_id: str
    min_gpu_vram_gb: float
    min_ram_gb: float
    quality_score: int  # 1-10, higher is better
    max_tokens: int
    supports_batch: bool = False
    model_type: str = "qwen2vl"  # qwen2vl, llava, blip2, etc.


@dataclass
class HardwareConstraints:
    """Hardware resource constraints"""

    max_gpu_vram_gb: float
    max_ram_gb: float
    force_cpu: bool = False


@dataclass
class ProcessingStats:
    """Track processing performance and quality metrics"""

    documents_processed: int = 0
    pages_processed: int = 0
    text_extractions_successful: int = 0
    descriptions_successful: int = 0
    total_processing_time: float = 0.0
    memory_refreshes: int = 0
    model_switches: int = 0
    average_text_length: float = 0.0
    average_description_length: float = 0.0
    last_refresh_time: float = 0.0


class VisualProcessor:
    """Enhanced Visual Processor with hardware optimization and multi-model support"""

    def __init__(
        self,
        logger: logging.Logger,
        max_gpu_vram_gb: Optional[float] = None,
        max_ram_gb: Optional[float] = None,
        force_cpu: bool = False,
        processing_mode: ProcessingMode = ProcessingMode.BALANCED,
        refresh_interval: int = 5,
        memory_threshold: float = 80.0,
        auto_model_selection: bool = True,
        preferred_model: Optional[str] = None,
    ) -> None:
        # Validate required logger parameter
        if logger is None:
            raise ValueError(
                "Logger is required - VisualProcessor cannot be initialized without a logger"
            )
        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger instance, got {type(logger)}")

        # Initialize instance attributes
        self.logger = logger
        self.processing_mode = processing_mode
        self.refresh_interval = refresh_interval
        self.memory_threshold = memory_threshold
        self.auto_model_selection = auto_model_selection

        # Initialize hardware constraints
        self.hardware_constraints = self._detect_hardware_constraints(
            max_gpu_vram_gb, max_ram_gb, force_cpu
        )

        # Initialize model registry
        self.model_registry = self._initialize_model_registry()

        # Initialize processing statistics
        self.stats = ProcessingStats()

        # Model state
        self.model = None
        self.processor = None
        self.current_model_spec = None
        self.device = None

        # Select and load optimal model
        if auto_model_selection:
            self.current_model_spec = self._select_optimal_model(preferred_model)
        else:
            # Use preferred model or default
            model_name = preferred_model or "Qwen/Qwen2-VL-7B-Instruct"
            self.current_model_spec = self._get_model_spec_by_id(model_name)

        self.logger.info(
            f"Initialized VisualProcessor with model: {self.current_model_spec.name}"
        )
        self.logger.info(f"Hardware constraints: {self.hardware_constraints}")

        # Load the selected model
        self._load_model()

    def _detect_hardware_constraints(
        self,
        max_gpu_vram_gb: Optional[float],
        max_ram_gb: Optional[float],
        force_cpu: bool,
    ) -> HardwareConstraints:
        """Detect available hardware or use provided constraints"""

        # RAM detection
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            available_ram = total_ram * 0.7  # Use 70% of total RAM
        else:
            available_ram = max_ram_gb

        # GPU detection
        if max_gpu_vram_gb is None and not force_cpu:
            try:
                if torch.cuda.is_available():
                    # Get GPU memory from torch
                    gpu_props = torch.cuda.get_device_properties(0)
                    total_vram = gpu_props.total_memory / (1024**3)  # Convert to GB
                    available_vram = total_vram * 0.8  # Use 80% of total VRAM
                else:
                    available_vram = 0
            except Exception as e:
                self.logger.warning(f"Could not detect GPU VRAM: {e}")
                available_vram = 0
        else:
            available_vram = max_gpu_vram_gb or 0

        if force_cpu:
            available_vram = 0

        return HardwareConstraints(
            max_gpu_vram_gb=available_vram,
            max_ram_gb=available_ram,
            force_cpu=force_cpu,
        )

    def _initialize_model_registry(self) -> List[VisionModelSpec]:
        """Initialize registry of available vision models with their requirements"""
        return [
            # Qwen2-VL models
            VisionModelSpec(
                name="Qwen2-VL-2B",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                min_gpu_vram_gb=4.0,
                min_ram_gb=8.0,
                quality_score=7,
                max_tokens=512,
                model_type="qwen2vl",
            ),
            VisionModelSpec(
                name="Qwen2-VL-7B",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                min_gpu_vram_gb=14.0,
                min_ram_gb=16.0,
                quality_score=9,
                max_tokens=512,
                model_type="qwen2vl",
            ),
            VisionModelSpec(
                name="Qwen2-VL-72B",
                model_id="Qwen/Qwen2-VL-72B-Instruct",
                min_gpu_vram_gb=144.0,
                min_ram_gb=200.0,
                quality_score=10,
                max_tokens=1024,
                model_type="qwen2vl",
            ),
            # Alternative models for fallback
            VisionModelSpec(
                name="Qwen2-VL-7B-CPU",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                min_gpu_vram_gb=0.0,  # CPU only
                min_ram_gb=32.0,
                quality_score=8,
                max_tokens=512,
                model_type="qwen2vl",
            ),
        ]

    def _select_optimal_model(
        self, preferred_model: Optional[str] = None
    ) -> VisionModelSpec:
        """Select the best model that fits within hardware constraints"""

        # If a preferred model is specified, try to use it if it fits
        if preferred_model:
            preferred_spec = self._get_model_spec_by_id(preferred_model)
            if preferred_spec and self._model_fits_constraints(preferred_spec):
                self.logger.info(f"Using preferred model: {preferred_spec.name}")
                return preferred_spec
            else:
                self.logger.warning(
                    f"Preferred model {preferred_model} doesn't fit constraints, selecting alternative"
                )

        # Filter models that fit within hardware constraints
        suitable_models = [
            model
            for model in self.model_registry
            if self._model_fits_constraints(model)
        ]

        if not suitable_models:
            # Fallback to CPU-only model
            cpu_models = [m for m in self.model_registry if m.min_gpu_vram_gb == 0]
            if cpu_models:
                best_model = max(cpu_models, key=lambda m: m.quality_score)
                self.logger.warning(
                    f"No GPU models fit constraints, using CPU model: {best_model.name}"
                )
                return best_model
            else:
                raise RuntimeError(
                    "No suitable vision models found for current hardware constraints"
                )

        # Select model with highest quality score that fits
        best_model = max(suitable_models, key=lambda m: m.quality_score)
        self.logger.info(
            f"Selected optimal model: {best_model.name} (quality: {best_model.quality_score})"
        )

        return best_model

    def _model_fits_constraints(self, model_spec: VisionModelSpec) -> bool:
        """Check if a model fits within hardware constraints"""
        if self.hardware_constraints.force_cpu:
            return model_spec.min_gpu_vram_gb == 0

        fits_vram = (
            model_spec.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
        )
        fits_ram = model_spec.min_ram_gb <= self.hardware_constraints.max_ram_gb

        return fits_vram and fits_ram

    def _get_model_spec_by_id(self, model_id: str) -> Optional[VisionModelSpec]:
        """Get model specification by model ID"""
        for spec in self.model_registry:
            if spec.model_id == model_id:
                return spec
        return None

    def _load_model(self) -> None:
        """Load or reload the vision model and processor"""
        self.logger.info(f"Loading vision model: {self.current_model_spec.name}")

        try:
            # Clean up existing model if present
            if self.model is not None:
                self.logger.debug("Cleaning up existing model...")
                del self.model
                del self.processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Determine compute device
            if (
                self.hardware_constraints.force_cpu
                or self.current_model_spec.min_gpu_vram_gb == 0
            ):
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.logger.info(f"Using device: {self.device}")

            # Log hardware info
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": (
                    torch.bfloat16 if self.device == "cuda" else torch.float32
                ),
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.device == "cpu":
                model_kwargs["device_map"] = {"": "cpu"}

            self.logger.debug("Loading vision model from transformers...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.current_model_spec.model_id, **model_kwargs
            )

            self.logger.debug("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.current_model_spec.model_id
            )

            self.logger.info(f"Model loaded successfully on {self.device}")
            self.stats.last_refresh_time = time.time()

        except Exception as e:
            self.logger.error(
                f"Failed to load model {self.current_model_spec.name}: {e}"
            )
            # Try fallback to a smaller model or CPU
            self._try_fallback_model()

    def _try_fallback_model(self) -> None:
        """Try to load a fallback model if primary model fails"""
        self.logger.info("Attempting to load fallback model...")

        # Try smaller models first
        fallback_models = sorted(
            [m for m in self.model_registry if m != self.current_model_spec],
            key=lambda m: m.quality_score,
            reverse=True,
        )

        for fallback_spec in fallback_models:
            if self._model_fits_constraints(fallback_spec):
                try:
                    self.logger.info(f"Trying fallback model: {fallback_spec.name}")
                    self.current_model_spec = fallback_spec
                    self._load_model()
                    self.stats.model_switches += 1
                    return
                except Exception as e:
                    self.logger.warning(
                        f"Fallback model {fallback_spec.name} also failed: {e}"
                    )
                    continue

        raise RuntimeError("All fallback models failed to load")

    def _should_refresh_model(self) -> bool:
        """Check if model should be refreshed based on usage and performance"""
        current_time = time.time()

        # Refresh based on document count
        if self.stats.documents_processed >= self.refresh_interval:
            self.logger.info(
                f"Refresh triggered: processed {self.stats.documents_processed} documents"
            )
            return True

        # Check GPU memory if available
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                memory_percent = (memory_used / memory_total) * 100

                if memory_percent > self.memory_threshold:
                    self.logger.warning(
                        f"Refresh triggered: GPU memory usage {memory_percent:.1f}%"
                    )
                    return True
            except Exception as e:
                self.logger.warning(f"Could not check GPU memory: {e}")

        # Check processing time degradation (refresh if processing is taking too long)
        if self.stats.documents_processed > 0:
            avg_time_per_doc = (
                self.stats.total_processing_time / self.stats.documents_processed
            )
            if avg_time_per_doc > 60.0:  # More than 1 minute per document
                self.logger.warning(
                    f"Refresh triggered: slow processing ({avg_time_per_doc:.1f}s per doc)"
                )
                return True

        return False

    def refresh_model(self) -> None:
        """Refresh the model to clear context and free memory"""
        self.logger.info("Refreshing vision model...")
        try:
            self._load_model()
            self.stats.memory_refreshes += 1
            # Reset some counters but keep overall stats
            processing_count = self.stats.documents_processed
            self.stats.documents_processed = 0
            self.logger.info(
                f"Model refresh completed (total processed: {processing_count})"
            )
        except Exception as e:
            self.logger.error(f"Failed to refresh model: {e}")
            raise

    def switch_model(self, new_model_id: str) -> bool:
        """Switch to a different model"""
        new_spec = self._get_model_spec_by_id(new_model_id)
        if not new_spec:
            self.logger.error(f"Model {new_model_id} not found in registry")
            return False

        if not self._model_fits_constraints(new_spec):
            self.logger.error(f"Model {new_model_id} doesn't fit hardware constraints")
            return False

        try:
            old_model = self.current_model_spec.name
            self.current_model_spec = new_spec
            self._load_model()
            self.stats.model_switches += 1
            self.logger.info(
                f"Successfully switched from {old_model} to {new_spec.name}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to model {new_model_id}: {e}")
            return False

    def _monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        memory_info = {}

        # System RAM
        ram = psutil.virtual_memory()
        memory_info["system_ram_used_gb"] = (ram.total - ram.available) / (1024**3)
        memory_info["system_ram_total_gb"] = ram.total / (1024**3)
        memory_info["system_ram_percent"] = ram.percent

        # GPU memory if available
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (
                    1024**3
                )
                memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (
                    1024**3
                )
                memory_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
                memory_info["gpu_memory_percent"] = (
                    memory_info["gpu_memory_used_gb"]
                    / memory_info["gpu_memory_total_gb"]
                ) * 100
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory info: {e}")

        return memory_info

    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for processing results"""
        metrics = {}

        # Text extraction quality
        text = result.get("text", "")
        if text and text not in ["NO_TEXT_FOUND", "TEXT_EXTRACTION_FAILED"]:
            metrics["text_extraction_success"] = 1.0
            metrics["text_length"] = len(text)
            # Simple quality heuristic based on text length and diversity
            unique_words = len(set(text.lower().split()))
            total_words = len(text.split())
            metrics["text_diversity"] = unique_words / max(total_words, 1)
        else:
            metrics["text_extraction_success"] = 0.0
            metrics["text_length"] = 0
            metrics["text_diversity"] = 0.0

        # Description quality
        description = result.get("description", "")
        if description and description != "DESCRIPTION_GENERATION_FAILED":
            metrics["description_success"] = 1.0
            metrics["description_length"] = len(description)
            # Check for descriptive keywords as quality indicator
            descriptive_keywords = [
                "color",
                "layout",
                "text",
                "image",
                "diagram",
                "chart",
                "table",
            ]
            keyword_count = sum(
                1 for keyword in descriptive_keywords if keyword in description.lower()
            )
            metrics["description_richness"] = keyword_count / len(descriptive_keywords)
        else:
            metrics["description_success"] = 0.0
            metrics["description_length"] = 0
            metrics["description_richness"] = 0.0

        return metrics

    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """Convert PDF pages to PIL Image objects with adaptive resolution"""
        pdf_path_obj = Path(pdf_path)
        self.logger.info(f"Converting PDF to images: {pdf_path_obj}")

        if not pdf_path_obj.exists():
            self.logger.error(f"PDF file not found: {pdf_path_obj}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        try:
            doc = fitz.open(str(pdf_path_obj))
            images = []
            page_count = len(doc)

            # Adjust resolution based on processing mode
            zoom_factors = {
                ProcessingMode.FAST: 1.5,
                ProcessingMode.BALANCED: 2.0,
                ProcessingMode.HIGH_QUALITY: 3.0,
            }
            zoom = zoom_factors.get(self.processing_mode, 2.0)

            self.logger.info(f"PDF has {page_count} pages, using {zoom}x zoom")

            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)

                    self.logger.debug(
                        f"Page {page_num + 1} converted - size: {img.size}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    continue

            doc.close()
            self.logger.info(f"Successfully converted {len(images)}/{page_count} pages")
            return images

        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def _process_single_image(
        self,
        image: Union[str, Path, Image.Image],
        extract_text: bool = True,
        describe_image: bool = True,
    ) -> Dict[str, Any]:
        """Process individual image with enhanced error handling and quality monitoring"""

        start_time = time.time()

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                processed_image = Image.open(image_path)
            else:
                processed_image = image

            results = {}

            # Text extraction
            if extract_text:
                try:
                    text_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image. Return only the text content, no explanations. If there's no text, return 'NO_TEXT_FOUND'.",
                                },
                            ],
                        }
                    ]

                    text_inputs = self.processor.apply_chat_template(
                        text_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(text_messages)

                    inputs = self.processor(
                        text=[text_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.current_model_spec.max_tokens,
                            do_sample=False,  # Deterministic for consistency
                        )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        text_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["text"] = text_output.strip()

                except Exception as e:
                    self.logger.error(f"Text extraction failed: {e}")
                    results["text"] = "TEXT_EXTRACTION_FAILED"

            # Image description
            if describe_image:
                try:
                    desc_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {
                                    "type": "text",
                                    "text": "Describe all visual elements in this image in detail. Include: layout, colors, objects, people, text formatting, charts/graphs, diagrams, symbols, and any other visual content. Be comprehensive but concise.",
                                },
                            ],
                        }
                    ]

                    desc_inputs = self.processor.apply_chat_template(
                        desc_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(desc_messages)

                    inputs = self.processor(
                        text=[desc_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.current_model_spec.max_tokens,
                            do_sample=False,
                        )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        desc_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["description"] = desc_output.strip()

                except Exception as e:
                    self.logger.error(f"Image description failed: {e}")
                    results["description"] = "DESCRIPTION_GENERATION_FAILED"

            # Calculate processing time and quality metrics
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["quality_metrics"] = self._calculate_quality_metrics(results)

            return results

        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            raise

    def extract_article_semantics(self, document: Union[str, Path]) -> Dict[str, str]:
        """Process document with enhanced monitoring and performance tracking"""
        document_obj = Path(document)
        self.logger.info(f"Starting document processing: {document_obj}")

        start_time = time.time()

        # Check if model should be refreshed
        if self._should_refresh_model():
            self.logger.info(
                "Auto-refreshing model due to usage threshold or memory pressure"
            )
            self.refresh_model()

        # Monitor memory before processing
        memory_before = self._monitor_memory_usage()
        self.logger.debug(
            f"Memory before processing: GPU {memory_before.get('gpu_memory_percent', 0):.1f}%, "
            f"RAM {memory_before.get('system_ram_percent', 0):.1f}%"
        )

        if not document_obj.exists():
            raise FileNotFoundError(f"Document file not found: {document_obj}")

        file_ext = document_obj.suffix.lower()
        self.logger.info(f"Document type: {file_ext}")

        all_text = []
        all_descriptions = []
        page_quality_metrics = []

        try:
            if file_ext == ".pdf":
                images = self._pdf_to_images(document_obj)
                total_pages = len(images)
                self.logger.info(f"Processing {total_pages} pages from PDF")

                for i, img in enumerate(images):
                    page_num = i + 1
                    self.logger.debug(f"Processing page {page_num}/{total_pages}")

                    try:
                        result = self._process_single_image(img)
                        page_quality_metrics.append(result.get("quality_metrics", {}))

                        # Handle text extraction
                        if result.get("text") and result["text"] not in [
                            "NO_TEXT_FOUND",
                            "TEXT_EXTRACTION_FAILED",
                        ]:
                            page_text = f"=== Page {page_num} ===\n{result['text']}"
                            all_text.append(page_text)
                            self.stats.text_extractions_successful += 1

                        # Handle descriptions
                        if (
                            result.get("description")
                            and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                        ):
                            page_desc = f"=== Page {page_num} Visual Description ===\n{result['description']}"
                            all_descriptions.append(page_desc)
                            self.stats.descriptions_successful += 1

                        self.stats.pages_processed += 1

                    except Exception as e:
                        self.logger.error(f"Failed to process page {page_num}: {e}")
                        continue

            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                self.logger.info(f"Processing image file: {document_obj}")

                result = self._process_single_image(document_obj)
                page_quality_metrics.append(result.get("quality_metrics", {}))

                if result.get("text") and result["text"] not in [
                    "NO_TEXT_FOUND",
                    "TEXT_EXTRACTION_FAILED",
                ]:
                    all_text.append(result["text"])
                    self.stats.text_extractions_successful += 1

                if (
                    result.get("description")
                    and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                ):
                    all_descriptions.append(result["description"])
                    self.stats.descriptions_successful += 1

                self.stats.pages_processed += 1

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Compile results
            final_text = (
                "\n\n".join(all_text) if all_text else "No text found in document."
            )
            final_descriptions = (
                "\n\n".join(all_descriptions)
                if all_descriptions
                else "No visual content described."
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.documents_processed += 1

            # Update average metrics
            if all_text:
                avg_text_length = sum(len(text) for text in all_text) / len(all_text)
                self.stats.average_text_length = (
                    self.stats.average_text_length
                    * (self.stats.documents_processed - 1)
                    + avg_text_length
                ) / self.stats.documents_processed

            if all_descriptions:
                avg_desc_length = sum(len(desc) for desc in all_descriptions) / len(
                    all_descriptions
                )
                self.stats.average_description_length = (
                    self.stats.average_description_length
                    * (self.stats.documents_processed - 1)
                    + avg_desc_length
                ) / self.stats.documents_processed

            # Monitor memory after processing
            memory_after = self._monitor_memory_usage()
            self.logger.debug(
                f"Memory after processing: GPU {memory_after.get('gpu_memory_percent', 0):.1f}%, "
                f"RAM {memory_after.get('system_ram_percent', 0):.1f}%"
            )

            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            self.logger.info(
                f"Results - Text: {len(final_text)} chars, Descriptions: {len(final_descriptions)} chars"
            )

            return {
                "all_text": final_text,
                "all_imagery": final_descriptions,
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {document_obj}: {e}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            "documents_processed": self.stats.documents_processed,
            "pages_processed": self.stats.pages_processed,
            "text_extractions_successful": self.stats.text_extractions_successful,
            "descriptions_successful": self.stats.descriptions_successful,
            "total_processing_time": self.stats.total_processing_time,
            "memory_refreshes": self.stats.memory_refreshes,
            "model_switches": self.stats.model_switches,
            "current_model": {
                "name": self.current_model_spec.name,
                "model_id": self.current_model_spec.model_id,
                "quality_score": self.current_model_spec.quality_score,
            },
            "device": self.device,
            "processing_mode": self.processing_mode.value,
        }

        # Calculate rates and averages
        if self.stats.documents_processed > 0:
            stats["avg_processing_time_per_doc"] = (
                self.stats.total_processing_time / self.stats.documents_processed
            )
            stats["text_extraction_success_rate"] = (
                self.stats.text_extractions_successful / self.stats.pages_processed
                if self.stats.pages_processed > 0
                else 0
            )
            stats["description_success_rate"] = (
                self.stats.descriptions_successful / self.stats.pages_processed
                if self.stats.pages_processed > 0
                else 0
            )
            stats["average_text_length"] = self.stats.average_text_length
            stats["average_description_length"] = self.stats.average_description_length

        # Add current memory usage
        stats["memory_usage"] = self._monitor_memory_usage()

        # Hardware constraints
        stats["hardware_constraints"] = {
            "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
            "max_ram_gb": self.hardware_constraints.max_ram_gb,
            "force_cpu": self.hardware_constraints.force_cpu,
        }

        return stats

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of models that fit current hardware constraints"""
        available_models = []
        for model_spec in self.model_registry:
            fits = self._model_fits_constraints(model_spec)
            available_models.append(
                {
                    "name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "quality_score": model_spec.quality_score,
                    "min_gpu_vram_gb": model_spec.min_gpu_vram_gb,
                    "min_ram_gb": model_spec.min_ram_gb,
                    "fits_constraints": fits,
                    "is_current": model_spec == self.current_model_spec,
                }
            )
        return available_models

    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        stats = self.get_processing_stats()
        suggestions = []

        # Check success rates
        text_success_rate = stats.get("text_extraction_success_rate", 0)
        desc_success_rate = stats.get("description_success_rate", 0)

        if text_success_rate < 0.8:
            suggestions.append(
                "Consider switching to a higher quality model for better text extraction"
            )

        if desc_success_rate < 0.8:
            suggestions.append(
                "Consider adjusting processing mode to HIGH_QUALITY for better descriptions"
            )

        # Check processing speed
        avg_time = stats.get("avg_processing_time_per_doc", 0)
        if avg_time > 30:
            suggestions.append(
                "Processing is slow - consider using FAST mode or switching to a smaller model"
            )

        # Check memory usage
        memory_usage = stats.get("memory_usage", {})
        gpu_percent = memory_usage.get("gpu_memory_percent", 0)
        if gpu_percent > 90:
            suggestions.append(
                "GPU memory usage is very high - consider reducing refresh interval or using CPU mode"
            )

        return {
            "current_performance": stats,
            "optimization_suggestions": suggestions,
            "available_models": self.get_available_models(),
        }


"""
LLM-based field extraction with hardware auto-detection and context management

This module provides intelligent document field extraction using local LLM models through OLLAMA.
It automatically detects hardware capabilities, selects optimal models, and manages context to
prevent performance degradation during batch processing.

Key Features:
- Automatic hardware detection and model selection
- Context refresh to prevent model degradation
- Multiple extraction strategies (single/multi-prompt)
- Comprehensive validation and error handling
- Performance monitoring and statistics

Dependencies:
- requests: HTTP communication with OLLAMA API
- psutil: System hardware detection
- GPUtil: GPU memory detection (optional)
"""

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


@dataclass
class ModelSpec:
    """
    Specification for an LLM model including resource requirements and capabilities.

    Attributes:
        name (str): OLLAMA model identifier (e.g., "llama3.1:8b")
        min_ram_gb (float): Minimum system RAM required in gigabytes
        min_gpu_vram_gb (float): Minimum GPU VRAM required in gigabytes (0 for CPU-only)
        min_cpu_cores (int): Minimum CPU cores required for reasonable performance
        context_window (int): Maximum context length in tokens the model supports
        quality_score (int): Subjective quality rating from 1-10 (10 being best)
                           Based on accuracy, coherence, and instruction following
    """

    name: str
    min_ram_gb: float
    min_gpu_vram_gb: float
    min_cpu_cores: int
    context_window: int
    quality_score: int  # 1-10 scale, higher indicates better extraction quality


@dataclass
class HardwareConstraints:
    """
    Available system hardware resources for model execution.

    Attributes:
        max_ram_gb (float): Available system RAM in gigabytes
        max_gpu_vram_gb (float): Available GPU memory in gigabytes (0 if no GPU)
        max_cpu_cores (int): Number of available CPU cores for processing
    """

    max_ram_gb: float
    max_gpu_vram_gb: float
    max_cpu_cores: int


class LanguageProcessor:
    """
    Enhanced LLM-based document field extractor with intelligent model management.

    This class handles document field extraction using local LLM models via OLLAMA.
    It automatically detects system capabilities, selects appropriate models, and
    manages extraction context to maintain consistent performance.

    Key Features:
    - Hardware-aware model selection
    - Context refresh to prevent degradation
    - Multiple extraction strategies
    - Comprehensive error handling and logging
    - Performance monitoring

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> processor = LanguageProcessor(
        ...     logger=logger,
        ...     auto_model_selection=True
        ... )
        >>> result = processor.extract_fields(
        ...     ocr_text="Document content...",
        ...     visual_description="Visual analysis...",
        ...     metadata={"filename": "doc.pdf"},
        ...     uuid="doc-123"
        ... )
    """

    # ====================================================================================
    # CLASS CONSTANTS - Configuration parameters that control system behavior
    # ====================================================================================

    # Field extraction with full data and system prompts
    FIELD_PROMPTS = {
        "title": """Create a descriptive title that captures the document's purpose and content.

        If the document has an official title, use it. If not, synthesize a descriptive title based on:
        - Document type and purpose
        - Issuing organization
        - Main subject matter

        Examples:
        - "Immigration and Refugee Board Virtual Hearing Notice"
        - "Refugee Protection Division Hearing Preparation Guide"
        - "IRB Document Submission Requirements and Procedures"

        Return ONLY the title. No quotes, no explanations.""",
        "document_type": """Identify the specific type of document this is.

    Examples of document types:
    - birth_certificate, passport_us, driver_license_ca
    - invoice, receipt, bank_statement, credit_report
    - contract, lease_agreement, employment_contract
    - medical_record, prescription, insurance_policy
    - w2_tax_form, 1099, tax_return
    - court_order, legal_notice, business_license
    - university_transcript, diploma, certificate
    - property_deed, mortgage_document, warranty

    Return ONLY the specific document type in lowercase with underscores. If uncertain, return "UNKNOWN".""",
        "language": """Analyze the document text and identify ALL languages present.

    Document contains sections in multiple languages. List ALL languages found.
    Examples:
    - If English only: "English"
    - If French only: "French"
    - If both: "English, French"
    - If trilingual: "English, French, Spanish"

    Return ONLY language names, comma-separated.""",
        "confidentiality_level": """Determine the confidentiality or security classification of this document.

    Look for markings or indicators such as:
    - CONFIDENTIAL, CLASSIFIED, SECRET, TOP SECRET
    - INTERNAL USE ONLY, PROPRIETARY, RESTRICTED
    - PUBLIC, FOR PUBLIC RELEASE
    - Security stamps, watermarks, or headers

    Classification levels:
    - Public: No restrictions, can be freely shared
    - Internal: Internal use within organization
    - Confidential: Sensitive information, restricted access
    - Restricted: Highly sensitive, top-secret information

    Return ONLY one of these words: Public, Internal, Confidential, Restricted""",
        "translator_name": """If this document has been translated, identify the translator's name.

    Look for:
    - Translator certifications or signatures
    - Translation agency information
    - "Translated by" notices
    - Official translation stamps or seals
    - Translator contact information

    Return ONLY the translator's full name. If this is not a translated document or no translator is identified, return "UNKNOWN".""",
        "issuer_name": """Identify who issued, created, or published this document.

    Look for:
    - Organization names, agencies, departments
    - Company names or letterheads
    - Government agencies or institutions
    - Individual names (for personally issued documents)
    - Official stamps or seals with issuer information

    Return ONLY the full official name of the issuer. If unclear, return "UNKNOWN".""",
        "officiater_name": """Identify any official authority that validated, certified, witnessed, or authorized this document.

    Look for:
    - Notary public names and seals
    - Certifying agency names
    - Official witnesses or authorizing bodies
    - Government officials who signed or stamped
    - Licensing boards or regulatory authorities

    Return ONLY the name of the official authority. If no official validation exists, return "UNKNOWN".""",
        "date_created": """Find when this document was originally created, written, or authored.

    Look for:
    - Creation dates, authored dates, written dates
    - "Created on", "Date created", "Authored"
    - Document composition or drafting dates

    Return the date in YYYY-MM-DD format. If no creation date is found, return "UNKNOWN".""",
        "date_of_reception": """Find when this document was received by the current holder.

    Look for:
    - "Received", "Date received", "Arrival date"
    - Postal stamps or delivery confirmations
    - Filing dates or intake dates
    - "Delivered on" stamps

    Return the date in YYYY-MM-DD format. If no reception date is found, return "UNKNOWN".""",
        "date_of_issue": """Find the official issue, publication, or release date of this document.

    Look for:
    - "Issued", "Date of issue", "Publication date"
    - "Released", "Effective date"
    - Official dating stamps or seals
    - Government or agency issue dates

    Return the date in YYYY-MM-DD format. If no issue date is found, return "UNKNOWN".""",
        "date_of_expiry": """Find when this document expires, becomes invalid, or requires renewal.

    Look for:
    - "Expires", "Expiration date", "Valid until"
    - "Renewal required", "Valid through"
    - License or certification expiry dates
    - "Not valid after" dates

    Return the date in YYYY-MM-DD format. If no expiration date exists, return "UNKNOWN".""",
        "tags": """Create comprehensive keywords that describe this document for search and categorization purposes.

    Include keywords for:
    - Document category (legal, medical, financial, educational, personal, business, government, technical)
    - Subject matter (taxes, healthcare, employment, education, property, travel, identification, insurance)
    - Content type (contract, certificate, statement, report, application, notice, invoice, receipt)
    - Industry/field (healthcare, legal, finance, education, technology, government, military)
    - Geographic relevance (federal, state, local, international, specific regions)
    - Time relevance (annual, quarterly, monthly, historical, current)
    - Action items (renewal_required, payment_due, action_needed, informational_only)
    - Format type (official, certified, notarized, electronic, handwritten, typed)

    Return 15-25 comma-separated keywords. If document content is unclear, return "UNKNOWN".""",
        "version_notes": """Analyze document versioning, revision history, and administrative metadata.

    Look for:
    - Version numbers, revision dates, edition information
    - Document control numbers, form numbers
    - "Supersedes" notices, amendment references
    - Administrative tracking information

    Provide a professional assessment of document currency and version status.
    If no versioning found, state: "No explicit version control information identified."

    Use formal, administrative language.""",
        "utility_notes": """Provide a professional analysis of this document's administrative function and legal purpose.

    Analyze:
    - Regulatory or statutory requirements this document fulfills
    - Administrative processes it initiates or supports
    - Legal obligations or rights it establishes
    - Institutional workflows it facilitates
    - Compliance requirements it addresses

    Write in formal, bureaucratic language appropriate for government documentation.""",
        "additional_notes": """Document significant administrative, security, or procedural characteristics not covered elsewhere.

    Note:
    - Security classifications, handling restrictions
    - Authentication elements, official markings
    - Distribution methods, transmission records
    - Document quality, preservation concerns
    - Cross-references to related administrative instruments

    Present observations in formal, official terminology suitable for administrative records.""",
    }

    SYSTEM_PROMPT = """You are a document extraction tool. Extract ONLY the requested information.

    CRITICAL RULES:
    - Return ONLY the answer, nothing else
    - NO explanations, NO reasoning, NO "based on", NO "therefore"
    - NO sentences, just the raw information
    - If not found, return exactly: UNKNOWN

    Examples:
    GOOD: "Immigration and Refugee Board of Canada"
    BAD: "The document appears to be issued by the Immigration and Refugee Board of Canada"

    GOOD: "UNKNOWN"
    BAD: "document does not appear to have any official authority validating, certifying, witnessing, or authorizing it. Therefore, the answer is: UNKNOWN"

    Extract the information. Nothing else.

    Return ONLY the requested information. Any additional text, explanation, or reasoning will be considered an error and rejected causing immediate shutdown
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_ram_gb: Optional[float] = None,
        max_gpu_vram_gb: Optional[float] = None,
        max_cpu_cores: Optional[int] = None,
        auto_model_selection: bool = True,
    ):
        """Initialize the LanguageProcessor with proper error handling."""

        # Initialize ALL instance variables FIRST
        self.logger = logger

        # Initialize tracking counters BEFORE any method calls
        self.processed_count = 0
        self.prompts_since_refresh = 0
        self.total_processing_time = 0.0
        self.successful_extractions = 0
        self.failed_extractions = 0

        self.logger.info("=== Initializing LanguageProcessor ===")

        try:
            # Now do hardware detection
            self.hardware_constraints = self._detect_hardware_constraints(
                max_ram_gb, max_gpu_vram_gb, max_cpu_cores
            )

            # Initialize model registry
            self.model_registry = self._initialize_model_registry()
            self.logger.info(f"Loaded {len(self.model_registry)} model specifications")

            # Select and load optimal model
            if auto_model_selection:
                self.model = self._select_optimal_model()
            else:
                self.model = "mistral:7b"  # "mistral-nemo:12b"
                self.logger.warning(f"Using specified fallback model: {self.model}")

            # Load the model
            self.client = ollama.Client()

            self.logger.info("Language processor initialization completed successfully")
            self.logger.info(f"Active model: {self.model}")
            self.logger.info(
                f"Hardware constraints: RAM={self.hardware_constraints.max_ram_gb:.1f}GB, "
                f"GPU={self.hardware_constraints.max_gpu_vram_gb:.1f}GB, "
                f"CPU={self.hardware_constraints.max_cpu_cores} cores"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LanguageProcessor: {e}")

    def _detect_hardware_constraints(
        self,
        max_ram_gb: Optional[float],
        max_gpu_vram_gb: Optional[float],
        max_cpu_cores: Optional[int],
    ) -> HardwareConstraints:
        """
        Detect available hardware resources or use provided constraints.

        This method intelligently detects system capabilities to inform model selection.
        It uses conservative estimates to ensure stable operation under load.

        Args:
            max_ram_gb: Override for maximum RAM usage (None for auto-detection)
            max_gpu_vram_gb: Override for maximum GPU memory (None for auto-detection)
            max_cpu_cores: Override for CPU core count (None for auto-detection)

        Returns:
            HardwareConstraints object with detected or specified limits

        Note:
            Auto-detected RAM is limited to 70% of total to leave headroom for OS and
            other applications. GPU detection requires GPUtil package.
        """
        self.logger.info("--- Detecting Hardware Constraints ---")

        # RAM detection with conservative allocation
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (
                1024**3
            )  # Bytes to GB conversion
            available_ram = total_ram * self.DEFAULT_RAM_USAGE_RATIO
            self.logger.info(
                f"RAM detection: total={total_ram:.1f}GB, "
                f"allocated={available_ram:.1f}GB "
                f"({self.DEFAULT_RAM_USAGE_RATIO*100:.0f}% of total)"
            )
        else:
            available_ram = max_ram_gb
            self.logger.info(f"RAM override: using specified {available_ram:.1f}GB")

        # CPU detection with logical vs physical core consideration
        if max_cpu_cores is None:
            # Use physical cores for more accurate performance estimation
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_cores = psutil.cpu_count(logical=True) or 1
            available_cores = physical_cores  # Conservative estimate
            self.logger.info(
                f"CPU detection: physical={physical_cores}, "
                f"logical={logical_cores}, allocated={available_cores}"
            )
        else:
            available_cores = max_cpu_cores
            self.logger.info(f"CPU override: using specified {available_cores} cores")

        # GPU detection with error handling for missing dependencies
        if max_gpu_vram_gb is None:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    # Find GPU with most memory (GB conversion from MB)
                    available_gpu = max([gpu.memoryTotal / 1024 for gpu in gpus])
                    gpu_names = [gpu.name for gpu in gpus]
                    self.logger.info(
                        f"GPU detection: found {len(gpus)} GPU(s), "
                        f"max_memory={available_gpu:.1f}GB"
                    )
                    self.logger.debug(f"GPU details: {gpu_names}")
                else:
                    available_gpu = 0
                    self.logger.info("GPU detection: no GPUs found")
            except ImportError:
                available_gpu = 0
                self.logger.warning(
                    "GPU detection: GPUtil not available, assuming CPU-only mode"
                )
            except Exception as e:
                available_gpu = 0
                self.logger.error(f"GPU detection failed: {e}, assuming CPU-only mode")
        else:
            available_gpu = max_gpu_vram_gb
            self.logger.info(f"GPU override: using specified {available_gpu:.1f}GB")

        constraints = HardwareConstraints(
            max_ram_gb=available_ram,
            max_gpu_vram_gb=available_gpu,
            max_cpu_cores=available_cores,
        )

        self.logger.info(f"Hardware constraints finalized: {constraints}")
        return constraints

    def _initialize_model_registry(self) -> List[ModelSpec]:
        """
        Initialize registry of available models with their resource requirements.

        This registry contains empirically determined resource requirements for various
        LLM models. Requirements are conservative estimates to ensure stable operation.

        Returns:
            List of ModelSpec objects with requirements and quality ratings

        Note:
            Quality scores are subjective ratings (1-10) based on:
            - Accuracy of field extraction
            - Instruction following capability
            - Consistency across document types
            - Speed vs accuracy trade-offs
        """
        self.logger.debug("Initializing model registry with resource requirements")

        models = [
            # Small, fast models for resource-constrained environments
            ModelSpec("llama3.2:1b", 2, 0, 2, 2048, 6),  # Basic accuracy, very fast
            ModelSpec("llama3.2:3b", 4, 0, 4, 2048, 7),  # Good balance for small docs
            # Medium models for balanced performance
            ModelSpec("llama3.1:8b", 8, 0, 6, 4096, 8),  # Excellent general purpose
            ModelSpec("qwen2.5:7b", 7, 0, 4, 4096, 8),  # Strong multilingual support
            ModelSpec("mistral:7b", 7, 0, 4, 4096, 7),  # Fast and reliable
            ModelSpec("gemma2:9b", 9, 6, 6, 8192, 8),  # Good with structured data
            ModelSpec(
                "codellama:13b", 13, 8, 8, 4096, 8
            ),  # Excellent for technical docs
            # Large models for maximum accuracy
            ModelSpec("qwen2.5:14b", 14, 8, 8, 8192, 9),  # Superior multilingual
            ModelSpec("llama3.1:70b", 32, 24, 16, 8192, 10),  # Highest quality, slow
        ]

        self.logger.info(f"Model registry initialized with {len(models)} models")
        self.logger.debug(
            "Model quality scores: "
            + ", ".join([f"{m.name}={m.quality_score}" for m in models])
        )

        return models

    def _select_optimal_model(self) -> str:
        """
        Select the best model that fits within hardware constraints.

        Selection algorithm:
        1. Filter models that fit within hardware limits
        2. Rank by quality score (accuracy/capability metric)
        3. Select highest-scoring model
        4. Fallback to smallest model if none fit

        Returns:
            String identifier of the selected model

        Note:
            Quality scores range from 1-10 where:
            - 1-3: Basic functionality, limited accuracy
            - 4-6: Adequate for simple documents
            - 7-8: Good for most document types
            - 9-10: Excellent accuracy, handles complex documents
        """
        self.logger.info("--- Selecting Optimal Model ---")
        self.logger.debug(f"Evaluating {len(self.model_registry)} candidate models")

        suitable_models = []
        rejected_models = []

        for model in self.model_registry:
            fits_constraints = (
                model.min_ram_gb <= self.hardware_constraints.max_ram_gb
                and model.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
                and model.min_cpu_cores <= self.hardware_constraints.max_cpu_cores
            )

            if fits_constraints:
                suitable_models.append(model)
                self.logger.debug(
                    f"✓ {model.name}: quality={model.quality_score}, "
                    f"ram={model.min_ram_gb}GB, gpu={model.min_gpu_vram_gb}GB, "
                    f"cores={model.min_cpu_cores}"
                )
            else:
                rejected_models.append(model)
                reasons = []
                if model.min_ram_gb > self.hardware_constraints.max_ram_gb:
                    reasons.append(
                        f"RAM: need {model.min_ram_gb}GB, have {self.hardware_constraints.max_ram_gb:.1f}GB"
                    )
                if model.min_gpu_vram_gb > self.hardware_constraints.max_gpu_vram_gb:
                    reasons.append(
                        f"GPU: need {model.min_gpu_vram_gb}GB, have {self.hardware_constraints.max_gpu_vram_gb:.1f}GB"
                    )
                if model.min_cpu_cores > self.hardware_constraints.max_cpu_cores:
                    reasons.append(
                        f"CPU: need {model.min_cpu_cores} cores, have {self.hardware_constraints.max_cpu_cores}"
                    )

                self.logger.debug(f"✗ {model.name}: {'; '.join(reasons)}")

        if not suitable_models:
            self.logger.warning(
                f"No models fit hardware constraints! "
                f"Rejected {len(rejected_models)} models. "
                f"Using minimal fallback model."
            )
            return "llama3.2:1b"  # Smallest available model

        # Select model with highest quality score
        best_model = max(suitable_models, key=lambda m: m.quality_score)

        self.logger.info(f"Model selection complete:")
        self.logger.info(f"  Selected: {best_model.name}")
        self.logger.info(f"  Quality score: {best_model.quality_score}/10")
        self.logger.info(
            f"  Requirements: {best_model.min_ram_gb}GB RAM, "
            f"{best_model.min_gpu_vram_gb}GB GPU, {best_model.min_cpu_cores} cores"
        )
        self.logger.info(f"  Context window: {best_model.context_window:,} tokens")
        self.logger.info(f"  Alternatives considered: {len(suitable_models)-1}")

        return best_model.name

    def extract_fields(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        uuid: str,
    ) -> Dict[str, Any]:
        """
        Extract all document fields using LLM with comprehensive error handling.
        """
        start_time = datetime.now()
        self.logger.info(f"=== Starting field extraction for document {uuid} ===")

        self.logger.debug("Extracting fields from summaries...")

        self.logger.info(
            f"Input sizes: OCR={len(ocr_text)} chars, "
            f"d{len(visual_description)} chars, "
            f"Metadata keys={list(metadata.keys())}"
        )

        try:

            # Updated extraction loop
            extracted_fields = {}
            for (
                field_name,
                field_instruction,
            ) in LanguageProcessor.FIELD_PROMPTS.items():
                try:
                    self.logger.debug(f"Extracting field: {field_name}")

                    # Create complete prompt with system prompt and full data
                    complete_prompt = f"""
                    {LanguageProcessor.SYSTEM_PROMPT}

                    DOCUMENT DATA:
                    =============
                    OCR TEXT CONTENT:
                    {ocr_text}

                    VISUAL DESCRIPTION:
                    {visual_description}

                    TASK:
                    =====
                    {field_instruction}"""

                    # Send prompt to LLM
                    response = self.client.generate(
                        model=self.model, prompt=complete_prompt, stream=False
                    )

                    response_text = response["response"]
                    extracted_fields[field_name] = response_text

                    self.logger.info(
                        f"Extracted field data for {field_name} with {response_text}"
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_fields[field_name] = "UNKNOWN"

            # Update performance counters
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processed_count += 1
            self.successful_extractions += 1
            self.total_processing_time += processing_time

            self.logger.info(f"=== Extraction completed successfully for {uuid} ===")
            self.logger.info(f"Processing time: {processing_time:.2f}s")

            return {
                "success": True,
                "extracted_fields": extracted_fields,
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "processing_time_seconds": processing_time,
            }

        except Exception as e:
            # Comprehensive error handling with context preservation
            processing_time = (datetime.now() - start_time).total_seconds()
            self.failed_extractions += 1

            self.logger.error(f"=== Field extraction FAILED for {uuid} ===")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Processing time before failure: {processing_time:.2f}s")
            self.logger.error(
                f"Session stats: {self.processed_count} total, "
                f"{self.failed_extractions} failed"
            )

            # Log additional context for debugging
            if hasattr(e, "__traceback__"):
                import traceback

                self.logger.debug("Full traceback:", exc_info=True)

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "model_used": self.model,
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics and system status.

        Returns detailed information about processing performance, hardware
        utilization, and system configuration for monitoring and optimization.

        Returns:
            Dictionary containing:
            - Processing counters and success rates
            - Performance metrics (timing, throughput)
            - Hardware configuration and constraints
            - Model information and settings
            - Context management status

        Note:
            Statistics are cumulative since processor initialization.
            Call this method regularly for performance monitoring.
        """
        # Calculate derived metrics
        total_attempts = self.successful_extractions + self.failed_extractions
        success_rate = (
            (self.successful_extractions / total_attempts * 100)
            if total_attempts > 0
            else 0
        )

        avg_processing_time = (
            (self.total_processing_time / self.processed_count)
            if self.processed_count > 0
            else 0
        )

        # Estimate throughput (documents per hour)
        if self.total_processing_time > 0:
            throughput_per_hour = (
                self.processed_count / self.total_processing_time
            ) * 3600
        else:
            throughput_per_hour = 0

        stats = {
            # Processing Statistics
            "processing": {
                "total_processed": self.processed_count,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "success_rate_percent": round(success_rate, 2),
                "total_processing_time_seconds": round(self.total_processing_time, 2),
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "estimated_throughput_per_hour": round(throughput_per_hour, 1),
            },
            # Model Configuration
            "model": {
                "current_model": self.model,
                "available_models": len(self.model_registry),
            },
            # Hardware Configuration
            "hardware": {
                "max_ram_gb": self.hardware_constraints.max_ram_gb,
                "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
                "max_cpu_cores": self.hardware_constraints.max_cpu_cores,
                "gpu_available": self.hardware_constraints.max_gpu_vram_gb > 0,
            },
            # Timestamp
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.debug(
            f"Generated stats: {stats['processing']['total_processed']} processed, "
            f"{stats['processing']['success_rate_percent']}% success rate"
        )

        return stats
