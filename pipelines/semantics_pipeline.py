# =====================================================================
# STAGE 5: LLM FIELD EXTRACTION (SEMANTIC METADATA)
# =====================================================================


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
