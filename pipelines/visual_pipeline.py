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
