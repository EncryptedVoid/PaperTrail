# =====================================================================
# STAGE 6: COMPLETION AND FINAL PROCESSING
# =====================================================================

# Get all artifacts from LLM processed directory, sorted by size
llm_processed_artifacts: List[Path] = [
    item for item in PATHS["process_completed_dir"].iterdir()
]

if len(llm_processed_artifacts) > 0:

    logger.info("=" * 80)
    logger.info("FINAL PROCESSING STAGE")
    logger.info("=" * 80)
    logger.info("Moving fully processed artifacts to completion directory...")

    llm_processed_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(f"Found {len(llm_processed_artifacts)} artifacts ready for completion")

    for artifact in tqdm(
        llm_processed_artifacts, desc="Completing processing", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Mark as completed in profile
            profile_data["stages"]["completed"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "total_stages_completed": len(
                    [
                        s
                        for s in profile_data["stages"].values()
                        if s.get("status") == "completed"
                    ]
                ),
            }

            # Add final processing summary
            profile_data["processing_summary"] = {
                "total_file_size_mb": profile_data.get("file_size", 0) / (1024 * 1024),
                "stages_completed": list(profile_data["stages"].keys()),
                "has_ocr_text": bool(
                    profile_data.get("semantics", {}).get("all_text", "").strip()
                ),
                "has_visual_description": bool(
                    profile_data.get("semantics", {}).get("all_imagery", "").strip()
                ),
                "llm_extraction_success": profile_data.get("llm_extraction", {}).get(
                    "success", False
                ),
                "fields_with_data": (
                    len(
                        [
                            v
                            for v in profile_data.get("llm_extraction", {})
                            .get("extracted_fields", {})
                            .values()
                            if v != "UNKNOWN"
                        ]
                    )
                    if profile_data.get("llm_extraction", {}).get("success")
                    else 0
                ),
            }

            # Save final profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to completed directory
            final_artifact = artifact.rename(
                PATHS["process_completed_dir"] / artifact.name
            )

            # Update session tracking
            update_stage_counts("llm_processed", "completed", session_data)

            # Log detailed progress
            file_size = final_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "completed")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Processing completed for: {artifact.name}")

        except Exception as e:
            logger.error(f"Failed to complete processing for {artifact.name}: {e}")
            session_data["errors"].append(
                {"file": artifact.name, "stage": "completion", "error": str(e)}
            )
