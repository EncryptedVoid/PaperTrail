# =====================================================================
# STAGE 3: METADATA EXTRACTION
# =====================================================================

# Get all artifacts from rename directory, sorted by size
renamed_artifacts: List[Path] = [item for item in PATHS["rename_dir"].iterdir()]

if len(renamed_artifacts) > 0:
    logger.info("=" * 80)
    logger.info("AUTOMATED TECHNICAL METADATA EXTRACTION STAGE")
    logger.info("=" * 80)
    logger.info("Extracting technical metadata from renamed artifacts...")

    renamed_artifacts.sort(key=lambda p: p.stat().st_size)

    logger.info(
        f"Found {len(renamed_artifacts)} artifacts ready for metadata extraction"
    )

    # Initialize metadata extractor
    extractor = MetadataExtractor(logger=logger)
    logger.info("Document Metadata Extractor initialized")

    for artifact in tqdm(
        renamed_artifacts, desc="Extracting metadata", unit="artifact"
    ):
        try:
            # Extract UUID from filename for profile lookup
            artifact_id = artifact_id = artifact.stem[9:]
            profile_path = PATHS["profiles_dir"] / f"PROFILE-{artifact_id}.json"

            # Load existing profile
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Extract metadata using the extractor
            logger.info(f"Extracting metadata for: {artifact.name}")
            extracted_metadata: Dict[str, Any] = extractor.extract(artifact)

            # Update profile with metadata and stage completion
            profile_data["metadata"] = extracted_metadata
            profile_data["stages"]["metadata_extraction"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Move artifact to next stage directory
            moved_artifact = artifact.rename(PATHS["metadata_dir"] / artifact.name)

            # Update session tracking
            update_stage_counts("rename", "metadata", session_data)

            # Log detailed progress
            file_size = moved_artifact.stat().st_size
            log_detailed_progress(artifact.name, file_size, "metadata_extraction")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Metadata extracted and saved for: {artifact.name}")

            STAGE_3 = True

        except Exception as e:
            logger.error(f"Failed to extract metadata for {artifact.name}: {e}")

            # Create failed subdirectory if needed and move file there
            failed_dir = PATHS["rename_dir"] / "failed"
            failed_dir.mkdir(exist_ok=True)

            try:
                artifact.rename(failed_dir / artifact.name)
                update_stage_counts("rename", "failed", session_data)
                logger.info(f"Moved failed file to: {failed_dir / artifact.name}")
            except Exception as move_error:
                logger.error(
                    f"Failed to move {artifact.name} to failed directory: {move_error}"
                )

            session_data["errors"].append(
                {"file": artifact.name, "stage": "metadata_extraction", "error": str(e)}
            )

    logger.info(
        f"Metadata extraction complete - {session_data['stage_counts']['metadata']} files processed"
    )
