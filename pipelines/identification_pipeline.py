# =====================================================================
    # STAGE 2: UUID RENAMING AND PROFILE CREATION
    # =====================================================================

    logger.info("=" * 80)
    logger.info("UUID RENAMING AND PROFILE CREATION STAGE")
    logger.info("=" * 80)
    logger.info("Renaming artifacts with unique UUIDs and creating profile files...")

    remaining_artifacts: List[Path] = [
        item for item in PATHS["unprocessed_dir"].iterdir()
    ]

    for artifact in tqdm(
        remaining_artifacts, desc="Preparing artifact profiles", unit="artifact"
    ):
        try:
            # Generate unique identifier for this artifact
            artifact_id: str = generate_uuid4plus()
            original_name = artifact.name
            file_size = artifact.stat().st_size

            # Create initial profile data
            profile_data = {
                "uuid": artifact_id,
                "checksum": generate_checksum(artifact, algorithm=CHECKSUM_ALGORITHM),
                "original_filename": original_name,
                "file_size": file_size,
                "file_extension": artifact.suffix.lower(),
                "stages": {
                    "renamed": {
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                    }
                },
            }

            # Rename artifact with UUID (preserving extension)
            new_artifact_name = f"ARTIFACT-{artifact_id}{artifact.suffix}"
            renamed_artifact = artifact.rename(PATHS["rename_dir"] / new_artifact_name)

            # Create corresponding profile file
            profile_filename = f"PROFILE-{artifact_id}.json"
            profile_path = PATHS["profiles_dir"] / profile_filename

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

            # Update session tracking
            session_data["processed_files"] += 1
            session_data["stage_counts"]["unprocessed"] -= 1
            session_data["stage_counts"]["rename"] += 1

            # Log detailed progress
            log_detailed_progress(new_artifact_name, file_size, "renamed")

            # Update session JSON after each file
            update_session_json()

            logger.debug(f"Renamed: {original_name} -> {new_artifact_name}")
            logger.debug(f"Created profile: {profile_filename}")

        except Exception as e:
            logger.error(f"Failed to rename {artifact.name}: {e}")
            session_data["errors"].append(
                {"file": artifact.name, "stage": "rename", "error": str(e)}
            )

    logger.info(
        f"Renaming stage complete - {session_data['stage_counts']['rename']} files renamed"
    )