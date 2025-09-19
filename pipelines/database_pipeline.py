# =====================================================================
# STAGE 7: DATABASE FORMATION
# =====================================================================

profiles: List[Path] = [item for item in PATHS["profiles_dir"].iterdir()]

if len(profiles) > 0:  # Fixed condition - process when we HAVE profiles

    logger.info("=" * 80)
    logger.info("DATABASE FORMATION STAGE")
    logger.info("=" * 80)
    logger.info("Creating final spreadsheet databases from processed profiles...")

    logger.info(f"Found {len(profiles)} profiles ready for database formation")

    try:

        if output_files:
            logger.info("Database formation completed successfully!")

            # Update session data with database creation info
            session_data["database_files"] = {}

            for file_type, file_path in output_files.items():
                logger.info(f"  {file_type.upper()}: {file_path}")
                session_data["database_files"][file_type] = str(file_path)

                # Update stage counts for completed database formation
                session_data["stage_counts"]["database_formed"] = len(profiles)

            # Log database statistics
            logger.info(f"Database contains {len(profiles)} document profiles")

            # Calculate some summary statistics
            completed_profiles = 0
            successful_llm_extractions = 0

            for profile in profiles:
                try:
                    with open(profile, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)

                    # Count completed profiles
                    if "completed" in profile_data.get("stages", {}):
                        completed_profiles += 1

                    # Count successful LLM extractions
                    if profile_data.get("llm_extraction", {}).get("success", False):
                        successful_llm_extractions += 1

                except Exception as e:
                    logger.debug(
                        f"Failed to read profile {profile.name} for statistics: {e}"
                    )

            logger.info(f"  - {completed_profiles} fully completed documents")
            logger.info(
                f"  - {successful_llm_extractions} successful LLM field extractions"
            )
            logger.info(
                f"  - {len(profiles) - completed_profiles} documents with partial processing"
            )

            # Add database formation timestamp
            # session_data["stages"]["database_formation"] = {
            #     "status": "completed",
            #     "timestamp": datetime.now().isoformat(),
            #     "files_created": list(output_files.keys()),
            #     "total_profiles_exported": len(profiles),
            # }

        else:
            logger.warning("No database files were created")
            # session_data["stages"]["database_formation"] = {
            #     "status": "failed",
            #     "timestamp": datetime.now().isoformat(),
            #     "reason": "No output files generated",
            # }

    except Exception as e:
        logger.error(f"Database formation failed: {e}")
        # session_data["errors"].append(
        #     {
        #         "file": "database_formation",
        #         "stage": "database_formation",
        #         "error": str(e),
        #         "timestamp": datetime.now().isoformat(),
        #     }
        # )

        # session_data["stages"]["database_formation"] = {
        #     "status": "failed",
        #     "timestamp": datetime.now().isoformat(),
        #     "error": str(e),
        # }

    # Update session JSON with final database information
    update_session_json()

    logger.info("Database formation stage complete")

else:
    logger.warning("No profiles found for database formation")
    # session_data["stages"]["database_formation"] = {
    #     "status": "skipped",
    #     "timestamp": datetime.now().isoformat(),
    #     "reason": "No profiles found",
    # }
