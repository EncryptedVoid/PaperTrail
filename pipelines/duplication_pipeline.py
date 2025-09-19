# =====================================================================
# STAGE 1: DUPLICATE DETECTION AND CHECKSUM VERIFICATION
# =====================================================================

# Get all artifacts in unprocessed directory and sort by size (smallest first)
unprocessed_artifacts: List[Path] = [
    item for item in PATHS["unprocessed_dir"].iterdir()
]

if len(unprocessed_artifacts) > 0:

    logger.info("=" * 80)
    logger.info(
        "DUPLICATE DETECTION, UNSUPPORTED FILES, AND CHECKSUM VERIFICATION STAGE"
    )
    logger.info("=" * 80)
    logger.info("Scanning unprocessed artifacts for duplicates and zero-byte files...")

    # Sort by file size - process smallest files first for faster initial feedback
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)

    session_data["total_files"] = len(unprocessed_artifacts)
    logger.info(
        f"Found {len(unprocessed_artifacts)} artifacts to process (sorted smallest to largest)"
    )

    # Count file types for initial summary
    for artifact in unprocessed_artifacts:
        ext = artifact.suffix.lower()
        session_data["file_types"][ext] += 1

    initial_file_summary = ", ".join(
        [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
    )
    logger.info(f"File type breakdown: {initial_file_summary}")

    # Update session JSON with initial state
    update_session_json()

    # Process each artifact for duplicate detection
    skipped_duplicates = 0
    skipped_zero_byte = 0
    unsupported_artifacts = 0
    remaining_artifacts: List[Path] = []

    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Checking for duplicates, zeros, and unsupported artifacts",
        unit="artifacts",
    ):
        try:
            review_location = PATHS["review_dir"] / artifact.name

            if artifact.suffix in UNSUPPORTED_EXTENSIONS:
                logger.debug(
                    f"Found artifact to an unsupported file type. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                unsupported_artifacts += 1
                continue

            # Check for zero-byte files (corrupted/incomplete downloads)
            file_size = artifact.stat().st_size
            if file_size == 0:
                # Handle duplicate names in review folder
                # counter = 1
                # while review_location.exists():
                #     name_part = artifact.stem
                #     ext_part = artifact.suffix
                #     review_location = PATHS["review_dir"] / f"{name_part}_{counter}{ext_part}"
                #     counter += 1

                logger.debug(
                    f"Found artifact to be a zero-size item. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                skipped_zero_byte += 1
                continue

            # Generate checksum for duplicate detection
            checksum = generate_checksum(artifact, algorithm=CHECKSUM_ALGORITHM)
            logger.debug(f"Generated checksum for {artifact.name}: {checksum[:16]}...")

            # Check if this file has been processed before (permanent history)
            if checksum in checksum_history:
                logger.info(
                    f"Duplicate detected. Moving {artifact.name} to review folder."
                )
                artifact.rename(review_location)
                skipped_duplicates += 1
                continue

            # Add to permanent checksums and save immediately
            checksum_history.add(checksum)
            with open(checksum_history_path, "a", encoding="utf-8") as f:
                f.write(f"{checksum}\n")

            # This is a new file - keep for processing
            remaining_artifacts.append(artifact)
            session_data["stage_counts"]["unprocessed"] += 1

        except Exception as e:
            logger.error(
                f"Failed to process {artifact.name} in duplicate detection: {e}"
            )
            session_data["errors"].append(
                {"file": artifact.name, "stage": "duplicate_detection", "error": str(e)}
            )

    logger.info(f"Duplicate detection complete:")
    logger.info(f"  - {len(remaining_artifacts)} new files to process")
    logger.info(f"  - {skipped_duplicates} duplicates skipped")
    logger.info(f"  - {skipped_zero_byte} zero-byte files moved to review")
    logger.info(f"  - {unsupported_artifacts} unsupported files moved to review")

    # Update session data and JSON
    session_data["total_files"] = len(remaining_artifacts)
    update_session_json()
