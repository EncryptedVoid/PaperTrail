# Create the final spreadsheet using the database processor
output_files = create_final_spreadsheet(
    profiles_dir=PATHS["profiles_dir"],
    output_dir=PATHS["base_dir"],
    logger=logger,
)
# SESSION JSON file path
session_json_path = PATHS["logs_dir"] / f"SESSION-{session_timestamp}.json"
# Load permanent checksum history (one checksum per line)
checksum_history: Set[str] = set()
checksum_history_path = PATHS["base_dir"] / CHECKSUM_HISTORY_FILE

if checksum_history_path.exists():
    try:
        with open(checksum_history_path, "r", encoding="utf-8") as f:
            checksum_history = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(checksum_history)} permanent checksums from history")
    except Exception as e:
        logger.warning(f"Failed to load permanent checksums: {e}")
        raise e
else:
    logger.info("No permanent checksum history found - creating new file")
    checksum_history_path.touch()
