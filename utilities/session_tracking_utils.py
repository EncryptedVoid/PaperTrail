# =====================================================================
# SESSION INITIALIZATION AND STATE TRACKING
# =====================================================================

# Initialize session tracking
session_start_time = datetime.now()
session_data = {
    "session_id": session_timestamp,
    "start_time": session_start_time.isoformat(),
    "status": "running",
    "total_files": 0,
    "processed_files": 0,
    "file_types": defaultdict(int),
    "stage_counts": {
        "unprocessed": 0,
        "rename": 0,
        "metadata": 0,
        "semantics": 0,
        "completed": 0,
        "failed": 0,
        "review": 0,
    },
    "performance": {"files_per_minute": 0.0, "total_runtime_seconds": 0},
    "errors": [],
}


def update_session_json():
    """Update the SESSION JSON file with current progress data"""
    try:
        with open(session_json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to update SESSION JSON: {e}")


def log_detailed_progress(file_name: str, file_size: int, stage: str):
    """Log detailed multi-line progress information after each file"""
    elapsed_seconds = (datetime.now() - session_start_time).total_seconds()
    files_per_minute = (
        (session_data["processed_files"] / elapsed_seconds * 60)
        if elapsed_seconds > 0
        else 0
    )

    # Update performance metrics
    session_data["performance"]["files_per_minute"] = round(files_per_minute, 2)
    session_data["performance"]["total_runtime_seconds"] = int(elapsed_seconds)

    # Format file size for human readability
    if file_size < 1024:
        size_str = f"{file_size}B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size/1024:.1f}KB"
    else:
        size_str = f"{file_size/(1024*1024):.1f}MB"

    # Create file type summary
    file_type_summary = ", ".join(
        [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
    )

    # Create stage status summary
    stage_summary = ", ".join(
        [
            f"{stage}({count})"
            for stage, count in session_data["stage_counts"].items()
            if count > 0
        ]
    )

    logger.info("=" * 60)
    logger.info(
        f"[File {session_data['processed_files']}/{session_data['total_files']}] Completed: {file_name} ({size_str})"
    )
    logger.info(f"Stage: {stage}")
    logger.info(
        f"Session Runtime: {elapsed_seconds//3600:02.0f}:{(elapsed_seconds%3600)//60:02.0f}:{elapsed_seconds%60:02.0f} | Speed: {files_per_minute:.1f} files/min"
    )
    logger.info(f"File Types: {file_type_summary}")
    logger.info(f"Stage Status: {stage_summary}")
    logger.info("=" * 60)


def update_stage_counts(from_stage: str, to_stage: str, session_data: Dict[str, Any]):
    """
    Properly update stage counts when moving files between stages
    Always decrements source stage and increments destination stage
    """
    if from_stage in session_data["stage_counts"]:
        session_data["stage_counts"][from_stage] -= 1
        # Ensure count doesn't go negative
        if session_data["stage_counts"][from_stage] < 0:
            session_data["stage_counts"][from_stage] = 0

    if to_stage in session_data["stage_counts"]:
        session_data["stage_counts"][to_stage] += 1
    else:
        session_data["stage_counts"][to_stage] = 1


# =====================================================================
# FINAL SESSION SUMMARY AND COMPLETION
# =====================================================================

# Mark session as completed
session_data["status"] = "completed"
session_data["end_time"] = datetime.now().isoformat()
update_session_json()

# Calculate final statistics
total_elapsed = (datetime.now() - session_start_time).total_seconds()
successful_files = session_data["stage_counts"]["completed"]
failed_files = session_data["stage_counts"]["failed"]

logger.info("=" * 80)
logger.info("PAPERTRAIL SESSION PROCESSING COMPLETED!")
logger.info("=" * 80)
logger.info(f"Session ID: {session_timestamp}")
logger.info(
    f"Total Runtime: {total_elapsed//3600:02.0f}:{(total_elapsed%3600)//60:02.0f}:{total_elapsed%60:02.0f}"
)
logger.info(f"Successfully Processed: {successful_files} files")
logger.info(
    f"Average Speed: {session_data['performance']['files_per_minute']:.1f} files/min"
)

# Final file type summary
final_file_summary = ", ".join(
    [f"{count} {ext}" for ext, count in session_data["file_types"].items()]
)
logger.info(f"File Types Processed: {final_file_summary}")

# Directory locations summary
logger.info("\nOutput Locations:")
logger.info(f"  Completed Files: {PATHS['process_completed_dir']}")
logger.info(f"  Profile Data: {PATHS['profiles_dir']}")
logger.info(f"  Session Logs: {PATHS['logs_dir']}")
logger.info(f"  Files for Review: {PATHS['review_dir']}")

# Error summary if any
if session_data["errors"]:
    logger.info(f"\nErrors Encountered ({len(session_data['errors'])}):")
    for error in session_data["errors"]:
        logger.info(f"  {error['file']} ({error['stage']}): {error['error']}")

logger.info("=" * 80)
