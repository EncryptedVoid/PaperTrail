#!/usr/bin/env python3
"""
Stage-aware resumability system for PaperTrail pipeline
Tracks which stage each file has reached and enables resume from any point
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from enum import Enum
import logging


class ProcessingStage(Enum):
    UNPROCESSED = "unprocessed"
    RENAMED = "renamed"
    METADATA_EXTRACTED = "metadata_extracted"
    SEMANTICS_EXTRACTED = "semantics_extracted"
    LLM_PROCESSED = "llm_processed"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumabilityTracker:
    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = Path(base_dir)
        self.logger = logger
        self.state_file = self.base_dir / "processing_state.json"
        self.state_data = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load processing state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.logger.info(
                        f"Loaded processing state with {len(data.get('files', {}))} tracked files"
                    )
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to load state file: {e}, starting fresh")

        return {
            "files": {},  # checksum -> {uuid, stage, timestamp, original_name}
            "sessions": [],
            "permanent_checksums": set(),  # Still need duplicate detection
        }

    def save_state(self):
        """Save current state to disk"""
        # Convert set to list for JSON serialization
        save_data = self.state_data.copy()
        save_data["permanent_checksums"] = list(self.state_data["permanent_checksums"])

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    def is_duplicate(self, checksum: str) -> bool:
        """Check if file was already processed in any previous session"""
        return checksum in self.state_data["permanent_checksums"]

    def register_new_file(self, checksum: str, uuid: str, original_name: str) -> bool:
        """Register a new file for processing"""
        if self.is_duplicate(checksum):
            return False

        self.state_data["files"][checksum] = {
            "uuid": uuid,
            "stage": ProcessingStage.RENAMED.value,
            "timestamp": datetime.now().isoformat(),
            "original_name": original_name,
        }
        self.state_data["permanent_checksums"].add(checksum)
        self.save_state()
        return True

    def update_stage(self, checksum: str, new_stage: ProcessingStage):
        """Update the processing stage for a file"""
        if checksum in self.state_data["files"]:
            self.state_data["files"][checksum]["stage"] = new_stage.value
            self.state_data["files"][checksum]["timestamp"] = datetime.now().isoformat()
            self.save_state()

    def get_files_at_stage(self, stage: ProcessingStage) -> List[Dict[str, Any]]:
        """Get all files currently at a specific stage"""
        return [
            {"checksum": checksum, **data}
            for checksum, data in self.state_data["files"].items()
            if data["stage"] == stage.value
        ]

    def can_resume(self) -> Dict[ProcessingStage, int]:
        """Check what stages have files ready for resuming"""
        stage_counts = {}
        for stage in ProcessingStage:
            if stage not in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                count = len(self.get_files_at_stage(stage))
                if count > 0:
                    stage_counts[stage] = count
        return stage_counts

    def get_resume_plan(self) -> Optional[Dict[str, Any]]:
        """Get a resume plan showing what can be resumed"""
        resumable = self.can_resume()
        if not resumable:
            return None

        return {
            "total_resumable_files": sum(resumable.values()),
            "by_stage": resumable,
            "suggested_start_stage": min(resumable.keys(), key=lambda x: x.value),
        }


def add_resumability_to_pipeline(
    base_dir: Path, logger: logging.Logger, session_id: str
):
    """
    Add resumability logic to your main pipeline
    Call this at the start of your papertrail.py
    """
    tracker = ResumabilityTracker(base_dir, logger)

    # Check for resumable work
    resume_plan = tracker.get_resume_plan()

    if resume_plan:
        logger.info("=" * 60)
        logger.info("RESUMABLE WORK DETECTED!")
        logger.info("=" * 60)
        logger.info(
            f"Total files that can be resumed: {resume_plan['total_resumable_files']}"
        )

        for stage, count in resume_plan["by_stage"].items():
            logger.info(f"  {stage.value}: {count} files")

        suggested_stage = resume_plan["suggested_start_stage"]
        logger.info(f"Suggested starting stage: {suggested_stage.value}")

        # Ask user if they want to resume
        print("\nResumable work found. Options:")
        print("1. Resume from where you left off")
        print("2. Start fresh (will skip already completed files)")
        print("3. Exit and check manually")

        choice = input("Choose option (1/2/3): ").strip()

        if choice == "1":
            return tracker, "resume", suggested_stage
        elif choice == "2":
            return tracker, "fresh", None
        else:
            logger.info("Exiting for manual review")
            exit(0)

    else:
        logger.info("No resumable work found - starting fresh processing")
        return tracker, "fresh", None


# Integration functions for each stage
def check_and_skip_completed_files(
    tracker: ResumabilityTracker,
    artifacts_list: List[Path],
    current_stage: ProcessingStage,
    logger: logging.Logger,
) -> List[Path]:
    """
    Filter out files that are already past the current stage
    """
    completed_files = tracker.get_files_at_stage(ProcessingStage.COMPLETED)
    completed_uuids = {f["uuid"] for f in completed_files}

    # Also check files at later stages
    later_stages = [ProcessingStage.LLM_PROCESSED, ProcessingStage.COMPLETED]
    if current_stage in [ProcessingStage.RENAMED, ProcessingStage.METADATA_EXTRACTED]:
        later_stages.extend([ProcessingStage.SEMANTICS_EXTRACTED])

    skip_uuids = set()
    for stage in later_stages:
        if stage.value > current_stage.value:  # Only skip if actually later
            stage_files = tracker.get_files_at_stage(stage)
            skip_uuids.update(f["uuid"] for f in stage_files)

    # Filter artifacts
    remaining_artifacts = []
    for artifact in artifacts_list:
        try:
            # Extract UUID from filename
            artifact_uuid = artifact.name.split("-")[1].split(".")[0]
            if artifact_uuid not in skip_uuids:
                remaining_artifacts.append(artifact)
            else:
                logger.info(
                    f"Skipping {artifact.name} - already processed past {current_stage.value}"
                )
        except IndexError:
            # Filename doesn't match expected pattern, include it
            remaining_artifacts.append(artifact)

    logger.info(
        f"Stage {current_stage.value}: {len(remaining_artifacts)}/{len(artifacts_list)} files to process"
    )
    return remaining_artifacts


# Usage example for integration:
"""
At the start of papertrail.py, replace your duplicate detection with:

# Initialize resumability
tracker, mode, start_stage = add_resumability_to_pipeline(BASE_DIR, logger, session_timestamp)

# In each stage, filter out already-completed files:
remaining_artifacts = check_and_skip_completed_files(
    tracker, renamed_artifacts, ProcessingStage.METADATA_EXTRACTED, logger
)

# After successfully processing each file, update the tracker:
tracker.update_stage(checksum, ProcessingStage.METADATA_EXTRACTED)
"""
