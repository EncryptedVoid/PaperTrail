"""
Session Tracking Agent Module

Comprehensive session management and progress tracking for the PaperTrail processing pipeline.
Provides real-time monitoring, persistent state management, and detailed reporting across
all processing stages.

Author: Ashiq Gazi
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TypedDict
from dataclasses import dataclass, asdict
from config import (
    SESSION_TRACKING_FILE,
    PROCESSING_HISTORY_FILE,
    BASE_DIR,
)


class SessionStage(TypedDict):
    """Type definition for individual processing stage tracking."""

    stage_name: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: Optional[float]
    status: str  # "pending", "running", "completed", "failed"
    files_processed: int
    files_failed: int
    files_skipped: int
    errors: List[str]
    stage_data: Dict[str, Any]


@dataclass
class SessionMetrics:
    """Comprehensive session performance metrics."""

    total_files_input: int = 0
    total_files_completed: int = 0
    total_files_failed: int = 0
    total_files_skipped: int = 0
    total_processing_time: float = 0.0
    conversion_success_rate: float = 0.0
    sanitization_success_rate: float = 0.0
    metadata_success_rate: float = 0.0
    semantics_success_rate: float = 0.0
    encryption_success_rate: float = 0.0
    overall_success_rate: float = 0.0
    average_processing_time_per_file: float = 0.0
    throughput_files_per_minute: float = 0.0


class SessionTracker:
    """
    Comprehensive session tracking and monitoring system for the PaperTrail pipeline.

    This class provides:
    - Real-time progress tracking across all pipeline stages
    - Persistent session state management
    - Performance metrics and analytics
    - Error tracking and reporting
    - Historical session data archival
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the SessionTracker with comprehensive state management.

        Args:
            logger: Logger instance for recording session operations
        """
        self.logger = logger
        self.session_id = self._generate_session_id()
        self.start_timestamp = datetime.now(timezone.utc)
        self.end_timestamp: Optional[datetime] = None

        # Session state
        self.current_stage: Optional[str] = None
        self.stages: Dict[str, SessionStage] = {}
        self.global_errors: List[str] = []
        self.metrics = SessionMetrics()

        # File paths
        self.session_file = SESSION_TRACKING_FILE
        self.history_file = PROCESSING_HISTORY_FILE

        # Ensure directories exist
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize stage definitions
        self._initialize_stages()

        self.logger.info(f"SessionTracker initialized with ID: {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"PAPERTRAIL_SESSION_{timestamp}"

    def _initialize_stages(self) -> None:
        """Initialize all expected pipeline stages."""
        stage_names = [
            "conversion",
            "sanitization",
            "metadata_extraction",
            "semantic_extraction",
            "tabulation_encryption",
        ]

        for stage_name in stage_names:
            self.stages[stage_name] = {
                "stage_name": stage_name,
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "status": "pending",
                "files_processed": 0,
                "files_failed": 0,
                "files_skipped": 0,
                "errors": [],
                "stage_data": {},
            }

    def start(self) -> None:
        """Start session tracking and create initial session file."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PAPERTRAIL PROCESSING SESSION")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Start Time: {self.start_timestamp.isoformat()}")

        # Save initial session state
        self._save_session_state()

        self.logger.info("Session tracking initialized successfully")

    def update(self, report: Union[Dict[str, Any], object]) -> None:
        """
        Update session state with stage completion report.

        Args:
            report: Stage completion report (SanitizationReport, MetadataReport, etc.)
        """
        # Convert dataclass/TypedDict to dict if needed
        if hasattr(report, "__dict__"):
            report_dict = report.__dict__
        elif hasattr(report, "_asdict"):
            report_dict = report._asdict()
        else:
            report_dict = dict(report) if isinstance(report, dict) else {}

        # Determine stage from report structure
        stage_name = self._identify_stage_from_report(report_dict)

        if stage_name:
            self._update_stage(stage_name, report_dict)
            self._update_metrics(stage_name, report_dict)
            self._save_session_state()

            self.logger.info(f"Session updated with {stage_name} results")
        else:
            # Generic update
            self.logger.debug("Received generic session update")

    def _identify_stage_from_report(self, report: Dict[str, Any]) -> Optional[str]:
        """Identify which pipeline stage generated this report."""
        # Look for stage-specific fields to identify the report type
        if "duplicates_moved" in report or "zero_byte_moved" in report:
            return "sanitization"
        elif "image_files_processed" in report or "document_files_processed" in report:
            return "metadata_extraction"
        elif "ocr_failures" in report or "llm_extraction_failures" in report:
            return "semantic_extraction"
        elif "encrypted_files" in report or "spreadsheet_exports" in report:
            return "tabulation_encryption"
        elif (
            "conversion_time_seconds" in report
            or "quality_enhancements_applied" in report
        ):
            return "conversion"
        else:
            return None

    def _update_stage(self, stage_name: str, report: Dict[str, Any]) -> None:
        """Update specific stage with completion data."""
        if stage_name not in self.stages:
            self.logger.warning(f"Unknown stage: {stage_name}")
            return

        stage = self.stages[stage_name]

        # Set stage as completed
        stage["status"] = "completed"
        stage["end_time"] = datetime.now(timezone.utc).isoformat()

        # Extract common metrics
        stage["files_processed"] = report.get(
            "processed_files", report.get("processed_artifacts", 0)
        )
        stage["files_failed"] = report.get(
            "failed_extractions", report.get("failed_encryptions", 0)
        )
        stage["files_skipped"] = report.get("skipped_files", 0)
        stage["errors"] = report.get("errors", [])

        # Store full report data
        stage["stage_data"] = report

        # Calculate duration if we have start time
        if stage["start_time"]:
            start_time = datetime.fromisoformat(stage["start_time"])
            end_time = datetime.fromisoformat(stage["end_time"])
            stage["duration_seconds"] = (end_time - start_time).total_seconds()

    def _update_metrics(self, stage_name: str, report: Dict[str, Any]) -> None:
        """Update global session metrics based on stage completion."""
        # Update total files counts based on stage
        if stage_name == "sanitization":
            self.metrics.total_files_input = report.get("total_artifacts", 0)
            processed = report.get("processed_artifacts", 0)
            total = report.get("total_artifacts", 1)
            self.metrics.sanitization_success_rate = (
                processed / total if total > 0 else 0
            )

        elif stage_name == "metadata_extraction":
            processed = report.get("processed_files", 0)
            total = report.get("total_files", 1)
            self.metrics.metadata_success_rate = processed / total if total > 0 else 0

        elif stage_name == "semantic_extraction":
            processed = report.get("processed_files", 0)
            total = report.get("total_files", 1)
            self.metrics.semantics_success_rate = processed / total if total > 0 else 0

        elif stage_name == "tabulation_encryption":
            encrypted = report.get("encrypted_files", 0)
            total = report.get("total_files", 1)
            self.metrics.encryption_success_rate = encrypted / total if total > 0 else 0
            self.metrics.total_files_completed = report.get("processed_files", 0)

        # Update failure counts
        self.metrics.total_files_failed += report.get("failed_extractions", 0)
        self.metrics.total_files_skipped += report.get("skipped_files", 0)

        # Calculate overall success rate
        self._calculate_overall_metrics()

    def _calculate_overall_metrics(self) -> None:
        """Calculate comprehensive session metrics."""
        if self.metrics.total_files_input > 0:
            self.metrics.overall_success_rate = (
                self.metrics.total_files_completed / self.metrics.total_files_input
            )

        # Calculate processing time and throughput
        if self.end_timestamp and self.metrics.total_files_input > 0:
            self.metrics.total_processing_time = (
                self.end_timestamp - self.start_timestamp
            ).total_seconds()

            self.metrics.average_processing_time_per_file = (
                self.metrics.total_processing_time / self.metrics.total_files_input
            )

            self.metrics.throughput_files_per_minute = (
                self.metrics.total_files_input
                / (self.metrics.total_processing_time / 60)
            )

    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as started."""
        if stage_name in self.stages:
            self.stages[stage_name]["status"] = "running"
            self.stages[stage_name]["start_time"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.current_stage = stage_name

            self.logger.info(f"Started stage: {stage_name}")
            self._save_session_state()

    def display_update(self) -> None:
        """Display current session progress to console and logs."""
        completed_stages = [
            name
            for name, stage in self.stages.items()
            if stage["status"] == "completed"
        ]
        total_stages = len(self.stages)

        self.logger.info("=" * 60)
        self.logger.info("SESSION PROGRESS UPDATE")
        self.logger.info("=" * 60)
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Stages Completed: {len(completed_stages)}/{total_stages}")

        # Display stage status
        for stage_name, stage in self.stages.items():
            status_icon = (
                "✓"
                if stage["status"] == "completed"
                else "○" if stage["status"] == "pending" else "⚡"
            )
            duration = (
                f" ({stage['duration_seconds']:.1f}s)"
                if stage["duration_seconds"]
                else ""
            )
            self.logger.info(
                f"  {status_icon} {stage_name.replace('_', ' ').title()}{duration}"
            )

        # Display current metrics
        if self.metrics.total_files_input > 0:
            self.logger.info(
                f"Files Processed: {self.metrics.total_files_completed}/{self.metrics.total_files_input}"
            )
            self.logger.info(
                f"Overall Success Rate: {self.metrics.overall_success_rate:.1%}"
            )

        self.logger.info("=" * 60)

    def end(self) -> None:
        """End session tracking and finalize metrics."""
        self.end_timestamp = datetime.now(timezone.utc)
        self._calculate_overall_metrics()

        self.logger.info("=" * 80)
        self.logger.info("PAPERTRAIL SESSION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(
            f"Total Duration: {self.metrics.total_processing_time:.1f} seconds"
        )

        # Save final session state
        self._save_session_state()

        # Archive session to history
        self._archive_session()

    def display_session_report(self) -> None:
        """Display comprehensive session completion report."""
        self.logger.info("=" * 80)
        self.logger.info("FINAL SESSION REPORT")
        self.logger.info("=" * 80)

        # Session overview
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(
            f"Start Time: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        if self.end_timestamp:
            self.logger.info(
                f"End Time: {self.end_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            self.logger.info(
                f"Total Duration: {self.metrics.total_processing_time:.1f} seconds"
            )

        # File processing summary
        self.logger.info("\nFILE PROCESSING SUMMARY:")
        self.logger.info(f"  Input Files: {self.metrics.total_files_input}")
        self.logger.info(
            f"  Successfully Completed: {self.metrics.total_files_completed}"
        )
        self.logger.info(f"  Failed: {self.metrics.total_files_failed}")
        self.logger.info(f"  Skipped: {self.metrics.total_files_skipped}")
        self.logger.info(
            f"  Overall Success Rate: {self.metrics.overall_success_rate:.1%}"
        )

        # Stage breakdown
        self.logger.info("\nSTAGE BREAKDOWN:")
        for stage_name, stage in self.stages.items():
            status = stage["status"].upper()
            duration = (
                f" ({stage['duration_seconds']:.1f}s)"
                if stage["duration_seconds"]
                else ""
            )
            processed = stage["files_processed"]
            failed = stage["files_failed"]

            self.logger.info(
                f"  {stage_name.replace('_', ' ').title()}: {status}{duration}"
            )
            if processed > 0 or failed > 0:
                self.logger.info(f"    Processed: {processed}, Failed: {failed}")
            if stage["errors"]:
                self.logger.info(f"    Errors: {len(stage['errors'])}")

        # Performance metrics
        if self.metrics.average_processing_time_per_file > 0:
            self.logger.info("\nPERFORMANCE METRICS:")
            self.logger.info(
                f"  Average Time per File: {self.metrics.average_processing_time_per_file:.1f}s"
            )
            self.logger.info(
                f"  Throughput: {self.metrics.throughput_files_per_minute:.1f} files/minute"
            )

        # Error summary
        total_errors = sum(
            len(stage["errors"]) for stage in self.stages.values()
        ) + len(self.global_errors)
        if total_errors > 0:
            self.logger.info(f"\nTotal Errors Encountered: {total_errors}")

        self.logger.info("=" * 80)

    def add_error(self, error_message: str, stage: Optional[str] = None) -> None:
        """Add an error to session tracking."""
        if stage and stage in self.stages:
            self.stages[stage]["errors"].append(error_message)
        else:
            self.global_errors.append(error_message)

        self.logger.error(f"Session error tracked: {error_message}")
        self._save_session_state()

    def _save_session_state(self) -> None:
        """Save current session state to persistent storage."""
        try:
            session_data = {
                "session_id": self.session_id,
                "start_timestamp": self.start_timestamp.isoformat(),
                "end_timestamp": (
                    self.end_timestamp.isoformat() if self.end_timestamp else None
                ),
                "current_stage": self.current_stage,
                "stages": self.stages,
                "metrics": asdict(self.metrics),
                "global_errors": self.global_errors,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")

    def _archive_session(self) -> None:
        """Archive completed session to processing history."""
        try:
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)

            # Add current session
            session_summary = {
                "session_id": self.session_id,
                "start_timestamp": self.start_timestamp.isoformat(),
                "end_timestamp": (
                    self.end_timestamp.isoformat() if self.end_timestamp else None
                ),
                "metrics": asdict(self.metrics),
                "stages_completed": len(
                    [s for s in self.stages.values() if s["status"] == "completed"]
                ),
                "total_stages": len(self.stages),
                "total_errors": sum(
                    len(stage["errors"]) for stage in self.stages.values()
                )
                + len(self.global_errors),
            }

            history.append(session_summary)

            # Keep only last 100 sessions
            if len(history) > 100:
                history = history[-100:]

            # Save updated history
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            self.logger.info("Session archived to processing history")

        except Exception as e:
            self.logger.error(f"Failed to archive session: {e}")

    def get_session_data(self) -> Dict[str, Any]:
        """Get complete session data for external use."""
        return {
            "session_id": self.session_id,
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": (
                self.end_timestamp.isoformat() if self.end_timestamp else None
            ),
            "current_stage": self.current_stage,
            "stages": self.stages,
            "metrics": asdict(self.metrics),
            "global_errors": self.global_errors,
        }

    @classmethod
    def load_session(
        cls, logger: logging.Logger, session_file: Optional[Path] = None
    ) -> "SessionTracker":
        """Load existing session from file."""
        file_path = session_file or SESSION_TRACKING_FILE

        if not file_path.exists():
            logger.warning("No existing session file found, creating new session")
            return cls(logger)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Create new instance and restore state
            tracker = cls(logger)
            tracker.session_id = session_data["session_id"]
            tracker.start_timestamp = datetime.fromisoformat(
                session_data["start_timestamp"]
            )

            if session_data["end_timestamp"]:
                tracker.end_timestamp = datetime.fromisoformat(
                    session_data["end_timestamp"]
                )

            tracker.current_stage = session_data["current_stage"]
            tracker.stages = session_data["stages"]
            tracker.global_errors = session_data["global_errors"]

            # Restore metrics
            metrics_data = session_data["metrics"]
            tracker.metrics = SessionMetrics(**metrics_data)

            logger.info(f"Loaded existing session: {tracker.session_id}")
            return tracker

        except Exception as e:
            logger.error(f"Failed to load session from {file_path}: {e}")
            logger.info("Creating new session instead")
            return cls(logger)
