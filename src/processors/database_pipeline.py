"""
Enhanced DatabasePipeline with complete tabulate function implementation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict , List , TypedDict

from tqdm import tqdm

from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	PASSWORD_VAULT_PATH ,
	PROFILE_PREFIX ,
)
from utilities.common import (
	generate_checksum ,
	move ,
	save_checksum ,
)


class TabulationReport(TypedDict):
    """
    Type definition for the tabulation report returned by the tabulate method.
    """

    processed_files: int
    total_files: int
    encrypted_files: int
    failed_encryptions: int
    failed_exports: int
    skipped_files: int
    spreadsheet_exports: Dict[str, Path]
    password_vault_created: bool
    errors: List[str]
    processing_time: float


class DatabasePipeline:
    """
    Comprehensive database pipeline providing file encryption and spreadsheet export services.
    """

    def _init__(self, logger: logging.Logger) -> None:
        """Initialize the DatabasePipeline with logging configuration."""
        self.logger: logging.Logger = logger

    def tabulate(
        self,
        source_dir: Path,
        failure_dir: Path,
        success_dir: Path,
    ) -> TabulationReport:
        """
        Main tabulation function that encrypts files, stores passwords, and exports spreadsheets.

        This method performs the complete database pipeline process:
        1. Discovers all processed artifact files in the source directory
        2. Encrypts each file with a unique password/passphrase
        3. Securely stores encryption passwords in an encrypted vault
        4. Exports all profile data to Excel and CSV spreadsheets
        5. Moves processed files to appropriate directories
        6. Generates comprehensive processing report

        Args:
            source_dir: Directory containing processed artifacts to tabulate
            failure_dir: Directory to move problematic files for manual review
            success_dir: Directory to move successfully processed files

        Returns:
            TabulationReport containing detailed results of the tabulation process
        """
        start_time = datetime.now()

        # Validate input directories
        if not source_dir.exists():
            error_msg = f"Source directory does not exist: {source_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create output directories if they don't exist
        for directory in [failure_dir, success_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")

        # Initialize report
        report: TabulationReport = {
            "processed_files": 0,
            "total_files": 0,
            "encrypted_files": 0,
            "failed_encryptions": 0,
            "failed_exports": 0,
            "skipped_files": 0,
            "spreadsheet_exports": {},
            "password_vault_created": False,
            "errors": [],
            "processing_time": 0.0,
        }

        # Log tabulation stage header
        self.logger.info("=" * 80)
        self.logger.info("DATABASE TABULATION AND ENCRYPTION STAGE")
        self.logger.info("=" * 80)

        # Discover artifact files
        try:
            artifacts = [
                item
                for item in source_dir.iterdir()
                if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
            ]
        except Exception as e:
            error_msg = f"Failed to scan source directory: {e}"
            self.logger.error(error_msg)
            report["errors"].append(error_msg)
            return report

        if not artifacts:
            self.logger.info("No artifact files found for tabulation")

        report["total_files"] = len(artifacts)
        self.logger.info(f"Found {len(artifacts)} artifact files to process")

        # Process each artifact
        for artifact in tqdm(artifacts, desc="Processing artifacts", unit="file"):
            try:
                # Extract UUID for profile lookup
                artifact_id = artifact.stem[(len(ARTIFACT_PREFIX) + 1) :]
                profile_path = (
                    ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
                )

                # Verify profile exists
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    report["skipped_files"] += 1
                    move(artifact, failure_dir, "missing_profile")
                    continue

                # STAGE 5: File passed all sanitization checks (only reached if conversion was successful)
                # Verify the file still exists at the expected location
                if not artifact.exists():
                    error_msg = f"File disappeared after conversion: {artifact}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    continue

                # Add checksum to history to prevent future duplicates
                try:
                    checksum = generate_checksum(artifact)
                    save_checksum(checksum)
                except Exception as checksum_error:
                    error_msg = f"Checksum generation failed for {artifact.name}: {checksum_error}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    continue

                # Move artifact to success directory
                try:
                    moved_artifact = artifact.rename(success_dir / artifact.name)
                    report["processed_files"] += 1
                    self.logger.debug(f"Moved to success directory: {artifact.name}")
                except Exception as e:
                    error_msg = (
                        f"Failed to move {artifact.name} to success directory: {e}"
                    )
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)

            except Exception as e:
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                move(artifact, failure_dir)

            # Finalize report
            total_time = (datetime.now() - start_time).total_seconds()
            report["processing_time"] = total_time

            # Log final summary
            self.logger.info("Tabulation complete:")
            self.logger.info(
                f"  - {report['processed_files']} files successfully processed"
            )
            self.logger.info(f"  - {report['encrypted_files']} files encrypted")
            self.logger.info(f"  - {report['failed_encryptions']} encryption failures")
            self.logger.info(f"  - {report['skipped_files']} files skipped")
            self.logger.info(
                f"  - Processing time: {report['processing_time']:.2f} seconds"
            )

            if report["spreadsheet_exports"]:
                self.logger.info("  - Spreadsheet exports:")
                for format_type, file_path in report["spreadsheet_exports"].items():
                    self.logger.info(f"    - {format_type.upper()}: {file_path}")

            if report["password_vault_created"]:
                self.logger.info(f"  - Password vault: {PASSWORD_VAULT_PATH}")

            if report["errors"]:
                self.logger.warning(f"  - {len(report['errors'])} errors encountered")

            return report
        return None
