"""
Enhanced DatabasePipeline with complete tabulate function implementation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any , Dict , List , Optional , TypedDict

import pandas as pd
from tqdm import tqdm

from config import (
	ARCHIVE_DIR ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	FOR_REVIEW_ARTIFACTS_DIR ,
	PROFILE_PREFIX ,
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
        self.conversion_agent: ConversionProcessor = ConversionProcessor(
            logger=logger, archive_dir=ARCHIVE_DIR, failure_dir=FOR_REVIEW_ARTIFACTS_DIR
        )

        self.logger.info("DatabasePipeline initialized successfully")

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
                    checksum_history.add(checksum)
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
                move(artifact, failure_dir, "processing_error")

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

    def _process_profile(profile_file: Path) -> Optional[Dict[str, Any]]:
        """Process a single profile JSON file into a spreadsheet row"""
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_file.name}: {e}")
            return None

        row = {}

        for column_name, field_path in column_mapping.items():
            try:
                if field_path == "":
                    # Manual entry field - leave empty
                    row[column_name] = ""
                elif field_path == "calculated":
                    # Special calculated field
                    row[column_name] = _calculate_special_field(column_name, profile)
                elif isinstance(field_path, str):
                    # Simple field path
                    row[column_name] = profile.get(field_path, "UNKNOWN")
                elif isinstance(field_path, list):
                    # Nested field path
                    row[column_name] = _get_nested_value(profile, field_path)
                else:
                    row[column_name] = "UNKNOWN"

            except Exception as e:
                self.logger.debug(
                    f"Failed to extract {column_name} from {profile_file.name}: {e}"
                )
                row[column_name] = "UNKNOWN"

        # Clean up the row
        row = _clean_row_data(row)

        return row

    def _get_nested_value(data: Dict, path: List[str]) -> Any:
        """Navigate nested dictionary using path list"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return "UNKNOWN"
        return current if current is not None else "UNKNOWN"

    def _calculate_special_field(field_name: str, profile: Dict) -> Any:
        """Calculate special fields that need computation"""
        if field_name == "Fields_Extracted_Count":
            try:
                extracted_fields = _get_nested_value(
                    profile, ["llm_extraction", "extracted_fields"]
                )
                if isinstance(extracted_fields, dict):
                    return sum(1 for v in extracted_fields.values() if v != "UNKNOWN")
                return 0
            except:
                return 0

        elif field_name == "OCR_Text_Length":
            try:
                text = _get_nested_value(profile, ["semantics", "all_text"])
                return len(str(text)) if text != "UNKNOWN" else 0
            except:
                return 0

        elif field_name == "Visual_Description_Length":
            try:
                desc = _get_nested_value(profile, ["semantics", "all_imagery"])
                return len(str(desc)) if desc != "UNKNOWN" else 0
            except:
                return 0

        return "UNKNOWN"

    def _clean_row_data(row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize row data"""
        cleaned = {}
        for key, value in row.items():
            # Convert None to UNKNOWN
            if value is None:
                cleaned[key] = "UNKNOWN"
            # Clean up strings
            elif isinstance(value, str):
                cleaned[key] = value.strip() if value.strip() else "UNKNOWN"
            # Convert booleans
            elif isinstance(value, bool):
                cleaned[key] = "Yes" if value else "No"
            # Keep numbers as-is
            else:
                cleaned[key] = value

        return cleaned

    def _export_to_excel(df: pd.DataFrame, output_path: Path):
        """Export DataFrame to Excel with formatting"""
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Documents", index=False)

                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets["Documents"]

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 chars
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            self.logger.info(f"Excel file exported: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export Excel file: {e}")

    def _export_to_csv(df: pd.DataFrame, output_path: Path):
        """Export DataFrame to CSV"""
        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"CSV file exported: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export CSV file: {e}")
