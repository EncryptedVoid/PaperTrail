#!/usr/bin/env python3
"""
Export all processed document profiles to final spreadsheet
Creates Excel and CSV files with all extracted document fields
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class DocumentSpreadsheetExporter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # Define the column mapping from your original requirements
        self.column_mapping = {
            # Core identification
            "ITEM_ID": "uuid",
            "Title": ["llm_extraction", "extracted_fields", "title"],
            "File_Extension": "file_extension",
            # Document classification
            "Document_Type": ["llm_extraction", "extracted_fields", "document_type"],
            "Original_Language": [
                "llm_extraction",
                "extracted_fields",
                "current_language",
            ],  # Assuming same for now
            "Current_Language": [
                "llm_extraction",
                "extracted_fields",
                "current_language",
            ],
            "Confidentiality_Level": [
                "llm_extraction",
                "extracted_fields",
                "confidentiality_level",
            ],
            # Technical data
            "Checksum_SHA256": "checksum",  # You'll need to add this to profiles
            "File_Size_MB": ["metadata", "size_mb"],
            # People and organizations
            "Translator_Name": [
                "llm_extraction",
                "extracted_fields",
                "translator_name",
            ],
            "Issuer_Name": ["llm_extraction", "extracted_fields", "issuer_name"],
            "Officiater_Name": [
                "llm_extraction",
                "extracted_fields",
                "officiater_name",
            ],
            # Dates
            "Date_Added": ["stages", "renamed", "timestamp"],
            "Date_Created": ["llm_extraction", "extracted_fields", "date_created"],
            "Date_of_Reception": [
                "llm_extraction",
                "extracted_fields",
                "date_of_reception",
            ],
            "Date_of_Issue": ["llm_extraction", "extracted_fields", "date_of_issue"],
            "Date_of_Expiry": ["llm_extraction", "extracted_fields", "date_of_expiry"],
            # Content and notes
            "Tags": ["llm_extraction", "extracted_fields", "tags"],
            "Version_Notes": ["llm_extraction", "extracted_fields", "version_notes"],
            "Utility_Notes": ["llm_extraction", "extracted_fields", "utility_notes"],
            "Additional_Notes": [
                "llm_extraction",
                "extracted_fields",
                "additional_notes",
            ],
            # Manual entry fields (will be empty for now)
            "Action_Required": "",  # Manual entry
            "Parent_Document_ID": "",  # Manual entry
            "Off_Site_Storage_ID": "",  # Manual entry
            "On_Site_Storage_ID": "",  # Manual entry
            "Backup_Storage_ID": "",  # Manual entry
            "Project_ID": "",  # Manual entry
            "Version_Number": "",  # Manual entry
            # Processing metadata (bonus columns)
            "Processing_Status": ["llm_extraction", "success"],
            "OCR_Text_Length": ["semantics", "all_text"],  # Will calculate length
            "Visual_Description_Length": [
                "semantics",
                "all_imagery",
            ],  # Will calculate length
            "Fields_Extracted_Count": "calculated",  # Will calculate
            "Processing_Date": ["stages", "completed", "timestamp"],
            "Original_Filename": "original_filename",
        }

    def export_to_spreadsheet(
        self,
        profiles_dir: Path,
        output_dir: Path,
        filename_prefix: str = "PaperTrail_Artifact_Registry",
    ) -> Dict[str, Path]:
        """
        Export all profile JSON files to Excel and CSV spreadsheets

        Returns:
            Dictionary with paths to created files
        """
        self.logger.info("Starting spreadsheet export process...")

        # Find all profile files
        profile_files = list(profiles_dir.glob("PROFILE-*.json"))
        self.logger.info(f"Found {len(profile_files)} profile files to export")

        if not profile_files:
            self.logger.warning("No profile files found for export")
            return {}

        # Process each profile
        rows = []
        for profile_file in profile_files:
            try:
                row = self._process_profile(profile_file)
                if row:
                    rows.append(row)
            except Exception as e:
                self.logger.error(f"Failed to process profile {profile_file.name}: {e}")
                continue

        if not rows:
            self.logger.error("No valid data to export")
            return {}

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by processing date, then by title
        df = df.sort_values(["Processing_Date", "Title"], na_position="last")

        # Reorder columns to match your requirements
        ordered_columns = [
            "ITEM_ID",
            "Title",
            "Action_Required",
            "File_Extension",
            "Document_Type",
            "Original_Language",
            "Current_Language",
            "Confidentiality_Level",
            "Checksum_SHA256",
            "Translator_Name",
            "Parent_Document_ID",
            "Off_Site_Storage_ID",
            "On_Site_Storage_ID",
            "Backup_Storage_ID",
            "Project_ID",
            "Issuer_Name",
            "Officiater_Name",
            "Version_Number",
            "Date_Added",
            "Date_Created",
            "Date_of_Reception",
            "Date_of_Issue",
            "Date_of_Expiry",
            "Tags",
            "Version_Notes",
            "Utility_Notes",
            "Additional_Notes",
        ]

        # Add columns that exist in df but not in ordered list
        additional_cols = [col for col in df.columns if col not in ordered_columns]
        final_columns = ordered_columns + additional_cols

        # Reorder DataFrame columns
        df = df.reindex(columns=[col for col in final_columns if col in df.columns])

        # Generate output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}

        # Excel file
        excel_path = output_dir / f"{filename_prefix}_{timestamp}.xlsx"
        self._export_to_excel(df, excel_path)
        output_files["excel"] = excel_path

        # CSV file
        csv_path = output_dir / f"{filename_prefix}_{timestamp}.csv"
        self._export_to_csv(df, csv_path)
        output_files["csv"] = csv_path

        # Summary stats
        self._log_export_summary(df, len(profile_files))

        return output_files

    def _process_profile(self, profile_file: Path) -> Optional[Dict[str, Any]]:
        """Process a single profile JSON file into a spreadsheet row"""
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_file.name}: {e}")
            return None

        row = {}

        for column_name, field_path in self.column_mapping.items():
            try:
                if field_path == "":
                    # Manual entry field - leave empty
                    row[column_name] = ""
                elif field_path == "calculated":
                    # Special calculated field
                    row[column_name] = self._calculate_special_field(
                        column_name, profile
                    )
                elif isinstance(field_path, str):
                    # Simple field path
                    row[column_name] = profile.get(field_path, "UNKNOWN")
                elif isinstance(field_path, list):
                    # Nested field path
                    row[column_name] = self._get_nested_value(profile, field_path)
                else:
                    row[column_name] = "UNKNOWN"

            except Exception as e:
                self.logger.debug(
                    f"Failed to extract {column_name} from {profile_file.name}: {e}"
                )
                row[column_name] = "UNKNOWN"

        # Clean up the row
        row = self._clean_row_data(row)

        return row

    def _get_nested_value(self, data: Dict, path: List[str]) -> Any:
        """Navigate nested dictionary using path list"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return "UNKNOWN"
        return current if current is not None else "UNKNOWN"

    def _calculate_special_field(self, field_name: str, profile: Dict) -> Any:
        """Calculate special fields that need computation"""
        if field_name == "Fields_Extracted_Count":
            try:
                extracted_fields = self._get_nested_value(
                    profile, ["llm_extraction", "extracted_fields"]
                )
                if isinstance(extracted_fields, dict):
                    return sum(1 for v in extracted_fields.values() if v != "UNKNOWN")
                return 0
            except:
                return 0

        elif field_name == "OCR_Text_Length":
            try:
                text = self._get_nested_value(profile, ["semantics", "all_text"])
                return len(str(text)) if text != "UNKNOWN" else 0
            except:
                return 0

        elif field_name == "Visual_Description_Length":
            try:
                desc = self._get_nested_value(profile, ["semantics", "all_imagery"])
                return len(str(desc)) if desc != "UNKNOWN" else 0
            except:
                return 0

        return "UNKNOWN"

    def _clean_row_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
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

    def _export_to_excel(self, df: pd.DataFrame, output_path: Path):
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

    def _export_to_csv(self, df: pd.DataFrame, output_path: Path):
        """Export DataFrame to CSV"""
        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"CSV file exported: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export CSV file: {e}")

    def _log_export_summary(self, df: pd.DataFrame, total_profiles: int):
        """Log summary statistics about the export"""
        self.logger.info("=" * 60)
        self.logger.info("SPREADSHEET EXPORT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total profiles processed: {total_profiles}")
        self.logger.info(f"Rows exported: {len(df)}")
        self.logger.info(f"Columns exported: {len(df.columns)}")

        # Document type breakdown
        if "Document_Type" in df.columns:
            doc_types = df["Document_Type"].value_counts()
            self.logger.info(f"Document types: {dict(doc_types)}")

        # Processing success rate
        if "Processing_Status" in df.columns:
            success_rate = (df["Processing_Status"] == "Yes").sum() / len(df) * 100
            self.logger.info(f"LLM processing success rate: {success_rate:.1f}%")


# Usage function to add to your pipeline
def create_final_spreadsheet(
    profiles_dir: Path, output_dir: Path, logger: logging.Logger
) -> Dict[str, Path]:
    """
    Convenience function to export spreadsheets
    Call this at the end of your papertrail.py pipeline
    """
    exporter = DocumentSpreadsheetExporter(logger)
    return exporter.export_to_spreadsheet(profiles_dir, output_dir)


# Example integration for papertrail.py:
"""
Add this at the very end of your pipeline, after Stage 6:

# =====================================================================
# FINAL SPREADSHEET EXPORT
# =====================================================================

logger.info("\n" + "=" * 80)
logger.info("FINAL SPREADSHEET EXPORT")
logger.info("=" * 80)

try:
    output_files = create_final_spreadsheet(
        profiles_dir=PATHS["profiles_dir"],
        output_dir=PATHS["base_dir"],
        logger=logger
    )

    if output_files:
        logger.info("Spreadsheet export completed successfully!")
        for file_type, file_path in output_files.items():
            logger.info(f"  {file_type.upper()}: {file_path}")
    else:
        logger.warning("No spreadsheet files were created")

except Exception as e:
    logger.error(f"Spreadsheet export failed: {e}")
"""
