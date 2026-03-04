"""
PaperTrail Main Processing Pipeline

Complete document processing pipeline with session tracking, conversion, sanitization,
metadata extraction, semantic analysis, and database tabulation with encryption.

This is the main entry point for the PaperTrail system, orchestrating a multi-stage
document processing workflow that takes raw artifacts and processes them through
various stages until they are ready for storage and retrieval.

Author: Ashiq Gazi
"""

import logging
from datetime import datetime

from applications.identify_duplicates import DuplicateReviewer
# from applications.manual_file_triage import FileTriage
from config import (
	APPLICATION_FOLDERS ,
	COMPLETED_FORMAT_CONVERSION_DIR ,
	COMPLETED_SANITIZATION_DIR ,
	LOG_DIR ,
	LOG_FORMAT ,
	LOG_LEVEL ,
	RECURSIVE_SORT_DIR ,
	SESSION_LOG_FILE_PREFIX ,
	SYSTEM_DIRECTORIES ,
	SYSTEM_PROGRAM_TRACKING_FILES ,
	UNPROCESSED_ARTIFACTS_DIR ,
)
from stages.auto_sort import automatically_sorting
from stages.file_conversion import converting_files
from stages.folder_decompression import decompressing_artifacts
from stages.sanitize import sanitizing
from utilities.visual_processor import VisualProcessor

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

# Ensure all required directories exist before processing begins
# This prevents errors during processing if any directory is missing
# parents=True creates parent directories if needed, exist_ok=True prevents errors if already exists
for directory in SYSTEM_DIRECTORIES :
	directory.mkdir( parents=True , exist_ok=True )

for file in SYSTEM_PROGRAM_TRACKING_FILES :
	file.parent.mkdir( parents=True , exist_ok=True )
	file.touch( exist_ok=True )

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

print( "Root handlers before basicConfig:" , logging.root.handlers )

# Setup dual logging: console output for real-time monitoring and file output for persistence
# Console handler allows operators to monitor progress in real-time
# File handler creates a permanent record of all processing activities
handlers = [
	logging.FileHandler(
			# Generate a unique log file name for this session using timestamp
			# Format: prefix-YYYY-MM-DD_HH-MM-SS.log
			# This ensures each run has its own log file for auditing and debugging
			(
					LOG_DIR
					/ f"{SESSION_LOG_FILE_PREFIX}-{datetime.now( ).strftime( '%Y-%m-%d_%H-%M-%S' )}.log"
			) ,  # Outputs log messages to timestamped file
			encoding="utf-8" ,  # Ensures proper handling of international characters
	) ,
]

# Configure the root logger with INFO level and custom format
# INFO level captures important events without excessive debug details
# Format includes timestamp, logger name,  severity level, and the actual message
logging.basicConfig(
		level=LOG_LEVEL ,
		format=LOG_FORMAT ,
		handlers=handlers ,
		force=True ,  # clears any existing root handlers first
)

# Create a named logger for this application
# This allows filtering and identifying PaperTrail-specific log entries
logger = logging.getLogger( "PAPERTRAIL" )

print( "Root handlers after basicConfig:" , logging.root.handlers )

# ============================================================================
# PIPELINE STARTING
# ============================================================================

# Log the session startup banner
# This marks the beginning of a new processing session in the logs
logger.info( "WELCOME TO PAPERTRAIL! AN AUTOMATED ARTIFACT ORGANISATION SYSTEM" )

# app = FolderManagerApp( source_dir=RECURSIVE_SORT_DIR , dest_dir=UNPROCESSED_ARTIFACTS_DIR , logger=logger )
# app.mainloop( )

decompressing_artifacts(
		logger=logger ,
		source_dir=RECURSIVE_SORT_DIR ,
		dest_dir=UNPROCESSED_ARTIFACTS_DIR ,
)

sanitizing(
		logger=logger ,
		source_dir=UNPROCESSED_ARTIFACTS_DIR ,
		dest_dir=COMPLETED_SANITIZATION_DIR ,
)

converting_files(
		logger=logger ,
		source_dir=COMPLETED_SANITIZATION_DIR ,
		dest_dir=COMPLETED_FORMAT_CONVERSION_DIR ,
)

duplicate_reviewer = DuplicateReviewer( logger=logger , source_dir=COMPLETED_FORMAT_CONVERSION_DIR )
duplicate_reviewer.run( )

visual_processor = VisualProcessor( logger=logger )

automatically_sorting(
		logger=logger ,
		visual_processor=visual_processor ,
		source_dir=COMPLETED_FORMAT_CONVERSION_DIR ,
)

# manual_artifact_triage = FileTriage( logger=logger , source_dir=COMPLETED_FORMAT_CONVERSION_DIR )
# manual_artifact_triage.run( )

# ============================================================================
# PIPELINE COMPLETION
# ============================================================================
# Log successful completion of entire processing pipeline
# All documents have been processed through all seven stages and are ready for use
logger.info( "All processing stages completed successfully!" )
