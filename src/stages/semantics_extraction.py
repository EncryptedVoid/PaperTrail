"""
Metadata Pipeline Module

A robust file processing pipeline that handles comprehensive metadata extraction
from various file types including images, file_paths, PDFs, office files, audio,
video, archives, and more.

This module provides functionality to extract metadata by:
- Detecting file types and routing to appropriate extractors
- Extracting filesystem metadata (size, dates, permissions)
- Processing image EXIF, IPTC, XMP data, color profiles, and technical details
- Extracting file_path properties, structure information, and content analysis
- Processing audio/video metadata including codecs, duration, and technical specs
- Analyzing archive contents and compression details
- Extracting code and text file metrics and language detection
- Handling extraction failures gracefully with fallback methods
- Maintaining detailed operation logs and error tracking
- Updating artifact profiles with extracted metadata

Author: Ashiq Gazi
"""

from utilities.extract_semantics import extract_fields

"""
LLM-based field extraction with hardware auto-detection and context management

This module provides intelligent document field extraction using local LLM models through OLLAMA.
It automatically detects hardware capabilities, selects optimal models, and manages context to
prevent performance degradation during batch processing.

Key Features:
- Automatic hardware detection and model selection
- Context refresh to prevent model degradation
- Multiple extraction strategies (single/multi-prompt)
- Comprehensive validation and error handling
- Performance monitoring and statistics

Dependencies:
- requests: HTTP communication with OLLAMA API
- psutil: System hardware detection
- GPUtil: GPU memory detection (optional)
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

import exiftool
from tqdm import tqdm

from config import (
	ARTIFACT_PROFILES_DIR ,
	CODE_EXTENSIONS ,
	IMAGE_TYPES ,
	JAVA_PATH ,
	MAX_PDF_SIZE_BEFORE_SUBSETTING ,
	METADATA_EXTRACTION_TIMEOUT ,
	TEMP_DIR ,
	TEXT_TYPES ,
	TIKA_APP_JAR_PATH ,
	UNSUPPORTED_ARTIFACTS_DIR ,
)
from utilities.dependancy_ensurance import (
	ensure_apache_tika ,
	ensure_exiftool ,
	ensure_ollama ,
)
from utilities.visual_processor import VisualProcessor , compile_doc_subset


def extracting_semantics(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Path ,
		visual_processor: VisualProcessor ,
) -> None :
	ensure_ollama( )
	ensure_apache_tika( )
	ensure_exiftool( )

	# Log conversion stage header for clear progress tracking
	# This helps distinguish conversion logs from other pipeline stages
	logger.info( "=" * 80 )
	logger.info( "SEMANTICAL DATA EXTRACTION STAGE" )
	logger.info( "=" * 80 )

	logger.info( f"Starting semantical data extraction process for directory: {source_dir}" )

	# Ensure output directory exists, create if necessary
	# exist_ok=True prevents error if directory already exists
	# parents=True creates parent directories if needed
	ARTIFACT_PROFILES_DIR.mkdir( parents=True , exist_ok=True )
	logger.info( f"Artifact Profile directory validated and ready" )

	# Use Path.iterdir() to get all items in directory, filter to only regular artifacts
	# This excludes subdirectories, symlinks, and other non-artifact items
	unprocessed_artifacts: List[ Path ] = [
		item for item in source_dir.iterdir( ) if item.is_file( )
	]

	# Handle empty directory case - exit early if no artifacts to process
	# This prevents unnecessary processing and provides clear feedback
	if not unprocessed_artifacts :
		logger.info( "No artifacts found in source directory, sanitization skipped" )
		return None

	# Sort artifacts by size (smallest first) for faster initial processing feedback
	# Smaller artifacts process faster, giving users immediate progress indication
	# The lambda function retrieves artifact size in bytes using Path.stat().st_size
	unprocessed_artifacts.sort( key=lambda p : p.stat( ).st_size )

	total_artifacts = len( unprocessed_artifacts )
	logger.info( f"Found {total_artifacts} artifact(s) to process" )
	logger.info( f"Files sorted by size for optimal processing order" )

	# Initialize statistics tracking using simple counters
	# These track the outcomes of metadata extraction for reporting
	stats = {
		"successful" : 0 ,  # Files with successful metadata extraction
		"failed"     : 0 ,  # Files that failed metadata extraction
	}

	logger.info( "Beginning artifact-by-artifact data extraction process" )
	# Process each artifact file with progress tracking
	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Extracting semantical data" ,
			unit="artifacts" ,
	) :
		try :
			logger.info( f"Processing artifact: {artifact.name}" )

			# Extract text content from the file
			logger.debug( f"Extracting text content for: {artifact.name}" )

			# Determine file category and extract content accordingly
			content = None
			artifact_ext = artifact.suffix.strip( "." )

			if artifact_ext in TEXT_TYPES or artifact_ext in CODE_EXTENSIONS :
				logger.debug( f"Processing as text file: {artifact.name}" )

				with open( artifact , "r" , encoding="utf-8" ) as f :
					content = f.read( ).strip( )

			elif artifact_ext in [ "pdf" ] :
				logger.debug( f"Processing as document: {artifact.name}" )

				# Try Apache Tika text extraction first
				text_cmd = [
					JAVA_PATH ,
					"-jar" ,
					str( TIKA_APP_JAR_PATH ) ,
					"--text" ,
					str( artifact ) ,
				]

				text_process = subprocess.run(
						text_cmd ,
						capture_output=True ,
						text=True ,
						timeout=METADATA_EXTRACTION_TIMEOUT ,
				)

				if text_process.returncode == 0 :
					content = text_process.stdout.strip( )

				# Fallback to QWEN visual extraction if Tika fails or returns empty
				if not content :
					logger.warning(
							f"Apache Tika failed for {artifact.name}, trying QWEN OCR" ,
					)

					subset = compile_doc_subset(
							input_pdf=artifact ,
							subset_size=MAX_PDF_SIZE_BEFORE_SUBSETTING ,
							temp_dir=TEMP_DIR ,
					)

					content = visual_processor.extract_text( file_path=subset )

					# If OCR fails, get visual description
					if not content :
						logger.warning( f"QWEN OCR failed for {artifact.name}, using visual description" )
						content = visual_processor.extract_visual_description( file_path=subset )

			elif artifact_ext in IMAGE_TYPES :
				logger.debug( f"Processing as image: {artifact.name}" )
				content = visual_processor.extract_visual_description( file_path=artifact )

			else :
				raise RuntimeError( f"Unsupported file type: {artifact.suffix}. Cannot extract semantical data" )

			# Validate that we extracted some content
			if not content :
				raise ValueError( f"No content could be extracted from {artifact.name}" )

			logger.debug( f"Successfully extracted {len( content )} characters from {artifact.name}" )

			# Extract structured semantic fields from the content
			semantical_descriptors = extract_fields( logger=logger , content=content )

			metadata_json = json.dumps( semantical_descriptors )

			shutil.move( src=artifact , dst=dest_dir / artifact.name )

			with exiftool.ExifTool( ) as et :
				et.execute(
						b"-XMP-custom:Metadata=" + metadata_json.encode( "utf-8" ) ,
						str( artifact ).encode( "utf-8" ) ,
				)

		except Exception as e :
			# Catch any errors during processing to prevent pipeline failure
			error_msg: str = f"Error processing {artifact.name}: {e}"
			logger.error( error_msg , exc_info=True )

			# Move the failed artifact for later review and debugging
			shutil.move( artifact , UNSUPPORTED_ARTIFACTS_DIR )
			logger.info( f"Moved failed artifact to: {UNSUPPORTED_ARTIFACTS_DIR}" )
			# Continue processing remaining artifacts despite this failure
			continue

	# Log completion of the entire extraction stage
	logger.info( "Semantics extraction stage completed" )
	return None
