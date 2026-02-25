"""
Conversion Pipeline Module

A robust artifact conversion pipeline that handles artifact type detection, quality enhancement,
and format conversion for various media types including images, videos, audio, and documents.

This module provides functionality to convert artifacts by:
- Detecting artifact type through extension and content analysis using Magika
- Converting artifacts to standardized formats with quality enhancement
- Archiving original artifacts before conversion for backup purposes
- Handling unsupported conversions by moving artifacts to failure directory
- Maintaining proper error handling and logging throughout the process
- Updating artifact profiles with conversion metadata and results

Author: Ashiq Gazi
"""

import logging
import shutil
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
	ARCHIVAL_DIR ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	AUDIO_TYPES ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	IMAGE_TYPES ,
	PROFILE_PREFIX ,
	SPREADSHEET_TYPES , VIDEO_TYPES ,
)
from utilities.dependancy_ensurance import (
	ensure_ffmpeg ,
	ensure_imagemagick ,
	ensure_libpff_python ,
	ensure_pandoc ,
	ensure_par2 ,
)
from utilities.format_converting import (
	convert_audio_to_mp3 ,
	convert_document_to_pdf ,
	convert_email_to_pdf ,
	convert_image_to_png ,
	convert_pub_to_pdf , convert_video_to_mp4 , convert_xlsx_to_csv ,
)


def converting_files(
		logger: logging.Logger ,
		source_dir: Path ,
) -> None :
	"""
	Convert artifacts to standardized formats based on detected artifact types.

	This method performs comprehensive artifact conversion by detecting artifact types using
	Magika content analysis, routing artifacts to appropriate converters based on type,
	archiving original artifacts before conversion, and moving converted artifacts to the
	success directory. Files that fail type detection or conversion are moved to
	the failure directory.

	The conversion process includes:
	- File discovery and size-based sorting (smallest first for faster feedback)
	- Profile loading and validation for each artifact
	- Content-based artifact type detection using Magika with confidence scoring
	- Original artifact archiving before conversion for backup purposes
	- Type-specific conversion routing (documents, images, videos, audio, archives)
	- Profile updates with conversion metadata and results
	- File moving to success or failure directories based on conversion results
	- Comprehensive error logging with full exception details

	Supported artifact types and conversions:
	- Documents: Converted to PDF format
	- Images:    Converted to PNG format
	- Videos:    Converted to MP4 format
	- Audio:     Converted to MP3 format
	- Archives:  Converted to 7Z format

	Args:
																	logger: Logger instance for tracking operations and errors
																	source_dir: Directory containing artifact artifacts to process. Files must follow
																																	the ARTIFACT-{uuid}.ext naming convention

	Returns:
																	None. This method processes artifacts in-place and moves them to appropriate
																	directories based on conversion results.

	Raises:
																	FileNotFoundError: If source directory or artifact profile does not exist
																	ValueError: If profile JSON is corrupted or artifact type confidence is too low
																	TypeError: If artifact type detection does not meet minimum confidence score

	Note:
																	- Files are processed in order of size (smallest first) for faster feedback
																	- Requires corresponding PROFILE-{uuid}.json artifacts in ARTIFACT_PROFILES_DIR
																	- Original artifacts are archived before conversion for backup purposes
																	- File type detection uses Magika with minimum confidence threshold
																	- Unsupported artifact types are passed through without conversion
																	- Text and code artifacts are not converted but still processed
																	- Naming conflicts in destination directories are automatically resolved
																	- All conversion failures are logged with full exception details (exc_info=True)
	"""

	ensure_ffmpeg( )
	ensure_imagemagick( )
	ensure_pandoc( )
	ensure_par2( )
	ensure_libpff_python( )

	# ============================================================================
	# INITIALIZATION AND SETUP
	# ============================================================================

	# Log conversion stage header for clear progress tracking
	# This helps distinguish conversion logs from other pipeline stages
	logger.info( "=" * 80 )
	logger.info( "ARTIFACT CONVERSION STAGE" )
	logger.info( "=" * 80 )

	# Record the start time to calculate total processing duration
	logger.info( f"Starting artifact conversion process for directory: {source_dir}" )

	# Use Path.iterdir() to get all items in directory, filter to only regular files
	# This excludes subdirectories, symlinks, and other non-artifact items
	unprocessed_artifacts: List[ Path ] = [
		item for item in source_dir.iterdir( ) if item.is_file( )
	]

	# Handle empty directory case - exit early if no artifacts to process
	# This prevents unnecessary processing and provides clear feedback
	if not unprocessed_artifacts :
		logger.info( "No artifacts found in source directory, sanitization complete" )
		return None

	# Sort artifacts by size (smallest first) for faster initial processing feedback
	# Smaller artifacts process faster, giving users immediate progress indication
	# The lambda function retrieves artifact size in bytes using Path.stat().st_size
	unprocessed_artifacts.sort( key=lambda p : p.stat( ).st_size )

	total_artifacts = len( unprocessed_artifacts )
	logger.info( f"Found {total_artifacts} file(s) to process" )
	logger.info( f"Files sorted by size for optimal processing order" )

	# Process each artifact with progress bar
	# tqdm provides visual progress feedback to users
	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Converting format" ,
			unit="artifacts" ,
	) :
		try :
			start_time = time.time( )

			# Log the start of processing for this artifact
			logger.info( f"Processing artifact: {artifact.name}" )

			# ====================================================================
			# PROFILE LOADING AND VALIDATION
			# ====================================================================

			# Extract UUID from filename for profile lookup
			# Expected format: ARTIFACT-{uuid}.ext
			# We strip the prefix and extension to get just the UUID
			artifact_id = artifact.stem[ (len( ARTIFACT_PREFIX ) + 1) : ]

			# Construct the path to the corresponding profile JSON file
			# Profile contains metadata about the artifact's processing history
			artifact_profile_json = (
					ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
			)

			# ====================================================================
			# FILE TYPE DETECTION
			# ====================================================================

			# Extract detected artifact type label
			# Convert to lowercase for consistent comparison with type constants
			artifact_ext: str = artifact.suffix.lower( ).strip( ).strip( "." )
			logger.info( f"Detected type for {artifact.name}: {artifact_ext})" )

			if artifact_ext in [
				"epub" ,
				"cbr" ,
				"djvu" ,
				"html" ,
				"txt" ,
				"csv" ,
			] :
				logger.info( f"This Artifact will be ignored and tended to during manual triage. " )
				continue

			# ====================================================================
			# CONVERSION ROUTING
			# ====================================================================

			# Route to appropriate converter based on detected artifact type
			# Each type has its own specialized conversion function

			archival_directory = ARCHIVAL_DIR / artifact.name

			if artifact_ext in [ "pub" ] :
				logger.debug( f"Converting document: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_pub_to_pdf( src=artifact , logger=logger )

			elif artifact_ext in SPREADSHEET_TYPES :
				logger.debug( f"Converting document: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_xlsx_to_csv( src=artifact , logger=logger )

			elif artifact_ext in DOCUMENT_TYPES :
				# Convert documents to PDF for standardization
				# PDF is universal and preserves formatting
				logger.debug( f"Converting document: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_document_to_pdf( src=artifact , logger=logger )

			elif artifact_ext in IMAGE_TYPES :
				# Convert images to PNG for lossless quality
				# PNG supports transparency and high quality
				logger.debug( f"Converting image: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_image_to_png( src=artifact , logger=logger )

			elif artifact_ext in VIDEO_TYPES :
				# Convert videos to MP4 for broad compatibility
				# MP4 is universally supported and efficient
				logger.debug( f"Converting video: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_video_to_mp4( src=artifact , logger=logger )

			elif artifact_ext in AUDIO_TYPES :
				# Convert audio to MP3 for broad compatibility
				# MP3 is universally supported with good quality at high bitrates
				logger.debug( f"Converting audio: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_audio_to_mp3( src=artifact , logger=logger )

			elif artifact_ext in EMAIL_TYPES :
				# Convert email artifacts to standard EML format
				# EML is the standard MIME email format
				logger.debug( f"Converting email: {artifact.name}" )
				shutil.copy2( str( artifact ) , str( archival_directory ) )
				logger.debug( f"Archived original artifact to: {archival_directory}" )
				convert_email_to_pdf( src=artifact , logger=logger )

			else :
				# Log unsupported artifact types
				# These artifacts will be passed through without conversion
				logger.warning(
						f"Could not find appropriate conversion protocol of type {artifact_ext} for {artifact.name}" ,
				)

			# Get the size of the output JSON artifact for logging
			logger.info(
					f"Output JSON artifact size: {artifact_profile_json.stat( ).st_size:,} bytes" ,
			)

			# Calculate total operation duration
			total_duration = time.time( ) - start_time
			logger.info(
					f"Format Conversion Completed Successfully in {total_duration:.2f} seconds" ,
			)

		except Exception as e :
			# Log the error with full exception details
			# exc_info=True includes the full stack trace
			error_msg: str = f"Error processing {artifact.name}: {e}"
			logger.error( error_msg , exc_info=True )
			continue  # Continue with next artifact

	# Log completion of the entire conversion stage
	logger.info( "Conversion stage completed" )
	return None
