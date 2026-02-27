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

from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	AUDIO_TYPES ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	IMAGE_TYPES ,
	PROFILE_PREFIX ,
	VIDEO_TYPES ,
)
from tqdm import tqdm
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
	convert_video_to_mp4 ,
)


def converting_files(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Path ,
) -> None :
	"""
	Convert artifacts to standardized formats based on detected artifact types.

	Args:
		:param logger: Logger instance for tracking operations and errors
		:param source_dir: Directory containing artifacts to process.
		:param dest_dir: Destination directory to store converted artifacts.
	Returns:
		None. This method processes artifacts in-place and moves them to appropriate
		directories based on conversion results.

	Raises:
		FileNotFoundError: If source directory or artifact profile does not exist
		ValueError: If profile JSON is corrupted or artifact type confidence is too low
		TypeError: If artifact type detection does not meet minimum confidence score


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
				"arw" ,
				"cr2" ,
				"heic" ,
				"apkg" ,
			] :
				logger.info( f"This Artifact will be ignored and tended to during manual triage. " )
				shutil.move( src=artifact , dst=dest_dir / artifact.name )
				continue

			# ====================================================================
			# CONVERSION ROUTING
			# ====================================================================

			# Route to appropriate converter based on detected artifact type
			# Each type has its own specialized conversion function
			if artifact_ext in DOCUMENT_TYPES :
				new_artifact = convert_document_to_pdf( src=artifact , logger=logger )
				shutil.move( src=new_artifact , dst=dest_dir / new_artifact.name )

			elif artifact_ext in IMAGE_TYPES :
				new_artifact = convert_image_to_png( src=artifact , logger=logger )
				shutil.move( src=new_artifact , dst=dest_dir / new_artifact.name )

			elif artifact_ext in VIDEO_TYPES :
				new_artifact = convert_video_to_mp4( src=artifact , logger=logger )
				shutil.move( src=new_artifact , dst=dest_dir / new_artifact.name )

			elif artifact_ext in AUDIO_TYPES :
				new_artifact = convert_audio_to_mp3( src=artifact , logger=logger )
				shutil.move( src=new_artifact , dst=dest_dir / new_artifact.name )

			elif artifact_ext in EMAIL_TYPES :
				new_artifact = convert_email_to_pdf( src=artifact , logger=logger )
				shutil.move( src=new_artifact , dst=dest_dir / new_artifact.name )

			else :
				# Log unsupported artifact types
				# These artifacts will be passed through without conversion
				logger.warning( f"Could not find appropriate conversion protocol of type {artifact_ext} for {artifact.name}" )
				shutil.move( src=artifact , dst=dest_dir / artifact.name )

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
