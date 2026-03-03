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
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
	ANKI_EXTENSIONS , ARTIFACT_PREFIX , ARTIFACT_PROFILES_DIR , AUDIO_TYPES ,
	CAD_FILES , CODE_EXTENSIONS , DIGITAL_CONTACT_EXTENSIONS , DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS , IMAGE_TYPES ,
	PROFILE_PREFIX , VIDEO_TYPES ,
)
from utilities.dependancy_ensurance import (
	ensure_ffmpeg ,
	ensure_imagemagick ,
	ensure_libpff_python ,
	ensure_pandoc ,
	ensure_par2 ,
)
from utilities.extract_metadata import extract_metadata
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
	for raw_artifact in tqdm(
			unprocessed_artifacts ,
			desc="Converting format" ,
			unit="artifacts" ,
	) :
		try :
			start_time = time.time( )

			# Log the start of processing for this artifact
			logger.info( f"Processing artifact: {raw_artifact.name}" )

			# Extract detected artifact type label
			# Convert to lowercase for consistent comparison with type constants
			artifact_ext: str = raw_artifact.suffix.lower( ).strip( ).strip( "." )
			logger.info( f"Detected type for {raw_artifact.name}: {artifact_ext})" )

			metadata = None

			if (artifact_ext in
					[ "epub" , "cbr" , "djvu" , "html" , "txt" , "csv" , "arw" , "cr2" , "nef" , "heic" , "onepkg" , ]
					or artifact_ext in ANKI_EXTENSIONS
					or artifact_ext in CAD_FILES
					or artifact_ext in DIGITAL_CONTACT_EXTENSIONS
					or artifact_ext in EXECUTABLE_EXTENSIONS
					or artifact_ext in CODE_EXTENSIONS
			) :
				logger.info( f"This Artifact will be ignored and tended to during manual triage. " )
				shutil.move( src=raw_artifact , dst=dest_dir / raw_artifact.name )
				continue

			elif artifact_ext in [
				"pdf" ,
				"mp4" ,
				"mp3" ,
				"png" ,
			] :
				logger.info( f"This Artifact is already in the target format. Conversion not needed. " )
				shutil.move( src=raw_artifact , dst=dest_dir / raw_artifact.name )
				continue

			unique_id = uuid.uuid4( )

			original_name = raw_artifact.stem

			artifact = raw_artifact.rename( raw_artifact.parent / f"{ARTIFACT_PREFIX}-{unique_id}{raw_artifact.suffix}" )
			logger.debug( f"Renaming artifact with UUID4 to avoid collisions: {artifact}" )

			artifact_profile_json = (ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{unique_id}.json")
			logger.debug( f"Output JSON will be saved to: {artifact_profile_json}" )

			if artifact_ext in DOCUMENT_TYPES :
				metadata = extract_metadata( artifact_location=artifact , logger=logger )
				formatted_artifact = convert_document_to_pdf( src=artifact , logger=logger )
				shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			elif artifact_ext in IMAGE_TYPES :
				metadata = extract_metadata( artifact_location=artifact , logger=logger )
				formatted_artifact = convert_image_to_png( src=artifact , logger=logger )
				shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			elif artifact_ext in VIDEO_TYPES :
				metadata = extract_metadata( artifact_location=artifact , logger=logger )
				formatted_artifact = convert_video_to_mp4( src=artifact , logger=logger )
				shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			elif artifact_ext in AUDIO_TYPES :
				metadata = extract_metadata( artifact_location=artifact , logger=logger )
				formatted_artifact = convert_audio_to_mp3( src=artifact , logger=logger )
				shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			elif artifact_ext in EMAIL_TYPES :
				metadata = extract_metadata( artifact_location=artifact , logger=logger )
				formatted_artifact = convert_email_to_pdf( src=artifact , logger=logger )
				shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			else :
				# Log unsupported artifact types
				# These artifacts will be passed through without conversion
				logger.warning( f"Could not find appropriate conversion protocol of type {artifact_ext} for {artifact.name}" )
				shutil.move( src=artifact , dst=dest_dir / artifact.name )

			# Write the metadata dictionary to a JSON artifact
			# 'w': Open artifact in write mode
			# encoding='utf-8': Use UTF-8 encoding to support international characters
			# indent=2: Pretty-print JSON with 2-space indentation
			# ensure_ascii=False: Allow non-ASCII characters in output
			if metadata :
				profile = {
					"original_name" : original_name ,
					"uuid"          : str( unique_id ) ,
					"extension"     : artifact_ext ,
					"metadata"      : metadata ,
				}

				logger.info( f"Writing metadata to output artifact: {artifact_profile_json}" )
				with open( artifact_profile_json , "w" , encoding="utf-8" ) as f :
					json.dump( profile , f , indent=2 , ensure_ascii=False )
				logger.info( f"Output JSON artifact size: {artifact_profile_json.stat( ).st_size:,} bytes" )

			# Calculate total operation duration
			total_duration = time.time( ) - start_time
			logger.info( f"Format Conversion Completed Successfully in {total_duration:.2f} seconds" )

		except Exception as e :
			# Log the error with full exception details
			# exc_info=True includes the full stack trace
			error_msg: str = f"Error processing {raw_artifact.name}: {e}"
			logger.error( error_msg , exc_info=True )
			continue  # Continue with next artifact

	# Log completion of the entire conversion stage
	logger.info( "Conversion stage completed" )
	return None
