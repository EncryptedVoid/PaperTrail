"""
Sanitizer Pipeline Module

A robust artifact processing pipeline that handles duplicate detection, zero-byte artifact
identification, and unsupported artifact type filtering using checksum verification.

This module provides functionality to sanitize a directory of artifacts by:
- Detecting and moving duplicate artifacts based on checksum comparison
- Identifying and moving zero-byte (corrupted/incomplete) artifacts
- Filtering out unsupported artifact types and password-protected artifacts
- Maintaining a persistent history of processed artifacts
- Generating unique identifiers and proartifacts for valid artifacts
- Recursively collecting files from an allowed directory
- Decompressing compressed archives (zip, 7z, tar, gz, bz2, xz)

The sanitization process validates each artifact's integrity, checks for duplicates using
SHA-256 checksums, and organizes artifacts into appropriate directories based on their
status. Processing statistics and timing information are logged for monitoring.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List , Set

from tqdm import tqdm

from config import (
	CORRUPTED_ARTIFACTS_DIR ,
	DUPLICATE_ARTIFACTS_DIR ,
	PASSWORD_PROTECTED_ARTIFACTS_DIR ,
	UNSUPPORTED_ARTIFACTS_DIR
)
from utilities.artifact_data_manipulation import stop_apache_tika_server
from utilities.checksum import generate_checksum , load_checksum_history , save_checksum
from utilities.dependancy_ensurance import ensure_apache_tika , ensure_ffmpeg
from utilities.sanitization import is_corrupted , is_password_protected , is_supported_type


def sanitizing(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Path ,
		tika_server_process: subprocess.Popen | None = None ,  # ← add this
) -> None :
	ensure_apache_tika( )
	ensure_ffmpeg( )

	# Log conversion stage header for clear progress tracking
	# This helps distinguish conversion logs from other pipeline stages
	logger.info( "=" * 80 )
	logger.info( "ARTIFACT SET SANITIZATION STAGE" )
	logger.info( "=" * 80 )

	logger.info( f"Starting sanitization process for directory: {source_dir}" )

	# List to store checksums of successfully processed artifacts
	# Used to detect duplicates during current processing session
	processed_checksums: List[ str ] = [ ]
	past_processed_checksums: Set[ str ] = load_checksum_history( logger=logger )

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
	logger.info( f"Artifact sorted by size for optimal processing order" )

	logger.info( "Beginning artifact-by-artifact sanitization process" )

	# Process each artifact with a progress bar for user feedback
	# tqdm provides a visual progress bar in the console
	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Sanitizing" ,
			unit="artifacts" ,
	) :
		try :
			start_time = time.time( )

			logger.info( f"Processing artifact: {artifact.name}" )

			# Generate SHA-256 checksum for the artifact to detect duplicates
			# Checksum is a unique fingerprint based on artifact content
			artifact_checksum: str = generate_checksum( logger=logger , artifact_path=artifact )
			logger.info( f"Generated checksum for {artifact.name}: {artifact_checksum[ :16 ]}..." )

			# Check if this checksum has been seen before (duplicate detection)
			# Duplicates are artifacts with identical content regardless of artifact name
			if artifact_checksum in processed_checksums or artifact_checksum in past_processed_checksums :
				logger.info( f"Duplicate detected: {artifact.name}" )
				shutil.move( src=artifact , dst=DUPLICATE_ARTIFACTS_DIR / artifact.name )
				logger.info( f"Moved duplicate artifact to: {DUPLICATE_ARTIFACTS_DIR / artifact.name}" )
				continue

			logger.info( f"Artifact \"{artifact.name}\" is not a duplicate" )
			processed_checksums.append( artifact_checksum )

			if artifact.suffix.lower( ).strip( ).strip( '.' ) in [ "gcode" ] :
				logger.info( f"Ignoring {artifact.name}, Apache Tika does not recognise gcode files as of Feb 25th, 2026" )
				shutil.move( src=artifact , dst=dest_dir / artifact.name )
				continue

			# Check if artifact type is supported using Apache Tika content detection
			# Tika analyzes artifact content rather than just extension to prevent spoofing
			if not is_supported_type( artifact_location=artifact ) :
				logger.info( f"Unsupported artifact type detected: {artifact.name}" )
				shutil.move( src=artifact , dst=UNSUPPORTED_ARTIFACTS_DIR / artifact.name )
				logger.info( f"Moved unsupported artifact to: {UNSUPPORTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			logger.info( f"Artifact \"{artifact.name}\" is supported by PaperTrail as of Mar 1st, 2026" )

			if is_corrupted( logger=logger , artifact_location=artifact , tika_server_process=tika_server_process ) :
				logger.info( f"Corrupted artifact detected: {artifact.name}" )
				shutil.move( src=artifact , dst=CORRUPTED_ARTIFACTS_DIR / artifact.name )
				logger.info( f"Moved corrupted artifact to: {CORRUPTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			logger.info( f"Artifact \"{artifact.name}\" is not corrupted" )

			# Check if artifact is password-protected (encrypted)
			# Password-protected artifacts cannot be automatically processed
			if is_password_protected( artifact_location=artifact ) :
				logger.info( f"Password-protected artifact detected: {artifact.name}" )
				shutil.move( src=artifact , dst=PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name )
				logger.info( f"Moved password-protected artifact to: {PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			logger.info( f"Artifact \"{artifact.name}\" is not password protected" )

			# File passed all validation checks - save its checksum for future reference
			# Persistent checksum storage prevents reprocessing the same artifacts
			save_checksum( logger=logger , checksum=artifact_checksum )
			logger.info( f"File validated successfully: {artifact.name}" )
			shutil.move( src=artifact , dst=(dest_dir / artifact.name) )

			# Calculate total processing time by subtracting start time from current time
			elapsed_time = time.time( ) - start_time
			logger.info( f"Total processing time for {artifact.name}: {elapsed_time:.2f} seconds" )
		except Exception as e :
			# Catch any unexpected errors during artifact processing
			# Log the error but continue processing remaining artifacts
			logger.error( f"Error processing artifact {artifact.name}: {str( e )}" )
			continue

	stop_apache_tika_server( logger=logger , tika_server_process=tika_server_process )
	return None
