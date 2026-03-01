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

import bz2
import gzip
import logging
import lzma
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import List , Optional

import py7zr
from tqdm import tqdm

from config import (
	ARCHIVE_TYPES ,
	CORRUPTED_ARTIFACTS_DIR ,
	DUPLICATE_ARTIFACTS_DIR ,
	PASSWORD_PROTECTED_ARTIFACTS_DIR ,
	UNSUPPORTED_ARTIFACTS_DIR ,
)
from utilities.checksum import generate_checksum , save_checksum
from utilities.dependancy_ensurance import ensure_apache_tika , ensure_ffmpeg
from utilities.sanitization import (
	is_corrupted ,
	is_password_protected ,
	is_supported_type ,
)


def _decompress_artifact(
		logger: logging.Logger ,
		artifact: Path ,
		dest_dir: Path ,
) -> List[ Path ] :
	"""
	Decompress a compressed artifact into the destination directory.

	Supports zip, 7z, tar, tar.gz, tar.bz2, tar.xz, standalone gz, bz2, and xz files.
	After successful extraction the original compressed artifact is removed.

	Args:
		logger: Logger instance for recording decompression events.
		artifact: Path to the compressed artifact.
		dest_dir: Directory to extract contents into.

	Returns:
		A list of Path objects for every file extracted.

	Raises:
		Exception: Propagates any extraction errors after logging them.
	"""

	extracted_files: List[ Path ] = [ ]
	name_lower = artifact.name.lower( )

	try :
		# --- ZIP ---
		if name_lower.endswith( ".zip" ) :
			logger.info( f"Decompressing ZIP archive: {artifact.name}" )
			with zipfile.ZipFile( artifact , "r" ) as zf :
				zf.extractall( dest_dir )
				for info in zf.infolist( ) :
					if not info.is_dir( ) :
						extracted_files.append( dest_dir / info.filename )

		# --- 7Z (requires py7zr) ---
		elif name_lower.endswith( ".7z" ) :
			logger.info( f"Decompressing 7z archive: {artifact.name}" )
			with py7zr.SevenZipFile( artifact , mode="r" ) as sz :
				sz.extractall( path=dest_dir )
				for entry in sz.getnames( ) :
					candidate = dest_dir / entry
					if candidate.is_file( ) :
						extracted_files.append( candidate )

		# --- TAR variants (.tar, .tar.gz/.tgz, .tar.bz2/.tbz2, .tar.xz/.txz) ---
		elif (
				name_lower.endswith( ".tar" )
				or name_lower.endswith( ".tar.gz" )
				or name_lower.endswith( ".tgz" )
				or name_lower.endswith( ".tar.bz2" )
				or name_lower.endswith( ".tbz2" )
				or name_lower.endswith( ".tar.xz" )
				or name_lower.endswith( ".txz" )
		) :
			logger.info( f"Decompressing TAR archive: {artifact.name}" )
			with tarfile.open( artifact , "r:*" ) as tf :
				tf.extractall( dest_dir , filter="data" )
				for member in tf.getmembers( ) :
					if member.isfile( ) :
						extracted_files.append( dest_dir / member.name )

		# --- Standalone GZ (not tar.gz) ---
		elif name_lower.endswith( ".gz" ) :
			out_name = artifact.stem  # strips .gz
			out_path = dest_dir / out_name
			logger.info( f"Decompressing GZ file: {artifact.name} -> {out_name}" )
			with gzip.open( artifact , "rb" ) as f_in , open( out_path , "wb" ) as f_out :
				shutil.copyfileobj( f_in , f_out )
			extracted_files.append( out_path )

		# --- Standalone BZ2 ---
		elif name_lower.endswith( ".bz2" ) :
			out_name = artifact.stem
			out_path = dest_dir / out_name
			logger.info( f"Decompressing BZ2 file: {artifact.name} -> {out_name}" )
			with bz2.open( artifact , "rb" ) as f_in , open( out_path , "wb" ) as f_out :
				shutil.copyfileobj( f_in , f_out )
			extracted_files.append( out_path )

		# --- Standalone XZ ---
		elif name_lower.endswith( ".xz" ) :
			out_name = artifact.stem
			out_path = dest_dir / out_name
			logger.info( f"Decompressing XZ file: {artifact.name} -> {out_name}" )
			with lzma.open( artifact , "rb" ) as f_in , open( out_path , "wb" ) as f_out :
				shutil.copyfileobj( f_in , f_out )
			extracted_files.append( out_path )

		else :
			logger.warning( f"Unrecognised compressed format, skipping: {artifact.name}" )
			return [ ]

		# Remove the original archive after successful extraction
		artifact.unlink( )
		logger.info( f"Extracted {len( extracted_files )} file(s) from {artifact.name}" )

	except Exception as e :
		logger.error( f"Failed to decompress {artifact.name}: {e}" )
		raise

	return extracted_files


def sanitizing(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Path ,
		recursive_search_dir: Optional[ Path ] = None ,
) -> None :
	"""
	Sanitize a directory of artifacts by detecting and moving duplicates, corrupted artifacts,
	unsupported artifact types, and password-protected artifacts.

	Args:
		logger: Logger instance for recording processing events and statistics.
		source_dir: Path object pointing to the directory containing artifacts to process.
		recursive_search_dir:Path to a directory the program is allowed to search recursively.

	Returns:
		None
	"""

	ensure_apache_tika( )
	ensure_ffmpeg( )

	logger.info( f"Starting sanitization process for directory: {source_dir}" )

	# ------------------------------------------------------------------
	# Preliminary: collect from recursive dir & decompress all archives
	# ------------------------------------------------------------------
	# ------------------------------------------------------------------
	# Step 1 – Collect from recursive_search_dir
	# ------------------------------------------------------------------
	if recursive_search_dir is not None :
		if not recursive_search_dir.exists( ) :
			logger.warning(
					f"recursive_search_dir does not exist, skipping: {recursive_search_dir}" ,
			)
		else :
			logger.info(
					f"Collecting artifacts recursively from: {recursive_search_dir}" ,
			)

			# Gather every file recursively (rglob("*") yields all entries)
			recursive_files: List[ Path ] = [
				p for p in recursive_search_dir.rglob( "*" ) if p.is_file( )
			]
			logger.info(
					f"Found {len( recursive_files )} file(s) in recursive directory" ,
			)

			for artifact in tqdm(
					recursive_files ,
					desc="Collecting from recursive dir" ,
					unit="file" ,
			) :
				try :
					# Decompress compressed files directly into source_dir
					if artifact.suffix.lower( ).strip( ) in ARCHIVE_TYPES( artifact ) :
						_decompress_artifact( logger , artifact , source_dir )
					else :
						# Resolve naming collisions by appending a counter
						dest = source_dir / artifact.name
						if dest.exists( ) :
							stem = artifact.stem
							suffix = artifact.suffix
							counter = 1
							while dest.exists( ) :
								dest = source_dir / f"{stem}_{counter}{suffix}"
								counter += 1

						shutil.move( str( artifact ) , str( dest ) )
						logger.debug( f"Moved {artifact.name} -> {dest}" )
				except Exception as e :
					logger.error(
							f"Error collecting {artifact.name} from recursive dir: {e}" ,
					)

			# Clean up any now-empty subdirectories left behind
			for dirpath in sorted(
					recursive_search_dir.rglob( "*" ) , reverse=True ,
			) :
				if dirpath.is_dir( ) and not any( dirpath.iterdir( ) ) :
					dirpath.rmdir( )

	# ------------------------------------------------------------------
	# Step 2 – Iteratively decompress everything in source_dir
	# ------------------------------------------------------------------
	# Loop until no compressed files remain (handles nested archives)
	pass_number = 0
	while True :
		compressed_in_source: List[ Path ] = [
			p for p in source_dir.iterdir( ) if (
					p.is_file( )
					and artifact.suffix.lower( ).strip( ) in ARCHIVE_TYPES
			)
		]
		if not compressed_in_source :
			break

		pass_number += 1
		logger.info(
				f"Decompression pass {pass_number}: "
				f"{len( compressed_in_source )} compressed file(s) in source_dir" ,
		)

		for artifact in tqdm(
				compressed_in_source ,
				desc=f"Decompressing (pass {pass_number})" ,
				unit="file" ,
		) :
			try :
				_decompress_artifact( logger , artifact , source_dir )
			except Exception as e :
				logger.error( f"Skipping {artifact.name} after decompression error: {e}" )

		# Safety valve – avoid infinite loops from unrecognised nested formats
		if pass_number >= 10 :
			logger.warning(
					"Reached maximum decompression passes (10). "
					"Some nested archives may remain." ,
			)
			break

	# Flatten any subdirectories that extraction may have created inside source_dir
	# Move nested files up into source_dir so the main loop can see them
	for nested_file in list( source_dir.rglob( "*" ) ) :
		if nested_file.is_file( ) and nested_file.parent != source_dir :
			dest = source_dir / nested_file.name
			if dest.exists( ) :
				stem = nested_file.stem
				suffix = nested_file.suffix
				counter = 1
				while dest.exists( ) :
					dest = source_dir / f"{stem}_{counter}{suffix}"
					counter += 1
			shutil.move( str( nested_file ) , str( dest ) )
			logger.debug( f"Flattened {nested_file} -> {dest}" )

	# Remove empty subdirectories left behind after flattening
	for dirpath in sorted( source_dir.rglob( "*" ) , reverse=True ) :
		if dirpath.is_dir( ) and not any( dirpath.iterdir( ) ) :
			dirpath.rmdir( )

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

	# List to store checksums of successfully processed artifacts
	# Used to detect duplicates during current processing session
	processed_checksums: List[ str ] = [ ]

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

			logger.debug( f"Processing artifact: {artifact.name}" )

			# Generate SHA-256 checksum for the artifact to detect duplicates
			# Checksum is a unique fingerprint based on artifact content
			artifact_checksum: str = generate_checksum( logger=logger , artifact_path=artifact )
			logger.debug( f"Generated checksum for {artifact.name}: {artifact_checksum[ :16 ]}..." )

			# Check if this checksum has been seen before (duplicate detection)
			# Duplicates are artifacts with identical content regardless of artifact name
			if artifact_checksum in processed_checksums :
				logger.info( f"Duplicate detected: {artifact.name}" )
				shutil.move( src=artifact , dst=DUPLICATE_ARTIFACTS_DIR / artifact.name )
				logger.debug( f"Moved duplicate artifact to: {DUPLICATE_ARTIFACTS_DIR / artifact.name}" )
				continue

			# File is not a duplicate, add checksum to processed list
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
				logger.debug( f"Moved unsupported artifact to: {UNSUPPORTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			if is_corrupted( artifact_location=artifact ) :
				logger.info( f"Corrupted artifact detected: {artifact.name}" )
				shutil.move( src=artifact , dst=CORRUPTED_ARTIFACTS_DIR / artifact.name )
				logger.debug( f"Moved corrupted artifact to: {CORRUPTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			# Check if artifact is password-protected (encrypted)
			# Password-protected artifacts cannot be automatically processed
			if is_password_protected( artifact_location=artifact ) :
				logger.info( f"Password-protected artifact detected: {artifact.name}" )
				shutil.move( src=artifact , dst=PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name )
				logger.debug( f"Moved password-protected artifact to: {PASSWORD_PROTECTED_ARTIFACTS_DIR / artifact.name}" )
				continue

			# File passed all validation checks - save its checksum for future reference
			# Persistent checksum storage prevents reprocessing the same artifacts
			save_checksum( logger=logger , checksum=artifact_checksum )
			logger.debug( f"File validated successfully: {artifact.name}" )

			shutil.move( src=artifact , dst=dest_dir / artifact.name )

			# Calculate total processing time by subtracting start time from current time
			elapsed_time = time.time( ) - start_time
			logger.info( f"Total processing time for {artifact.name}: {elapsed_time:.2f} seconds" )
		except Exception as e :
			# Catch any unexpected errors during artifact processing
			# Log the error but continue processing remaining artifacts
			logger.error( f"Error processing artifact {artifact.name}: {str( e )}" )
			continue

	return None
