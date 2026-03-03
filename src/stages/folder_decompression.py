import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List , Set

from tqdm import tqdm

from config import ARCHIVE_TYPES , DELETE_DIR , DUPLICATE_ARTIFACTS_DIR , OUCH_DECOMPRESSOR_PATH
from utilities.checksum import generate_checksum , load_checksum_history , save_checksum


def is_compressed( artifact: Path ) -> bool :
	"""
	Determine whether an artifact is a compressed archive based on its extension.

	Handles compound extensions like .tar.gz, .tar.bz2, .tar.xz as well as
	single extensions like .zip, .7z, .gz, etc.

	Args:
		artifact: Path to the artifact to check.

	Returns:
		True if the artifact has a recognized compressed extension.
	"""
	name_lower = artifact.name.lower( )

	# Check compound extensions first (e.g. .tar.gz)
	for ext in (".tar.gz" , ".tar.bz2" , ".tar.xz") :
		if name_lower.endswith( ext ) :
			return True

	return artifact.suffix.lower( ) in ARCHIVE_TYPES


def decompress_folder( logger: logging.Logger , archive_path: Path , output_dir: Path ) :
	"""Uses the standalone ouch binary to decompress an archive."""
	cmd = [
		str( OUCH_DECOMPRESSOR_PATH ) ,
		"decompress" , str( archive_path ) ,
		"--dir" , str( output_dir ) ,
		"--yes" ,
	]
	try :
		subprocess.run( cmd , check=True , capture_output=True )
		logger.info( f"Successfully decompressed: {archive_path.name}" )
		# ouch doesn't delete the source, so we do it manually after success
		shutil.move( src=str( archive_path ) , dst=str( DELETE_DIR / archive_path ) )
	except subprocess.CalledProcessError as e :
		logger.error( f"Ouch failed on {archive_path.name}: {e.stderr.decode( )}" )
		raise


def decompressing_artifacts(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Path ,
) -> None :
	# Log conversion stage header for clear progress tracking
	# This helps distinguish conversion logs from other pipeline stages
	logger.info( "=" * 80 )
	logger.info( "FOLDER DECOMPRESSION STAGE" )
	logger.info( "=" * 80 )

	logger.info( f"Starting decompression proces for directory: {source_dir}" )

	# List to store checksums of successfully processed artifacts
	# Used to detect duplicates during current processing session
	processed_checksums: List[ str ] = [ ]
	past_processed_checksums: Set[ str ] = load_checksum_history( logger=logger )

	# --- Step 1: Iterative Decompression (Handles Nested Archives) ---
	pass_number = 0
	while True :
		# Get a fresh list of archives for every pass
		archives = [ p for p in source_dir.rglob( "*" ) if p.is_file( ) and is_compressed( p ) ]

		if not archives or pass_number >= 10 :
			if pass_number >= 10 :
				logger.warning( "Reached max decompression depth (10)." )
			break

		pass_number += 1
		for archive in tqdm( archives , desc=f"Decompressing Pass #{pass_number}" , unit="folder" ) :
			# 1. CRITICAL: Check if file still exists (it might have been deleted/moved in this pass)
			if not archive.exists( ) :
				logger.warning( f"File disappeared before processing: {archive.name}" )
				continue

			try :
				# 2. Checksum/Duplicate Logic
				checksum = generate_checksum( logger , archive )
				if checksum in processed_checksums or checksum in past_processed_checksums :
					logger.info( f"Moving duplicate: {archive.name}" )
					# Ensure destination exists
					DUPLICATE_ARTIFACTS_DIR.mkdir( parents=True , exist_ok=True )
					shutil.move(
							str( archive ) ,
							str( DUPLICATE_ARTIFACTS_DIR / f"{archive.stem} ({uuid.uuid4( ).hex[ :8 ]}){archive.suffix}" ) ,
					)
					continue

				processed_checksums.append( checksum )
				save_checksum( logger=logger , checksum=checksum )

				# 3. Decompress
				# After this call, the file is unlinked (deleted) by your decompress_folder function
				decompress_folder( logger , archive , archive.parent )

			except FileNotFoundError :
				logger.warning( f"Caught FileNotFoundError for {archive.name} - skipping." )
				continue
			except Exception as e :
				logger.error( f"Failed to process {archive.name}: {e}" )

	# --- Step 2: Flattening & Final Move ---
	# Now that everything is decompressed, move all non-compressed files to dest_dir
	all_extracted_items = list( source_dir.rglob( "*" ) )

	for item in tqdm( all_extracted_items , desc="Flattening to destination" ) :
		if item.is_file( ) :
			# If it's a file, move it to the root of dest_dir
			shutil.move(
					str( item ) ,
					str(
							dest_dir
							/ f"{item.stem} ({str( uuid.uuid4( ) )[ : :8 ]}){item.suffix}" ,
					) ,
			)

	# --- Step 3: Cleanup ---
	# Remove empty directories in source_dir
	for dirpath in sorted( source_dir.rglob( "*" ) , reverse=True ) :
		if dirpath.is_dir( ) and not any( dirpath.iterdir( ) ) :
			dirpath.rmdir( )

	logger.info( f"Preparation complete. All files moved to {dest_dir}" )
