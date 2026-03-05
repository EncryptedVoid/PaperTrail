"""
Auto-sorting module for categorizing and organizing artifact files.

This module provides automated file sorting functionality that analyzes artifacts
and moves them to appropriate destination directories based on file type, content,
and metadata. Supports various file types including bookmarks, code, Anki decks,
backup codes, books, and financial documents.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (AFFINE_DIR , ANKI_DIR , BITWARDEN_DIR , DIGITAL_ASSET_MANAGEMENT_DIR ,
										FIREFLYIII_DIR , GITLAB_DIR , IMAGE_TYPES , IMMICH_DIR , JELLYFIN_DIR , LINKWARDEN_DIR ,
										MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR , ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR ,
										ODOO_PLM_DIR , ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR , PERSONAL_LIBRARY_DIR ,
										SCANNING_REQUIRED_DIR , SOFTWARE_ARCHIVE_DIR , TEXT_MODEL , ULTIMAKER_CURA_DIR , VISION_MODEL)
from utilities.automatic_sorting import (is_3d_file , is_academic , is_anki_deck , is_book , is_bookmark_file ,
																				 is_code , is_digital_contact_file , is_executable , is_financial_document ,
																				 is_immigration , is_instruction_manual , is_legal , is_personal_security_item ,
																				 is_professional , is_textbook , is_unscanned_document , is_video_course)
from utilities.dependancy_ensurance import ensure_apache_tika , ensure_ollama , ensure_ollama_model
from utilities.sanitization import sanitize_artifact_name


def automatically_sorting(
		logger: logging.Logger ,
		source_dir: Path ,
) :
	"""
	Automatically sort and organize artifact files to detected locations.

	Args:
		logger: Logger instance for recording process information and errors
		source_dir: Path object pointing to the directory containing artifacts to sort

	Returns:
		None: Function completes silently after processing all artifacts
	"""

	logger.info( f"Starting automatic sorting process for directory: {source_dir}" )
	ensure_apache_tika( )
	ensure_ollama( )
	ensure_ollama_model( VISION_MODEL , logger )
	ensure_ollama_model( TEXT_MODEL , logger )

	# Discover all artifact files in the source directory
	# Using list() on iterdir() materializes the generator into a list for processing
	unprocessed_artifacts: List[ Path ] = [
		item for item in source_dir.iterdir( ) if item.is_file( )
	]

	# Sort files by size for consistent processing order (smaller files first)
	# stat().st_size returns file size in bytes
	# This provides faster initial feedback and helps identify issues early
	unprocessed_artifacts.sort( key=lambda p : p.stat( ).st_size )
	logger.info( f"Found {len( unprocessed_artifacts )} artifact files to process" )

	# Process each artifact file with progress tracking
	# tqdm provides console progress bar, desc sets bar label, unit customizes counter
	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Automatically sorting" ,
			unit="artifacts" ,
	) :
		try :
			artifact_ext = artifact.suffix.lower( ).strip( ).strip( '.' )
			artifact_label = artifact.stem.lower( )

			if artifact_ext in IMAGE_TYPES and is_unscanned_document( artifact_location=artifact , logger=logger ) :
				shutil.move(
						src=artifact ,
						dst=(SCANNING_REQUIRED_DIR / f"{artifact_label}.{artifact_ext}") ,
				)
				logger.info( f"Moved file to: {SCANNING_REQUIRED_DIR}" )

			sanitized_label = sanitize_artifact_name( artifact_label )

			if any(
					(artifact_ext == extension
					 for extension in [ "qpf" , "qsf" , "vwf" ]) ,
			) :
				logger.info(
						f"Moving lab/simulation artifact to: "
						f"{(DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC" / "SCIENTIFIC_LAB_REPORT")}" ,
				)
				shutil.move(
						src=artifact ,
						dst=(
								DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC" / "SCIENTIFIC_LAB_REPORT"
								/ f"{sanitized_label}.{artifact_ext}"
						) ,
				)


			elif artifact_ext in [ "arw" , "cr2" , "nef" ] :
				shutil.move( src=artifact , dst=IMMICH_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved file to: {IMMICH_DIR}" )

			elif artifact_ext in [ "iso" ] :
				shutil.move(
						src=artifact ,
						dst=(SOFTWARE_ARCHIVE_DIR / "OPERATING_SYSTEMS") / f"{sanitized_label}.{artifact_ext}" ,
				)
				logger.info( f"Moved file to: {(SOFTWARE_ARCHIVE_DIR / "OPERATING_SYSTEMS")}" )

			elif artifact_ext in [ "onepkg" ] :
				shutil.move( src=artifact , dst=AFFINE_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Copied file to: {AFFINE_DIR}" )

			elif is_bookmark_file( artifact_location=artifact ) :
				logger.info( f"Detected bookmark file: {artifact.name}" )
				shutil.move( src=artifact , dst=LINKWARDEN_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved bookmark file to: {LINKWARDEN_DIR}" )

			elif is_executable( artifact_location=artifact ) :
				logger.info( f"Detected executable file: {artifact.name}" )
				shutil.move( src=artifact , dst=SOFTWARE_ARCHIVE_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved file to: {SOFTWARE_ARCHIVE_DIR}" )

			elif is_anki_deck( artifact_location=artifact ) :
				logger.info( f"Detected Anki deck: {artifact.name}" )
				shutil.move( src=artifact , dst=ANKI_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved Anki deck to: {ANKI_DIR}" )

			elif is_3d_file( artifact_location=artifact ) :
				logger.info( f"Detected 3D file: {artifact.name}" )
				shutil.move( src=artifact , dst=ULTIMAKER_CURA_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved code file to: {ULTIMAKER_CURA_DIR}" )

			# Check if file is source code based on file extension
			elif artifact_ext != "html" and is_code( artifact_location=artifact ) :
				logger.info( f"Detected code file: {artifact.name}" )
				shutil.move( src=artifact , dst=GITLAB_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved code file to: {GITLAB_DIR}" )

			elif is_digital_contact_file( artifact_location=artifact ) :
				logger.info( f"Detected digital contact document: {artifact.name}" )

				logger.info( f"Copying digital contact document to: {MONICA_CRM_DIR}" )
				shutil.copy2( src=artifact , dst=MONICA_CRM_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Moving digital contact document to: {ODOO_CRM_DIR}" )
				shutil.move( src=artifact , dst=ODOO_CRM_DIR / f"{sanitized_label}.{artifact_ext}" )

			# Check if file contains 2FA backup/recovery codes
			elif is_personal_security_item( artifact_location=artifact ) :
				logger.info( f"Detected backup codes file: {artifact.name}" )
				shutil.move( src=artifact , dst=BITWARDEN_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved backup codes to: {BITWARDEN_DIR}" )

			elif is_instruction_manual( artifact_location=artifact , logger=logger ) :
				shutil.move( src=artifact , dst=MANUALS_ARCHIVE_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved file to: {MANUALS_ARCHIVE_DIR}" )

			elif is_professional( artifact_location=artifact , logger=logger ) :
				logger.info( f"Moving resume/professional document to: {(DIGITAL_ASSET_MANAGEMENT_DIR / "PROFESSIONAL")}" )
				shutil.move(
						src=artifact ,
						dst=(DIGITAL_ASSET_MANAGEMENT_DIR / "PROFESSIONAL") / f"{sanitized_label}.{artifact_ext}" )

			elif is_legal( artifact_location=artifact , logger=logger ) :
				shutil.move(
						src=artifact ,
						dst=(DIGITAL_ASSET_MANAGEMENT_DIR / "LEGAL") / f"{sanitized_label}.{artifact_ext}" )

			elif is_immigration( artifact_location=artifact , logger=logger ) :
				logger.info( f"Moving immigration/legal document to: {(DIGITAL_ASSET_MANAGEMENT_DIR / "IMMIGRATION")}" )
				shutil.move(
						src=artifact ,
						dst=(
										DIGITAL_ASSET_MANAGEMENT_DIR / "LEGAL" / "IMMIGRATION") / f"{sanitized_label}.{artifact_ext}" )

			elif is_academic( artifact_location=artifact , logger=logger ) :
				logger.info( f"Moving academic/educational document to: {(DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC")}" )
				shutil.move( src=artifact ,
										 dst=(DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC") / f"{sanitized_label}.{artifact_ext}" )

			elif is_video_course( artifact_location=artifact , logger=logger ) :
				logger.info( f"Detected video course file: {artifact.name}" )
				shutil.move( src=artifact , dst=JELLYFIN_DIR / f"{sanitized_label}.{artifact_ext}" )
				logger.info( f"Moved video course file to: {JELLYFIN_DIR}" )

			elif is_book( artifact_location=artifact , logger=logger ) :
				logger.info( f"Detected book: {artifact.name}" )
				shutil.move( src=artifact , dst=(PERSONAL_LIBRARY_DIR / "BOOK" / f"{sanitized_label}.{artifact_ext}") )
				logger.info( f"Moved book to: {(PERSONAL_LIBRARY_DIR / "BOOK" / f"{sanitized_label}.{artifact_ext}")}" )

			elif is_textbook( artifact_location=artifact , logger=logger ) :
				logger.info( f"Detected book: {artifact.name}" )
				shutil.move( src=artifact , dst=(PERSONAL_LIBRARY_DIR / "TEXTBOOK" / f"{sanitized_label}.{artifact_ext}") )
				logger.info( f"Moved book to: {(PERSONAL_LIBRARY_DIR / "TEXTBOOK" / f"{sanitized_label}.{artifact_ext}")}" )

			elif is_financial_document( artifact_location=artifact , logger=logger ) :
				logger.info( f"Detected financial document: {artifact.name}" )

				# Financial documents need to be copied to multiple locations
				# shutil.copy2() preserves metadata (timestamps, permissions)
				logger.info( f"Copying financial document to: {FIREFLYIII_DIR}" )
				shutil.copy2( src=artifact , dst=FIREFLYIII_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Copying financial document to: {PERFORMANCE_PORTFOLIO_DIR}" )
				shutil.copy2( src=artifact , dst=PERFORMANCE_PORTFOLIO_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Copying financial document to: {ODOO_MAINTENANCE_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_MAINTENANCE_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Copying financial document to: {ODOO_PLM_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_PLM_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Copying financial document to: {ODOO_PURCHASE_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_PURCHASE_DIR / f"{sanitized_label}.{artifact_ext}" )

				logger.info( f"Copying financial document to: {ODOO_INVENTORY_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_INVENTORY_DIR / f"{sanitized_label}.{artifact_ext}" )

				# Final move to primary storage location
				logger.info( f"Moving financial document to: {(DIGITAL_ASSET_MANAGEMENT_DIR / "FINANCIAL")}" )
				shutil.move(
						src=artifact ,
						dst=(DIGITAL_ASSET_MANAGEMENT_DIR / "FINANCIAL") / f"{sanitized_label}.{artifact_ext}" )

			else :
				# File is supported but didn't match any category
				# Leave in source directory for manual review
				logger.warning( f"Could not find automated group to sort {artifact.name}" , )

		except Exception as e :
			# Catch and log any errors during file processing
			# Continue processing remaining files despite errors
			logger.error( f"Error processing {artifact.name}: {e}" , exc_info=True )

	return None
