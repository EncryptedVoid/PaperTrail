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

from config import (AFFINE_DIR , ANKI_DIR , BITWARDEN_DIR , CALIBRE_LIBRARY_DIR , DIGITAL_ASSET_MANAGEMENT_DIR ,
										FIREFLYIII_DIR , GITLAB_DIR , IMMICH_DIR , LINKWARDEN_DIR , MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR ,
										ODOO_CRM_DIR ,
										ODOO_INVENTORY_DIR ,
										ODOO_MAINTENANCE_DIR , ODOO_PLM_DIR , ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR ,
										SOFTWARE_ARCHIVE_DIR , ULTIMAKER_CURA_DIR)
from tqdm import tqdm
from utilities.automatic_sorting import (
	is_3d_file ,
	is_anki_deck ,
	is_backup_codes_file ,
	is_book ,
	is_bookmark_file ,
	is_code ,
	is_digital_contact_file ,
	is_executable ,
	is_financial_document ,
)
from utilities.visual_processor import VisualProcessor


def automatically_sorting(
		logger: logging.Logger ,
		visual_processor: VisualProcessor ,
		source_dir: Path ,
) :
	"""
	Automatically sort and organize artifact files to detected locations.

	Args:
		logger: Logger instance for recording process information and errors
		visual_processor: VisualProcessor instance for analyzing image/PDF content
		source_dir: Path object pointing to the directory containing artifacts to sort

	Returns:
		None: Function completes silently after processing all artifacts
	"""

	logger.info( f"Starting automatic sorting process for directory: {source_dir}" )

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
			# Log current file being processed with size information
			logger.info( f"Processing file: {artifact.name} ({artifact.stat( ).st_size / 1024:.2f} KB)" )

			artifact_ext = artifact.suffix.lower( ).strip( ).strip( '.' )

			if "manual" in artifact.stem.lower( ).strip( ) :
				shutil.move( src=artifact , dst=MANUALS_ARCHIVE_DIR )
				logger.info( f"Moved file to: {MANUALS_ARCHIVE_DIR}" )

			elif artifact_ext in [ "arw" , "cr2" , "nef" ] :
				shutil.move( src=artifact , dst=IMMICH_DIR )
				logger.info( f"Moved file to: {IMMICH_DIR}" )

			elif artifact_ext in [ "iso" ] :
				shutil.move( src=artifact , dst=SOFTWARE_ARCHIVE_DIR )
				logger.info( f"Moved file to: {SOFTWARE_ARCHIVE_DIR}" )

			elif artifact_ext in [ "onepkg" ] :
				shutil.copy2( src=artifact , dst=AFFINE_DIR )
				logger.info( f"Copied file to: {AFFINE_DIR}" )

				shutil.move( src=artifact , dst=DIGITAL_ASSET_MANAGEMENT_DIR )
				logger.info( f"Moved file to: {DIGITAL_ASSET_MANAGEMENT_DIR}" )

			# Check if file is an HTML bookmark export
			elif (is_bookmark_file( artifact_location=artifact )) :
				logger.info( f"Detected bookmark file: {artifact.name}" )
				shutil.move( src=artifact , dst=LINKWARDEN_DIR )
				logger.info( f"Moved bookmark file to: {LINKWARDEN_DIR}" )

			elif is_executable( artifact_location=artifact ) :
				logger.info( f"Detected executable file: {artifact.name}" )

				shutil.move( src=artifact , dst=SOFTWARE_ARCHIVE_DIR )
				logger.info( f"Moved file to: {SOFTWARE_ARCHIVE_DIR}" )

			# Check if file is an Anki flashcard deck
			elif is_anki_deck( artifact_location=artifact ) :
				logger.info( f"Detected Anki deck: {artifact.name}" )
				shutil.move( src=artifact , dst=ANKI_DIR )
				logger.info( f"Moved Anki deck to: {ANKI_DIR}" )

			elif is_3d_file( artifact_location=artifact ) :
				logger.info( f"Detected 3D file: {artifact.name}" )
				shutil.move( src=artifact , dst=ULTIMAKER_CURA_DIR )
				logger.info( f"Moved code file to: {ULTIMAKER_CURA_DIR}" )

			# Check if file is source code based on file extension
			elif (
					artifact_ext not in "html"
					and (artifact.stem == "README" or is_code( artifact_location=artifact ))
			) :
				logger.info( f"Detected code file: {artifact.name}" )
				shutil.move( src=artifact , dst=GITLAB_DIR )
				logger.info( f"Moved code file to: {GITLAB_DIR}" )

			elif is_digital_contact_file( artifact_location=artifact ) :
				logger.info( f"Detected digital contact document: {artifact.name}" )

				logger.info( f"Copying digital contact document to: {MONICA_CRM_DIR}" )
				shutil.copy2( src=artifact , dst=MONICA_CRM_DIR )

				logger.info( f"Moving digital contact document to: {ODOO_CRM_DIR}" )
				shutil.move( src=artifact , dst=ODOO_CRM_DIR )

			# Check if file contains 2FA backup/recovery codes
			elif (
					is_backup_codes_file( artifact_location=artifact )
					or ("dashlane" in artifact.stem.lower( ).strip( ))
			) :
				logger.info( f"Detected backup codes file: {artifact.name}" )
				shutil.move( src=artifact , dst=BITWARDEN_DIR )
				logger.info( f"Moved backup codes to: {BITWARDEN_DIR}" )

			# Check if file is a book (EPUB, PDF with ISBN, etc.)
			elif is_book( artifact_location=artifact ) :
				logger.info( f"Detected book: {artifact.name}" )
				shutil.move( src=artifact , dst=CALIBRE_LIBRARY_DIR )
				logger.info( f"Moved book to: {CALIBRE_LIBRARY_DIR}" )

			# Check if file is a financial document (invoice, receipt, statement)
			elif is_financial_document(
					artifact_location=artifact ,
					visual_processor=visual_processor ,
					logger=logger ,
			) :
				logger.info( f"Detected financial document: {artifact.name}" )

				# Financial documents need to be copied to multiple locations
				# shutil.copy2() preserves metadata (timestamps, permissions)
				logger.info( f"Copying financial document to: {FIREFLYIII_DIR}" )
				shutil.copy2( src=artifact , dst=FIREFLYIII_DIR )

				logger.info( f"Copying financial document to: {PERFORMANCE_PORTFOLIO_DIR}" )
				shutil.copy2( src=artifact , dst=PERFORMANCE_PORTFOLIO_DIR )

				logger.info( f"Copying financial document to: {ODOO_MAINTENANCE_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_MAINTENANCE_DIR )

				logger.info( f"Copying financial document to: {ODOO_PLM_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_PLM_DIR )

				logger.info( f"Copying financial document to: {ODOO_PURCHASE_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_PURCHASE_DIR )

				logger.info( f"Copying financial document to: {ODOO_INVENTORY_DIR}" )
				shutil.copy2( src=artifact , dst=ODOO_INVENTORY_DIR )

				# Final move to primary storage location
				logger.info( f"Moving financial document to: {DIGITAL_ASSET_MANAGEMENT_DIR}" )
				shutil.move( src=artifact , dst=DIGITAL_ASSET_MANAGEMENT_DIR )

			else :
				# File is supported but didn't match any category
				# Leave in source directory for manual review
				logger.warning( f"Could not find automated group to sort {artifact.name}" , )

		except Exception as e :
			# Catch and log any errors during file processing
			# Continue processing remaining files despite errors
			logger.error( f"Error processing {artifact.name}: {e}" , exc_info=True )

	return None
