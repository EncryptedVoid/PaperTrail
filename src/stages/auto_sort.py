"""
Auto-sorting module for categorizing and organizing artifact files.

This module provides automated file sorting functionality that analyzes artifacts
and moves them to appropriate destination directories based on file type, content,
and metadata. Supports various file types including bookmarks, code, Anki decks,
backup codes, books, and financial documents.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (AFFINE_DIR , ANKI_DIR , ARTIFACT_PROFILES_DIR , BITWARDEN_DIR , DIGITAL_ASSET_MANAGEMENT_DIR ,
										DOCUMENT_TYPES , FIREFLYIII_DIR , GITLAB_DIR , IMAGE_TYPES , IMMICH_DIR , JELLYFIN_DIR ,
										LINKWARDEN_DIR ,
										MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR , ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR ,
										ODOO_PLM_DIR , ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR , PERSONAL_LIBRARY_DIR ,
										PROFILE_PREFIX , SCANNING_REQUIRED_DIR , SOFTWARE_ARCHIVE_DIR , TEXT_MODEL , ULTIMAKER_CURA_DIR ,
										VISION_MODEL)
from utilities.ai_processing import extract_text_for_detection , generate_filename , generate_tags
from utilities.automatic_sorting import (is_3d_file , is_academic , is_anki_deck , is_book , is_bookmark_file ,
																				 is_code , is_digital_contact_file , is_executable , is_financial_document ,
																				 is_immigration , is_instruction_manual , is_legal , is_personal_security_item ,
																				 is_professional , is_textbook , is_unscanned_document , is_video_course)
from utilities.checksum import generate_checksum
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
	logger.info( f"Found {len( unprocessed_artifacts )} artifact(s) to process" )

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
			artifact_size = artifact.stat( ).st_size
			logger.info( f"Processing '{artifact.name}' (ext='{artifact_ext}', size={artifact_size} bytes)" )

			# --- Pre-sort scan: check if image needs physical scanning ---
			logger.debug(
					f"Checking if '{artifact.name}' is an image type requiring scan detection (ext in IMAGE_TYPES={artifact_ext in IMAGE_TYPES})" )
			if artifact_ext in IMAGE_TYPES :
				logger.debug( f"'{artifact.name}' is an image type, running unscanned document detection" )
				if is_unscanned_document( artifact_location=artifact , logger=logger ) :
					dest = SCANNING_REQUIRED_DIR / f"{artifact_label}.{artifact_ext}"
					logger.info( f"[UNSCANNED_DOCUMENT] '{artifact.name}' detected as unscanned document, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

			sanitized_label = sanitize_artifact_name( artifact_label )

			# --- Extension-based sorting (no content extraction needed) ---

			if artifact_ext in [ "qpf" , "qsf" , "vwf" ] :
				dest = DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC" / "SCIENTIFIC_LAB_REPORT" / f"{sanitized_label}.{artifact_ext}"
				logger.info(
						f"[LAB_SIMULATION] '{artifact.name}' matched lab/simulation extension '{artifact_ext}', moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif artifact_ext in [ "arw" , "cr2" , "nef" ] :
				dest = IMMICH_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[RAW_PHOTO] '{artifact.name}' matched RAW photo extension '{artifact_ext}', moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif artifact_ext in [ "iso" ] :
				dest = (SOFTWARE_ARCHIVE_DIR / "OPERATING_SYSTEMS") / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[DISK_IMAGE] '{artifact.name}' matched ISO disk image extension, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif artifact_ext in [ "onepkg" , "one" , "onetoc2" ] :
				dest = AFFINE_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[ONENOTE_PACKAGE] '{artifact.name}' matched OneNote package extension, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			# --- Detection-based sorting (heuristic / AI classification) ---

			elif is_bookmark_file( artifact_location=artifact ) :
				dest = LINKWARDEN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[BOOKMARK] '{artifact.name}' detected as bookmark file, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif is_executable( artifact_location=artifact ) :
				dest = SOFTWARE_ARCHIVE_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[EXECUTABLE] '{artifact.name}' detected as executable, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif is_anki_deck( artifact_location=artifact ) :
				dest = ANKI_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[ANKI_DECK] '{artifact.name}' detected as Anki deck, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif is_3d_file( artifact_location=artifact ) :
				dest = ULTIMAKER_CURA_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[3D_MODEL] '{artifact.name}' detected as 3D model file, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif artifact_ext != "html" and is_code( artifact_location=artifact ) :
				dest = GITLAB_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[SOURCE_CODE] '{artifact.name}' detected as source code (non-HTML), moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif is_digital_contact_file( artifact_location=artifact ) :
				logger.info(
						f"[DIGITAL_CONTACT] '{artifact.name}' detected as digital contact file, distributing to CRM systems" )

				copy_dest = MONICA_CRM_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[DIGITAL_CONTACT] Copying '{artifact.name}' to Monica CRM at {copy_dest}" )
				shutil.copy2( src=artifact , dst=copy_dest )

				move_dest = ODOO_CRM_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[DIGITAL_CONTACT] Moving '{artifact.name}' to Odoo CRM at {move_dest}" )
				shutil.move( src=artifact , dst=move_dest )

			elif is_personal_security_item( artifact_location=artifact ) :
				dest = BITWARDEN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[SECURITY_CREDENTIAL] '{artifact.name}' detected as 2FA/backup codes, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			elif is_video_course( artifact_location=artifact , logger=logger ) :
				dest = JELLYFIN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[VIDEO_COURSE] '{artifact.name}' detected as video course, moving to {dest}" )
				shutil.move( src=artifact , dst=dest )

			# --- Content-based sorting (requires text extraction) ---

			if artifact_ext in DOCUMENT_TYPES :
				logger.info(
						f"[DOCUMENT] '{artifact.name}' is a document type (ext='{artifact_ext}'), extracting text for content-based classification" )
				content = extract_text_for_detection( artifact_location=artifact , logger=logger )

				if not content :
					logger.warning(
							f"[DOCUMENT] Text extraction returned no content for '{artifact.name}', skipping content-based classification" )
					continue

				logger.debug(
						f"[DOCUMENT] Extracted {len( content )} characters from '{artifact.name}', running content classifiers" )

				artifact_descriptor_tags: List[ str ] | None = generate_tags( logger=logger , content=content )

				if not artifact_descriptor_tags :
					logger.warning(
							f"[TAG-GENERATION] No description tags were able to be generated for '{artifact.name}'" )

				if is_instruction_manual( artifact_location=artifact , logger=logger , content=content ) :
					improved_label = generate_filename( logger=logger , content=content )
					dest = MANUALS_ARCHIVE_DIR / f"{improved_label}.{artifact_ext}"
					logger.info( f"[INSTRUCTION_MANUAL] '{sanitized_label}' detected as instruction manual, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_professional( artifact_location=artifact , logger=logger , content=content ) :
					dest = (DIGITAL_ASSET_MANAGEMENT_DIR / "PROFESSIONAL") / f"{sanitized_label}.{artifact_ext}"
					logger.info(
							f"[PROFESSIONAL] '{sanitized_label}' detected as professional/resume document, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_legal( artifact_location=artifact , logger=logger , content=content ) :
					improved_label = generate_filename( logger=logger , content=content )
					dest = (DIGITAL_ASSET_MANAGEMENT_DIR / "LEGAL") / f"{improved_label}.{artifact_ext}"
					logger.info( f"[LEGAL] '{sanitized_label}' detected as legal document, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_immigration( artifact_location=artifact , logger=logger , content=content ) :
					improved_label = generate_filename( logger=logger , content=content )
					dest = (DIGITAL_ASSET_MANAGEMENT_DIR / "LEGAL" / "IMMIGRATION") / f"{improved_label}.{artifact_ext}"
					logger.info( f"[IMMIGRATION] '{sanitized_label}' detected as immigration document, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_academic( artifact_location=artifact , logger=logger , content=content ) :
					improved_label = generate_filename( logger=logger , content=content )
					dest = (DIGITAL_ASSET_MANAGEMENT_DIR / "ACADEMIC") / f"{improved_label}.{artifact_ext}"
					logger.info( f"[ACADEMIC] '{sanitized_label}' detected as academic document, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_book( artifact_location=artifact , logger=logger , content=content ) :
					dest = PERSONAL_LIBRARY_DIR / "BOOK" / f"{sanitized_label}.{artifact_ext}"
					logger.info( f"[BOOK] '{sanitized_label}' detected as book, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_textbook( artifact_location=artifact , logger=logger , content=content ) :
					dest = PERSONAL_LIBRARY_DIR / "TEXTBOOK" / f"{sanitized_label}.{artifact_ext}"
					logger.info( f"[TEXTBOOK] '{sanitized_label}' detected as textbook, moving to {dest}" )
					shutil.move( src=artifact , dst=dest )

				elif is_financial_document( artifact_location=artifact , logger=logger , content=content ) :
					logger.info(
							f"[FINANCIAL] '{sanitized_label}' detected as financial document, distributing to finance systems" )

					improved_label = generate_filename( logger=logger , content=content )

					copy_targets = [
						("FireflyIII" , FIREFLYIII_DIR) ,
						("Performance Portfolio" , PERFORMANCE_PORTFOLIO_DIR) ,
						("Odoo Maintenance" , ODOO_MAINTENANCE_DIR) ,
						("Odoo PLM" , ODOO_PLM_DIR) ,
						("Odoo Purchase" , ODOO_PURCHASE_DIR) ,
						("Odoo Inventory" , ODOO_INVENTORY_DIR) ,
					]
					for target_name , target_dir in copy_targets :
						copy_dest = target_dir / f"{improved_label}.{artifact_ext}"
						logger.info( f"[FINANCIAL] Copying '{sanitized_label}' to {target_name} at {copy_dest}" )
						shutil.copy2( src=artifact , dst=copy_dest )

					dest = (DIGITAL_ASSET_MANAGEMENT_DIR / "FINANCIAL") / f"{sanitized_label}.{artifact_ext}"
					logger.info( f"[FINANCIAL] Moving '{sanitized_label}' to primary storage at {dest}" )
					shutil.move( src=artifact , dst=dest )

				else :
					logger.warning(
							f"[DOCUMENT] '{artifact.name}' did not match any content-based category, leaving in place for manual review" )

				papertrail_metadata = {
					"papertrail_metadata" : {
						"tags"    : artifact_descriptor_tags ,
						"content" : content ,
					} ,
				}

				profile_path = Path(
						ARTIFACT_PROFILES_DIR
						/ f"{PROFILE_PREFIX}-{generate_checksum( logger=logger , artifact_path=artifact )}.json" ,
				)
				try :
					profile_path.parent.mkdir( parents=True , exist_ok=True )
					with open( profile_path , "w" , encoding="utf-8" ) as f :
						json.dump( papertrail_metadata , f , indent="\t" , ensure_ascii=False )
					logger.info( f"[PAPERTRAIL] Saved profile for '{artifact.name}' to {profile_path}" )
				except Exception as e :
					logger.error(
							f"[PAPERTRAIL] Failed to save profile for '{artifact.name}' to {profile_path}: {e}" ,
							exc_info=True ,
					)

			else :
				logger.debug(
						f"[SKIP] '{artifact.name}' (ext='{artifact_ext}') is not a document type and was not matched by detection-based sorting, leaving in place for manual review" )

		except Exception as e :
			# Catch and log any errors during file processing
			# Continue processing remaining files despite errors
			logger.error( f"Failed to process '{artifact.name}': {e}" , exc_info=True )

	logger.info( f"Automatic sorting complete for directory: {source_dir}" )
	return None
