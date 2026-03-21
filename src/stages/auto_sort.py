"""
Auto-sorting module v2 — Unified single-call LLM classification.

Key change: instead of 8 sequential LLM calls per document (one per is_* check),
we make ONE call via classify_document() that returns the category directly.

This reduces per-document LLM time from ~30-60s to ~4-8s.

Author: Ashiq Gazi
"""
import json
import logging
import shutil
from pathlib import Path
from typing import List , Optional

from tqdm import tqdm

from config import (
	ACADEMIC_FILES_DIR , ALTERATIONS_REQUIRED_DIR , ANKI_DIR ,
	ARTIFACT_PROFILES_DIR , AUDIO_TYPES , BITWARDEN_DIR ,
	BOOK_LIBRARY_DIR , CLASSIFICATION_MODEL , COMPLETED_SANITIZATION_DIR ,
	DIGITAL_ASSET_MANAGEMENT_DIR , DOCUMENT_TYPES , EMAIL_TYPES ,
	FIREFLYIII_DIR , GITLAB_DIR , IMAGE_TYPES , IMMICH_DIR ,
	IMMIGRATION_FILES_DIR , JELLYFIN_DIR , LEGAL_FILES_DIR ,
	LINKWARDEN_DIR , MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR ,
	ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR ,
	ODOO_PLM_DIR , ODOO_PURCHASE_DIR , OS_ISO_ARCHIVE_DIR ,
	PERFORMANCE_PORTFOLIO_DIR , PROFESSIONAL_FILES_DIR ,
	PROFILE_PREFIX , SCANNING_REQUIRED_DIR ,
	SCIENTIFIC_SIMULATION_DATA_DIR , SOFTWARE_ARCHIVE_DIR ,
	TEXTBOOK_LIBRARY_DIR , TEXT_MODEL , ULTIMAKER_CURA_DIR ,
	VIDEO_TYPES , VISION_MODEL , VISUAL_NOTE_FILES_DIR ,
)
from utilities.ai_processing import (
	classify_document ,
	extract_text_for_detection ,
	generate_filename_v2 ,
	generate_tags ,
)
from utilities.automatic_sorting import (
	is_3d_file , is_anki_deck , is_bookmark_file ,
	is_code , is_digital_contact_file , is_executable ,
	is_personal_security_item , is_unscanned_document ,
	is_video_course ,
)
from utilities.checksum import generate_checksum
from utilities.dependancy_ensurance import (
	ensure_apache_tika , ensure_ollama , ensure_ollama_model ,
)
from utilities.sanitization import sanitize_artifact_name


# ── Category → destination mapping ──────────────────────────────────────────
# Maps classify_document() output → (primary_dest, copy_targets)
# copy_targets is a list of dirs to copy to BEFORE moving to primary_dest.

def _get_category_routing( category: str ) -> dict :
	"""Return routing info for a classification category."""
	ROUTING = {
		"instruction_manual" : {
			"dest"   : MANUALS_ARCHIVE_DIR ,
			"copies" : [ ] ,
		} ,
		"professional"       : {
			"dest"   : PROFESSIONAL_FILES_DIR ,
			"copies" : [ ] ,
		} ,
		"legal"              : {
			"dest"   : LEGAL_FILES_DIR ,
			"copies" : [ ] ,
		} ,
		"immigration"        : {
			"dest"   : IMMIGRATION_FILES_DIR ,
			"copies" : [ ] ,
		} ,
		"academic"           : {
			"dest"   : ACADEMIC_FILES_DIR ,
			"copies" : [ ] ,
		} ,
		"book"               : {
			"dest"   : BOOK_LIBRARY_DIR ,
			"copies" : [ ] ,
		} ,
		"textbook"           : {
			"dest"   : TEXTBOOK_LIBRARY_DIR ,
			"copies" : [ ] ,
		} ,
		"financial"          : {
			"dest"   : DIGITAL_ASSET_MANAGEMENT_DIR / "FINANCIAL" ,
			"copies" : [
				FIREFLYIII_DIR ,
				PERFORMANCE_PORTFOLIO_DIR ,
				ODOO_MAINTENANCE_DIR ,
				ODOO_PLM_DIR ,
				ODOO_PURCHASE_DIR ,
				ODOO_INVENTORY_DIR ,
			] ,
		} ,
	}
	return ROUTING.get( category , None )


def automatically_sorting(
		logger: logging.Logger ,
		source_dir: Path ,
		dest_dir: Optional[ Path ] = None ,
) -> None :
	"""
	Automatically sort and organize artifact files using unified LLM classification.

	v2: Single LLM call per document instead of 8 sequential calls.
	"""
	logger.info( f"Starting automatic sorting v2 for directory: {source_dir}" )
	ensure_apache_tika( )
	ensure_ollama( )
	ensure_ollama_model( VISION_MODEL , logger )
	ensure_ollama_model( TEXT_MODEL , logger )
	ensure_ollama_model( CLASSIFICATION_MODEL , logger )

	unprocessed_artifacts: List[ Path ] = [
		item for item in source_dir.iterdir( ) if item.is_file( )
	]
	unprocessed_artifacts.sort( key=lambda p : p.stat( ).st_size )
	logger.info( f"Found {len( unprocessed_artifacts )} artifact(s) to process" )

	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Automatically sorting" ,
			unit="artifacts" ,
	) :
		try :
			artifact_ext = artifact.suffix.lower( ).strip( ).strip( '.' )
			artifact_label = artifact.stem.lower( )
			artifact_size = artifact.stat( ).st_size
			logger.info(
					f"Processing '{artifact.name}' "
					f"(ext='{artifact_ext}', size={artifact_size} bytes)" ,
			)

			# ── Pre-sort: check if image needs physical scanning ─────
			if artifact_ext in IMAGE_TYPES :
				if is_unscanned_document( artifact_location=artifact , logger=logger ) :
					dest = SCANNING_REQUIRED_DIR / f"{artifact_label}.{artifact_ext}"
					logger.info( f"[UNSCANNED] '{artifact.name}' → {dest}" )
					shutil.move( src=artifact , dst=dest )
					continue

			sanitized_label = sanitize_artifact_name( artifact_label )

			# ── Format check: needs conversion first? ────────────────
			if (
					(artifact_ext in DOCUMENT_TYPES and artifact_ext != "pdf")
					or (artifact_ext in VIDEO_TYPES and artifact_ext != "mp4")
					or (artifact_ext in AUDIO_TYPES and artifact_ext != "mp3")
					or (artifact_ext in EMAIL_TYPES
							and artifact_ext not in ("pdf" , "eml"))
			) :
				dest = COMPLETED_SANITIZATION_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[UNFORMATTED] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			# ── Extension-based sorting (no LLM needed) ──────────────

			if artifact_ext in [ "qpf" , "qsf" , "vwf" , "bdf" ] :
				dest = SCIENTIFIC_SIMULATION_DATA_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[LAB_SIM] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if artifact_ext in [ "arw" , "cr2" , "nef" ] :
				dest = IMMICH_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[RAW_PHOTO] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if artifact_ext in [ "iso" , "img" , "vbox" , "vdi" ] :
				dest = OS_ISO_ARCHIVE_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[DISK_IMAGE] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if artifact_ext in [ "onepkg" , "one" , "onetoc2" ] :
				dest = VISUAL_NOTE_FILES_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[ONENOTE] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if artifact_ext == "json" :
				dest = ALTERATIONS_REQUIRED_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[JSON_REVIEW] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			# ── Heuristic detection (no LLM, fast checks) ────────────

			if is_bookmark_file( artifact_location=artifact ) :
				dest = LINKWARDEN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[BOOKMARK] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if is_executable( artifact_location=artifact ) :
				dest = SOFTWARE_ARCHIVE_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[EXECUTABLE] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if is_anki_deck( artifact_location=artifact ) :
				dest = ANKI_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[ANKI] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if is_3d_file( artifact_location=artifact ) :
				dest = ULTIMAKER_CURA_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[3D_MODEL] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if artifact_ext != "html" and is_code( artifact_location=artifact ) :
				dest = GITLAB_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[CODE] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if is_digital_contact_file( artifact_location=artifact ) :
				copy_dest = MONICA_CRM_DIR / f"{sanitized_label}.{artifact_ext}"
				shutil.copy2( src=artifact , dst=copy_dest )
				move_dest = ODOO_CRM_DIR / f"{sanitized_label}.{artifact_ext}"
				shutil.move( src=artifact , dst=move_dest )
				logger.info( f"[CONTACT] '{artifact.name}' → CRMs" )
				continue

			if is_personal_security_item( artifact_location=artifact ) :
				dest = BITWARDEN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[SECURITY] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			if is_video_course( artifact_location=artifact , logger=logger ) :
				dest = JELLYFIN_DIR / f"{sanitized_label}.{artifact_ext}"
				logger.info( f"[VIDEO_COURSE] '{artifact.name}' → {dest}" )
				shutil.move( src=artifact , dst=dest )
				continue

			# ══════════════════════════════════════════════════════════
			# DOCUMENT CONTENT-BASED SORTING — SINGLE LLM CALL
			# ══════════════════════════════════════════════════════════

			if artifact_ext not in DOCUMENT_TYPES :
				if dest_dir :
					fallback = dest_dir / f"{sanitized_label}.{artifact_ext}"
					dest_dir.mkdir( parents=True , exist_ok=True )
					logger.debug(
							f"[SKIP] '{artifact.name}' not a document type, "
							f"moving to dest_dir for manual review" ,
					)
					shutil.move( src=artifact , dst=fallback )
				else :
					logger.debug(
							f"[SKIP] '{artifact.name}' not a document type, "
							f"leaving for manual review" ,
					)
				continue

			logger.info(
					f"[DOCUMENT] Extracting content from '{artifact.name}' "
					f"for unified classification" ,
			)
			content = extract_text_for_detection(
					artifact_location=artifact , logger=logger ,
			)

			if not content :
				if dest_dir :
					fallback = dest_dir / f"{sanitized_label}.{artifact_ext}"
					dest_dir.mkdir( parents=True , exist_ok=True )
					logger.warning(
							f"[DOCUMENT] No content extracted from '{artifact.name}', "
							f"moving to dest_dir for manual review" ,
					)
					shutil.move( src=artifact , dst=fallback )
				else :
					logger.warning(
							f"[DOCUMENT] No content extracted from '{artifact.name}', "
							f"skipping classification" ,
					)
				continue

			# ── Generate tags (parallel-safe, independent of classification) ──
			tags: Optional[ List[ str ] ] = generate_tags(
					logger=logger , content=content ,
			)

			# ── Classify: keywords first, then ONE LLM call ─────────
			category = classify_document(
					logger=logger ,
					content=content ,
					filename_stem=artifact_label ,
					file_ext=artifact_ext ,
			)
			logger.info(
					f"[CLASSIFY] '{artifact.name}' → category='{category}'" ,
			)

			# ── Generate improved filename ───────────────────────────
			improved_label = generate_filename_v2(
					logger=logger , content=content ,
			)
			if not improved_label :
				improved_label = sanitized_label
				logger.warning(
						f"[FILENAME] Fallback to sanitized name for '{artifact.name}'" ,
				)

			# ── Route based on category ──────────────────────────────
			routing = _get_category_routing( category )

			if routing is None :
				if dest_dir :
					fallback = dest_dir / f"{improved_label}.{artifact_ext}"
					dest_dir.mkdir( parents=True , exist_ok=True )
					logger.info(
							f"[UNKNOWN] '{artifact.name}' classified as '{category}', "
							f"moving to dest_dir for manual review → {fallback}" ,
					)
					shutil.move( src=artifact , dst=fallback )
				else :
					logger.warning(
							f"[UNKNOWN] '{artifact.name}' classified as '{category}', "
							f"no dest_dir set — leaving in place" ,
					)
			else :
				# Copy to secondary destinations first
				for copy_dir in routing[ "copies" ] :
					copy_dest = copy_dir / f"{improved_label}.{artifact_ext}"
					logger.info(
							f"[{category.upper( )}] Copying '{artifact.name}' → {copy_dest}" ,
					)
					shutil.copy2( src=artifact , dst=copy_dest )

				# Move to primary destination
				dest = routing[ "dest" ] / f"{improved_label}.{artifact_ext}"
				dest.parent.mkdir( parents=True , exist_ok=True )
				logger.info(
						f"[{category.upper( )}] Moving '{artifact.name}' → {dest}" ,
				)
				shutil.move( src=artifact , dst=dest )

			# ── Save profile with tags + content ─────────────────────
			# Determine where the file ended up for checksum
			if routing is not None :
				final_path = routing[ "dest" ] / f"{improved_label}.{artifact_ext}"
			elif dest_dir :
				final_path = dest_dir / f"{improved_label}.{artifact_ext}"
			else :
				final_path = artifact

			papertrail_metadata = {
				"papertrail_metadata" : {
					"tags"               : tags ,
					"category"           : category ,
					"generated_filename" : improved_label ,
					"content"            : content ,
				} ,
			}

			if final_path.exists( ) :
				profile_path = Path(
						ARTIFACT_PROFILES_DIR
						/ f"{PROFILE_PREFIX}-{generate_checksum( logger=logger , artifact_path=final_path )}.json" ,
				)
				try :
					profile_path.parent.mkdir( parents=True , exist_ok=True )
					with open( profile_path , "w" , encoding="utf-8" ) as f :
						json.dump( papertrail_metadata , f , indent="\t" , ensure_ascii=False )
					logger.info( f"[PROFILE] Saved → {profile_path}" )
				except Exception as e :
					logger.error(
							f"[PROFILE] Failed for '{artifact.name}': {e}" ,
							exc_info=True ,
					)
			else :
				logger.warning(
						f"[PROFILE] Cannot generate — file not found at {final_path}" ,
				)

		except Exception as e :
			logger.error( f"Failed to process '{artifact.name}': {e}" , exc_info=True )

	logger.info( f"Automatic sorting v2 complete for: {source_dir}" )
