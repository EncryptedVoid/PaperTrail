"""
Conversion Pipeline Module (excerpt — key changes only)
"""
import json
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List , Optional

from tqdm import tqdm

from config import (
	ANKI_EXTENSIONS ,
	ARCHIVAL_DIR ,
	ARCHIVE_TYPES ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	AUDIO_TYPES ,
	CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS ,
	IMAGE_TYPES ,
	PROFILE_PREFIX ,
	VIDEO_TYPES ,
)
from utilities.artifact_data_manipulation import get_metadata , inject_metadata
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
		tika_server_process: subprocess.Popen | None = None ,  # ← NEW PARAM
) -> None :
	ensure_ffmpeg( )
	ensure_imagemagick( )
	ensure_pandoc( )
	ensure_par2( )
	ensure_libpff_python( )

	logger.info( "=" * 80 )
	logger.info( "ARTIFACT CONVERSION STAGE" )
	logger.info( "=" * 80 )
	logger.info( f"Starting artifact conversion process for directory: {source_dir}" )

	unprocessed_artifacts: List[ Path ] = [
		item for item in source_dir.iterdir( ) if item.is_file( )
	]

	if not unprocessed_artifacts :
		logger.info( "No artifacts found in source directory, conversion complete" )
		return None

	unprocessed_artifacts.sort( key=lambda p : p.stat( ).st_size )
	total_artifacts = len( unprocessed_artifacts )
	logger.info( f"Found {total_artifacts} file(s) to process" )

	for raw_artifact in tqdm(
			unprocessed_artifacts ,
			desc="Converting format" ,
			unit="artifacts" ,
	) :
		try :
			start_time = time.time( )
			logger.info( f"Processing artifact: {raw_artifact.name}" )

			artifact_ext: str = raw_artifact.suffix.lower( ).strip( ).strip( "." )
			logger.info( f"Detected type for {raw_artifact.name}: {artifact_ext}" )

			# ── Skip types that don't need conversion ────────────────
			if (artifact_ext in
					[ "epub" , "cbr" , "djvu" , "html" , "txt" , "csv" , "arw" ,
						"cr2" , "nef" , "heic" , "onepkg" , "gif" ]
					or artifact_ext in ANKI_EXTENSIONS
					or artifact_ext in CAD_FILES
					or artifact_ext in DIGITAL_CONTACT_EXTENSIONS
					or artifact_ext in EXECUTABLE_EXTENSIONS
					or artifact_ext in CODE_EXTENSIONS
					or artifact_ext in ARCHIVE_TYPES
			) :
				logger.info( f"Skipping — will be handled during manual triage." )
				shutil.move( src=raw_artifact , dst=dest_dir / raw_artifact.name )
				continue

			elif artifact_ext in [ "pdf" , "mp4" , "mp3" , "png" ] :
				logger.info( f"Already in target format. No conversion needed." )
				shutil.move( src=raw_artifact , dst=dest_dir / raw_artifact.name )
				continue

			# ── Rename with UUID ─────────────────────────────────────
			unique_id = uuid.uuid4( )
			original_name = raw_artifact.stem
			artifact = raw_artifact.rename(
					raw_artifact.parent / f"{ARTIFACT_PREFIX}-{unique_id}{raw_artifact.suffix}" ,
			)

			artifact_profile_json = (
					ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{unique_id}.json"
			)

			# ── Extract metadata via Tika SERVER (fast, non-fatal) ───
			metadata: Optional[ dict ] = None
			if tika_server_process is not None :
				metadata = get_metadata(
						logger=logger ,
						artifact=artifact ,
						tika_server_process=tika_server_process ,
				)
				if metadata :
					logger.info( f"Extracted {len( metadata )} metadata fields via Tika server" )
				else :
					logger.warning( f"Metadata extraction returned nothing for {artifact.name} — continuing" )
			else :
				logger.debug( "No Tika server available — skipping metadata extraction" )

			# ── Convert based on type ────────────────────────────────
			formatted_artifact = None

			if artifact_ext in DOCUMENT_TYPES :
				formatted_artifact = convert_document_to_pdf( src=artifact , logger=logger )

			elif artifact_ext in IMAGE_TYPES :
				formatted_artifact = convert_image_to_png( src=artifact , logger=logger )

			elif artifact_ext in VIDEO_TYPES :
				formatted_artifact = convert_video_to_mp4( src=artifact , logger=logger )

			elif artifact_ext in AUDIO_TYPES :
				formatted_artifact = convert_audio_to_mp3( src=artifact , logger=logger )

			elif artifact_ext in EMAIL_TYPES :
				formatted_artifact = convert_email_to_pdf( src=artifact , logger=logger )

			else :
				logger.warning(
						f"No conversion protocol for type '{artifact_ext}' — passing through" ,
				)
				# Clean up the now-pointless profile JSON
				if artifact_profile_json.exists( ) :
					artifact_profile_json.unlink( )
					logger.debug( f"Removed orphaned profile: {artifact_profile_json.name}" )
				# Revert UUID name → original name before moving
				reverted = artifact.rename(
						artifact.parent / f"{original_name}{artifact.suffix}" ,
				)
				shutil.move( src=reverted , dst=dest_dir / reverted.name )
				continue

			# ── Handle conversion failure ────────────────────────────
			if formatted_artifact is None :
				logger.warning(
						f"Conversion failed for {artifact.name} — "
						f"reverting to original name and passing through" ,
				)

				# Clean up the now-pointless profile JSON
				if artifact_profile_json.exists( ) :
					artifact_profile_json.unlink( )
					logger.debug( f"Removed orphaned profile: {artifact_profile_json.name}" )

				# The file might still exist at its UUID path if the
				# converter returned None without deleting the source
				if artifact.exists( ) :
					reverted = artifact.rename(
							artifact.parent / f"{original_name}{artifact.suffix}" ,
					)
					shutil.move( src=reverted , dst=dest_dir / reverted.name )
				else :
					# Converter deleted the source but produced no output —
					# check if the archive copy survived
					archive_copy = ARCHIVAL_DIR / artifact.name
					if archive_copy.exists( ) :
						restored = shutil.copy2(
								src=archive_copy ,
								dst=source_dir / f"{original_name}{artifact.suffix}" ,
						)
						shutil.move( src=restored , dst=dest_dir / Path( restored ).name )
						logger.info( f"Restored from archive: {original_name}{artifact.suffix}" )
					else :
						logger.error(
								f"Conversion failed AND source is gone for "
								f"{original_name}{artifact.suffix} — file lost" ,
						)
				continue

			shutil.move( src=formatted_artifact , dst=dest_dir / formatted_artifact.name )

			# ── Inject original metadata into the converted file ─────
			converted_path = dest_dir / formatted_artifact.name
			if metadata :
				injected = inject_metadata(
						file_path=converted_path ,
						raw_metadata=metadata ,
						original_name=original_name ,
						unique_id=str( unique_id ) ,
						original_ext=artifact_ext ,
						logger=logger ,
				)
				if injected :
					logger.info( f"Metadata injected into {converted_path.name}" )
				else :
					logger.debug( f"Metadata injection skipped/failed for {converted_path.name}" )

			# ── Write profile JSON (only if we got metadata) ─────────
			if metadata :
				profile = {
					"original_name" : original_name ,
					"uuid"          : str( unique_id ) ,
					"extension"     : artifact_ext ,
					"metadata"      : metadata ,
				}
				with open( artifact_profile_json , "w" , encoding="utf-8" ) as f :
					json.dump( profile , f , indent=2 , ensure_ascii=False )
				logger.info( f"Profile written: {artifact_profile_json.name}" )

			total_duration = time.time( ) - start_time
			logger.info( f"Completed in {total_duration:.2f}s" )

		except Exception as e :
			logger.error( f"Error processing {raw_artifact.name}: {e}" , exc_info=True )
			# Clean up orphaned profile if it was created
			try :
				if artifact_profile_json.exists( ) :
					artifact_profile_json.unlink( )
			except NameError :
				pass  # artifact_profile_json wasn't assigned yet
			# Revert UUID name if the file still exists under its UUID path
			try :
				if artifact.exists( ) and artifact.name != raw_artifact.name :
					artifact.rename(
							artifact.parent / f"{original_name}{artifact.suffix}" ,
					)
			except NameError :
				pass  # artifact / original_name weren't assigned yet
			continue

	logger.info( "Conversion stage completed" )
	return None
