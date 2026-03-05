"""
Conversion Pipeline Module (excerpt — key changes only)
"""
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List , Optional

from tqdm import tqdm

from config import (
	ANKI_EXTENSIONS ,
	ARCHIVAL_DIR ,
	ARCHIVE_TYPES ,
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
from utilities.checksum import generate_checksum
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
		tika_server_process: subprocess.Popen | None = None ,
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

	for artifact in tqdm(
			unprocessed_artifacts ,
			desc="Converting format" ,
			unit="artifacts" ,
	) :
		try :
			start_time = time.time( )
			logger.info( f"Processing artifact: {artifact.name}" )

			artifact_ext: str = artifact.suffix.lower( ).strip( ).strip( "." )
			logger.info( f"Detected type for {artifact.name}: {artifact_ext}" )

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
				shutil.move( src=artifact , dst=dest_dir / artifact.name )
				continue

			elif artifact_ext in [ "pdf" , "mp4" , "mp3" , "png" ] :
				logger.info( f"Already in target format. No conversion needed." )
				shutil.move( src=artifact , dst=dest_dir / artifact.name )
				continue

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
				# FIX: No conversion protocol — leave in source_dir
				logger.warning(
						f"No conversion protocol for type '{artifact_ext}' "
						f"— leaving in source directory for manual triage" ,
				)
				continue

			# ── Handle conversion failure ────────────────────────────
			# FIX: Failed conversions stay in source_dir.
			# If the converter deleted the source, restore from archive
			# back into source_dir (not dest_dir).
			if formatted_artifact is None :
				logger.warning(
						f"Conversion failed for {artifact.name} — "
						f"leaving in source directory" ,
				)

				if not artifact.exists( ) :
					# Converter deleted the source but produced no output —
					# restore from archive back into source_dir
					archive_copy = ARCHIVAL_DIR / artifact.name
					if archive_copy.exists( ) :
						shutil.copy2(
								src=archive_copy ,
								dst=source_dir / artifact.name ,
						)
						logger.info( f"Restored from archive into source dir: {artifact.name}" )
					else :
						logger.error(
								f"Conversion failed AND source is gone for "
								f"{artifact.name} — file lost" ,
						)
				continue

			# ── Move converted file to dest_dir ──────────────────────
			converted_path = dest_dir / formatted_artifact.name
			shutil.move( src=formatted_artifact , dst=converted_path )

			# ── Inject original metadata into the converted file ─────
			if metadata :
				original_file_checksum = generate_checksum( logger=logger , artifact_path=artifact )
				artifact_checksum = generate_checksum( logger=logger , artifact_path=converted_path )

				injected = inject_metadata(
						file_path=converted_path ,
						raw_metadata=metadata ,
						original_file_checksum=original_file_checksum ,
						original_ext=artifact_ext ,
						logger=logger ,
				)

				artifact_profile_json = (
						ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_checksum}.json"
				)

				profile = {
					"original_metadata" : metadata ,
				}
				with open( artifact_profile_json , "w" , encoding="utf-8" ) as f :
					json.dump( profile , f , indent=2 , ensure_ascii=False )
				logger.info( f"Profile written: {artifact_profile_json.name}" )

				if injected :
					logger.info( f"Metadata injected into {converted_path.name}" )
				else :
					logger.debug( f"Metadata injection skipped/failed for {converted_path.name}" )

			total_duration = time.time( ) - start_time
			logger.info( f"Completed in {total_duration:.2f}s" )

		except Exception as e :
			logger.error( f"Error processing {artifact.name}: {e}" , exc_info=True )
			continue

	logger.info( "Conversion stage completed" )
	return None
