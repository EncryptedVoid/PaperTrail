"""
File Type Detection Module

Fast content-based file type detection using magic bytes (filetype library)
with Apache Tika server fallback for uncommon formats.
"""

import logging
import subprocess
from datetime import datetime , timezone
from pathlib import Path
from typing import Dict
from urllib.parse import quote

import filetype
import fitz
import requests
from PIL import Image , PngImagePlugin

from config import (
	EXCLUDED_KEYS_AFTER_CONVERSION ,
	EXCLUDED_PREFIXES_AFTER_CONVERSION ,
	FILETYPE_TRUSTED_EXTENSIONS ,
	MIME_TO_EXT_MAP ,
	TIKA_SERVER_PORT
)

# Build a reverse lookup: MIME string → dotted extension, only for trusted types.
# Uses the first extension in MIME_TO_EXT_MAP's list for each MIME.
_TRUSTED_MIME_TO_DOT_EXT: dict[ str , str ] = { }
for _mime , _ext_list in MIME_TO_EXT_MAP.items( ) :
	_canonical = _ext_list[ 0 ]  # e.g. ".jpg"
	_bare = _canonical.lstrip( "." )  # e.g. "jpg"
	if _bare in FILETYPE_TRUSTED_EXTENSIONS :
		_TRUSTED_MIME_TO_DOT_EXT[ _mime ] = _canonical


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _should_exclude( key: str ) -> bool :
	"""Return True if this key should NOT be injected into the new file."""
	if key in EXCLUDED_KEYS_AFTER_CONVERSION :
		return True
	for prefix in EXCLUDED_PREFIXES_AFTER_CONVERSION :
		if key.startswith( prefix ) :
			return True
	return False


def _flatten_metadata( raw: dict , prefix: str = "" ) -> Dict[ str , str ] :
	"""
	Recursively flatten a nested metadata dict into string key-value pairs.

	Handles Tika's flat dicts and ffprobe's nested structure (format.tags,
	streams[].tags, etc.).
	"""
	result: Dict[ str , str ] = { }

	for key , value in raw.items( ) :
		full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

		if isinstance( value , dict ) :
			result.update( _flatten_metadata( value , full_key ) )
		elif isinstance( value , list ) :
			# Tika sometimes returns lists for multi-value fields
			str_items = [ str( v ) for v in value if v is not None ]
			if str_items :
				result[ full_key ] = "; ".join( str_items )
		elif value is not None :
			str_val = str( value ).strip( )
			if str_val :
				result[ full_key ] = str_val

	return result


def _build_injection_dict(
		raw_metadata: dict ,
		original_name: str ,
		unique_id: str ,
		original_ext: str ,
		target_ext: str ,
) -> Dict[ str , str ] :
	"""
	Build the final metadata dict to inject:
	  1. Flatten all raw metadata
	  2. Remove format-specific keys
	  3. Add PaperTrail provenance signature
	"""
	# Flatten everything
	flat = _flatten_metadata( raw_metadata )

	# Filter out format-specific keys
	filtered: Dict[ str , str ] = { }
	for key , value in flat.items( ) :
		if not _should_exclude( key ) :
			filtered[ key ] = value

	# ── PaperTrail provenance signature ──────────────────────────────
	now = datetime.now( timezone.utc ).isoformat( )
	filtered[ "papertrail:processed_by" ] = "PaperTrail Automated Pipeline"
	filtered[ "papertrail:conversion_date" ] = now
	filtered[ "papertrail:artifact_uuid" ] = unique_id
	filtered[ "papertrail:original_filename" ] = f"{original_name}.{original_ext}"
	filtered[ "papertrail:original_format" ] = original_ext
	filtered[ "papertrail:converted_to" ] = target_ext
	filtered[ "papertrail:metadata_injected" ] = "true"
	filtered[ "papertrail:metadata_field_count" ] = str( len( filtered ) )

	return filtered


# ══════════════════════════════════════════════════════════════════════════════
# PDF — inject via PyMuPDF XMP + info dict
# ══════════════════════════════════════════════════════════════════════════════

def _inject_into_pdf(
		file_path: Path ,
		meta: Dict[ str , str ] ,
		logger: logging.Logger ,
) -> bool :
	try :
		doc = fitz.open( file_path )

		# ── Standard PDF info dict (title, author, subject, keywords) ─
		info = doc.metadata or { }
		_PDF_STANDARD = {
			"dc:title"   : "title" , "Title" : "title" ,
			"dc:creator" : "author" , "Author" : "author" , "meta:author" : "author" ,
			"dc:subject" : "subject" , "subject" : "subject" ,
			"Keywords"   : "keywords" , "meta:keyword" : "keywords" ,
			"Producer"   : "producer" ,
		}
		for meta_key , pdf_key in _PDF_STANDARD.items( ) :
			if meta_key in meta and not info.get( pdf_key ) :
				info[ pdf_key ] = meta[ meta_key ]
		doc.set_metadata( info )

		# ── XMP block with ALL metadata ──────────────────────────────
		props = [ ]
		for k , v in meta.items( ) :
			# Sanitize for XML
			safe_k = k.replace( " " , "_" ).replace( ":" , "_" ).replace( "/" , "_" )
			safe_v = (
				v.replace( "&" , "&amp;" )
				.replace( "<" , "&lt;" )
				.replace( ">" , "&gt;" )
				.replace( '"' , "&quot;" )
			)
			props.append( f"      <pt:{safe_k}>{safe_v}</pt:{safe_k}>" )

		xmp = (
				'<?xpacket begin="\xef\xbb\xbf" id="papertrail_metadata"?>\n'
				'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
				'  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
				'           xmlns:pt="http://papertrail.local/metadata/">\n'
				'    <rdf:Description rdf:about="">\n'
				+ "\n".join( props ) + "\n"
															 '    </rdf:Description>\n'
															 '  </rdf:RDF>\n'
															 '</x:xmpmeta>\n'
															 '<?xpacket end="w"?>'
		)
		doc.set_xml_metadata( xmp )

		tmp = file_path.parent / f"_meta_{file_path.name}"
		doc.save( tmp , garbage=0 , deflate=False )
		doc.close( )
		tmp.replace( file_path )

		logger.info( f"Injected {len( meta )} metadata fields into PDF: {file_path.name}" )
		return True

	except Exception as e :
		logger.warning( f"PDF metadata injection failed for {file_path.name}: {e}" )
		tmp = file_path.parent / f"_meta_{file_path.name}"
		if tmp.exists( ) :
			tmp.unlink( )
		return False


# ══════════════════════════════════════════════════════════════════════════════
# PNG — inject via Pillow PngInfo text chunks
# ══════════════════════════════════════════════════════════════════════════════

def _inject_into_png(
		file_path: Path ,
		meta: Dict[ str , str ] ,
		logger: logging.Logger ,
) -> bool :
	try :
		img = Image.open( file_path )
		png_info = PngImagePlugin.PngInfo( )

		# Preserve existing PNG text chunks
		for k , v in (img.info or { }).items( ) :
			if isinstance( v , str ) :
				png_info.add_text( k , v )

		# Add ALL metadata — every key gets stored
		for k , v in meta.items( ) :
			png_info.add_text( k , v )

		tmp = file_path.parent / f"_meta_{file_path.name}"
		img.save( tmp , format="PNG" , pnginfo=png_info , optimize=False )
		img.close( )
		tmp.replace( file_path )

		logger.info( f"Injected {len( meta )} metadata fields into PNG: {file_path.name}" )
		return True

	except Exception as e :
		logger.warning( f"PNG metadata injection failed for {file_path.name}: {e}" )
		tmp = file_path.parent / f"_meta_{file_path.name}"
		if tmp.exists( ) :
			tmp.unlink( )
		return False


# ══════════════════════════════════════════════════════════════════════════════
# MP3 / MP4 — inject via ffmpeg -metadata
# ══════════════════════════════════════════════════════════════════════════════

def _inject_into_av(
		file_path: Path ,
		meta: Dict[ str , str ] ,
		logger: logging.Logger ,
) -> bool :
	try :
		tmp = file_path.parent / f"_meta_{file_path.name}"

		cmd = [
			"ffmpeg" , "-y" ,
			"-i" , str( file_path ) ,
			"-c" , "copy" ,  # no re-encoding
		]

		# Inject EVERY key as a metadata tag
		# ffmpeg stores arbitrary key=value pairs in the container
		for k , v in meta.items( ) :
			# ffmpeg metadata keys can't contain = or newlines
			safe_k = k.replace( "=" , "_" ).replace( "\n" , " " )
			safe_v = v.replace( "\n" , " " )
			cmd.extend( [ "-metadata" , f"{safe_k}={safe_v}" ] )

		cmd.append( str( tmp ) )

		result = subprocess.run(
				cmd ,
				capture_output=True ,
				text=True ,
				encoding="utf-8" ,
				errors="replace" ,
				timeout=120 ,
		)

		if result.returncode != 0 :
			logger.warning(
					f"ffmpeg metadata injection failed for {file_path.name}: "
					f"{result.stderr[ :300 ]}" ,
			)
			if tmp.exists( ) :
				tmp.unlink( )
			return False

		if not tmp.exists( ) or tmp.stat( ).st_size == 0 :
			logger.warning( f"ffmpeg produced empty output for {file_path.name}" )
			if tmp.exists( ) :
				tmp.unlink( )
			return False

		tmp.replace( file_path )
		logger.info( f"Injected {len( meta )} metadata fields into {file_path.suffix}: {file_path.name}" )
		return True

	except subprocess.TimeoutExpired :
		logger.warning( f"ffmpeg metadata injection timed out for {file_path.name}" )
		tmp = file_path.parent / f"_meta_{file_path.name}"
		if tmp.exists( ) :
			tmp.unlink( )
		return False

	except Exception as e :
		logger.warning( f"AV metadata injection failed for {file_path.name}: {e}" )
		tmp = file_path.parent / f"_meta_{file_path.name}"
		if tmp.exists( ) :
			tmp.unlink( )
		return False


def _detect_via_tika_server( logger: logging.Logger , artifact: Path ) -> str :
	"""Query the running Tika server. Returns the MIME string."""
	safe_name = quote( artifact.name , safe="" )
	try :
		with open( artifact , "rb" ) as f :
			resp = requests.put(
					f"http://localhost:{TIKA_SERVER_PORT}/detect/stream" ,
					data=f ,
					headers={ "Content-Disposition" : f"attachment; filename={safe_name}" } ,
					timeout=60 ,
			)
		return resp.text.strip( )
	except Exception as e :
		logger.error( f"Tika server detection failed for {artifact.name}: {e}" )
		return ""


def _mime_to_dot_extension( mime: str ) -> str :
	"""
	Convert any MIME type to a dotted extension using config's MIME_TO_EXT_MAP.
	Returns "" if the MIME is unknown.
	"""
	ext_list = MIME_TO_EXT_MAP.get( mime )
	if ext_list :
		ext = ext_list[ 0 ]
		return ext if ext.startswith( "." ) else f".{ext}"
	return ""


def detect_filetype(
		logger: logging.Logger ,
		artifact: Path ,
		tika_server_process: subprocess.Popen | None ,
) -> str :
	"""
	Detect a file's type by its content and return the canonical extension.

	Args:
			:param logger:   Logger instance.
			:param artifact: Path to the file.

	Returns:
			Dotted extension string (e.g. ".jpg", ".pdf", ".stl").
			Returns ``""`` if detection fails entirely.
	"""
	# ── Fast path: filetype magic bytes ──────────────────────────────
	kind = filetype.guess( str( artifact ) )

	if kind is not None :
		mime = kind.mime

		# Check the trusted lookup first (most common case)
		dot_ext = _TRUSTED_MIME_TO_DOT_EXT.get( mime )
		if dot_ext :
			logger.debug( f"filetype detected {artifact.name} as {mime} → {dot_ext}" )
			return dot_ext

		# filetype returned a MIME we don't consider "trusted" for this
		# category — try the full MIME_TO_EXT_MAP anyway in case config
		# has a mapping (e.g. a font or exotic format).
		dot_ext = _mime_to_dot_extension( mime )
		if dot_ext :
			logger.debug(
					f"filetype detected {artifact.name} as {mime} → {dot_ext} (non-trusted category, using anyway)" ,
			)
			return dot_ext

		# filetype gave us a MIME with no mapping at all — fall through to Tika
		logger.debug( f"filetype returned unmapped MIME {mime} for {artifact.name}, falling back to Tika" )

	# ── Slow path: Tika server ───────────────────────────────────────
	if tika_server_process is None or tika_server_process.poll( ) is not None :
		raise ModuleNotFoundError( f"Tika server not running for fallback detection of {artifact.name}." )

	tika_mime = _detect_via_tika_server( logger , artifact )
	if not tika_mime :
		logger.warning( f"Tika returned empty MIME for {artifact.name}" )
		return ""

	dot_ext = _mime_to_dot_extension( tika_mime )
	if dot_ext :
		logger.debug( f"Tika detected {artifact.name} as {tika_mime} → {dot_ext}" )
	else :
		logger.warning( f"No extension mapping for Tika MIME [{tika_mime}] on {artifact.name}" )

	return dot_ext


def get_metadata(
		logger: logging.Logger ,
		artifact: Path ,
		tika_server_process: subprocess.Popen | None ,
) -> dict | None :
	"""
	Extract metadata from a file via the Tika server.

	Args:
			logger:               Logger instance.
			artifact:             Path to the file.
			tika_server_process:  Running Tika server process.

	Returns:
			Dictionary of metadata key-value pairs, or None on failure.
	"""
	if tika_server_process is None or tika_server_process.poll( ) is not None :
		logger.error( f"Tika server not running; cannot extract metadata for {artifact.name}" )
		return { }

	safe_name = quote( artifact.name , safe="" )
	try :
		with open( artifact , "rb" ) as f :
			resp = requests.put(
					f"http://localhost:{TIKA_SERVER_PORT}/meta" ,
					data=f ,
					headers={
						"Content-Disposition" : f"attachment; filename={safe_name}" ,
						"Accept"              : "application/json" ,
					} ,
					timeout=60 ,
			)
		resp.raise_for_status( )
		return resp.json( )
	except requests.Timeout :
		logger.error( f"Tika metadata request timed out for {artifact.name}" )
		return None
	except Exception as e :
		logger.error( f"Tika metadata extraction failed for {artifact.name}: {e}" )
		return None


def stop_apache_tika_server(
		logger: logging.Logger ,
		tika_server_process: subprocess.Popen | None ,
) -> None :
	if tika_server_process is not None :
		tika_server_process.terminate( )
		logger.debug( f"Apache Tika server terminated." )
		try :
			tika_server_process.wait( timeout=10 )
		except subprocess.TimeoutExpired :
			tika_server_process.kill( )

	logger.debug( f"Apache Tika server was not running. Nothing to terminate." )


def inject_metadata(
		file_path: Path ,
		raw_metadata: dict ,
		original_name: str ,
		unique_id: str ,
		original_ext: str ,
		logger: logging.Logger ,
) -> bool :
	"""
	Inject ALL metadata from the original file into the converted file.

	Preserves every metadata field except format-specific container/codec
	fields that would be incorrect for the new file type. Adds PaperTrail
	provenance signature fields.

	Non-fatal — returns False on failure and never raises.

	Args:
		file_path:     Path to the converted file (.pdf, .png, .mp3, .mp4).
		raw_metadata:  Raw metadata dict from Tika server or ffprobe.
		original_name: Original filename stem (without extension).
		unique_id:     UUID4 string assigned during processing.
		original_ext:  Original file extension (without dot).
		logger:        Logger instance.

	Returns:
		True if metadata was successfully injected, False otherwise.
	"""
	if not raw_metadata :
		logger.debug( f"No metadata to inject for {file_path.name}" )
		return False

	if not file_path.exists( ) :
		logger.warning( f"Cannot inject metadata — file not found: {file_path}" )
		return False

	target_ext = file_path.suffix.lower( ).lstrip( "." )

	# Build the complete injection dict: all fields + provenance
	meta = _build_injection_dict(
			raw_metadata=raw_metadata ,
			original_name=original_name ,
			unique_id=unique_id ,
			original_ext=original_ext ,
			target_ext=target_ext ,
	)

	if not meta :
		logger.debug( f"Injection dict empty for {file_path.name}" )
		return False

	logger.info(
			f"Injecting {len( meta )} fields into {file_path.name} "
			f"(original: {original_name}.{original_ext} → {target_ext})" ,
	)

	ext = file_path.suffix.lower( )

	if ext == ".pdf" :
		return _inject_into_pdf( file_path , meta , logger )

	elif ext == ".png" :
		return _inject_into_png( file_path , meta , logger )

	elif ext in (".mp3" , ".mp4") :
		return _inject_into_av( file_path , meta , logger )

	else :
		logger.debug( f"No injection handler for {ext} — skipping" )
		return False
