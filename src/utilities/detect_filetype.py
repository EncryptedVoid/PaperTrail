"""
File Type Detection Module

Fast content-based file type detection using magic bytes (filetype library)
with Apache Tika server fallback for uncommon formats.
"""

import logging
import subprocess
from pathlib import Path
from urllib.parse import quote

import filetype
import requests

from config import FILETYPE_TRUSTED_EXTENSIONS , MIME_TO_EXT_MAP , TIKA_SERVER_PORT

# Build a reverse lookup: MIME string → dotted extension, only for trusted types.
# Uses the first extension in MIME_TO_EXT_MAP's list for each MIME.
_TRUSTED_MIME_TO_DOT_EXT: dict[ str , str ] = { }
for _mime , _ext_list in MIME_TO_EXT_MAP.items( ) :
	_canonical = _ext_list[ 0 ]  # e.g. ".jpg"
	_bare = _canonical.lstrip( "." )  # e.g. "jpg"
	if _bare in FILETYPE_TRUSTED_EXTENSIONS :
		_TRUSTED_MIME_TO_DOT_EXT[ _mime ] = _canonical


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
