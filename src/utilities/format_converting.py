"""
Format Converting Module

Converts files in-place to standardized formats, deletes the original,
and returns the path to the new file (or None on failure).

Supported conversions:
	Images       → PNG   (Pillow, rawpy, pillow-heif)
	PNG          → PDF   (Pillow + ReportLab)
	Video        → MP4   (FFmpeg: H.264 + AAC)
	Audio        → MP3   (FFmpeg: libmp3lame)
	Documents    → PDF   (LibreOffice headless)
	HTML         → PDF   (WeasyPrint)
	Email        → PDF   (emailconverter jar)
	XLSX         → CSV   (openpyxl)
	Publisher    → PDF   (LibreOffice headless)

Author: Ashiq Gazi
"""

from __future__ import annotations

import logging
import os
import subprocess
from email.header import decode_header , make_header
from pathlib import Path
from typing import Optional

import rawpy
from PIL import Image , ImageOps
from config import EMAIL_TYPES
from pillow_heif import register_heif_opener
from reportlab.pdfgen import canvas
from utilities.dependancy_ensurance import find_libreoffice
from weasyprint import CSS , HTML


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _decode_header_value( raw: str | None ) -> str :
	"""Decode an RFC-2047 email header into a plain Unicode string."""
	if not raw :
		return ""
	try :
		return str( make_header( decode_header( raw ) ) )
	except Exception :
		return raw or ""


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE → PNG
# ══════════════════════════════════════════════════════════════════════════════


def convert_image_to_png( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""Convert any supported image (NEF, HEIC, JPEG, BMP, TIFF, WebP, GIF, etc.) to PNG in-place."""
	src = src.resolve( )
	dst = src.with_suffix( ".png" )
	suffix = src.suffix.lower( )
	logger.info( "Converting image → PNG: '%s' (%s)" , src.name , suffix )

	try :
		if suffix == ".nef" :
			# Nikon RAW — demosaic sensor data via rawpy

			logger.debug( "Decoding NEF via rawpy with camera white balance" )
			with rawpy.imread( str( src ) ) as raw :
				rgb = raw.postprocess( use_camera_wb=True , output_bps=8 )
			Image.fromarray( rgb ).save( dst , format="PNG" )

		elif suffix in (".heic" , ".heif") :
			heif_ok = False
			try :
				register_heif_opener( )
				logger.debug( "Attempting HEIC decode via pillow-heif" )

				with Image.open( src ) as img :
					img = ImageOps.exif_transpose( img )
					img.convert( "RGB" ).save( dst , format="PNG" )

				heif_ok = True

			except ImportError :
				logger.warning( "pillow-heif not installed — falling back to FFmpeg" )
			except Exception as exc :
				logger.warning( "pillow-heif failed for '%s': %s — falling back to FFmpeg" , src.name , exc )

			if not heif_ok :
				logger.info( "Attempting HEIC → PNG via FFmpeg" )
				subprocess.run(
						[ "ffmpeg" , "-y" , "-i" , str( src ) , str( dst ) ] ,
						capture_output=True , text=True , timeout=120 , check=True ,
				)
				if not dst.exists( ) :
					raise RuntimeError( f"FFmpeg produced no output for '{src.name}'" )

		else :
			# Standard formats handled by Pillow

			with Image.open( src ) as img :
				if img.mode not in ("RGBA" , "RGB" , "L" , "LA") :
					target = "RGBA" if getattr( img , "has_transparency_data" , False ) else "RGB"
					logger.debug( "Converting pixel mode %s → %s" , img.mode , target )
					img = img.convert( target )
				img.save( dst , format="PNG" )

		logger.info( "PNG created: '%s' (%.1f KB)" , dst.name , dst.stat( ).st_size / 1024 )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "Image → PNG failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# PNG → PDF
# ══════════════════════════════════════════════════════════════════════════════


def convert_png_to_pdf( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""Convert a PNG image to a single-page PDF sized to the image dimensions."""
	src = src.resolve( )
	dst = src.with_suffix( ".pdf" )
	logger.info( "Converting PNG → PDF: '%s'" , src.name )

	try :
		with Image.open( src ) as img :
			w_px , h_px = img.size
			dpi = img.info.get( "dpi" , (72 , 72) )
			# Convert pixels → points (1 pt = 1/72 in)
			w_pt = (w_px / max( dpi[ 0 ] , 1 )) * 72
			h_pt = (h_px / max( dpi[ 1 ] , 1 )) * 72

		logger.debug( "Image: %dx%d px, DPI: %s → page: %.0f×%.0f pt" , w_px , h_px , dpi , w_pt , h_pt )

		c = canvas.Canvas( str( dst ) , pagesize=(w_pt , h_pt) )
		c.drawImage( str( src ) , 0 , 0 , width=w_pt , height=h_pt )
		c.save( )

		logger.info( "PDF created: '%s' (%.1f KB)" , dst.name , dst.stat( ).st_size / 1024 )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "PNG → PDF failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO → MP4
# ══════════════════════════════════════════════════════════════════════════════


def convert_video_to_mp4( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""Convert a video file to MP4 (H.264 + AAC) via FFmpeg."""
	src = src.resolve( )
	dst = src.with_suffix( ".mp4" )
	logger.info( "Converting video → MP4: '%s'" , src.name )

	try :
		subprocess.run(
				[
					"ffmpeg" , "-y" , "-i" , str( src ) ,
					"-c:v" , "libx264" , "-crf" , "18" , "-preset" , "slow" ,
					"-c:a" , "aac" , "-b:a" , "192k" ,
					"-movflags" , "+faststart" ,
					str( dst ) ,
				] ,
				capture_output=True , text=True , timeout=3600 , check=True ,
		)

		if not dst.exists( ) :
			raise RuntimeError( f"FFmpeg produced no output for '{src.name}'" )

		logger.info( "MP4 created: '%s' (%.1f MB)" , dst.name , dst.stat( ).st_size / (1024 * 1024) )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "Video → MP4 failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO → MP3
# ══════════════════════════════════════════════════════════════════════════════


def convert_audio_to_mp3( src: Path , logger: logging.Logger , bitrate: str = "320k" ) -> Optional[ Path ] :
	"""Convert an audio file to MP3 via FFmpeg. Stream-copies if already MP3."""
	src = src.resolve( )
	dst = src.with_suffix( ".mp3" )
	logger.info( "Converting audio → MP3: '%s' (bitrate=%s)" , src.name , bitrate )

	if src.suffix.lower( ) == ".mp3" :
		# Already MP3 — stream-copy to avoid re-encoding quality loss
		encode_args = [ "-c:a" , "copy" ]
	else :
		encode_args = [ "-c:a" , "libmp3lame" , "-b:a" , bitrate , "-id3v2_version" , "3" , "-write_id3v1" , "1" ]

	try :
		subprocess.run(
				[ "ffmpeg" , "-y" , "-i" , str( src ) ] + encode_args + [ str( dst ) ] ,
				capture_output=True , text=True , timeout=600 , check=True ,
		)

		if not dst.exists( ) :
			raise RuntimeError( f"FFmpeg produced no output for '{src.name}'" )

		logger.info( "MP3 created: '%s' (%.1f MB)" , dst.name , dst.stat( ).st_size / (1024 * 1024) )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "Audio → MP3 failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# HTML → PDF
# ══════════════════════════════════════════════════════════════════════════════


def convert_html_to_pdf( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""Convert an HTML file to PDF via WeasyPrint."""
	src = src.resolve( )
	dst = src.with_suffix( ".pdf" )
	logger.info( "Converting HTML → PDF: '%s'" , src.name )

	try :
		HTML( filename=str( src ) ).write_pdf(
				str( dst ) ,
				stylesheets=[ CSS( string="@page { margin: 20mm 15mm; }" ) ] ,
		)

		if not dst.exists( ) :
			raise RuntimeError( f"WeasyPrint produced no output for '{src.name}'" )

		logger.info( "PDF created: '%s' (%.1f KB)" , dst.name , dst.stat( ).st_size / 1024 )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "HTML → PDF failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT → PDF  (LibreOffice headless)
# ══════════════════════════════════════════════════════════════════════════════


def convert_document_to_pdf( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""Convert a document (DOCX, XLSX, PPTX, ODT, RTF, etc.) to PDF via LibreOffice headless."""
	src = src.resolve( )
	dst = src.with_suffix( ".pdf" )
	logger.info( "Converting document → PDF via LibreOffice: '%s'" , src.name )

	try :
		soffice = find_libreoffice( )
		logger.debug( "LibreOffice found at: %s" , soffice )
	except RuntimeError as exc :
		logger.error( "LibreOffice not available: %s" , exc )
		return None

	try :
		subprocess.run(
				[
					soffice ,
					"--headless" , "--nologo" , "--norestore" ,
					"--convert-to" , "pdf" ,
					"--outdir" , str( src.parent ) ,
					str( src ) ,
				] ,
				capture_output=True , text=True , timeout=300 , check=True ,
		)

		if not dst.exists( ) :
			raise RuntimeError(
					f"LibreOffice produced no output for '{src.name}' — "
					"file may be password-protected or corrupt" ,
			)

		logger.info( "PDF created: '%s' (%.1f KB)" , dst.name , dst.stat( ).st_size / 1024 )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "Document → PDF failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL → PDF
# ══════════════════════════════════════════════════════════════════════════════


def _find_on_path( name: str ) -> Path | None :
	"""Return the first match for `name` on PATH, or None."""
	for directory in os.environ.get( "PATH" , "" ).split( os.pathsep ) :
		candidate = Path( directory ) / name
		if candidate.is_file( ) :
			return candidate.resolve( )
	return None


def convert_email_to_pdf( src: Path , logger: logging.Logger ) -> Path :
	"""Convert an email (.eml/.msg) to PDF using emailconverter.

	Raises:
		FileNotFoundError: Missing source file, Java runtime, or emailconverter jar.
		ValueError:        Unsupported file extension.
		RuntimeError:      Conversion process failure or timeout.
	"""
	src = src.resolve( )

	if not src.exists( ) :
		raise FileNotFoundError( f"Source file not found: {src}" )

	if src.suffix.lower( ) not in EMAIL_TYPES :
		raise ValueError( f"Unsupported file type '{src.suffix}'. Expected one of: {', '.join( EMAIL_TYPES )}" )

	if not _find_on_path( "java" ) and not _find_on_path( "java.exe" ) :
		raise FileNotFoundError( "Java runtime not found on PATH. Install a JRE (8+) to use emailconverter." )

	jar_path = _find_on_path( "emailconverter.jar" )
	if jar_path is None :
		jar_path = Path( __file__ ).parent / "emailconverter.jar"
	jar_path = jar_path.resolve( )

	if not jar_path.exists( ) :
		raise FileNotFoundError(
				f"emailconverter.jar not found (checked PATH and {jar_path}). "
				f"Download from https://github.com/nickrussler/email-to-pdf-converter/releases" ,
		)

	dst = src.with_suffix( ".pdf" )
	cmd = [ "java" , "-jar" , str( jar_path ) , str( src ) , "-o" , str( dst ) , "-a" ]

	logger.info( "Converting email → PDF: '%s'" , src.name )
	logger.debug( "Command: %s" , " ".join( cmd ) )

	try :
		result = subprocess.run( cmd , capture_output=True , text=True , timeout=120 )
	except subprocess.TimeoutExpired :
		raise RuntimeError( f"Conversion timed out after 120s for: {src.name}" )

	if result.returncode != 0 :
		logger.error( "stderr: %s" , result.stderr.strip( ) )
		logger.error( "stdout: %s" , result.stdout.strip( ) )
		raise RuntimeError(
				f"emailconverter exited with code {result.returncode} "
				f"for '{src.name}': {result.stderr.strip( )}" ,
		)

	if not dst.exists( ) :
		raise RuntimeError( f"Conversion succeeded but output not found: {dst}" )

	logger.info( "PDF created: '%s'" , dst.name )
	return dst
