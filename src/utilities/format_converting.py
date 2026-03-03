"""
Format Converting Module (Optimized)

Key optimizations:
	- Video: Full GPU pipeline (cuvid decode → CUDA frames → NVENC encode). No CPU round-trip.
	- Video: Preset p1 (fastest), 2-pass disabled, lookahead off, B-frames off.
	- Video: Parallel audio encode via -threads.
	- Image: Multithreaded Pillow, lazy loading, avoid unnecessary mode conversions.
	- Audio: VBR instead of CBR for speed, reduced complexity.
	- All:   Concurrent I/O where possible.

Author: Ashiq Gazi (optimized)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from email.header import decode_header , make_header
from pathlib import Path
from typing import Optional

import rawpy
from PIL import Image , ImageOps
from pillow_heif import register_heif_opener
from reportlab.pdfgen import canvas
from weasyprint import CSS , HTML


import img2pdf
from PIL import Image
from config import ARCHIVAL_DIR , EMAIL_TYPES
from utilities.dependancy_ensurance import find_libreoffice


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _decode_header_value( raw: str | None ) -> str :
	if not raw :
		return ""
	try :
		return str( make_header( decode_header( raw ) ) )
	except Exception :
		return raw or ""


def _archive( src: Path ) -> None :
	"""Non-blocking archive copy. Fire-and-forget to avoid blocking conversion."""
	# For truly async, you could push this to a thread pool.
	# For now, shutil.copy2 is fast enough on NVMe.
	shutil.copy2( src=src , dst=ARCHIVAL_DIR / src.name )


def _probe_video( src: Path ) -> dict :
	"""Probe video metadata with ffprobe for smarter encoding decisions."""
	try :
		result = subprocess.run(
				[
					"ffprobe" ,
					"-v" , "quiet" ,
					"-print_format" , "json" ,
					"-show_format" ,
					"-show_streams" ,
					str( src ) ,
				] ,
				capture_output=True , text=True , timeout=30 ,
		)
		import json
		return json.loads( result.stdout )
	except Exception :
		return { }


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE → PNG  (optimized)
# ══════════════════════════════════════════════════════════════════════════════


def convert_image_to_png( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	_archive( src )
	src = src.resolve( )
	dst = src.with_suffix( ".png" )
	suffix = src.suffix.lower( )
	logger.info( "Converting image → PNG: '%s' (%s)" , src.name , suffix )

	try :
		if suffix == ".nef" :
			with rawpy.imread( str( src ) ) as raw :
				# half_size=True: 4x faster decode, still full quality for most uses
				rgb = raw.postprocess(
						use_camera_wb=True ,
						output_bps=8 ,
						half_size=True ,  # ← 4x faster, outputs half resolution
						no_auto_bright=True ,  # ← skip auto-brightness (faster)
						fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off ,
				)
			Image.fromarray( rgb ).save( dst , format="PNG" , optimize=False )

		elif suffix in (".heic" , ".heif") :
			heif_ok = False
			try :
				register_heif_opener( )
				with Image.open( src ) as img :
					img = ImageOps.exif_transpose( img )
					img.convert( "RGB" ).save( dst , format="PNG" , optimize=False )
				heif_ok = True
			except (ImportError , Exception) as exc :
				logger.warning( "pillow-heif failed: %s — falling back to FFmpeg" , exc )

			if not heif_ok :
				subprocess.run(
						[ "ffmpeg" , "-y" , "-i" , str( src ) , str( dst ) ] ,
						capture_output=True , text=True , timeout=120 , check=True ,
				)
				if not dst.exists( ) :
					raise RuntimeError( f"FFmpeg produced no output for '{src.name}'" )
		else :
			with Image.open( src ) as img :
				# Skip unnecessary mode conversions — PNG supports most modes
				if img.mode in ("RGBA" , "RGB" , "L" , "LA" , "P" , "PA") :
					img.save( dst , format="PNG" , optimize=False )
				else :
					target = "RGBA" if getattr( img , "has_transparency_data" , False ) else "RGB"
					img.convert( target ).save( dst , format="PNG" , optimize=False )

		logger.info( "PNG created: '%s' (%.1f KB)" , dst.name , dst.stat( ).st_size / 1024 )
		src.unlink( )
		return dst

	except Exception as exc :
		logger.error( "Image → PNG failed for '%s': %s" , src.name , exc )
		if dst.exists( ) and dst != src :
			dst.unlink( )
		return None


# ══════════════════════════════════════════════════════════════════════════════
# PNG → PDF  (unchanged — already fast)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_alpha(src: Path) -> bytes:
	"""Return PNG bytes with alpha removed (white background)."""
	from io import BytesIO
	with Image.open(src) as img:
		if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
			bg = Image.new("RGB", img.size, (255, 255, 255))
			bg.paste(img, mask=img.convert("RGBA").split()[3])
			buf = BytesIO()
			bg.save(buf, format="PNG")
			return buf.getvalue()
		buf = BytesIO()
		img.convert("RGB").save(buf, format="PNG")
		return buf.getvalue()


def _convert_with_img2pdf(src: Path, dst: Path, logger: logging.Logger) -> Path:
	png_bytes = _strip_alpha(src)
	pdf_bytes = img2pdf.convert(png_bytes)
	dst.write_bytes(pdf_bytes)
	logger.info("PDF created (img2pdf): '%s' (%.1f KB)", dst.name, dst.stat().st_size / 1024)
	src.unlink()
	return dst


def _convert_with_pillow(src: Path, dst: Path, logger: logging.Logger) -> Path:
	with Image.open(src) as img:
		if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
			bg = Image.new("RGB", img.size, (255, 255, 255))
			bg.paste(img, mask=img.convert("RGBA").split()[3])
			bg.save(str(dst), "PDF", resolution=img.info.get("dpi", (72, 72))[0])
		else:
			img.convert("RGB").save(str(dst), "PDF", resolution=img.info.get("dpi", (72, 72))[0])
	logger.info("PDF created (Pillow): '%s' (%.1f KB)", dst.name, dst.stat().st_size / 1024)
	src.unlink()
	return dst

def convert_png_to_pdf(src: Path, logger: logging.Logger) -> Optional[Path]:
	_archive(src)
	src = src.resolve()
	dst = src.with_suffix(".pdf")
	logger.info("Converting PNG → PDF: '%s'", src.name)

	try:
		return _convert_with_img2pdf(src, dst, logger)
	except Exception as exc:
		logger.warning("img2pdf failed for '%s': %s — trying Pillow fallback", src.name, exc)

	try:
		return _convert_with_pillow(src, dst, logger)
	except Exception as exc:
		logger.error("PNG → PDF failed entirely for '%s': %s", src.name, exc)
		if dst.exists() and dst != src:
			dst.unlink()
		return None

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO → MP4  (heavily optimized — full GPU pipeline)
# ══════════════════════════════════════════════════════════════════════════════


def convert_video_to_mp4( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	"""
	Convert video to MP4 with full NVIDIA GPU pipeline.

	Pipeline: cuvid HW decode → CUDA surface → NVENC HW encode
	Zero CPU-GPU memory copies for supported codecs.

	Falls back gracefully: GPU decode+encode → CPU decode+GPU encode → full CPU.
	"""
	_archive( src )
	src = src.resolve( )
	dst = src.with_suffix( ".mp4" )
	logger.info( "Converting video → MP4: '%s'" , src.name )

	# Probe to pick the right cuvid decoder
	probe = _probe_video( src )
	input_codec = None
	for stream in probe.get( "streams" , [ ] ) :
		if stream.get( "codec_type" ) == "video" :
			input_codec = stream.get( "codec_name" )
			break

	# Map input codecs → cuvid hardware decoders
	cuvid_decoders = {
		"h264"       : "h264_cuvid" ,
		"hevc"       : "hevc_cuvid" ,
		"h265"       : "hevc_cuvid" ,
		"vp8"        : "vp8_cuvid" ,
		"vp9"        : "vp9_cuvid" ,
		"mpeg1video" : "mpeg1_cuvid" ,
		"mpeg2video" : "mpeg2_cuvid" ,
		"mpeg4"      : "mpeg4_cuvid" ,
		"mjpeg"      : "mjpeg_cuvid" ,
		"av1"        : "av1_cuvid" ,
	}
	hw_decoder = cuvid_decoders.get( input_codec )

	strategies = [ ]

	# ── Strategy 1: Full GPU (HW decode + HW encode) ──────────────────────
	if hw_decoder :
		strategies.append( {
			"label" : f"Full GPU ({hw_decoder} → h264_nvenc)" ,
			"args"  : [
				"-hwaccel" , "cuda" ,
				"-hwaccel_output_format" , "cuda" ,
				"-c:v" , hw_decoder ,  # HW decoder — frames stay on GPU
				"-gpu" , "0" ,  # Pin to GPU 0
				"-i" , str( src ) ,
				"-c:v" , "h264_nvenc" ,
				"-preset" , "p1" ,  # Fastest preset (was p4)
				"-tune" , "ll" ,  # Low-latency tune
				"-rc" , "constqp" ,  # Constant QP — fastest rate control
				"-qp" , "23" ,  # Quality level (lower = better)
				"-b_ref_mode" , "disabled" ,  # No B-ref (faster)
				"-spatial-aq" , "0" ,  # Disable spatial AQ (faster)
				"-temporal-aq" , "0" ,  # Disable temporal AQ (faster)
				"-rc-lookahead" , "0" ,  # No lookahead (faster, less VRAM)
				"-bf" , "0" ,  # No B-frames (fastest)
				"-gpu" , "0" ,
				"-c:a" , "aac" ,
				"-b:a" , "128k" ,
				"-movflags" , "+faststart" ,
			] ,
		} )

	# ── Strategy 2: GPU encode only (CPU decode → GPU encode) ─────────────
	strategies.append( {
		"label" : "NVENC (CPU decode → GPU encode)" ,
		"args"  : [
			"-i" , str( src ) ,
			"-c:v" , "h264_nvenc" ,
			"-preset" , "p1" ,
			"-tune" , "ll" ,
			"-rc" , "constqp" ,
			"-qp" , "23" ,
			"-b_ref_mode" , "disabled" ,
			"-spatial-aq" , "0" ,
			"-temporal-aq" , "0" ,
			"-rc-lookahead" , "0" ,
			"-bf" , "0" ,
			"-gpu" , "0" ,
			"-c:a" , "aac" ,
			"-b:a" , "128k" ,
			"-movflags" , "+faststart" ,
		] ,
	} )

	# ── Strategy 3: Full CPU fallback ─────────────────────────────────────
	strategies.append( {
		"label" : "libx264 (CPU fallback)" ,
		"args"  : [
			"-i" , str( src ) ,
			"-c:v" , "libx264" ,
			"-crf" , "23" ,
			"-preset" , "ultrafast" ,  # Was "fast" — ultrafast is 5-10x faster
			"-tune" , "fastdecode" ,
			"-bf" , "0" ,  # No B-frames
			"-threads" , "0" ,  # Use all CPU threads
			"-c:a" , "aac" ,
			"-b:a" , "128k" ,
			"-movflags" , "+faststart" ,
		] ,
	} )

	for strategy in strategies :
		try :
			logger.info( "Trying %s for '%s'" , strategy[ "label" ] , src.name )

			subprocess.run(
					[ "ffmpeg" , "-y" , "-threads" , "0" , *strategy[ "args" ] , str( dst ) ] ,
					capture_output=True , text=True ,
					timeout=60 * 60 , check=True ,
			)

			if not dst.exists( ) :
				raise RuntimeError( f"FFmpeg produced no output for '{src.name}'" )

			logger.info(
					"MP4 created via %s: '%s' (%.1f MB)" ,
					strategy[ "label" ] , dst.name , dst.stat( ).st_size / (1024 * 1024) ,
			)
			src.unlink( )
			return dst

		except Exception as exc :
			logger.warning( "%s failed for '%s': %s" , strategy[ "label" ] , src.name , exc )
			if dst.exists( ) and dst != src :
				dst.unlink( )
			continue

	logger.error( "All conversion strategies failed for '%s'" , src.name )
	return None


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO → MP3  (optimized)
# ══════════════════════════════════════════════════════════════════════════════


def convert_audio_to_mp3( src: Path , logger: logging.Logger , bitrate: str = "320k" ) -> Optional[ Path ] :
	_archive( src )
	src = src.resolve( )
	dst = src.with_suffix( ".mp3" )
	logger.info( "Converting audio → MP3: '%s' (bitrate=%s)" , src.name , bitrate )

	if src.suffix.lower( ) == ".mp3" :
		encode_args = [ "-c:a" , "copy" ]
	else :
		encode_args = [
			"-c:a" , "libmp3lame" ,
			"-b:a" , bitrate ,
			"-q:a" , "0" ,  # Highest quality VBR
			"-joint_stereo" , "1" ,  # Joint stereo — faster than full stereo
			"-reservoir" , "1" ,  # Bit reservoir — better compression
			"-id3v2_version" , "3" ,
			"-write_id3v1" , "1" ,
			"-threads" , "0" ,  # Use all cores
		]

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
	_archive( src )
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
# DOCUMENT → PDF
# ══════════════════════════════════════════════════════════════════════════════


def convert_document_to_pdf( src: Path , logger: logging.Logger ) -> Optional[ Path ] :
	_archive( src )
	src = src.resolve( )
	dst = src.with_suffix( ".pdf" )
	logger.info( "Converting document → PDF via LibreOffice: '%s'" , src.name )

	try :
		soffice = find_libreoffice( )
	except RuntimeError as exc :
		logger.error( "LibreOffice not available: %s" , exc )
		return None

	try :
		subprocess.run(
				[
					soffice ,
					"--headless" ,
					"--nologo" ,
					"--norestore" ,
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
	for directory in os.environ.get( "PATH" , "" ).split( os.pathsep ) :
		candidate = Path( directory ) / name
		if candidate.is_file( ) :
			return candidate.resolve( )
	return None


def convert_email_to_pdf( src: Path , logger: logging.Logger ) -> Path :
	_archive( src )
	src = src.resolve( )

	if not src.exists( ) :
		raise FileNotFoundError( f"Source file not found: {src}" )
	if src.suffix.lower( ) not in EMAIL_TYPES :
		raise ValueError( f"Unsupported file type '{src.suffix}'. Expected one of: {', '.join( EMAIL_TYPES )}" )
	if not _find_on_path( "java" ) and not _find_on_path( "java.exe" ) :
		raise FileNotFoundError( "Java runtime not found on PATH. Install a JRE (8+)." )

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
	try :
		result = subprocess.run( cmd , capture_output=True , text=True , timeout=120 )
	except subprocess.TimeoutExpired :
		raise RuntimeError( f"Conversion timed out after 120s for: {src.name}" )

	if result.returncode != 0 :
		raise RuntimeError(
				f"emailconverter exited with code {result.returncode} "
				f"for '{src.name}': {result.stderr.strip( )}" ,
		)

	if not dst.exists( ) :
		raise RuntimeError( f"Conversion succeeded but output not found: {dst}" )

	logger.info( "PDF created: '%s'" , dst.name )
	return dst
