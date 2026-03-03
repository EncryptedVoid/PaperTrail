"""
File Triage — Manual Review Tool  v5.2  (performance overhaul)
──────────────────────────────────────────────────────────────
Changes from v5.1:
  • All file I/O (moves, copies, deletes, conversions) offloaded to a
    ThreadPoolExecutor so the UI never freezes during heavy operations.
  • _release_media() replaces _stop_video() — also unloads pygame audio,
    fixing WinError 32 "file in use" on videos and audio files.
  • Video preview no longer auto-plays; shows a still frame + scrub bar.
    Playback capped at 15 fps to halve CPU usage.
  • PDF pages render incrementally (first 2 eager, rest deferred via after()).
  • Sidebar thumbnails generated asynchronously in background threads.
  • Significantly more logging and inline comments throughout.

Architecture (unchanged):
  • DestButton dataclass registry — each destination is a self-contained record
    holding its label, emoji, sort order, keybind, and action callable.
  • Context-aware conversion panel — only shows conversions the pipeline can
    actually perform for the current file type.
  • UNSUPPORTED_ARTIFACTS_DIR contents appended to the end of the queue.
  • Silent batch flush every 20 items (history sidebar resets, no popup).
  • Arrow-key destinations: Up → ALTERATIONS_REQUIRED, Down → UNESSENTIAL.
  • Rotation in-place for PDF / PNG / MP4 via toolbar buttons.
  • Bold-Unicode keybind letters rendered in destination button labels.
  • No "OTHER" button — unsupported files go to UNSUPPORTED_ARTIFACTS_DIR on failure.

Keyboard map:
  Enter   → Confirm          Left    → Previous file
  Up      → Flag (dialog)    Down    → Unessential
  Delete  → Delete           Letters → Toggle destination (see DEST_REGISTRY)
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import time
import tkinter as tk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass , field
from datetime import datetime
from pathlib import Path
from tkinter import messagebox , simpledialog
from typing import Callable
from zipfile import ZipFile

import customtkinter as ctk
from customtkinter import CTkScrollableFrame

from config import (
	AFFINE_DIR , ALTERATIONS_REQUIRED_DIR , ANKI_DIR , ARCHIVAL_DIR ,
	AUDIO_TYPES , BASE_DIR , BITWARDEN_DIR , CALIBRE_LIBRARY_DIR ,
	DIGITAL_ASSET_MANAGEMENT_DIR , DOCUMENT_TYPES , EMAIL_TYPES ,
	FIREFLYIII_DIR , GAMES_ARCHIVE_DIR , GITLAB_DIR , IMAGE_TYPES ,
	IMMICH_DIR , JAVA_PATH , JELLYFIN_DIR ,
	LINKWARDEN_DIR , MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR ,
	ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR ,
	ODOO_PLM_DIR , ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR ,
	PERSONAL_ARCHIVE_DIR , SCANNING_REQUIRED_DIR , SOFTWARE_ARCHIVE_DIR ,
	TIKA_APP_JAR_PATH , ULTIMAKER_CURA_DIR ,
	UNESSENTIAL_DIR , UNSUPPORTED_ARTIFACTS_DIR , VIDEO_TYPES ,
)
from utilities.format_converting import (
	convert_audio_to_mp3 , convert_document_to_pdf , convert_email_to_pdf ,
	convert_html_to_pdf , convert_image_to_png , convert_png_to_pdf ,
	convert_video_to_mp4 ,
)

# ── Appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode( "Dark" )
ctk.set_default_color_theme( "blue" )

from PIL import Image , ImageTk
import fitz  # PyMuPDF — PDF rendering
import cv2  # OpenCV  — video frame capture
import pygame  # pygame  — audio playback

pygame.mixer.init( )

from tkinterweb import HtmlFrame

# ── Fonts ────────────────────────────────────────────────────────────────────
# Try to use Segoe UI (Windows default), fall back gracefully.
FONT = "Segoe UI"
EMOJI_FONT = "Segoe UI Emoji"
try :
	import tkinter.font as _tkf

	_r = tk.Tk( )
	_avail = _tkf.families( )
	_r.destroy( )
	FONT = FONT if FONT in _avail else ("Arial" if "Arial" in _avail else "TkDefaultFont")
	if EMOJI_FONT not in _avail :
		EMOJI_FONT = FONT
except Exception :
	FONT = EMOJI_FONT = "Arial"

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG = "#141414"  # main background
SIDEBAR = "#1C1C1C"  # sidebar / panel backgrounds
PREVIEW = "#181818"  # preview area background
CARD = "#222222"  # default button / card background
CARD_SEL = "#2A2A2A"  # hovered / selected card
FG = "#EFEFEF"  # primary text
FG2 = "#909090"  # secondary / muted text
ACCENT = "#E05C52"  # primary accent (red-ish)
ACCENT_L = "#F07068"  # lighter accent
ACCENT_D = "#B84840"  # darker accent
GREEN = "#5DBD72"  # success / confirm
BLUE = "#4A8ED9"  # informational / play
YELLOW = "#D9A740"  # warning / in-progress
RED = "#CC2222"  # destructive actions

# ── Special directories ──────────────────────────────────────────────────────
DELETE_DIR: Path = BASE_DIR / "DELETE"
ALTERATIONS_CSV: Path = ALTERATIONS_REQUIRED_DIR / "alterations_log.csv"

# ══════════════════════════════════════════════════════════════════════════════
# BOLD-UNICODE HELPER  (Mathematical Sans-Serif Bold A-Z / a-z)
# Used to visually highlight the keybind letter inside button labels.
# ══════════════════════════════════════════════════════════════════════════════
_BOLD = {
	**{ chr( ord( "A" ) + i ) : chr( 0x1D5D4 + i ) for i in range( 26 ) } ,
	**{ chr( ord( "a" ) + i ) : chr( 0x1D5EE + i ) for i in range( 26 ) } ,
}


def _bold_key_in_label( label: str , key: str ) -> str :
	"""Replace the first occurrence of *key* in *label* with its bold-unicode
	counterpart so the user can visually identify the keybind at a glance."""
	ku = key.upper( )
	for i , ch in enumerate( label ) :
		if ch.upper( ) == ku :
			return label[ :i ] + _BOLD.get( ch , ch ) + label[ i + 1 : ]
	# Key letter not found in label — prepend it in brackets
	return f"[{_BOLD.get( key.upper( ) , key.upper( ) )}] {label}"


# ══════════════════════════════════════════════════════════════════════════════
# DESTINATION BUTTON REGISTRY
# Each DestButton is one triage target the user can send files to.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DestButton :
	"""Self-contained record for one triage-destination button."""
	name: str  # internal identifier
	label: str  # display text (with emoji)
	order: int  # sort order in the grid
	key: str  # keyboard shortcut character
	targets: list[ Path ] = field( default_factory=list )  # directories to copy/move into

	def display_text( self ) -> str :
		"""Label with the keybind letter bolded for visual identification."""
		return _bold_key_in_label( self.label , self.key )


def _build_dest_registry( ) -> list[ DestButton ] :
	"""Construct the ordered list of destination buttons."""
	return sorted( [
		DestButton( "AFFINE" , "🧠 AFFINE" , 10 , "a" , [ AFFINE_DIR ] ) ,
		DestButton( "ANKI" , "🃏 ANKI" , 20 , "n" , [ ANKI_DIR ] ) ,
		DestButton( "ARCHIVE" , "📦 ARCHIVE" , 30 , "r" , [ ARCHIVAL_DIR ] ) ,
		DestButton( "BITWARDEN" , "🔐 BITWARDEN" , 40 , "b" , [ BITWARDEN_DIR ] ) ,
		DestButton( "CALIBRE WEB" , "📚 CALIBRE WEB" , 50 , "c" , [ CALIBRE_LIBRARY_DIR ] ) ,
		DestButton( "CRM" , "👥 CRM" , 60 , "m" , [ MONICA_CRM_DIR , ODOO_CRM_DIR ] ) ,
		DestButton( "FIREFLY" , "💰 FIREFLY" , 70 , "f" , [ FIREFLYIII_DIR ] ) ,
		DestButton( "GAMES ARCHIVE" , "🎮 GAMES ARCHIVE" , 75 , "j" , [ GAMES_ARCHIVE_DIR ] ) ,
		DestButton( "GITLAB" , "🦊 GITLAB" , 80 , "g" , [ GITLAB_DIR ] ) ,
		DestButton( "IMMICH" , "📷 IMMICH" , 90 , "i" , [ IMMICH_DIR ] ) ,
		DestButton( "INTERNET ARCHIVE" , "🌍 INTERNET ARCHIVE" , 100 , "t" , [ PERSONAL_ARCHIVE_DIR ] ) ,
		DestButton( "LINKWARDEN" , "🔖 LINKWARDEN" , 120 , "w" , [ JELLYFIN_DIR ] ) ,
		DestButton( "JELLYFIN" , " JELLYFIN" , 125 , "w" , [ LINKWARDEN_DIR ] ) ,
		DestButton( "MANUALS ARCHIVE" , "📖 MANUALS ARCHIVE" , 130 , "h" , [ MANUALS_ARCHIVE_DIR ] ) ,
		DestButton( "ODOO INVENTORY" , "📦 ODOO INVENTORY" , 135 , "v" , [ ODOO_INVENTORY_DIR ] ) ,
		DestButton( "ODOO MAINTENANCE" , "🔧 ODOO MAINTENANCE" , 140 , "e" , [ ODOO_MAINTENANCE_DIR ] ) ,
		DestButton( "ODOO PLM" , "⚙ ODOO PLM" , 150 , "p" , [ ODOO_PLM_DIR ] ) ,
		DestButton( "ODOO PURCHASE" , "🛒 ODOO PURCHASE" , 160 , "u" , [ ODOO_PURCHASE_DIR ] ) ,
		DestButton( "PORTFOLIO PERF." , "🎭 PORTFOLIO PERF." , 170 , "x" , [ PERFORMANCE_PORTFOLIO_DIR ] ) ,
		DestButton( "SCANNING" , "🔍 SCANNING" , 175 , "z" , [ SCANNING_REQUIRED_DIR ] ) ,
		DestButton( "SEAFILE" , "☁ SEAFILE" , 180 , "s" , [ DIGITAL_ASSET_MANAGEMENT_DIR ] ) ,
		DestButton( "SOFTWARE ARCHIVE" , "💾 SOFTWARE ARCHIVE" , 185 , "d" , [ SOFTWARE_ARCHIVE_DIR ] ) ,
		DestButton( "ULTIMAKER CURA" , "🖨 ULTIMAKER CURA" , 190 , "q" , [ ULTIMAKER_CURA_DIR ] ) ,
	] , key=lambda b : b.order )


# Pre-built registries for fast lookup
DEST_REGISTRY: list[ DestButton ] = _build_dest_registry( )
DEST_BY_NAME: dict[ str , DestButton ] = { b.name : b for b in DEST_REGISTRY }
DEST_BY_KEY: dict[ str , DestButton ] = { b.key : b for b in DEST_REGISTRY }


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE CONVERSION OPTIONS
# Only conversions that make sense for the current file type are shown.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass( frozen=True )
class ConvOption :
	"""One conversion the user can choose for a given file."""
	label: str  # button label with emoji
	key: str  # internal identifier
	fn: str  # name of the converter function (for reference only)


# Master list of every supported conversion
_ALL_CONV = [
	ConvOption( "🖼→PNG" , "img_png" , "convert_image_to_png" ) ,
	ConvOption( "🖼→PDF" , "img_pdf" , "convert_png_to_pdf" ) ,
	ConvOption( "🎬→MP4" , "vid_mp4" , "convert_video_to_mp4" ) ,
	ConvOption( "🎵→MP3" , "aud_mp3" , "convert_audio_to_mp3" ) ,
	ConvOption( "📄→PDF" , "doc_pdf" , "convert_document_to_pdf" ) ,
	ConvOption( "📊→CSV" , "xls_csv" , "convert_xlsx_to_csv" ) ,
	ConvOption( "📧→PDF" , "eml_pdf" , "convert_email_to_pdf" ) ,
	ConvOption( "🌐→PDF" , "html_pdf" , "convert_html_to_pdf" ) ,
]

# Extension sets used for file-type detection
_IMG_EXTS = { e.lower( ) for e in IMAGE_TYPES }
_VID_EXTS = { e.lower( ) for e in VIDEO_TYPES }
_AUD_EXTS = { e.lower( ) for e in AUDIO_TYPES }
_DOC_EXTS = { e.lower( ) for e in DOCUMENT_TYPES }
_EML_EXTS = { e.lower( ) for e in EMAIL_TYPES }
_HTML_EXTS = { "html" , "htm" , "xhtml" }


def _conv_options_for( file: Path ) -> list[ ConvOption ] :
	"""Return only the conversions that are operable for *file*'s type."""
	ext = file.suffix.lower( ).lstrip( "." )
	if ext in _HTML_EXTS : return [ c for c in _ALL_CONV if c.key == "html_pdf" ]
	if ext in _IMG_EXTS :  return [ c for c in _ALL_CONV if c.key in ("img_png" , "img_pdf") ]
	if ext in _VID_EXTS :  return [ c for c in _ALL_CONV if c.key == "vid_mp4" ]
	if ext in _AUD_EXTS :  return [ c for c in _ALL_CONV if c.key == "aud_mp3" ]
	if ext in _DOC_EXTS :  return [ c for c in _ALL_CONV if c.key == "doc_pdf" ]
	if ext in _EML_EXTS :  return [ c for c in _ALL_CONV if c.key == "eml_pdf" ]
	return [ ]


# ══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _unique( directory: Path , file: Path ) -> Path :
	"""Return a non-colliding destination path inside *directory*.
	Appends _1, _2, … to the stem if the filename already exists."""
	dst = directory / file.name
	n = 1
	while dst.exists( ) :
		dst = directory / f"{file.stem}_{n}{file.suffix}"
		n += 1
	return dst


def _hover( hex_col: str ) -> str :
	"""Lighten a hex colour by ~18 % for hover state."""
	h = hex_col.lstrip( "#" )
	r , g , b = int( h[ 0 :2 ] , 16 ) , int( h[ 2 :4 ] , 16 ) , int( h[ 4 :6 ] , 16 )
	return "#{:02x}{:02x}{:02x}".format(
			min( 255 , int( r * 1.18 ) ) ,
			min( 255 , int( g * 1.18 ) ) ,
			min( 255 , int( b * 1.18 ) ) ,
	)


def _tika_metadata( file: Path ) -> str :
	"""Extract metadata via Apache Tika (best-effort).
	Returns a string of key-value pairs, or an error message."""
	try :
		r = subprocess.run(
				[ str( JAVA_PATH ) , "-jar" , str( TIKA_APP_JAR_PATH ) , "-m" , str( file ) ] ,
				capture_output=True , text=True , timeout=30 ,
		)
		return r.stdout.strip( ) or "No metadata returned by Tika."
	except FileNotFoundError :
		return "Java or Tika JAR not found."
	except subprocess.TimeoutExpired :
		return "Tika extraction timed out."
	except Exception as e :
		return f"Tika error: {e}"


def _write_alteration_csv( filename: str , description: str , logger: logging.Logger ) -> None :
	"""Append a row to the alterations CSV log.
	Creates the header row on first write."""
	is_new = not ALTERATIONS_CSV.exists( )
	with open( ALTERATIONS_CSV , "a" , newline="" , encoding="utf-8" ) as fh :
		w = csv.writer( fh )
		if is_new :
			w.writerow( [ "timestamp" , "filename" , "description" ] )
		w.writerow( [ datetime.now( ).isoformat( ) , filename , description ] )
	logger.info( "Alteration logged for '%s': %s" , filename , description[ :80 ] )


def _file_category( file: Path ) -> str :
	"""Classify a file into a preview category string for the dispatcher."""
	ext = file.suffix.lower( ).lstrip( "." )
	if ext == "pdf" :      return "pdf"
	if ext in _IMG_EXTS :  return "image"
	if ext in _VID_EXTS :  return "video"
	if ext in _AUD_EXTS :  return "audio"
	if ext in _HTML_EXTS : return "html"
	if ext in {
		"txt" , "log" , "json" , "csv" , "xml" , "md" , "yaml" , "yml" ,
		"toml" , "ini" , "py" , "js" , "ts" , "css" , "sh" , "bat" ,
	} :
		return "text"
	return "other"


# ══════════════════════════════════════════════════════════════════════════════
# ROTATION HELPERS (in-place modification)
# ══════════════════════════════════════════════════════════════════════════════


def _rotate_pdf( file: Path , degrees: int , logger: logging.Logger ) -> None :
	"""Rotate every page of a PDF by *degrees* (90/180/270) and save in-place."""
	logger.info( "Rotating PDF '%s' by %d°…" , file.name , degrees )
	doc = fitz.open( file )
	for page in doc :
		page.set_rotation( (page.rotation + degrees) % 360 )
	tmp = file.parent / f"_rot_{file.name}"
	doc.save( tmp )
	doc.close( )
	file.unlink( )
	tmp.rename( file )
	logger.info( "PDF rotation complete: '%s' rotated %d°" , file.name , degrees )


def _rotate_image( file: Path , degrees: int , logger: logging.Logger ) -> None :
	"""Rotate a PNG image by *degrees* (CW) and save in-place."""
	logger.info( "Rotating image '%s' by %d°…" , file.name , degrees )
	img = Image.open( file )
	# PIL rotates counter-clockwise by default, so negate for CW
	rotated = img.rotate( -degrees , expand=True )
	rotated.save( file , format="PNG" )
	logger.info( "Image rotation complete: '%s' rotated %d°" , file.name , degrees )


def _rotate_video( file: Path , degrees: int , logger: logging.Logger ) -> None :
	"""Rotate an MP4 video by *degrees* (CW) in-place via ffmpeg."""
	logger.info( "Rotating video '%s' by %d° via ffmpeg…" , file.name , degrees )
	vf = {
		90  : "transpose=1" ,
		180 : "transpose=1,transpose=1" ,
		270 : "transpose=2" ,
	}.get( degrees )
	if not vf :
		raise ValueError( f"Unsupported rotation angle: {degrees}" )
	tmp = file.parent / f"_rot_{file.name}"
	try :
		subprocess.run(
				[ "ffmpeg" , "-y" , "-i" , str( file ) , "-vf" , vf ,
					"-c:a" , "copy" , "-metadata:s:v" , "rotate=0" , str( tmp ) ] ,
				capture_output=True , timeout=300 , check=True ,
		)
		file.unlink( )
		tmp.rename( file )
		logger.info( "Video rotation complete: '%s' rotated %d°" , file.name , degrees )
	except FileNotFoundError :
		raise RuntimeError( "ffmpeg not found on PATH" )
	except subprocess.CalledProcessError as e :
		if tmp.exists( ) :
			tmp.unlink( )
		raise RuntimeError( f"ffmpeg error: {(e.stderr or b'')[ :300 ]}" )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ══════════════════════════════════════════════════════════════════════════════


class FileTriage :
	BATCH_SIZE = 20  # flush pending operations every N confirms
	MAX_PDF_PG = 10  # max pages to render in preview

	# ──────────────────────────────────────────────────────────────────────────
	# INITIALISATION
	# ──────────────────────────────────────────────────────────────────────────
	def __init__( self , source_dir: Path , logger: logging.Logger ) :
		self.source_dir = Path( source_dir )
		self.logger = logger
		self.logger.info( "Initialising FileTriage v5.2 with source: %s" , self.source_dir )

		# ── file queue & history ──────────────────────────────────────────────
		# queue:   files waiting to be reviewed (FIFO)
		# history: list of [Path, [dest_names], conv_key|None] for reviewed files
		self.queue: deque[ Path ] = deque( )
		self.history: list[ list ] = [ ]
		self.current_index: int = -1
		self.current_file: Path | None = None

		# ── UI state ──────────────────────────────────────────────────────────
		self.selected_dests: list[ str ] = [ ]  # currently toggled destinations
		self.dest_widgets: dict[ str , ctk.CTkButton ] = { }
		self._bound_keys: list[ str ] = [ ]  # keybinds to unbind on refresh

		self.selected_conversion: str | None = None  # currently selected conversion key
		self.conv_widgets: dict[ str , ctk.CTkButton ] = { }

		# ── batch / flag tracking ─────────────────────────────────────────────
		self.pending: list[ tuple[ Path , list[ str ] ] ] = [ ]  # (file, dest_names) awaiting flush
		self.batch_n: int = 0
		self.flagged: set[ Path ] = set( )

		# ── media playback state ──────────────────────────────────────────────
		self.video_cap = None  # cv2.VideoCapture handle
		self.video_after_id = None  # tkinter after() id for frame loop
		self.video_lbl = None  # tk.Label displaying video frames
		self.is_playing = False
		self.zoom_scale = 1.0
		self._vid_size = (640 , 360)  # pre-calculated preview dimensions
		self._vid_delay = 66  # ms between frames (~15 fps)

		# ── threading ─────────────────────────────────────────────────────────
		# 2 workers: one for I/O (moves/copies), one for thumbnails/conversions
		self._executor = ThreadPoolExecutor( max_workers=2 )
		self.logger.info( "ThreadPoolExecutor started with 2 workers" )

		# ── performance tracking ──────────────────────────────────────────────
		self._thumb_cache: dict[ str , ImageTk.PhotoImage | None ] = { }
		self.session_start = time.time( )
		self.file_start: float | None = None
		self.file_times: list[ float ] = [ ]

		# ── ensure every target directory exists ──────────────────────────────
		for btn in DEST_REGISTRY :
			for p in btn.targets :
				p.mkdir( parents=True , exist_ok=True )
		for d in (ALTERATIONS_REQUIRED_DIR , UNESSENTIAL_DIR , DELETE_DIR ,
							UNSUPPORTED_ARTIFACTS_DIR) :
			d.mkdir( parents=True , exist_ok=True )
		self.logger.info( "All target directories verified / created" )

		# ── load file queue ───────────────────────────────────────────────────
		self._load_files( )

		# ── build the main window ─────────────────────────────────────────────
		self.root = ctk.CTk( )
		self.root.title( "File Triage  v5.2" )
		self.root.geometry( "1920x1060" )
		self.root.configure( fg_color=BG )

		self._build_ui( )
		self._bind_global_keys( )
		self._tick_timer( )

		# Show the first file if the queue is not empty
		if self.queue :
			self.show_next( )

		self.logger.info( "Application ready — %d files in queue" , len( self.queue ) )

	# ──────────────────────────────────────────────────────────────────────────
	# FILE LOADING
	# ──────────────────────────────────────────────────────────────────────────
	def _load_files( self ) :
		"""Populate the queue from source_dir, then append unsupported dir."""
		primary = [
			f for f in self.source_dir.iterdir( )
			if f.is_file( ) and f.parent == self.source_dir
		]
		self.queue.extend( primary )
		self.logger.info( "Loaded %d file(s) from source directory" , len( primary ) )

		# Append any leftover files from the unsupported-artifacts directory
		if UNSUPPORTED_ARTIFACTS_DIR.exists( ) and UNSUPPORTED_ARTIFACTS_DIR != self.source_dir :
			extras = [
				f for f in UNSUPPORTED_ARTIFACTS_DIR.iterdir( )
				if f.is_file( ) and f.parent == UNSUPPORTED_ARTIFACTS_DIR
			]
			if extras :
				self.queue.extend( extras )
				self.logger.info( "Appended %d file(s) from UNSUPPORTED_ARTIFACTS" , len( extras ) )

	# ══════════════════════════════════════════════════════════════════════════
	# UI CONSTRUCTION
	# ══════════════════════════════════════════════════════════════════════════
	def _build_ui( self ) :
		self.logger.debug( "Building UI layout…" )
		main = ctk.CTkFrame( self.root , fg_color=BG )
		main.pack( fill=tk.BOTH , expand=True , padx=10 , pady=10 )

		# ── LEFT: history sidebar ────────────────────────────────────────────
		left = ctk.CTkFrame( main , width=290 , fg_color=SIDEBAR , corner_radius=14 )
		left.pack( side=tk.LEFT , fill=tk.Y , padx=(0 , 8) )
		left.pack_propagate( False )

		ctk.CTkLabel(
				left , text="HISTORY" ,
				font=(FONT , 12 , "bold") , text_color=ACCENT ,
		).pack( pady=(14 , 6) )

		self.hist_frame = CTkScrollableFrame( left , fg_color=SIDEBAR )
		self.hist_frame.pack( fill=tk.BOTH , expand=True , padx=6 , pady=(0 , 8) )

		# ── CENTER column ────────────────────────────────────────────────────
		center = ctk.CTkFrame( main , fg_color=BG )
		center.pack( side=tk.LEFT , fill=tk.BOTH , expand=True )

		# ---- info bar (filename, action, queue count) ----
		info = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 , height=46 )
		info.pack( fill=tk.X , pady=(0 , 5) )
		info.pack_propagate( False )

		self.file_lbl = ctk.CTkLabel(
				info , text="—" , font=(FONT , 13 , "bold") , text_color=FG )
		self.file_lbl.pack( side=tk.LEFT , padx=14 )

		self.action_lbl = ctk.CTkLabel(
				info , text="" , font=(FONT , 11 , "bold") , text_color=ACCENT )
		self.action_lbl.pack( side=tk.LEFT , padx=8 )

		self.queue_lbl = ctk.CTkLabel(
				info , text="Queue: 0" , font=(FONT , 11) , text_color=FG2 )
		self.queue_lbl.pack( side=tk.RIGHT , padx=14 )

		# ---- preview outer frame ----
		prev_outer = ctk.CTkFrame( center , fg_color=PREVIEW , corner_radius=14 )
		prev_outer.pack( fill=tk.BOTH , expand=True , pady=(0 , 5) )

		# ---- toolbar (rotate, open external, delete, flag, unessential) ----
		tb = ctk.CTkFrame( prev_outer , fg_color=PREVIEW , height=44 )
		tb.pack( fill=tk.X , padx=10 , pady=(8 , 0) )
		tb.pack_propagate( False )

		for txt , deg in [ ("↺ 90°" , 270) , ("180°" , 180) , ("↻ 90°" , 90) ] :
			ctk.CTkButton(
					tb , text=txt , command=lambda d=deg : self._rotate_current( d ) ,
					fg_color=CARD , hover_color=CARD_SEL ,
					width=70 , height=34 , corner_radius=8 ,
					text_color=FG , font=(FONT , 10 , "bold") ,
			).pack( side=tk.LEFT , padx=3 )

		ctk.CTkButton(
				tb , text="🔗 Open External" , command=self._open_external ,
				fg_color=CARD , hover_color=CARD_SEL ,
				width=135 , height=34 , corner_radius=8 ,
				text_color=FG , font=(EMOJI_FONT , 10) ,
		).pack( side=tk.LEFT , padx=3 )

		self.zoom_lbl = ctk.CTkLabel(
				tb , text="🔍 100%" , font=(EMOJI_FONT , 10) , text_color=FG2 )
		self.zoom_lbl.pack( side=tk.LEFT , padx=8 )

		ctk.CTkButton(
				tb , text="🗑  DELETE  [Del]" , command=self._delete_file ,
				fg_color=RED , hover_color=_hover( RED ) ,
				width=120 , height=34 , corner_radius=8 ,
				text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
		).pack( side=tk.RIGHT , padx=3 )

		ctk.CTkButton(
				tb , text="🚩  FLAG  [↑]" , command=self._flag_with_dialog ,
				fg_color=ACCENT_D , hover_color=ACCENT ,
				width=110 , height=34 , corner_radius=8 ,
				text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
		).pack( side=tk.RIGHT , padx=3 )

		ctk.CTkButton(
				tb , text="↓  UNESSENTIAL  [↓]" , command=self._send_to_unessential ,
				fg_color=CARD , hover_color=CARD_SEL ,
				width=150 , height=34 , corner_radius=8 ,
				text_color=FG2 , font=(EMOJI_FONT , 10 , "bold") ,
		).pack( side=tk.RIGHT , padx=3 )

		# ---- scrollable preview area (images, PDFs, video, etc.) ----
		self.prev_scroll = CTkScrollableFrame( prev_outer , fg_color=PREVIEW )
		self.prev_scroll.pack( fill=tk.BOTH , expand=True , padx=6 , pady=6 )
		self.prev_scroll.bind( "<MouseWheel>" , self._zoom )
		self.prev_scroll._parent_canvas.bind( "<MouseWheel>" , self._zoom )

		# ---- conversion panel (optional format conversion) ----
		conv_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		conv_panel.pack( fill=tk.X , pady=(0 , 5) )

		conv_hdr = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		conv_hdr.pack( fill=tk.X , padx=14 , pady=(8 , 4) )

		ctk.CTkLabel(
				conv_hdr ,
				text="CONVERT  (optional · single select · original archived first)" ,
				font=(FONT , 10 , "bold") , text_color=FG2 ,
		).pack( side=tk.LEFT )

		ctk.CTkButton(
				conv_hdr , text="✕ clear" , command=self._clear_conversion ,
				fg_color=CARD , hover_color=CARD_SEL ,
				width=70 , height=24 , corner_radius=6 ,
				text_color=FG2 , font=(FONT , 9) ,
		).pack( side=tk.LEFT , padx=10 )

		self.conv_row = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		self.conv_row.pack( fill=tk.X , padx=10 , pady=(0 , 10) )

		# Placeholder label when no conversions are available
		self.no_conv_lbl = ctk.CTkLabel(
				self.conv_row , text="No conversions available for this file type" ,
				font=(FONT , 10) , text_color=FG2 ,
		)

		# ---- destination grid (multi-select triage targets) ----
		dest_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		dest_panel.pack( fill=tk.X , pady=(0 , 5) )

		ctk.CTkLabel(
				dest_panel ,
				text="TRIAGE DESTINATIONS  (click or key · multi-select)" ,
				font=(FONT , 10 , "bold") , text_color=FG2 ,
		).pack( anchor="w" , padx=14 , pady=(8 , 4) )

		self.dest_grid = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		self.dest_grid.pack( fill=tk.X , padx=10 , pady=(0 , 10) )

		# ---- footer (timer, status, nav buttons) ----
		foot = ctk.CTkFrame( center , fg_color=BG , height=52 )
		foot.pack( fill=tk.X )
		foot.pack_propagate( False )

		self.timer_lbl = ctk.CTkLabel(
				foot , text="⏱ 0s" , font=(FONT , 11 , "bold") , text_color=YELLOW )
		self.timer_lbl.pack( side=tk.LEFT , padx=10 )

		self.status_lbl = ctk.CTkLabel(
				foot , text="" , font=(FONT , 11) , text_color=ACCENT )
		self.status_lbl.pack( side=tk.LEFT , padx=10 )

		nav = ctk.CTkFrame( foot , fg_color=BG )
		nav.pack( side=tk.RIGHT , padx=8 )

		for txt , cmd , col , w in [
			("← Prev" , self.show_previous , CARD , 88) ,
			("✓ CONFIRM" , self._confirm , GREEN , 120) ,
		] :
			ctk.CTkButton(
					nav , text=txt , command=cmd ,
					fg_color=col , hover_color=_hover( col ) ,
					width=w , height=42 , corner_radius=10 , text_color=FG ,
					font=(FONT , 11 , "bold" if "CONFIRM" in txt else "normal") ,
			).pack( side=tk.LEFT , padx=3 )

		self.logger.debug( "UI layout complete" )

	# ── global key bindings ──────────────────────────────────────────────────
	def _bind_global_keys( self ) :
		"""Bind arrow keys, Enter, and Delete to their respective actions."""
		self.root.bind( "<Left>" , lambda _ : self.show_previous( ) )
		self.root.bind( "<Return>" , lambda _ : self._confirm( ) )
		self.root.bind( "<Delete>" , lambda _ : self._delete_file( ) )
		self.root.bind( "<Up>" , lambda _ : self._flag_with_dialog( ) )
		self.root.bind( "<Down>" , lambda _ : self._send_to_unessential( ) )
		self.logger.debug( "Global key bindings registered" )

	# ══════════════════════════════════════════════════════════════════════════
	# MEDIA RELEASE  (fixes WinError 32)
	# Must be called BEFORE any operation that moves / deletes / converts
	# the current file, because OpenCV and pygame hold OS-level file locks.
	# ══════════════════════════════════════════════════════════════════════════
	def _release_media( self ) :
		"""Release ALL media handles on the current file (video + audio).
		Safe to call even if nothing is playing."""
		# ── video (OpenCV) ────────────────────────────────────────────────────
		self.is_playing = False
		if self.video_after_id :
			self.root.after_cancel( self.video_after_id )
			self.video_after_id = None
		if self.video_cap :
			try :
				self.video_cap.release( )
				self.logger.debug( "Released OpenCV video capture handle" )
			except Exception :
				pass
			self.video_cap = None

		# ── audio (pygame) ────────────────────────────────────────────────────
		# .stop() alone does NOT release the file handle — .unload() is required
		try :
			pygame.mixer.music.stop( )
			pygame.mixer.music.unload( )
			self.logger.debug( "Unloaded pygame audio handle" )
		except Exception :
			pass

	# ══════════════════════════════════════════════════════════════════════════
	# ZOOM (mouse wheel in preview area)
	# ══════════════════════════════════════════════════════════════════════════
	def _zoom( self , event ) :
		old = self.zoom_scale
		self.zoom_scale = max( 0.2 , min( 4.0 ,
																			self.zoom_scale + (0.1 if event.delta > 0 else -0.1) ) )
		self.zoom_lbl.configure( text=f"🔍 {int( self.zoom_scale * 100 )}%" )
		if int( old * 100 ) != int( self.zoom_scale * 100 ) :
			self.logger.debug( "Zoom changed: %d%%" , int( self.zoom_scale * 100 ) )
			self._display_file( )

	# ══════════════════════════════════════════════════════════════════════════
	# ROTATION
	# ══════════════════════════════════════════════════════════════════════════
	def _rotate_current( self , degrees: int ) :
		"""Rotate the current file in-place and refresh the preview."""
		# Release media handles first — file must not be locked
		self._release_media( )

		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.warning( "Rotate called but no valid current file" )
			return

		cat = _file_category( Path( self.current_file ) )
		self.logger.info( "Rotation requested: %d° for '%s' (category=%s)" ,
											degrees , Path( self.current_file ).name , cat )
		self.status_lbl.configure( text="Rotating…" , text_color=YELLOW )
		self.root.update_idletasks( )

		try :
			if cat == "pdf" :
				_rotate_pdf( Path( self.current_file ) , degrees , self.logger )
			elif cat == "image" :
				_rotate_image( Path( self.current_file ) , degrees , self.logger )
			elif cat == "video" :
				_rotate_video( Path( self.current_file ) , degrees , self.logger )
			else :
				self.status_lbl.configure( text="Rotation not supported for this type" , text_color=FG2 )
				self.logger.info( "Rotation not supported for category '%s'" , cat )
				return
			self.status_lbl.configure( text=f"✓ Rotated {degrees}°" , text_color=GREEN )
			# Invalidate cached thumbnail so it regenerates
			self._thumb_cache.pop( str( Path( self.current_file ) ) , None )
			self._display_file( )
		except Exception as e :
			self.status_lbl.configure( text="✗ Rotation failed" , text_color=RED )
			self.logger.error( "Rotation failed for '%s': %s" , Path( self.current_file ).name , e )
			messagebox.showerror( "Rotation Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# CONVERSION PANEL
	# ══════════════════════════════════════════════════════════════════════════
	def _rebuild_conv_buttons( self ) :
		"""Rebuild the conversion button row for the current file's type."""
		# Tear down existing buttons
		for w in self.conv_row.winfo_children( ) :
			w.destroy( )
		self.conv_widgets.clear( )
		self.no_conv_lbl.pack_forget( )

		if not self.current_file :
			return

		options = _conv_options_for( Path( self.current_file ) )
		self.logger.debug( "Conversion options for '%s': %s" ,
											 Path( self.current_file ).name ,
											 [ o.key for o in options ] if options else "none" )

		if not options :
			# Show "no conversions" placeholder
			self.no_conv_lbl = ctk.CTkLabel(
					self.conv_row , text="No conversions available for this file type" ,
					font=(FONT , 10) , text_color=FG2 ,
			)
			self.no_conv_lbl.pack( padx=10 , pady=4 )
			return

		for opt in options :
			btn = ctk.CTkButton(
					self.conv_row , text=opt.label ,
					command=lambda k=opt.key : self._toggle_conversion( k ) ,
					fg_color=CARD , hover_color=CARD_SEL ,
					text_color=FG2 , height=34 , corner_radius=8 ,
					font=(EMOJI_FONT , 10) ,
			)
			btn.pack( side=tk.LEFT , padx=5 , expand=True , fill=tk.X )
			self.conv_widgets[ opt.key ] = btn

		# Clear any previous selection, then restore saved selection if navigating back
		self._clear_conversion( )
		if 0 <= self.current_index < len( self.history ) :
			saved = self.history[ self.current_index ][ 2 ]
			if saved and saved in self.conv_widgets :
				self._toggle_conversion( saved )

	def _toggle_conversion( self , key: str ) :
		"""Toggle a conversion option on/off (single-select)."""
		if self.selected_conversion == key :
			# Deselect
			self._clear_conversion( )
			return
		# Deselect previous
		if self.selected_conversion and self.selected_conversion in self.conv_widgets :
			self.conv_widgets[ self.selected_conversion ].configure(
					fg_color=CARD , text_color=FG2 )
		# Select new
		self.selected_conversion = key
		if key in self.conv_widgets :
			self.conv_widgets[ key ].configure( fg_color=YELLOW , text_color=BG )
		self.logger.debug( "Conversion selected: %s" , key )
		self._refresh_action_label( )

	def _clear_conversion( self ) :
		"""Deselect any active conversion."""
		if self.selected_conversion and self.selected_conversion in self.conv_widgets :
			self.conv_widgets[ self.selected_conversion ].configure(
					fg_color=CARD , text_color=FG2 )
		self.selected_conversion = None
		self._refresh_action_label( )

	# ══════════════════════════════════════════════════════════════════════════
	# DESTINATION BUTTONS
	# ══════════════════════════════════════════════════════════════════════════
	def _render_dest_buttons( self ) :
		"""Rebuild the destination button grid and keybindings."""
		# Unbind previous per-file keybindings
		for k in self._bound_keys :
			try :
				self.root.unbind( k )
			except Exception :
				pass
		self._bound_keys.clear( )

		# Tear down old buttons
		for w in self.dest_grid.winfo_children( ) :
			w.destroy( )
		self.dest_widgets.clear( )
		self.selected_dests.clear( )

		if not self.current_file :
			self._rebuild_conv_buttons( )
			return

		cols = 5
		for c in range( cols ) :
			self.dest_grid.grid_columnconfigure( c , weight=1 )

		for i , db in enumerate( DEST_REGISTRY ) :
			btn = ctk.CTkButton(
					self.dest_grid , text=db.display_text( ) ,
					command=lambda n=db.name : self._toggle_dest( n ) ,
					fg_color=CARD , hover_color=CARD_SEL ,
					text_color=FG2 , height=36 , corner_radius=8 ,
					font=(EMOJI_FONT , 10) ,
			)
			btn.grid( row=i // cols , column=i % cols , padx=4 , pady=4 , sticky="ew" )
			self.dest_widgets[ db.name ] = btn

			# Bind the key shortcut for this destination
			self.root.bind( db.key , lambda _ , n=db.name : self._toggle_dest( n ) )
			self._bound_keys.append( db.key )

		# Restore previously selected destinations if navigating back
		if 0 <= self.current_index < len( self.history ) :
			for name in self.history[ self.current_index ][ 1 ] :
				if name in self.dest_widgets :
					self.selected_dests.append( name )
					self.dest_widgets[ name ].configure( fg_color=ACCENT , text_color=FG )

		# Also rebuild the conversion panel for this file
		self._rebuild_conv_buttons( )
		self._refresh_action_label( )

	def _toggle_dest( self , name: str ) :
		"""Toggle a destination on/off (multi-select)."""
		btn = self.dest_widgets.get( name )
		if not btn :
			return
		if name in self.selected_dests :
			self.selected_dests.remove( name )
			btn.configure( fg_color=CARD , text_color=FG2 )
			self.logger.debug( "Deselected destination: %s" , name )
		else :
			self.selected_dests.append( name )
			btn.configure( fg_color=ACCENT , text_color=FG )
			self.logger.debug( "Selected destination: %s" , name )
		self._refresh_action_label( )

	def _refresh_action_label( self ) :
		"""Update the info-bar action summary text."""
		parts = [ ]
		if self.selected_conversion :
			opt = next( (c for c in _ALL_CONV if c.key == self.selected_conversion) , None )
			parts.append( f"⚡ {opt.label if opt else self.selected_conversion}" )
		if self.selected_dests :
			parts.append( "→ " + ", ".join( self.selected_dests ) )
		self.action_lbl.configure(
				text="  |  ".join( parts ) if parts else "no action selected" ,
				text_color=ACCENT if parts else FG2 ,
		)

	# ══════════════════════════════════════════════════════════════════════════
	# CONFIRM  +  SILENT BATCH FLUSH  (threaded)
	# ══════════════════════════════════════════════════════════════════════════
	def _confirm( self ) :
		"""Confirm the current file's triage choices and advance to the next."""
		# Release media handles FIRST to avoid WinError 32
		self._release_media( )

		if not self.current_file :
			self.logger.debug( "Confirm called with no current file — ignored" )
			return

		# Save selections into history
		if 0 <= self.current_index < len( self.history ) :
			self.history[ self.current_index ][ 1 ] = list( self.selected_dests )
			self.history[ self.current_index ][ 2 ] = self.selected_conversion

		# Run conversion if one was selected
		if self.selected_conversion :
			self._run_conversion( self.selected_conversion )

		# Queue the file for batch move/copy
		if self.selected_dests :
			self.pending.append( (Path( self.current_file ) , list( self.selected_dests )) )
			self.batch_n += 1
			self.logger.info(
					"Queued '%s' for batch flush (batch %d/%d) → %s" ,
					Path( self.current_file ).name , self.batch_n , self.BATCH_SIZE ,
					self.selected_dests )
			# Flush silently when we hit the batch threshold
			if self.batch_n >= self.BATCH_SIZE :
				self.logger.info( "Batch threshold reached (%d) — triggering flush" , self.BATCH_SIZE )
				self._flush_silent( )
		else :
			self.logger.info( "Confirmed '%s' with no destinations (conversion only or skip)" ,
												Path( self.current_file ).name )

		self.show_next( )

	# ── conversion (background thread) ───────────────────────────────────────
	def _run_conversion( self , key: str ) :
		"""Archive the original, then run the selected conversion.
		The heavy conversion work runs in a background thread."""
		# Release handles before touching the file
		self._release_media( )

		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.warning( "Conversion requested but file missing: %s" , self.current_file )
			return

		# Step 1: archive the original (on main thread — fast copy)
		try :
			ARCHIVAL_DIR.mkdir( parents=True , exist_ok=True )
			archive_dst = _unique( ARCHIVAL_DIR , Path( self.current_file ) )
			shutil.copy2( str( Path( self.current_file ) ) , str( archive_dst ) )
			self.logger.info( "Archived original → '%s' before conversion" , archive_dst.name )
		except Exception as e :
			self.logger.error( "Archive failed before conversion: %s" , e )
			messagebox.showerror(
					"Archive Error" ,
					f"Could not archive before conversion:\n{e}\n\nConversion aborted." )
			return

		self.status_lbl.configure( text="⏳ Converting…" , text_color=YELLOW )
		self.root.update_idletasks( )

		# Step 2: run the actual conversion in a background thread
		self.logger.info( "Starting background conversion '%s' for '%s'" ,
											key , Path( self.current_file ).name )
		self._executor.submit( self._convert_worker , key , Path( self.current_file ) )

	def _convert_worker( self , key: str , src_file: Path ) :
		"""Execute the conversion OFF the main thread.
		MUST NOT touch any tkinter widgets directly — use root.after()."""
		try :
			if key == "img_pdf" :
				# Two-step: image → PNG → PDF
				png_path = convert_image_to_png( src=src_file , logger=self.logger )
				if not png_path :
					raise RuntimeError( "image→PNG step failed (returned None)" )
				new_path = convert_png_to_pdf( src=png_path , logger=self.logger )
				if not new_path :
					raise RuntimeError( "PNG→PDF step failed (returned None)" )
				self.logger.info( "Two-step conversion complete: image → PNG → PDF" )
			else :
				fn_map: dict[ str , Callable ] = {
					"img_png"  : convert_image_to_png ,
					"vid_mp4"  : convert_video_to_mp4 ,
					"aud_mp3"  : convert_audio_to_mp3 ,
					"doc_pdf"  : convert_document_to_pdf ,
					"eml_pdf"  : convert_email_to_pdf ,
					"html_pdf" : convert_html_to_pdf ,
				}
				fn = fn_map.get( key )
				if fn is None :
					self.logger.warning( "No converter found for key '%s'" , key )
					return
				new_path = fn( src=src_file , logger=self.logger )
				self.logger.info( "Conversion complete: '%s' → '%s'" , src_file.name , new_path.name )

			# Schedule UI update on main thread
			self.root.after( 0 , lambda : self._convert_done( new_path ) )

		except Exception as e :
			self.logger.error( "Conversion failed for '%s' (key=%s): %s" , src_file.name , key , e )
			self.root.after( 0 , lambda : self._convert_failed( str( e ) ) )

	def _convert_done( self , new_path: Path ) :
		"""Called on main thread when a background conversion succeeds."""
		self.current_file = new_path
		if 0 <= self.current_index < len( self.history ) :
			self.history[ self.current_index ][ 0 ] = new_path
		ext = new_path.suffix.upper( ).lstrip( "." )
		self.status_lbl.configure(
				text=f"✓ Converted → {ext} (original archived)" , text_color=GREEN )
		self.logger.info( "Conversion UI updated: current file is now '%s'" , new_path.name )

	def _convert_failed( self , err: str ) :
		"""Called on main thread when a background conversion fails."""
		self.status_lbl.configure( text="✗ Conversion failed" , text_color=RED )
		messagebox.showerror( "Conversion Error" , err )

	# ── batch flush (background thread) ──────────────────────────────────────
	def _flush_silent( self ) :
		"""Kick off the batch flush in a background thread.
		Moves/copies all pending files to their destinations."""
		if not self.pending :
			self.logger.debug( "Flush called with no pending items — skipped" )
			return

		batch = list( self.pending )
		self.pending.clear( )
		self.batch_n = 0
		self.logger.info( "Starting background flush of %d file(s)…" , len( batch ) )
		self.status_lbl.configure( text=f"⏳ Flushing {len( batch )} files…" , text_color=YELLOW )
		self._executor.submit( self._flush_worker , batch )

	def _flush_worker( self , batch: list[ tuple[ Path , list[ str ] ] ] ) :
		"""Runs OFF the main thread — do NOT touch tkinter here.
		Moves single-dest files, copies multi-dest files, then removes originals."""
		errors: list[ str ] = [ ]

		for file , dest_names in batch :
			if not file.exists( ) :
				self.logger.warning( "Flush: file no longer exists: '%s'" , file.name )
				continue

			# Gather all target directories for this file
			all_targets: list[ tuple[ Path , str ] ] = [ ]
			for dname in dest_names :
				db = DEST_BY_NAME.get( dname )
				if not db :
					self.logger.warning( "Flush: unknown destination '%s' — skipping" , dname )
					continue
				all_targets.extend( (tgt , dname) for tgt in db.targets )

			if not all_targets :
				continue

			if len( all_targets ) == 1 :
				# Single destination — move (faster, no duplicate)
				dst_dir , label = all_targets[ 0 ]
				try :
					dst_dir.mkdir( parents=True , exist_ok=True )
					dst = _unique( dst_dir , file )
					shutil.move( str( file ) , str( dst ) )
					self.logger.info( "Flush: moved '%s' → %s/%s" , file.name , label , dst.name )
				except Exception as e :
					errors.append( f"move {file.name}→{label}: {e}" )
					self.logger.error( "Flush move failed: '%s' → %s: %s" , file.name , label , e )
					# Fallback: dump to unsupported
					try :
						shutil.move( str( file ) , str( _unique( UNSUPPORTED_ARTIFACTS_DIR , file ) ) )
						self.logger.warning( "Flush fallback: '%s' → UNSUPPORTED" , file.name )
					except Exception as e2 :
						errors.append( f"fallback {file.name}: {e2}" )
			else :
				# Multiple destinations — copy to each, then delete original
				copy_ok = True
				for dst_dir , label in all_targets :
					try :
						dst_dir.mkdir( parents=True , exist_ok=True )
						dst = _unique( dst_dir , file )
						shutil.copy2( str( file ) , str( dst ) )
						self.logger.info( "Flush: copied '%s' → %s/%s" , file.name , label , dst.name )
					except Exception as e :
						errors.append( f"copy {file.name}→{label}: {e}" )
						self.logger.error( "Flush copy failed: '%s' → %s: %s" , file.name , label , e )
						copy_ok = False
				# Only delete the original if ALL copies succeeded
				if copy_ok and file.exists( ) :
					try :
						file.unlink( )
						self.logger.info( "Flush: removed original '%s' after %d copies" ,
															file.name , len( all_targets ) )
					except Exception as e :
						errors.append( f"delete original {file.name}: {e}" )
						self.logger.error( "Flush: failed to remove original '%s': %s" , file.name , e )

		# Schedule UI update back on main thread
		count = len( batch )
		self.root.after( 0 , lambda : self._flush_done( count , errors ) )

	def _flush_done( self , count: int , errors: list[ str ] ) :
		"""Called on main thread after the background flush completes."""
		if errors :
			self.logger.warning( "Flush completed with %d error(s): %s" , len( errors ) , errors[ :5 ] )

		# Reset history and caches
		self.history.clear( )
		self.current_index = -1
		self._thumb_cache.clear( )
		self._refresh_history_sidebar( )
		self.status_lbl.configure( text=f"✓ Flushed {count} files" , text_color=GREEN )
		self.logger.info( "Batch flush complete: %d actions, %d errors" , count , len( errors ) )

	# ══════════════════════════════════════════════════════════════════════════
	# FLAG → ALTERATIONS_REQUIRED
	# ══════════════════════════════════════════════════════════════════════════
	def _flag_with_dialog( self ) :
		"""Move the current file to ALTERATIONS_REQUIRED with an optional note."""
		# Release handles before moving
		self._release_media( )

		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.debug( "Flag called but no valid current file" )
			return

		# Run pending conversion first (if any)
		if self.selected_conversion :
			self._run_conversion( self.selected_conversion )

		# Ask user for a description
		description = simpledialog.askstring(
				"Flag for Review" ,
				f"File: {Path( self.current_file ).name}\n\n"
				"Enter notes / directions for this file\n"
				"(leave blank for no description):" ,
				parent=self.root ,
		)
		if description is None :
			self.logger.debug( "Flag dialog cancelled by user" )
			return

		try :
			dst = _unique( ALTERATIONS_REQUIRED_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.flagged.add( dst )
			_write_alteration_csv( dst.name , description.strip( ) , self.logger )
			self.status_lbl.configure(
					text="🚩 Flagged → ALTERATIONS REQUIRED" , text_color=ACCENT )
			self.logger.info( "Flagged '%s' → ALTERATIONS_REQUIRED (note: '%s')" ,
												dst.name , description.strip( )[ :60 ] )
			self._refresh_history_sidebar( )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Flag failed for '%s': %s" , Path( self.current_file ).name , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# UNESSENTIAL
	# ══════════════════════════════════════════════════════════════════════════
	def _send_to_unessential( self ) :
		"""Move the current file to UNESSENTIAL and advance."""
		# Release handles before moving
		self._release_media( )

		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.debug( "Unessential called but no valid current file" )
			return

		try :
			UNESSENTIAL_DIR.mkdir( parents=True , exist_ok=True )
			dst = _unique( UNESSENTIAL_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.status_lbl.configure( text="↓ Moved to UNESSENTIAL" , text_color=FG2 )
			self.logger.info( "Sent '%s' → UNESSENTIAL" , Path( self.current_file ).name )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Unessential move failed for '%s': %s" ,
												 Path( self.current_file ).name , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# DELETE
	# ══════════════════════════════════════════════════════════════════════════
	def _delete_file( self ) :
		"""Move the current file to DELETE (with confirmation dialog)."""
		# Release handles before moving
		self._release_media( )

		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.debug( "Delete called but no valid current file" )
			return

		if not messagebox.askyesno(
				"Confirm Delete" ,
				f"Move '{Path( self.current_file ).name}' to DELETE folder?" ,
		) :
			self.logger.debug( "Delete cancelled by user" )
			return

		try :
			dst = _unique( DELETE_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.status_lbl.configure( text="🗑 Deleted" , text_color=RED )
			self.logger.info( "Deleted '%s' → DELETE/%s" , Path( self.current_file ).name , dst.name )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Delete failed for '%s': %s" , Path( self.current_file ).name , e )
			messagebox.showerror( "Error" , f"Delete failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# NAVIGATION
	# ══════════════════════════════════════════════════════════════════════════
	def show_next( self ) :
		"""Advance to the next file in the queue or history."""
		# Record how long the user spent on the previous file
		if self.file_start :
			elapsed = time.time( ) - self.file_start
			self.file_times.append( elapsed )
			self.logger.debug( "Time on previous file: %.1fs" , elapsed )

		# Release any active media playback
		self._release_media( )

		if self.current_index < len( self.history ) - 1 :
			# Navigate forward within already-reviewed history
			self.current_index += 1
			self.current_file = self.history[ self.current_index ][ 0 ]
			self.logger.debug( "Navigated forward in history to index %d: '%s'" ,
												 self.current_index , Path( self.current_file ).name )
		elif self.queue :
			# Pop the next file from the queue
			self.current_file = self.queue.popleft( )
			self.history.append( [ Path( self.current_file ) , [ ] , None ] )
			self.current_index = len( self.history ) - 1
			self.file_start = time.time( )
			self.logger.info( "Dequeued file %d: '%s' (%d remaining)" ,
												self.current_index + 1 ,
												Path( self.current_file ).name ,
												len( self.queue ) )
		else :
			# Queue exhausted
			self.current_file = None
			self._clear_preview( )
			self._update_info_bar( )
			self.status_lbl.configure( text="✓ All files reviewed!" , text_color=GREEN )
			self._render_dest_buttons( )
			self.logger.info( "Queue exhausted — all files have been reviewed" )
			return

		# Refresh the UI for the new file
		self._render_dest_buttons( )
		self._display_file( )
		self._update_info_bar( )
		self._refresh_history_sidebar( )

	def show_previous( self ) :
		"""Navigate back to the previous file in history."""
		self._release_media( )
		if self.current_index > 0 :
			self.current_index -= 1
			self.current_file = self.history[ self.current_index ][ 0 ]
			self.logger.debug( "Navigated back to index %d: '%s'" ,
												 self.current_index , Path( self.current_file ).name )
			self._render_dest_buttons( )
			self._display_file( )
			self._update_info_bar( )
			self._refresh_history_sidebar( )
		else :
			self.logger.debug( "Already at first file — cannot go back" )

	def _jump_to( self , idx: int ) :
		"""Jump directly to a specific history index (sidebar click)."""
		if 0 <= idx < len( self.history ) :
			self._release_media( )
			self.current_index = idx
			self.current_file = self.history[ idx ][ 0 ]
			self.logger.debug( "Jumped to history index %d: '%s'" ,
												 idx , Path( self.current_file ).name )
			self._render_dest_buttons( )
			self._display_file( )
			self._update_info_bar( )
			self._refresh_history_sidebar( )

	def _update_info_bar( self ) :
		"""Refresh filename, position counter, and queue count in the info bar."""
		if self.current_file :
			total = len( self.history ) + len( self.queue )
			self.file_lbl.configure(
					text=f"{Path( self.current_file ).name}  ({self.current_index + 1}/{total})" )
		else :
			self.file_lbl.configure( text="—" )
		self.queue_lbl.configure(
				text=f"Queue: {len( self.queue )}  |  Flagged: {len( self.flagged )}" )

	# ══════════════════════════════════════════════════════════════════════════
	# TIMER (updates every second)
	# ══════════════════════════════════════════════════════════════════════════
	def _tick_timer( self ) :
		"""Update the session timer, average time per file, and ETA."""
		elapsed = int( time.time( ) - self.session_start )
		avg = sum( self.file_times ) / len( self.file_times ) if self.file_times else 0
		rem = int( len( self.queue ) * avg / 60 ) if avg > 0 else 0
		self.timer_lbl.configure( text=f"⏱ {elapsed}s  ·  avg {avg:.1f}s  ·  ~{rem}m left" )
		self.root.after( 1000 , self._tick_timer )

	# ══════════════════════════════════════════════════════════════════════════
	# HISTORY SIDEBAR
	# ══════════════════════════════════════════════════════════════════════════
	def _refresh_history_sidebar( self ) :
		"""Rebuild the sidebar showing thumbnails and actions for reviewed files."""
		for w in self.hist_frame.winfo_children( ) :
			w.destroy( )

		for idx , (file , dests , conv) in enumerate( self.history ) :
			is_curr = idx == self.current_index
			is_flag = file in self.flagged
			bg = CARD_SEL if is_curr else CARD
			bdr = ACCENT if is_flag else (BLUE if is_curr else CARD)

			frame = ctk.CTkFrame(
					self.hist_frame , fg_color=bg , corner_radius=10 ,
					border_width=2 if (is_curr or is_flag) else 0 ,
					border_color=bdr ,
			)
			frame.pack( fill=tk.X , padx=4 , pady=4 )

			# Thumbnail (async — may return None on first call)
			thumb = self._thumbnail( file )
			if thumb :
				lbl = tk.Label( frame , image=thumb , bg=bg , cursor="hand2" )
				lbl.image = thumb
				lbl.pack( pady=(6 , 2) , padx=6 )
				lbl.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )

			# Filename (truncated to 30 chars)
			name = (file.name[ :29 ] + "…") if len( file.name ) > 30 else file.name
			ctk.CTkLabel(
					frame ,
					text=("🚩 " if is_flag else "") + name ,
					font=(FONT , 10 , "bold" if is_curr else "normal") ,
					text_color=ACCENT if is_flag else FG ,
					wraplength=252 , justify="left" ,
			).pack( fill=tk.X , padx=8 , pady=(0 , 2) )

			# Action summary
			sub = [ ]
			if conv :
				opt = next( (c for c in _ALL_CONV if c.key == conv) , None )
				sub.append( f"⚡ {opt.label if opt else conv}" )
			if dests :
				sub.append( "→ " + ", ".join( dests ) )
			ctk.CTkLabel(
					frame ,
					text="  ".join( sub ) if sub else "[no action]" ,
					font=(FONT , 9) , text_color=FG2 , wraplength=252 ,
			).pack( fill=tk.X , padx=8 , pady=(0 , 6) )

			# Clickable frame
			frame.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )

	# ══════════════════════════════════════════════════════════════════════════
	# PREVIEW DISPATCHER
	# ══════════════════════════════════════════════════════════════════════════
	def _clear_preview( self ) :
		"""Remove all widgets from the preview scroll area."""
		for w in self.prev_scroll.winfo_children( ) :
			w.destroy( )

	def _display_file( self ) :
		"""Render the appropriate preview for the current file type."""
		self._clear_preview( )
		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.debug( "No file to display" )
			return

		cat = _file_category( Path( self.current_file ) )
		self.logger.debug( "Displaying '%s' as category '%s'" ,
											 Path( self.current_file ).name , cat )

		try :
			dispatch = {
				"pdf"   : self._show_pdf ,
				"image" : self._show_image ,
				"video" : self._show_video ,
				"audio" : self._show_audio ,
				"html"  : self._show_html ,
				"text"  : self._show_text ,
			}
			dispatch.get( cat , self._show_metadata )( )
		except Exception as e :
			self.logger.error( "Preview failed for '%s': %s" , Path( self.current_file ).name , e )
			self._show_metadata( err=str( e ) )

	# ── metadata (Tika fallback) ─────────────────────────────────────────────
	def _show_metadata( self , err: str = "" ) :
		"""Show file metadata extracted by Tika (fallback for unsupported types)."""
		self.logger.debug( "Showing metadata view for '%s'" , Path( self.current_file ).name )
		ctk.CTkLabel(
				self.prev_scroll , text=f"📋  {Path( self.current_file ).name}" ,
				font=(EMOJI_FONT , 13 , "bold") , text_color=ACCENT ,
		).pack( anchor="w" , padx=16 , pady=(12 , 4) )

		if err :
			ctk.CTkLabel(
					self.prev_scroll , text=f"⚠ {err}" ,
					font=(FONT , 10) , text_color=YELLOW ,
			).pack( anchor="w" , padx=16 , pady=(0 , 8) )

		self.status_lbl.configure( text="Extracting metadata…" , text_color=FG2 )
		self.root.update_idletasks( )
		meta = _tika_metadata( Path( self.current_file ) )
		self.status_lbl.configure( text="" , text_color=FG2 )

		box = ctk.CTkTextbox(
				self.prev_scroll , fg_color=CARD , text_color=FG ,
				font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=14 , pady=8 )
		box.insert( "1.0" , meta )
		box.configure( state="disabled" )

	# ── HTML ─────────────────────────────────────────────────────────────────
	def _show_html( self ) :
		"""Render HTML content using tkinterweb, with text fallback."""
		self.logger.debug( "Showing HTML preview for '%s'" , Path( self.current_file ).name )
		html_content = Path( self.current_file ).read_text( encoding="utf-8" , errors="ignore" )

		ctk.CTkLabel(
				self.prev_scroll , text=f"🌐  {Path( self.current_file ).name}" ,
				font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ,
		).pack( anchor="w" , padx=14 , pady=(8 , 4) )

		container = ctk.CTkFrame( self.prev_scroll , fg_color=CARD , corner_radius=10 )
		container.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )

		try :
			hframe = HtmlFrame( container , horizontal_scrollbar="auto" )
			hframe.pack( fill=tk.BOTH , expand=True )
			hframe.load_html( html_content )
			hframe.set_zoom( self.zoom_scale )
			self.logger.debug( "HTML rendered via tkinterweb" )
			return
		except Exception as e :
			self.logger.warning( "tkinterweb render failed, falling back to text: %s" , e )
			for w in container.winfo_children( ) :
				w.destroy( )

		# Text fallback
		ctk.CTkLabel(
				self.prev_scroll ,
				text="ℹ  pip install tkinterweb  for rendered preview" ,
				font=(FONT , 9) , text_color=YELLOW ,
		).pack( anchor="w" , padx=16 , pady=(0 , 6) )

		box = ctk.CTkTextbox(
				self.prev_scroll , fg_color=CARD , text_color=FG ,
				font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )
		box.insert( "1.0" , html_content )
		box.configure( state="disabled" )

	# ── PDF (incremental page rendering) ─────────────────────────────────────
	def _show_pdf( self ) :
		"""Render PDF pages as images in a grid.
		First 2 pages are rendered eagerly; the rest are deferred via after()
		to keep the UI responsive on large documents."""
		self.logger.debug( "Showing PDF preview for '%s'" , Path( self.current_file ).name )
		doc = fitz.open( Path( self.current_file ) )
		total = len( doc )
		show = min( total , self.MAX_PDF_PG )
		truncated = total > self.MAX_PDF_PG

		self.logger.info( "PDF: %d pages total, rendering %d" , total , show )

		ctk.CTkLabel(
				self.prev_scroll ,
				text=(f"PDF  ·  {total} pages  ·  showing first {show}"
							if truncated else f"PDF  ·  {total} page(s)") ,
				font=(FONT , 11 , "bold") , text_color=ACCENT ,
		).pack( pady=(10 , 4) )

		if truncated :
			ctk.CTkLabel(
					self.prev_scroll ,
					text=f"⚠ Showing {show} of {total} — Open External for all." ,
					font=(FONT , 9) , text_color=YELLOW ,
			).pack( pady=(0 , 6) )

		grid = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		grid.pack( fill=tk.BOTH , expand=True , padx=14 )
		cols = min( 4 , show )
		for c in range( cols ) :
			grid.grid_columnconfigure( c , weight=1 )

		# Lower base scale for faster rendering (0.20 instead of 0.28)
		scale = 0.20 * self.zoom_scale
		EAGER_PAGES = 2  # render these immediately

		def render_page( i ) :
			"""Render a single PDF page into the grid."""
			frame = ctk.CTkFrame( grid , fg_color=CARD , corner_radius=10 )
			frame.grid( row=i // cols , column=i % cols , padx=8 , pady=8 , sticky="nsew" )
			pix = doc[ i ].get_pixmap( matrix=fitz.Matrix( scale , scale ) )
			img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
			photo = ImageTk.PhotoImage( img )
			lbl = tk.Label( frame , image=photo , bg=CARD )
			lbl.image = photo
			lbl.pack( pady=6 )
			ctk.CTkLabel( frame , text=f"p{i + 1}" ,
										font=(FONT , 9) , text_color=FG2 ).pack( pady=(0 , 6) )

		# Render first few pages eagerly
		for i in range( min( EAGER_PAGES , show ) ) :
			render_page( i )
			self.logger.debug( "PDF page %d rendered (eager)" , i + 1 )

		# Defer the remaining pages to avoid blocking
		def render_deferred( idx ) :
			if idx >= show :
				doc.close( )
				self.logger.debug( "PDF rendering complete (all %d pages)" , show )
				return
			try :
				render_page( idx )
				self.logger.debug( "PDF page %d rendered (deferred)" , idx + 1 )
			except Exception as e :
				self.logger.error( "Failed to render PDF page %d: %s" , idx + 1 , e )
			# Schedule the next page with a small delay so the UI can breathe
			self.root.after( 50 , lambda : render_deferred( idx + 1 ) )

		if show > EAGER_PAGES :
			self.root.after( 100 , lambda : render_deferred( EAGER_PAGES ) )
		else :
			doc.close( )

	# ── image ────────────────────────────────────────────────────────────────
	def _show_image( self ) :
		"""Display an image with zoom-aware resizing."""
		self.logger.debug( "Showing image preview for '%s'" , Path( self.current_file ).name )
		img = Image.open( Path( self.current_file ) )
		ow , oh = img.size

		ctk.CTkLabel(
				self.prev_scroll ,
				text=f"{ow}×{oh}  ·  {Path( self.current_file ).suffix.upper( ).lstrip( '.' )}" ,
				font=(FONT , 10) , text_color=FG2 ,
		).pack( anchor="w" , padx=14 , pady=(8 , 2) )

		# Apply zoom and cap width at 1400px
		nw , nh = int( ow * self.zoom_scale ) , int( oh * self.zoom_scale )
		if nw > 1400 :
			nh = int( nh * 1400 / nw )
			nw = 1400

		img = img.resize( (nw , nh) , Image.Resampling.LANCZOS )
		photo = ImageTk.PhotoImage( img )
		lbl = tk.Label( self.prev_scroll , image=photo , bg=PREVIEW )
		lbl.image = photo
		lbl.pack( pady=10 )

	# ── video (lazy — no auto-play, 15 fps cap, scrub bar) ───────────────────
	def _show_video( self ) :
		"""Display video preview with manual play/pause and a scrub bar.
		Does NOT auto-play — shows a still frame from frame 0.
		Playback is capped at 15 fps to reduce CPU load."""
		self.logger.debug( "Showing video preview for '%s'" , Path( self.current_file ).name )
		self.video_cap = cv2.VideoCapture( str( Path( self.current_file ) ) )
		cap = self.video_cap
		fps = cap.get( cv2.CAP_PROP_FPS ) or 25
		nfrm = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
		dur = nfrm / fps if fps else 0
		vw = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
		vh = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )

		self.logger.info( "Video: %dx%d, %.1fs, %.0f fps, %d frames" ,
											vw , vh , dur , fps , nfrm )

		# Pre-calculate a fixed preview size (cap at 720px wide)
		preview_w = min( 720 , int( vw * 0.55 * self.zoom_scale ) )
		preview_h = int( vh * (preview_w / vw) ) if vw > 0 else 360
		self._vid_size = (preview_w , preview_h)

		# Cap playback at 15 fps regardless of source — halves CPU usage
		self._vid_delay = max( 1 , int( 1000 / min( fps , 15 ) ) )

		ctk.CTkLabel(
				self.prev_scroll ,
				text=f"🎬  {Path( self.current_file ).name}  ·  {vw}×{vh}  ·  {dur:.1f}s  ·  {fps:.0f}fps" ,
				font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ,
		).pack( pady=(10 , 4) )

		ctk.CTkLabel(
				self.prev_scroll ,
				text="⚠ Preview only (no audio) — Open External for full playback" ,
				font=(FONT , 9) , text_color=YELLOW ,
		).pack( )

		# Show first frame as a still thumbnail
		ret , frame = cap.read( )
		photo = None
		if ret :
			frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
			frame = cv2.resize( frame , self._vid_size )
			photo = ImageTk.PhotoImage( Image.fromarray( frame ) )

		self.video_lbl = tk.Label( self.prev_scroll , bg="#000000" )
		if photo :
			self.video_lbl.configure( image=photo )
			self.video_lbl.image = photo
		self.video_lbl.pack( pady=8 )

		# Rewind to frame 0 for playback
		cap.set( cv2.CAP_PROP_POS_FRAMES , 0 )

		# ── controls ──────────────────────────────────────────────────────────
		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		ctrl.pack( )

		# NOT auto-playing — user clicks Play when ready
		self.is_playing = False

		def toggle( ) :
			self.is_playing = not self.is_playing
			play_btn.configure( text="⏸ Pause" if self.is_playing else "▶ Play" )
			if self.is_playing :
				self.logger.debug( "Video playback started" )
				_next( )
			else :
				self.logger.debug( "Video playback paused" )

		play_btn = ctk.CTkButton(
				ctrl , text="▶ Play" , command=toggle ,
				fg_color=BLUE , hover_color=_hover( BLUE ) ,
				width=90 , height=34 , corner_radius=8 , text_color=FG )
		play_btn.pack( side=tk.LEFT , padx=4 )

		def _next( ) :
			"""Advance one frame. Scheduled via after() for non-blocking playback."""
			if not self.is_playing or not self.video_cap :
				return
			ret , frame = self.video_cap.read( )
			if ret :
				frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
				frame = cv2.resize( frame , self._vid_size )
				photo = ImageTk.PhotoImage( Image.fromarray( frame ) )
				self.video_lbl.configure( image=photo )
				self.video_lbl.image = photo
				self.video_after_id = self.root.after( self._vid_delay , _next )
			else :
				# End of video — rewind and stop
				self.video_cap.set( cv2.CAP_PROP_POS_FRAMES , 0 )
				self.is_playing = False
				play_btn.configure( text="▶ Play" )
				self.logger.debug( "Video reached end, rewound to start" )

		# ── scrub bar for quick seeking ───────────────────────────────────────
		if nfrm > 0 :
			scrub = ctk.CTkSlider(
					self.prev_scroll ,
					from_=0 , to=nfrm - 1 , number_of_steps=min( nfrm - 1 , 500 ) ,
					command=lambda v : self._vid_seek( int( v ) ) ,
					fg_color=CARD , progress_color=ACCENT , button_color=ACCENT_L ,
					width=400 , height=16 ,
			)
			scrub.set( 0 )
			scrub.pack( pady=(4 , 8) )

	def _vid_seek( self , frame_num: int ) :
		"""Jump to a specific frame number (called by scrub bar)."""
		if self.video_cap and self.video_lbl :
			self.video_cap.set( cv2.CAP_PROP_POS_FRAMES , frame_num )
			ret , frame = self.video_cap.read( )
			if ret :
				frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
				frame = cv2.resize( frame , self._vid_size )
				photo = ImageTk.PhotoImage( Image.fromarray( frame ) )
				self.video_lbl.configure( image=photo )
				self.video_lbl.image = photo

	# ── audio ────────────────────────────────────────────────────────────────
	def _show_audio( self ) :
		"""Display audio metadata and play/pause/stop controls."""
		self.logger.debug( "Showing audio preview for '%s'" , Path( self.current_file ).name )

		ctk.CTkLabel(
				self.prev_scroll , text=f"🎵  {Path( self.current_file ).name}" ,
				font=(EMOJI_FONT , 14 , "bold") , text_color=ACCENT ,
		).pack( pady=(28 , 8) )

		# Show metadata from Tika
		meta = _tika_metadata( Path( self.current_file ) )
		box = ctk.CTkTextbox(
				self.prev_scroll , height=120 ,
				fg_color=CARD , text_color=FG2 ,
				font=(FONT , 10) , corner_radius=8 )
		box.pack( fill=tk.X , padx=20 , pady=8 )
		box.insert( "1.0" , meta or "—" )
		box.configure( state="disabled" )

		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		ctrl.pack( pady=10 )

		def play( ) :
			try :
				pygame.mixer.music.load( str( Path( self.current_file ) ) )
				pygame.mixer.music.play( )
				self.logger.debug( "Audio playback started" )
			except Exception as e :
				self.logger.error( "Audio playback failed: %s" , e )
				messagebox.showerror( "Playback Error" , str( e ) )

		for txt , cmd , col in [
			("▶ Play" , play , GREEN) ,
			("⏸ Pause" , pygame.mixer.music.pause , YELLOW) ,
			("⏹ Stop" , pygame.mixer.music.stop , ACCENT) ,
		] :
			ctk.CTkButton(
					ctrl , text=txt , command=cmd ,
					fg_color=col , hover_color=_hover( col ) ,
					width=90 , height=36 , corner_radius=8 ,
					text_color=FG , font=(FONT , 11) ,
			).pack( side=tk.LEFT , padx=4 )

		# Auto-play after a short delay
		self.root.after( 300 , play )

	# ── text / code ──────────────────────────────────────────────────────────
	def _show_text( self ) :
		"""Display text files with line count and zoom-aware font size."""
		self.logger.debug( "Showing text preview for '%s'" , Path( self.current_file ).name )
		try :
			content = Path( self.current_file ).read_text( encoding="utf-8" , errors="ignore" )
			lines = len( content.splitlines( ) )
			self.logger.debug( "Text file: %d lines" , lines )

			ctk.CTkLabel(
					self.prev_scroll ,
					text=f"{lines} lines  ·  {Path( self.current_file ).suffix.upper( ).lstrip( '.' )}" ,
					font=(FONT , 10) , text_color=FG2 ,
			).pack( anchor="w" , padx=14 , pady=(8 , 2) )

			box = ctk.CTkTextbox(
					self.prev_scroll , fg_color=CARD , text_color=FG ,
					font=("Consolas" , max( 8 , int( 10 * self.zoom_scale ) )) ,
					corner_radius=10 )
			box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=8 )
			box.insert( "1.0" , content )
			box.configure( state="disabled" )
		except Exception as e :
			self.logger.error( "Text preview failed: %s" , e )
			self._show_metadata( err=str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# OPEN EXTERNAL
	# ══════════════════════════════════════════════════════════════════════════
	def _open_external( self ) :
		"""Open the current file in the OS default application."""
		if not self.current_file or not Path( self.current_file ).exists( ) :
			return
		try :
			path = Path( self.current_file )
			if os.name == "nt" :
				os.startfile( path )
			elif os.name == "posix" :
				subprocess.run( [ "xdg-open" , str( path ) ] )
			else :
				subprocess.run( [ "open" , str( path ) ] )
			self.logger.info( "Opened external: '%s'" , path.name )
		except Exception as e :
			self.logger.error( "External open failed: %s" , e )
			messagebox.showerror( "Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# THUMBNAIL GENERATION (async)
	# Thumbnails are generated in a background thread to avoid blocking
	# the sidebar rendering. A placeholder (None) is stored immediately,
	# and the sidebar is refreshed once the thumbnail is ready.
	# ══════════════════════════════════════════════════════════════════════════
	def _thumbnail( self , file: Path , size=(260 , 100) ) :
		"""Return cached thumbnail or None (kicks off async generation)."""
		key = str( file )
		if key in self._thumb_cache :
			return self._thumb_cache[ key ]

		# Mark as in-progress so we don't spawn duplicate threads
		self._thumb_cache[ key ] = None
		self._executor.submit( self._gen_thumb_bg , file , size , key )
		return None

	def _gen_thumb_bg( self , file: Path , size: tuple , key: str ) :
		"""Generate a thumbnail OFF the main thread.
		PIL Image objects are thread-safe; ImageTk.PhotoImage is NOT,
		so we pass the PIL image back to the main thread for conversion."""
		if not file.exists( ) :
			return

		thumb_img = None
		ext = file.suffix.lower( )
		try :
			if ext == ".pdf" :
				doc = fitz.open( file )
				if doc :
					pix = doc[ 0 ].get_pixmap( matrix=fitz.Matrix( 0.2 , 0.2 ) )
					img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
					img.thumbnail( size , Image.Resampling.LANCZOS )
					doc.close( )
					thumb_img = img
			elif ext in { ".jpg" , ".jpeg" , ".png" , ".gif" , ".bmp" , ".webp" } :
				img = Image.open( file )
				img.thumbnail( size , Image.Resampling.LANCZOS )
				thumb_img = img
			elif ext in { ".mp4" , ".avi" , ".mkv" , ".mov" } :
				cap = cv2.VideoCapture( str( file ) )
				ret , frame = cap.read( )
				cap.release( )
				if ret :
					img = Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
					img.thumbnail( size , Image.Resampling.LANCZOS )
					thumb_img = img
			elif ext in { ".docx" , ".pptx" , ".xlsx" } :
				with ZipFile( file , "r" ) as z :
					thumbs = [ n for n in z.namelist( ) if "thumbnail" in n.lower( ) ]
					if thumbs and thumbs[ 0 ].lower( ).endswith( (".jpg" , ".jpeg" , ".png") ) :
						img = Image.open( io.BytesIO( z.read( thumbs[ 0 ] ) ) )
						img.thumbnail( size , Image.Resampling.LANCZOS )
						thumb_img = img
		except Exception as e :
			self.logger.debug( "Thumbnail generation failed for '%s': %s" , file.name , e )
			return

		if thumb_img :
			# ImageTk.PhotoImage must be created on the main thread
			self.root.after( 0 , lambda : self._set_thumb( key , thumb_img ) )

	def _set_thumb( self , key: str , pil_img: Image.Image ) :
		"""Convert PIL Image to PhotoImage on main thread and refresh sidebar."""
		try :
			photo = ImageTk.PhotoImage( pil_img )
			self._thumb_cache[ key ] = photo
			self._refresh_history_sidebar( )
			self.logger.debug( "Thumbnail ready for cache key '%s'" , key[ :40 ] )
		except Exception as e :
			self.logger.debug( "Failed to create PhotoImage for '%s': %s" , key[ :40 ] , e )

	# ══════════════════════════════════════════════════════════════════════════
	# RUN
	# ══════════════════════════════════════════════════════════════════════════
	def run( self ) :
		if any( item.is_file( ) for item in self.source_dir.iterdir( ) ) :
			self.root.mainloop( )
		else :
			self.logger.warning( f"No items found in {self.source_dir}. Not running \"Manual Triaging\" application" )
