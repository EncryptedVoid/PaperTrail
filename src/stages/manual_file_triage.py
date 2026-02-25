"""
File Triage — Manual Review Tool  v5.1
───────────────────────────────────────
Architecture:
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
from dataclasses import dataclass , field
from datetime import datetime
from pathlib import Path
from tkinter import messagebox , simpledialog
from typing import Callable
from zipfile import ZipFile

# ── Project config ───────────────────────────────────────────────────────────
from config import (AFFINE_DIR , ALTERATIONS_REQUIRED_DIR , ANKI_DIR , ARCHIVAL_DIR , AUDIO_TYPES , BASE_DIR ,
										BITWARDEN_DIR , CALIBRE_LIBRARY_DIR , DIGITAL_ASSET_MANAGEMENT_DIR , DOCUMENT_TYPES , EMAIL_TYPES ,
										FIREFLYIII_DIR ,
										GAMES_ARCHIVE_DIR , GITLAB_DIR , IMAGE_TYPES , IMMICH_DIR , JAVA_PATH , LIBRE_OFFICE_DIR ,
										LINKWARDEN_DIR ,
										MANUALS_ARCHIVE_DIR , MONICA_CRM_DIR , ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR ,
										ODOO_PLM_DIR ,
										ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR , PERSONAL_ARCHIVE_DIR , SCANNING_REQUIRED_DIR ,
										SOFTWARE_ARCHIVE_DIR ,
										SPREADSHEET_TYPES , TIKA_APP_JAR_PATH , ULTIMAKER_CURA_DIR , UNESSENTIAL_DIR ,
										UNSUPPORTED_ARTIFACTS_DIR , VIDEO_TYPES)
# ── Conversion functions ─────────────────────────────────────────────────────
from utilities.format_converting import (convert_audio_to_mp3 , convert_document_to_pdf , convert_email_to_pdf ,
																				 convert_html_to_pdf , convert_image_to_png , convert_png_to_pdf ,
																				 convert_video_to_mp4 ,
																				 convert_xlsx_to_csv)

# ── UI framework ─────────────────────────────────────────────────────────────
try :
	import customtkinter as ctk
	from customtkinter import CTkScrollableFrame

	ctk.set_appearance_mode( "Dark" )
	ctk.set_default_color_theme( "blue" )
except ImportError :
	import tkinter as ctk

	CTkScrollableFrame = tk.Frame

# ── Optional media libraries ─────────────────────────────────────────────────
try :
	from PIL import Image , ImageTk
	import fitz

	PIL_AVAILABLE = True
except ImportError :
	PIL_AVAILABLE = False

try :
	import cv2

	CV2_AVAILABLE = True
except ImportError :
	CV2_AVAILABLE = False

try :
	import pygame

	pygame.mixer.init( )
	PYGAME_AVAILABLE = True
except ImportError :
	PYGAME_AVAILABLE = False

try :
	from tkinterweb import HtmlFrame

	TKINTERWEB_AVAILABLE = True
except ImportError :
	TKINTERWEB_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig( level=logging.INFO , format="%(asctime)s  %(levelname)s  %(message)s" )
logger = logging.getLogger( "FileTriage" )

# ── Fonts ────────────────────────────────────────────────────────────────────
FONT = "Segoe UI"
EMOJI_FONT = "Segoe UI Emoji"
try :
	import tkinter.font as _tkf

	_r = tk.Tk( );
	_avail = _tkf.families( );
	_r.destroy( )
	FONT = FONT if FONT in _avail else ("Arial" if "Arial" in _avail else "TkDefaultFont")
	if EMOJI_FONT not in _avail :
		EMOJI_FONT = FONT
except Exception :
	FONT = EMOJI_FONT = "Arial"

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG = "#141414"
SIDEBAR = "#1C1C1C"
PREVIEW = "#181818"
CARD = "#222222"
CARD_SEL = "#2A2A2A"
FG = "#EFEFEF"
FG2 = "#909090"
ACCENT = "#E05C52"
ACCENT_L = "#F07068"
ACCENT_D = "#B84840"
GREEN = "#5DBD72"
BLUE = "#4A8ED9"
YELLOW = "#D9A740"
RED = "#CC2222"

DELETE_DIR: Path = BASE_DIR / "DELETE"
ALTERATIONS_CSV: Path = ALTERATIONS_REQUIRED_DIR / "alterations_log.csv"

# ══════════════════════════════════════════════════════════════════════════════
# BOLD-UNICODE HELPER  (Mathematical Sans-Serif Bold A-Z / a-z)
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
	return f"[{_BOLD.get( key.upper( ) , key.upper( ) )}] {label}"


# ══════════════════════════════════════════════════════════════════════════════
# DESTINATION BUTTON REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
#
# Every triage destination is described by a single DestButton instance.
# Fields:
#   name    — unique identifier (used in history, pending list, logs)
#   label   — display text shown on the button (emoji prefix)
#   order   — integer controlling left-to-right, top-to-bottom grid position
#   key     — keyboard shortcut (single lowercase letter)
#   action  — callable(file, app) executed at flush time; receives the Path
#             and a reference to the FileTriage instance so it can call helpers.
#             Must return True on success.
#   targets — list of directory Paths the file should end up in.  When there
#             is a single target the file is *moved*; with multiple targets the
#             file is *copied* to each then the original is deleted.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DestButton :
	"""Self-contained record for one triage-destination button."""
	name: str
	label: str
	order: int
	key: str
	targets: list[ Path ] = field( default_factory=list )

	# --- derived at runtime ---------------------------------------------------
	def display_text( self ) -> str :
		"""Label with the keybind letter bolded for visual identification."""
		return _bold_key_in_label( self.label , self.key )


def _build_dest_registry( ) -> list[ DestButton ] :
	"""Construct the ordered list of destination buttons.

    Each entry maps a user-visible button to one or more filesystem targets.
    The *order* field controls grid placement (sorted ascending, filled
    left-to-right across 5 columns)."""
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
		DestButton( "LIBREOFFICE" , "📝 LIBREOFFICE" , 110 , "l" , [ LIBRE_OFFICE_DIR ] ) ,
		DestButton( "LINKWARDEN" , "🔖 LINKWARDEN" , 120 , "w" , [ LINKWARDEN_DIR ] ) ,
		DestButton( "MANUALS ARCHIVE" , "📖 MANUALS ARCHIVE" , 125 , "h" , [ MANUALS_ARCHIVE_DIR ] ) ,
		DestButton( "ODOO INVENTORY" , "📦 ODOO INVENTORY" , 130 , "v" , [ ODOO_INVENTORY_DIR ] ) ,
		DestButton( "ODOO MAINTENANCE" , "🔧 ODOO MAINTENANCE" , 140 , "e" , [ ODOO_MAINTENANCE_DIR ] ) ,
		DestButton( "ODOO PLM" , "⚙ ODOO PLM" , 150 , "p" , [ ODOO_PLM_DIR ] ) ,
		DestButton( "ODOO PURCHASE" , "🛒 ODOO PURCHASE" , 160 , "u" , [ ODOO_PURCHASE_DIR ] ) ,
		DestButton( "PORTFOLIO PERF." , "🎭 PORTFOLIO PERF." , 170 , "x" , [ PERFORMANCE_PORTFOLIO_DIR ] ) ,
		DestButton( "SCANNING" , "🔍 SCANNING" , 175 , "z" , [ SCANNING_REQUIRED_DIR ] ) ,
		DestButton( "SEAFILE" , "☁ SEAFILE" , 180 , "s" , [ DIGITAL_ASSET_MANAGEMENT_DIR ] ) ,
		DestButton( "SOFTWARE ARCHIVE" , "💾 SOFTWARE ARCHIVE" , 185 , "d" , [ SOFTWARE_ARCHIVE_DIR ] ) ,
		DestButton( "ULTIMAKER CURA" , "🖨 ULTIMAKER CURA" , 190 , "q" , [ ULTIMAKER_CURA_DIR ] ) ,
	] , key=lambda b : b.order )


# Build the global registry once at import time
DEST_REGISTRY: list[ DestButton ] = _build_dest_registry( )

# Quick-lookup maps derived from the registry
DEST_BY_NAME: dict[ str , DestButton ] = { b.name : b for b in DEST_REGISTRY }
DEST_BY_KEY: dict[ str , DestButton ] = { b.key : b for b in DEST_REGISTRY }


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE CONVERSION OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Each ConvOption maps (display_label, internal_key, callable).
# _conv_options_for(file) returns only the conversions that are actually
# possible for the given file extension.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass( frozen=True )
class ConvOption :
	"""One conversion the user can choose for a given file."""
	label: str  # shown on the button
	key: str  # internal identifier
	fn: str  # name of the conversion function in stages.conversion


# master catalogue — every conversion the pipeline supports
_ALL_CONV = [
	ConvOption( "🖼→PNG" , "img_png" , "convert_image_to_png" ) ,
	ConvOption( "🖼→PDF" , "img_pdf" , "convert_png_to_pdf" ) ,  # image→png first, then png→pdf
	ConvOption( "🎬→MP4" , "vid_mp4" , "convert_video_to_mp4" ) ,
	ConvOption( "🎵→MP3" , "aud_mp3" , "convert_audio_to_mp3" ) ,
	ConvOption( "📄→PDF" , "doc_pdf" , "convert_document_to_pdf" ) ,
	ConvOption( "📊→CSV" , "xls_csv" , "convert_xlsx_to_csv" ) ,
	ConvOption( "📧→PDF" , "eml_pdf" , "convert_email_to_pdf" ) ,
	ConvOption( "🌐→PDF" , "html_pdf" , "convert_html_to_pdf" ) ,
]

# extension sets (normalised without leading dot)
_IMG_EXTS = { e.lower( ) for e in IMAGE_TYPES }
_VID_EXTS = { e.lower( ) for e in VIDEO_TYPES }
_AUD_EXTS = { e.lower( ) for e in AUDIO_TYPES }
_DOC_EXTS = { e.lower( ) for e in DOCUMENT_TYPES }
_EML_EXTS = { e.lower( ) for e in EMAIL_TYPES }
_XLS_EXTS = { e.lower( ) for e in SPREADSHEET_TYPES }
_HTML_EXTS = { "html" , "htm" , "xhtml" }


def _conv_options_for( file: Path ) -> list[ ConvOption ] :
	"""Return only the conversions that are operable for *file*'s type."""
	ext = file.suffix.lower( ).lstrip( "." )
	if ext in _HTML_EXTS :
		return [ c for c in _ALL_CONV if c.key == "html_pdf" ]
	if ext in _IMG_EXTS :
		return [ c for c in _ALL_CONV if c.key in ("img_png" , "img_pdf") ]
	if ext in _VID_EXTS :
		return [ c for c in _ALL_CONV if c.key == "vid_mp4" ]
	if ext in _AUD_EXTS :
		return [ c for c in _ALL_CONV if c.key == "aud_mp3" ]
	if ext in _XLS_EXTS :
		return [ c for c in _ALL_CONV if c.key == "xls_csv" ]
	if ext in _DOC_EXTS :
		return [ c for c in _ALL_CONV if c.key == "doc_pdf" ]
	if ext in _EML_EXTS :
		return [ c for c in _ALL_CONV if c.key == "eml_pdf" ]
	return [ ]


# ══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _unique( directory: Path , file: Path ) -> Path :
	"""Return a non-colliding destination path inside *directory*."""
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
			min( 255 , int( r * 1.18 ) ) , min( 255 , int( g * 1.18 ) ) , min( 255 , int( b * 1.18 ) ) ,
	)


def _tika_metadata( file: Path ) -> str :
	"""Extract metadata via Apache Tika (best-effort)."""
	try :
		r = subprocess.run(
				[ JAVA_PATH , "-jar" , str( TIKA_APP_JAR_PATH ) , "-m" , str( file ) ] ,
				capture_output=True , text=True , timeout=30 ,
		)
		return r.stdout.strip( ) or "No metadata returned by Tika."
	except FileNotFoundError :
		return "Java or Tika JAR not found."
	except subprocess.TimeoutExpired :
		return "Tika extraction timed out."
	except Exception as e :
		return f"Tika error: {e}"


def _write_alteration_csv( filename: str , description: str ) -> None :
	"""Append a row to the alterations CSV log (creates header on first write)."""
	is_new = not ALTERATIONS_CSV.exists( )
	with open( ALTERATIONS_CSV , "a" , newline="" , encoding="utf-8" ) as fh :
		w = csv.writer( fh )
		if is_new :
			w.writerow( [ "timestamp" , "filename" , "description" ] )
		w.writerow( [ datetime.now( ).isoformat( ) , filename , description ] )
	logger.info( "Alteration logged: '%s'" , filename )


def _file_category( file: Path ) -> str :
	"""Classify a file into a preview category string."""
	ext = file.suffix.lower( ).lstrip( "." )
	if ext == "pdf" :       return "pdf"
	if ext in _IMG_EXTS :   return "image"
	if ext in _VID_EXTS :   return "video"
	if ext in _AUD_EXTS :   return "audio"
	if ext in _HTML_EXTS :  return "html"
	if ext in {
		"txt" , "log" , "json" , "csv" , "xml" , "md" , "yaml" , "yml" ,
		"toml" , "ini" , "py" , "js" , "ts" , "css" , "sh" , "bat" ,
	} :
		return "text"
	return "other"


# ══════════════════════════════════════════════════════════════════════════════
# ROTATION HELPERS (in-place)
# ══════════════════════════════════════════════════════════════════════════════


def _rotate_pdf( file: Path , degrees: int ) -> None :
	"""Rotate every page of a PDF by *degrees* (90/180/270) and save in-place."""
	doc = fitz.open( file )
	for page in doc :
		page.set_rotation( (page.rotation + degrees) % 360 )
	tmp = file.parent / f"_rot_{file.name}"
	doc.save( tmp );
	doc.close( )
	file.unlink( );
	tmp.rename( file )
	logger.info( "Rotated PDF %d°: '%s'" , degrees , file.name )


def _rotate_image( file: Path , degrees: int ) -> None :
	"""Rotate a PNG image by *degrees* (CW) and save in-place."""
	img = Image.open( file )
	rotated = img.rotate( -degrees , expand=True )  # PIL rotates CCW
	rotated.save( file , format="PNG" )
	logger.info( "Rotated image %d°: '%s'" , degrees , file.name )


def _rotate_video( file: Path , degrees: int ) -> None :
	"""Rotate an MP4 video by *degrees* (CW) in-place via ffmpeg."""
	vf = { 90 : "transpose=1" , 180 : "transpose=1,transpose=1" , 270 : "transpose=2" }.get( degrees )
	if not vf :
		raise ValueError( f"Unsupported rotation: {degrees}" )
	tmp = file.parent / f"_rot_{file.name}"
	try :
		subprocess.run(
				[ "ffmpeg" , "-y" , "-i" , str( file ) , "-vf" , vf ,
					"-c:a" , "copy" , "-metadata:s:v" , "rotate=0" , str( tmp ) ] ,
				capture_output=True , timeout=300 , check=True ,
		)
		file.unlink( );
		tmp.rename( file )
		logger.info( "Rotated video %d°: '%s'" , degrees , file.name )
	except FileNotFoundError :
		raise RuntimeError( "ffmpeg not found on PATH" )
	except subprocess.CalledProcessError as e :
		if tmp.exists( ) : tmp.unlink( )
		raise RuntimeError( f"ffmpeg error: {(e.stderr or b'')[ :300 ]}" )


# ══════════════════════════════════════════════════════════════════════════════
# FLUSH LOGIC — move / copy files to their target directories
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ══════════════════════════════════════════════════════════════════════════════


class FileTriage :
	BATCH_SIZE = 20
	MAX_PDF_PG = 10

	# ── initialisation ───────────────────────────────────────────────────────
	def __init__( self , source_dir: Path ) :
		self.source_dir = Path( source_dir )

		# file queue & history
		self.queue: deque[ Path ] = deque( )
		self.history: list[ list ] = [ ]  # [[file, [dest_names], conv_key|None], ...]
		self.current_index: int = -1
		self.current_file: Path | None = None

		# UI state
		self.selected_dests: list[ str ] = [ ]
		self.dest_widgets: dict[ str , ctk.CTkButton ] = { }
		self._bound_keys: list[ str ] = [ ]

		self.selected_conversion: str | None = None
		self.conv_widgets: dict[ str , ctk.CTkButton ] = { }

		# batch / flag
		self.pending: list[ tuple[ Path , list[ str ] ] ] = [ ]
		self.batch_n: int = 0
		self.flagged: set[ Path ] = set( )

		# media playback
		self.video_cap = None
		self.video_after_id = None
		self.video_lbl = None
		self.is_playing = False
		self.zoom_scale = 1.0

		# performance
		self._thumb_cache: dict[ str , ImageTk.PhotoImage | None ] = { }
		self.session_start = time.time( )
		self.file_start: float | None = None
		self.file_times: list[ float ] = [ ]

		# ensure every target directory exists
		for btn in DEST_REGISTRY :
			for p in btn.targets :
				p.mkdir( parents=True , exist_ok=True )
		for d in (ALTERATIONS_REQUIRED_DIR , UNESSENTIAL_DIR , DELETE_DIR ,
							UNSUPPORTED_ARTIFACTS_DIR) :
			d.mkdir( parents=True , exist_ok=True )

		self._load_files( )

		# build window
		self.root = ctk.CTk( )
		self.root.title( "File Triage  v5.1" )
		self.root.geometry( "1920x1060" )
		self.root.configure( fg_color=BG )

		self._build_ui( )
		self._bind_global_keys( )
		self._tick_timer( )

		if self.queue :
			self.show_next( )

		logger.info( "Application ready — %d files in queue" , len( self.queue ) )

	# ── file loading ─────────────────────────────────────────────────────────
	def _load_files( self ) :
		"""Load source_dir files into the queue, then append unsupported dir."""
		# primary source
		primary = [
			f for f in self.source_dir.iterdir( )
			if f.is_file( ) and f.parent == self.source_dir
		]
		self.queue.extend( primary )
		logger.info( "Loaded %d files from source directory" , len( primary ) )

		# append unsupported artifacts at the tail so they come last
		if UNSUPPORTED_ARTIFACTS_DIR.exists( ) and UNSUPPORTED_ARTIFACTS_DIR != self.source_dir :
			extras = [
				f for f in UNSUPPORTED_ARTIFACTS_DIR.iterdir( )
				if f.is_file( ) and f.parent == UNSUPPORTED_ARTIFACTS_DIR
			]
			if extras :
				self.queue.extend( extras )
				logger.info( "Appended %d files from UNSUPPORTED_ARTIFACTS" , len( extras ) )

	# ══════════════════════════════════════════════════════════════════════════
	# UI CONSTRUCTION
	# ══════════════════════════════════════════════════════════════════════════
	def _build_ui( self ) :
		main = ctk.CTkFrame( self.root , fg_color=BG )
		main.pack( fill=tk.BOTH , expand=True , padx=10 , pady=10 )

		# ── LEFT: history sidebar ────────────────────────────────────────────
		left = ctk.CTkFrame( main , width=290 , fg_color=SIDEBAR , corner_radius=14 )
		left.pack( side=tk.LEFT , fill=tk.Y , padx=(0 , 8) )
		left.pack_propagate( False )

		ctk.CTkLabel( left , text="HISTORY" ,
									font=(FONT , 12 , "bold") , text_color=ACCENT ).pack( pady=(14 , 6) )
		self.hist_frame = CTkScrollableFrame( left , fg_color=SIDEBAR )
		self.hist_frame.pack( fill=tk.BOTH , expand=True , padx=6 , pady=(0 , 8) )

		# ── CENTER ───────────────────────────────────────────────────────────
		center = ctk.CTkFrame( main , fg_color=BG )
		center.pack( side=tk.LEFT , fill=tk.BOTH , expand=True )

		# info bar
		info = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 , height=46 )
		info.pack( fill=tk.X , pady=(0 , 5) );
		info.pack_propagate( False )

		self.file_lbl = ctk.CTkLabel( info , text="—" ,
																	font=(FONT , 13 , "bold") , text_color=FG )
		self.file_lbl.pack( side=tk.LEFT , padx=14 )

		self.action_lbl = ctk.CTkLabel( info , text="" ,
																		font=(FONT , 11 , "bold") , text_color=ACCENT )
		self.action_lbl.pack( side=tk.LEFT , padx=8 )

		self.queue_lbl = ctk.CTkLabel( info , text="Queue: 0" ,
																	 font=(FONT , 11) , text_color=FG2 )
		self.queue_lbl.pack( side=tk.RIGHT , padx=14 )

		# preview outer frame
		prev_outer = ctk.CTkFrame( center , fg_color=PREVIEW , corner_radius=14 )
		prev_outer.pack( fill=tk.BOTH , expand=True , pady=(0 , 5) )

		# toolbar
		tb = ctk.CTkFrame( prev_outer , fg_color=PREVIEW , height=44 )
		tb.pack( fill=tk.X , padx=10 , pady=(8 , 0) );
		tb.pack_propagate( False )

		# rotation buttons
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

		self.zoom_lbl = ctk.CTkLabel( tb , text="🔍 100%" ,
																	font=(EMOJI_FONT , 10) , text_color=FG2 )
		self.zoom_lbl.pack( side=tk.LEFT , padx=8 )

		# right toolbar actions
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

		# scrollable preview area
		self.prev_scroll = CTkScrollableFrame( prev_outer , fg_color=PREVIEW )
		self.prev_scroll.pack( fill=tk.BOTH , expand=True , padx=6 , pady=6 )
		self.prev_scroll.bind( "<MouseWheel>" , self._zoom )
		self.prev_scroll._parent_canvas.bind( "<MouseWheel>" , self._zoom )

		# ── conversion panel ─────────────────────────────────────────────────
		conv_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		conv_panel.pack( fill=tk.X , pady=(0 , 5) )

		conv_hdr = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		conv_hdr.pack( fill=tk.X , padx=14 , pady=(8 , 4) )

		ctk.CTkLabel( conv_hdr ,
									text="CONVERT  (optional · single select · original archived first)" ,
									font=(FONT , 10 , "bold") , text_color=FG2 ).pack( side=tk.LEFT )

		ctk.CTkButton( conv_hdr , text="✕ clear" , command=self._clear_conversion ,
									 fg_color=CARD , hover_color=CARD_SEL ,
									 width=70 , height=24 , corner_radius=6 ,
									 text_color=FG2 , font=(FONT , 9) ).pack( side=tk.LEFT , padx=10 )

		self.conv_row = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		self.conv_row.pack( fill=tk.X , padx=10 , pady=(0 , 10) )

		# placeholder label shown when no conversions are available
		self.no_conv_lbl = ctk.CTkLabel(
				self.conv_row , text="No conversions available for this file type" ,
				font=(FONT , 10) , text_color=FG2 ,
		)

		# ── destination grid ─────────────────────────────────────────────────
		dest_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		dest_panel.pack( fill=tk.X , pady=(0 , 5) )

		ctk.CTkLabel( dest_panel ,
									text="TRIAGE DESTINATIONS  (click or key · multi-select)" ,
									font=(FONT , 10 , "bold") , text_color=FG2 ,
									).pack( anchor="w" , padx=14 , pady=(8 , 4) )

		self.dest_grid = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		self.dest_grid.pack( fill=tk.X , padx=10 , pady=(0 , 10) )

		# ── footer ───────────────────────────────────────────────────────────
		foot = ctk.CTkFrame( center , fg_color=BG , height=52 )
		foot.pack( fill=tk.X );
		foot.pack_propagate( False )

		self.timer_lbl = ctk.CTkLabel( foot , text="⏱ 0s" ,
																	 font=(FONT , 11 , "bold") , text_color=YELLOW )
		self.timer_lbl.pack( side=tk.LEFT , padx=10 )

		self.status_lbl = ctk.CTkLabel( foot , text="" ,
																		font=(FONT , 11) , text_color=ACCENT )
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

	# ── global key bindings (always active) ──────────────────────────────────
	def _bind_global_keys( self ) :
		self.root.bind( "<Left>" , lambda _ : self.show_previous( ) )
		self.root.bind( "<Return>" , lambda _ : self._confirm( ) )
		self.root.bind( "<Delete>" , lambda _ : self._delete_file( ) )
		self.root.bind( "<Up>" , lambda _ : self._flag_with_dialog( ) )
		self.root.bind( "<Down>" , lambda _ : self._send_to_unessential( ) )

	# ══════════════════════════════════════════════════════════════════════════
	# ZOOM
	# ══════════════════════════════════════════════════════════════════════════
	def _zoom( self , event ) :
		self.zoom_scale = max( 0.2 , min( 4.0 ,
																			self.zoom_scale + (0.1 if event.delta > 0 else -0.1) ) )
		self.zoom_lbl.configure( text=f"🔍 {int( self.zoom_scale * 100 )}%" )
		self._display_file( )

	# ══════════════════════════════════════════════════════════════════════════
	# ROTATION
	# ══════════════════════════════════════════════════════════════════════════
	def _rotate_current( self , degrees: int ) :
		if not self.current_file or not self.current_file.exists( ) :
			return
		cat = _file_category( self.current_file )
		self.status_lbl.configure( text="Rotating…" , text_color=YELLOW )
		self.root.update_idletasks( )
		try :
			if cat == "pdf" :   _rotate_pdf( self.current_file , degrees )
			elif cat == "image" : _rotate_image( self.current_file , degrees )
			elif cat == "video" : _rotate_video( self.current_file , degrees )
			else :
				self.status_lbl.configure( text="Rotation not supported" , text_color=FG2 )
				return
			self.status_lbl.configure( text=f"✓ Rotated {degrees}°" , text_color=GREEN )
			self._thumb_cache.pop( str( self.current_file ) , None )
			self._display_file( )
		except Exception as e :
			self.status_lbl.configure( text="✗ Rotation failed" , text_color=RED )
			logger.error( "Rotation failed for '%s': %s" , self.current_file.name , e )
			messagebox.showerror( "Rotation Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# CONVERSION PANEL  (rebuilt per file, context-aware)
	# ══════════════════════════════════════════════════════════════════════════
	def _rebuild_conv_buttons( self ) :
		for w in self.conv_row.winfo_children( ) :
			w.destroy( )
		self.conv_widgets.clear( )
		self.no_conv_lbl.pack_forget( )

		if not self.current_file :
			return

		options = _conv_options_for( self.current_file )

		if not options :
			# show placeholder
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

		# restore previously selected conversion (if navigating back)
		self._clear_conversion( )
		if 0 <= self.current_index < len( self.history ) :
			saved = self.history[ self.current_index ][ 2 ]
			if saved and saved in self.conv_widgets :
				self._toggle_conversion( saved )

	def _toggle_conversion( self , key: str ) :
		if self.selected_conversion == key :
			self._clear_conversion( )
			return
		# deselect previous
		if self.selected_conversion and self.selected_conversion in self.conv_widgets :
			self.conv_widgets[ self.selected_conversion ].configure(
					fg_color=CARD , text_color=FG2 )
		self.selected_conversion = key
		if key in self.conv_widgets :
			self.conv_widgets[ key ].configure( fg_color=YELLOW , text_color=BG )
		self._refresh_action_label( )

	def _clear_conversion( self ) :
		if self.selected_conversion and self.selected_conversion in self.conv_widgets :
			self.conv_widgets[ self.selected_conversion ].configure(
					fg_color=CARD , text_color=FG2 )
		self.selected_conversion = None
		self._refresh_action_label( )

	# ══════════════════════════════════════════════════════════════════════════
	# DESTINATION BUTTONS  (driven by DEST_REGISTRY)
	# ══════════════════════════════════════════════════════════════════════════
	def _render_dest_buttons( self ) :
		# unbind previous per-destination hotkeys
		for k in self._bound_keys :
			try : self.root.unbind( k )
			except Exception : pass
		self._bound_keys.clear( )

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

			# bind keyboard shortcut
			self.root.bind( db.key , lambda _ , n=db.name : self._toggle_dest( n ) )
			self._bound_keys.append( db.key )

		# restore saved selections when navigating back
		if 0 <= self.current_index < len( self.history ) :
			for name in self.history[ self.current_index ][ 1 ] :
				if name in self.dest_widgets :
					self.selected_dests.append( name )
					self.dest_widgets[ name ].configure( fg_color=ACCENT , text_color=FG )

		self._rebuild_conv_buttons( )
		self._refresh_action_label( )

	def _toggle_dest( self , name: str ) :
		btn = self.dest_widgets.get( name )
		if not btn :
			return
		if name in self.selected_dests :
			self.selected_dests.remove( name )
			btn.configure( fg_color=CARD , text_color=FG2 )
		else :
			self.selected_dests.append( name )
			btn.configure( fg_color=ACCENT , text_color=FG )
		self._refresh_action_label( )

	def _refresh_action_label( self ) :
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
	# CONFIRM + SILENT BATCH FLUSH
	# ══════════════════════════════════════════════════════════════════════════
	def _confirm( self ) :
		if not self.current_file :
			return

		# persist current selections into history
		if 0 <= self.current_index < len( self.history ) :
			self.history[ self.current_index ][ 1 ] = list( self.selected_dests )
			self.history[ self.current_index ][ 2 ] = self.selected_conversion

		# run conversion (archives original first)
		if self.selected_conversion and CONVERSION_AVAILABLE :
			self._run_conversion( self.selected_conversion )

		if self.selected_dests :
			self.pending.append( (self.current_file , list( self.selected_dests )) )
			self.batch_n += 1
			if self.batch_n >= self.BATCH_SIZE :
				self._flush_silent( )

		logger.info( "Confirmed '%s' → dests=%s conv=%s" ,
								 self.current_file.name , self.selected_dests , self.selected_conversion )
		self.show_next( )

	def _run_conversion( self , key: str ) :
		"""Archive the original, then run the selected conversion in-place."""
		if not self.current_file or not self.current_file.exists( ) :
			return

		# archive a safety copy
		try :
			ARCHIVAL_DIR.mkdir( parents=True , exist_ok=True )
			archive_dst = _unique( ARCHIVAL_DIR , self.current_file )
			shutil.copy2( str( self.current_file ) , str( archive_dst ) )
			logger.info( "Archived original → '%s'" , archive_dst.name )
		except Exception as e :
			logger.error( "Archive failed before conversion: %s" , e )
			messagebox.showerror( "Archive Error" ,
														f"Could not archive before conversion:\n{e}\n\nConversion aborted." )
			return

		self.status_lbl.configure( text="Converting…" , text_color=YELLOW )
		self.root.update_idletasks( )

		# special case: img_pdf requires two steps (image→png then png→pdf)
		if key == "img_pdf" :
			try :
				png_path = convert_image_to_png( src=self.current_file , logger=logger )
				if not png_path :
					raise RuntimeError( "image→PNG step failed" )
				pdf_path = convert_png_to_pdf( src=png_path , logger=logger )
				if not pdf_path :
					raise RuntimeError( "PNG→PDF step failed" )
				self.current_file = pdf_path
				if 0 <= self.current_index < len( self.history ) :
					self.history[ self.current_index ][ 0 ] = pdf_path
				self.status_lbl.configure(
						text="✓ Converted → PDF (original archived)" , text_color=GREEN )
				logger.info( "Two-step conversion complete: image → PNG → PDF" )
			except Exception as e :
				self.status_lbl.configure( text="✗ Conversion failed" , text_color=RED )
				logger.error( "Image→PDF conversion failed: %s" , e )
				messagebox.showerror( "Conversion Error" , str( e ) )
			return

		# look up the conversion function by name
		fn_map: dict[ str , Callable ] = {
			"img_png"  : convert_image_to_png ,
			"vid_mp4"  : convert_video_to_mp4 ,
			"aud_mp3"  : convert_audio_to_mp3 ,
			"doc_pdf"  : convert_document_to_pdf ,
			"xls_csv"  : convert_xlsx_to_csv ,
			"eml_pdf"  : convert_email_to_pdf ,
			"html_pdf" : convert_html_to_pdf ,
		}
		fn = fn_map.get( key )
		if fn is None :
			self.status_lbl.configure( text="" , text_color=FG2 )
			return

		try :
			new_path: Path = fn( src=self.current_file , logger=logger )
			self.current_file = new_path
			if 0 <= self.current_index < len( self.history ) :
				self.history[ self.current_index ][ 0 ] = new_path
			ext = new_path.suffix.upper( ).lstrip( "." )
			self.status_lbl.configure(
					text=f"✓ Converted → {ext} (original archived)" , text_color=GREEN )
			logger.info( "Conversion complete → '%s'" , new_path.name )
		except Exception as e :
			self.status_lbl.configure( text="✗ Conversion failed" , text_color=RED )
			logger.error( "Conversion failed for '%s': %s" , self.current_file.name , e )
			messagebox.showerror( "Conversion Error" , str( e ) )

	def _flush_silent( self ) :
		"""Execute all pending file operations silently, then reset batch state.

        For each pending item:
          - Single destination → move the file.
          - Multiple destinations → copy to each target, then delete original.
        On any move failure, the file is dumped to UNSUPPORTED_ARTIFACTS_DIR."""
		if not self.pending :
			return
		total = len( self.pending )
		errors: list[ str ] = [ ]

		for file , dest_names in self.pending :
			if not file.exists( ) :
				continue

			# collect ALL target directories from every selected destination
			all_targets: list[ tuple[ Path , str ] ] = [ ]
			for dname in dest_names :
				db = DEST_BY_NAME.get( dname )
				if not db :
					logger.warning( "Unknown dest '%s' — skipping" , dname )
					continue
				for tgt in db.targets :
					all_targets.append( (tgt , dname) )

			if not all_targets :
				continue

			if len( all_targets ) == 1 :
				# single target → move
				dst_dir , label = all_targets[ 0 ]
				try :
					dst_dir.mkdir( parents=True , exist_ok=True )
					shutil.move( str( file ) , str( _unique( dst_dir , file ) ) )
					logger.info( "Moved '%s' → %s" , file.name , label )
				except Exception as e :
					errors.append( f"move {file.name}→{label}: {e}" )
					try :
						shutil.move( str( file ) , str( _unique( UNSUPPORTED_ARTIFACTS_DIR , file ) ) )
						logger.warning( "Fallback: '%s' → UNSUPPORTED" , file.name )
					except Exception as e2 :
						errors.append( f"fallback {file.name}: {e2}" )
			else :
				# multiple targets → copy to each, then remove original
				copy_ok = True
				for dst_dir , label in all_targets :
					try :
						dst_dir.mkdir( parents=True , exist_ok=True )
						shutil.copy2( str( file ) , str( _unique( dst_dir , file ) ) )
					except Exception as e :
						errors.append( f"copy {file.name}→{label}: {e}" )
						copy_ok = False
				if copy_ok and file.exists( ) :
					try :
						file.unlink( )
						logger.info( "Copied '%s' → %d targets, original removed" ,
												 file.name , len( all_targets ) )
					except Exception as e :
						errors.append( f"delete original {file.name}: {e}" )

		if errors :
			logger.warning( "Flush had %d error(s): %s" , len( errors ) , errors[ :5 ] )

		self.pending.clear( )
		self.batch_n = 0
		self.history.clear( )
		self.current_index = -1
		self._thumb_cache.clear( )
		self._refresh_history_sidebar( )
		logger.info( "Batch flushed: %d actions" , total )
		self.status_lbl.configure( text=f"✓ Batch flushed ({total})" , text_color=GREEN )

	# ══════════════════════════════════════════════════════════════════════════
	# FLAG → ALTERATIONS_REQUIRED  (Up arrow)
	# ══════════════════════════════════════════════════════════════════════════
	def _flag_with_dialog( self ) :
		if not self.current_file or not self.current_file.exists( ) :
			return

		# run any pending conversion BEFORE moving
		if self.selected_conversion and CONVERSION_AVAILABLE :
			self._run_conversion( self.selected_conversion )

		description = simpledialog.askstring(
				"Flag for Review" ,
				f"File: {self.current_file.name}\n\n"
				"Enter notes / directions for this file\n"
				"(leave blank for no description):" ,
				parent=self.root ,
		)
		if description is None :
			return  # user cancelled

		try :
			dst = _unique( ALTERATIONS_REQUIRED_DIR , self.current_file )
			shutil.move( str( self.current_file ) , str( dst ) )
			self.flagged.add( dst )
			_write_alteration_csv( dst.name , description.strip( ) )
			self.status_lbl.configure( text="🚩 Flagged → ALTERATIONS REQUIRED" ,
																 text_color=ACCENT )
			logger.info( "Flagged '%s' → ALTERATIONS_REQUIRED" , dst.name )
			self._refresh_history_sidebar( )
			self.show_next( )
		except Exception as e :
			logger.error( "Flag failed for '%s': %s" , self.current_file.name , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# UNESSENTIAL  (Down arrow)
	# ══════════════════════════════════════════════════════════════════════════
	def _send_to_unessential( self ) :
		if not self.current_file or not self.current_file.exists( ) :
			return
		try :
			UNESSENTIAL_DIR.mkdir( parents=True , exist_ok=True )
			dst = _unique( UNESSENTIAL_DIR , self.current_file )
			shutil.move( str( self.current_file ) , str( dst ) )
			self.status_lbl.configure( text="↓ Moved to UNESSENTIAL" , text_color=FG2 )
			logger.info( "Sent '%s' → UNESSENTIAL" , self.current_file.name )
			self.show_next( )
		except Exception as e :
			logger.error( "Unessential move failed for '%s': %s" , self.current_file.name , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# DELETE
	# ══════════════════════════════════════════════════════════════════════════
	def _delete_file( self ) :
		if not self.current_file or not self.current_file.exists( ) :
			return
		if not messagebox.askyesno( "Confirm Delete" ,
																f"Move '{self.current_file.name}' to DELETE folder?" ) :
			return
		try :
			shutil.move( str( self.current_file ) , str( _unique( DELETE_DIR , self.current_file ) ) )
			self.status_lbl.configure( text="🗑 Deleted" , text_color=RED )
			logger.info( "Deleted '%s' → DELETE" , self.current_file.name )
			self.show_next( )
		except Exception as e :
			logger.error( "Delete failed for '%s': %s" , self.current_file.name , e )
			messagebox.showerror( "Error" , f"Delete failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════════
	# NAVIGATION
	# ══════════════════════════════════════════════════════════════════════════
	def show_next( self ) :
		if self.file_start :
			self.file_times.append( time.time( ) - self.file_start )
		self._stop_video( )

		if self.current_index < len( self.history ) - 1 :
			# re-visiting a file we already saw
			self.current_index += 1
			self.current_file = self.history[ self.current_index ][ 0 ]
		elif self.queue :
			# new file from the queue
			self.current_file = self.queue.popleft( )
			self.history.append( [ self.current_file , [ ] , None ] )
			self.current_index = len( self.history ) - 1
			self.file_start = time.time( )
		else :
			# queue exhausted
			self.current_file = None
			self._clear_preview( )
			self._update_info_bar( )
			self.status_lbl.configure( text="✓ All files reviewed!" , text_color=GREEN )
			self._render_dest_buttons( )
			logger.info( "Queue exhausted — all files reviewed" )
			return

		self._render_dest_buttons( )
		self._display_file( )
		self._update_info_bar( )
		self._refresh_history_sidebar( )

	def show_previous( self ) :
		self._stop_video( )
		if self.current_index > 0 :
			self.current_index -= 1
			self.current_file = self.history[ self.current_index ][ 0 ]
			self._render_dest_buttons( )
			self._display_file( )
			self._update_info_bar( )
			self._refresh_history_sidebar( )

	def _jump_to( self , idx: int ) :
		if 0 <= idx < len( self.history ) :
			self._stop_video( )
			self.current_index = idx
			self.current_file = self.history[ idx ][ 0 ]
			self._render_dest_buttons( )
			self._display_file( )
			self._update_info_bar( )
			self._refresh_history_sidebar( )

	def _update_info_bar( self ) :
		if self.current_file :
			total = len( self.history ) + len( self.queue )
			self.file_lbl.configure(
					text=f"{self.current_file.name}  ({self.current_index + 1}/{total})" )
		else :
			self.file_lbl.configure( text="—" )
		self.queue_lbl.configure(
				text=f"Queue: {len( self.queue )}  |  Flagged: {len( self.flagged )}" )

	# ══════════════════════════════════════════════════════════════════════════
	# TIMER
	# ══════════════════════════════════════════════════════════════════════════
	def _tick_timer( self ) :
		elapsed = int( time.time( ) - self.session_start )
		avg = sum( self.file_times ) / len( self.file_times ) if self.file_times else 0
		rem = int( len( self.queue ) * avg / 60 ) if avg > 0 else 0
		self.timer_lbl.configure( text=f"⏱ {elapsed}s  ·  avg {avg:.1f}s  ·  ~{rem}m left" )
		self.root.after( 1000 , self._tick_timer )

	# ══════════════════════════════════════════════════════════════════════════
	# HISTORY SIDEBAR
	# ══════════════════════════════════════════════════════════════════════════
	def _refresh_history_sidebar( self ) :
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

			# thumbnail
			thumb = self._thumbnail( file )
			if thumb :
				lbl = tk.Label( frame , image=thumb , bg=bg , cursor="hand2" )
				lbl.image = thumb
				lbl.pack( pady=(6 , 2) , padx=6 )
				lbl.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )

			# file name
			name = (file.name[ :29 ] + "…") if len( file.name ) > 30 else file.name
			ctk.CTkLabel(
					frame ,
					text=("🚩 " if is_flag else "") + name ,
					font=(FONT , 10 , "bold" if is_curr else "normal") ,
					text_color=ACCENT if is_flag else FG ,
					wraplength=252 , justify="left" ,
			).pack( fill=tk.X , padx=8 , pady=(0 , 2) )

			# action summary
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

			frame.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )

	# ══════════════════════════════════════════════════════════════════════════
	# PREVIEW DISPATCHER
	# ══════════════════════════════════════════════════════════════════════════
	def _clear_preview( self ) :
		for w in self.prev_scroll.winfo_children( ) :
			w.destroy( )

	def _display_file( self ) :
		self._clear_preview( )
		if not self.current_file or not self.current_file.exists( ) :
			return
		cat = _file_category( self.current_file )
		try :
			dispatch = {
				"pdf"   : self._show_pdf , "image" : self._show_image ,
				"video" : self._show_video , "audio" : self._show_audio ,
				"html"  : self._show_html , "text" : self._show_text ,
			}
			dispatch.get( cat , self._show_metadata )( )
		except Exception as e :
			logger.error( "Preview failed for '%s': %s" , self.current_file.name , e )
			self._show_metadata( err=str( e ) )

	# ── metadata (Tika fallback) ─────────────────────────────────────────────
	def _show_metadata( self , err: str = "" ) :
		ctk.CTkLabel(
				self.prev_scroll , text=f"📋  {self.current_file.name}" ,
				font=(EMOJI_FONT , 13 , "bold") , text_color=ACCENT ,
		).pack( anchor="w" , padx=16 , pady=(12 , 4) )
		if err :
			ctk.CTkLabel( self.prev_scroll , text=f"⚠ {err}" ,
										font=(FONT , 10) , text_color=YELLOW ,
										).pack( anchor="w" , padx=16 , pady=(0 , 8) )
		self.status_lbl.configure( text="Extracting metadata…" , text_color=FG2 )
		self.root.update_idletasks( )
		meta = _tika_metadata( self.current_file )
		self.status_lbl.configure( text="" , text_color=FG2 )
		box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
													font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=14 , pady=8 )
		box.insert( "1.0" , meta )
		box.configure( state="disabled" )

	# ── HTML (tkinterweb or raw source) ──────────────────────────────────────
	def _show_html( self ) :
		html_content = self.current_file.read_text( encoding="utf-8" , errors="ignore" )
		ctk.CTkLabel(
				self.prev_scroll , text=f"🌐  {self.current_file.name}" ,
				font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ,
		).pack( anchor="w" , padx=14 , pady=(8 , 4) )

		if TKINTERWEB_AVAILABLE :
			container = ctk.CTkFrame( self.prev_scroll , fg_color=CARD , corner_radius=10 )
			container.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )
			try :
				hframe = HtmlFrame( container , horizontal_scrollbar="auto" )
				hframe.pack( fill=tk.BOTH , expand=True )
				hframe.load_html( html_content )
				hframe.set_zoom( self.zoom_scale )
				return
			except Exception as e :
				logger.warning( "tkinterweb render failed: %s" , e )
				for w in container.winfo_children( ) :
					w.destroy( )

		# fallback: show raw HTML source
		ctk.CTkLabel( self.prev_scroll ,
									text="ℹ  pip install tkinterweb  for rendered preview" ,
									font=(FONT , 9) , text_color=YELLOW ,
									).pack( anchor="w" , padx=16 , pady=(0 , 6) )
		box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
													font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )
		box.insert( "1.0" , html_content )
		box.configure( state="disabled" )

	# ── PDF (max 10 pages) ───────────────────────────────────────────────────
	def _show_pdf( self ) :
		if not PIL_AVAILABLE :
			self._show_metadata( err="PIL / PyMuPDF not installed" )
			return

		doc = fitz.open( self.current_file )
		total = len( doc )
		show = min( total , self.MAX_PDF_PG )
		truncated = total > self.MAX_PDF_PG

		ctk.CTkLabel(
				self.prev_scroll ,
				text=(f"PDF  ·  {total} pages  ·  showing first {show}"
							if truncated else f"PDF  ·  {total} page(s)") ,
				font=(FONT , 11 , "bold") , text_color=ACCENT ,
		).pack( pady=(10 , 4) )

		if truncated :
			ctk.CTkLabel( self.prev_scroll ,
										text=f"⚠ Showing {show} of {total} — Open External for all." ,
										font=(FONT , 9) , text_color=YELLOW ).pack( pady=(0 , 6) )

		grid = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		grid.pack( fill=tk.BOTH , expand=True , padx=14 )
		cols = min( 4 , show )
		for c in range( cols ) :
			grid.grid_columnconfigure( c , weight=1 )

		scale = 0.28 * self.zoom_scale
		for i in range( show ) :
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

		doc.close( )

	# ── image ────────────────────────────────────────────────────────────────
	def _show_image( self ) :
		if not PIL_AVAILABLE :
			self._show_metadata( err="PIL not installed" )
			return
		img = Image.open( self.current_file )
		ow , oh = img.size
		ctk.CTkLabel( self.prev_scroll ,
									text=f"{ow}×{oh}  ·  {self.current_file.suffix.upper( ).lstrip( '.' )}" ,
									font=(FONT , 10) , text_color=FG2 ,
									).pack( anchor="w" , padx=14 , pady=(8 , 2) )
		nw , nh = int( ow * self.zoom_scale ) , int( oh * self.zoom_scale )
		if nw > 1400 :
			nh = int( nh * 1400 / nw );
			nw = 1400
		img = img.resize( (nw , nh) , Image.Resampling.LANCZOS )
		photo = ImageTk.PhotoImage( img )
		lbl = tk.Label( self.prev_scroll , image=photo , bg=PREVIEW )
		lbl.image = photo
		lbl.pack( pady=10 )

	# ── video ────────────────────────────────────────────────────────────────
	def _show_video( self ) :
		if not CV2_AVAILABLE :
			self._show_metadata( err="opencv-python not installed" )
			return
		self.video_cap = cv2.VideoCapture( str( self.current_file ) )
		cap = self.video_cap
		fps = cap.get( cv2.CAP_PROP_FPS ) or 25
		nfrm = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
		dur = nfrm / fps if fps else 0
		vw = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
		vh = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )

		ctk.CTkLabel(
				self.prev_scroll ,
				text=f"🎬  {self.current_file.name}  ·  {vw}×{vh}  ·  {dur:.1f}s  ·  {fps:.0f}fps" ,
				font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ,
		).pack( pady=(10 , 4) )
		ctk.CTkLabel( self.prev_scroll ,
									text="⚠ No audio — Open External for full playback" ,
									font=(FONT , 9) , text_color=YELLOW ).pack( )

		self.video_lbl = tk.Label( self.prev_scroll , bg="#000000" )
		self.video_lbl.pack( pady=8 )

		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		ctrl.pack( )

		def toggle( ) :
			self.is_playing = not self.is_playing
			play_btn.configure( text="⏸ Pause" if self.is_playing else "▶ Play" )
			if self.is_playing :
				_next( )

		play_btn = ctk.CTkButton( ctrl , text="▶ Play" , command=toggle ,
															fg_color=BLUE , hover_color=_hover( BLUE ) ,
															width=90 , height=34 , corner_radius=8 , text_color=FG )
		play_btn.pack( side=tk.LEFT , padx=4 )

		def _next( ) :
			if not self.is_playing or not self.video_cap :
				return
			ret , frame = self.video_cap.read( )
			if ret :
				frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
				h2 , w2 = frame.shape[ :2 ]
				nw2 = int( w2 * self.zoom_scale * 0.55 )
				nh2 = int( h2 * self.zoom_scale * 0.55 )
				frame = cv2.resize( frame , (nw2 , nh2) )
				photo = ImageTk.PhotoImage( Image.fromarray( frame ) )
				self.video_lbl.configure( image=photo )
				self.video_lbl.image = photo
				self.video_after_id = self.root.after( max( 1 , int( 1000 / fps ) ) , _next )
			else :
				self.video_cap.set( cv2.CAP_PROP_POS_FRAMES , 0 )
				self.is_playing = False
				play_btn.configure( text="▶ Play" )

		self.is_playing = True
		play_btn.configure( text="⏸ Pause" )
		_next( )

	# ── audio ────────────────────────────────────────────────────────────────
	def _show_audio( self ) :
		ctk.CTkLabel( self.prev_scroll , text=f"🎵  {self.current_file.name}" ,
									font=(EMOJI_FONT , 14 , "bold") , text_color=ACCENT ,
									).pack( pady=(28 , 8) )
		meta = _tika_metadata( self.current_file )
		box = ctk.CTkTextbox( self.prev_scroll , height=120 ,
													fg_color=CARD , text_color=FG2 ,
													font=(FONT , 10) , corner_radius=8 )
		box.pack( fill=tk.X , padx=20 , pady=8 )
		box.insert( "1.0" , meta or "—" )
		box.configure( state="disabled" )

		if not PYGAME_AVAILABLE :
			ctk.CTkLabel( self.prev_scroll ,
										text="pip install pygame  for in-app playback" ,
										font=(FONT , 10) , text_color=YELLOW ).pack( )
			return

		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		ctrl.pack( pady=10 )

		def play( ) :
			try :
				pygame.mixer.music.load( str( self.current_file ) )
				pygame.mixer.music.play( )
			except Exception as e :
				messagebox.showerror( "Playback Error" , str( e ) )

		for txt , cmd , col in [
			("▶ Play" , play , GREEN) ,
			("⏸ Pause" , pygame.mixer.music.pause , YELLOW) ,
			("⏹ Stop" , pygame.mixer.music.stop , ACCENT) ,
		] :
			ctk.CTkButton( ctrl , text=txt , command=cmd ,
										 fg_color=col , hover_color=_hover( col ) ,
										 width=90 , height=36 , corner_radius=8 ,
										 text_color=FG , font=(FONT , 11) ).pack( side=tk.LEFT , padx=4 )
		self.root.after( 300 , play )

	# ── text / code ──────────────────────────────────────────────────────────
	def _show_text( self ) :
		try :
			content = self.current_file.read_text( encoding="utf-8" , errors="ignore" )
			lines = len( content.splitlines( ) )
			ctk.CTkLabel( self.prev_scroll ,
										text=f"{lines} lines  ·  {self.current_file.suffix.upper( ).lstrip( '.' )}" ,
										font=(FONT , 10) , text_color=FG2 ,
										).pack( anchor="w" , padx=14 , pady=(8 , 2) )
			box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
														font=("Consolas" , max( 8 , int( 10 * self.zoom_scale ) )) ,
														corner_radius=10 )
			box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=8 )
			box.insert( "1.0" , content )
			box.configure( state="disabled" )
		except Exception as e :
			self._show_metadata( err=str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# OPEN EXTERNAL  (system default for the file type)
	# ══════════════════════════════════════════════════════════════════════════
	def _open_external( self ) :
		if not self.current_file or not self.current_file.exists( ) :
			return
		try :
			if os.name == "nt" :
				os.startfile( self.current_file )
			elif os.name == "posix" :
				subprocess.run( [ "xdg-open" , str( self.current_file ) ] )
			else :
				subprocess.run( [ "open" , str( self.current_file ) ] )
			logger.info( "Opened external: '%s'" , self.current_file.name )
		except Exception as e :
			logger.error( "External open failed: %s" , e )
			messagebox.showerror( "Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════════
	# UTILITIES
	# ══════════════════════════════════════════════════════════════════════════
	def _stop_video( self ) :
		self.is_playing = False
		if self.video_after_id :
			self.root.after_cancel( self.video_after_id )
			self.video_after_id = None
		if self.video_cap :
			try : self.video_cap.release( )
			except Exception : pass
			self.video_cap = None

	def _thumbnail( self , file: Path , size=(260 , 100) ) :
		"""Generate a small preview thumbnail (cached)."""
		key = str( file )
		if key in self._thumb_cache :
			return self._thumb_cache[ key ]

		if not file.exists( ) :
			return None
		ext = file.suffix.lower( )
		thumb = None
		try :
			if ext == ".pdf" and PIL_AVAILABLE :
				doc = fitz.open( file )
				if doc :
					pix = doc[ 0 ].get_pixmap( matrix=fitz.Matrix( 0.2 , 0.2 ) )
					img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
					img.thumbnail( size , Image.Resampling.LANCZOS )
					doc.close( )
					thumb = ImageTk.PhotoImage( img )
			elif ext in { ".jpg" , ".jpeg" , ".png" , ".gif" , ".bmp" , ".webp" } and PIL_AVAILABLE :
				img = Image.open( file )
				img.thumbnail( size , Image.Resampling.LANCZOS )
				thumb = ImageTk.PhotoImage( img )
			elif ext in { ".mp4" , ".avi" , ".mkv" , ".mov" } and CV2_AVAILABLE :
				cap = cv2.VideoCapture( str( file ) )
				ret , frame = cap.read( )
				cap.release( )
				if ret :
					img = Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
					img.thumbnail( size , Image.Resampling.LANCZOS )
					thumb = ImageTk.PhotoImage( img )
			elif ext in { ".docx" , ".pptx" , ".xlsx" } and PIL_AVAILABLE :
				with ZipFile( file , "r" ) as z :
					thumbs = [ n for n in z.namelist( ) if "thumbnail" in n.lower( ) ]
					if thumbs and thumbs[ 0 ].lower( ).endswith( (".jpg" , ".jpeg" , ".png") ) :
						img = Image.open( io.BytesIO( z.read( thumbs[ 0 ] ) ) )
						img.thumbnail( size , Image.Resampling.LANCZOS )
						thumb = ImageTk.PhotoImage( img )
		except Exception as e :
			logger.debug( "Thumbnail failed for '%s': %s" , file.name , e )

		self._thumb_cache[ key ] = thumb
		return thumb

	def run( self ) :
		self.root.mainloop( )
