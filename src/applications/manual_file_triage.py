from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
import time
import tkinter as tk
import tkinter.font as _tkf
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass , field
from datetime import datetime
from pathlib import Path
from tkinter import messagebox , simpledialog
from typing import Callable
from zipfile import ZipFile

import customtkinter as ctk
import cv2
import fitz
import pygame
from PIL import Image
from customtkinter import CTkScrollableFrame
from tkinterweb import HtmlFrame

from config import (AFFINE_DIR , ALTERATIONS_CSV , ALTERATIONS_REQUIRED_DIR , ANKI_DIR , ARCHIVAL_DIR , AUDIO_TYPES ,
										BITWARDEN_DIR , DELETE_DIR , DIGITAL_ASSET_MANAGEMENT_DIR , DOCUMENT_TYPES , EMAIL_TYPES ,
										FILE_TRIAGE_BATCH_SIZE , FILE_TRIAGE_MAX_PDF_PG , FIREFLYIII_DIR , GAMES_ARCHIVE_DIR , GITLAB_DIR ,
										IMAGE_TYPES , IMMICH_DIR , JAVA_PATH , JELLYFIN_DIR , LINKWARDEN_DIR , MANUALS_ARCHIVE_DIR ,
										MONICA_CRM_DIR , ODOO_CRM_DIR , ODOO_INVENTORY_DIR , ODOO_MAINTENANCE_DIR , ODOO_PLM_DIR ,
										ODOO_PURCHASE_DIR , PERFORMANCE_PORTFOLIO_DIR , PERSONAL_ARCHIVE_DIR , PERSONAL_LIBRARY_DIR ,
										SCANNING_REQUIRED_DIR , SOFTWARE_ARCHIVE_DIR , TIKA_APP_JAR_PATH , ULTIMAKER_CURA_DIR ,
										UNESSENTIAL_DIR , UNSUPPORTED_ARTIFACTS_DIR , VIDEO_TYPES)
from utilities.ai_processing import (
	extract_document_text ,
	extract_visual_description ,
	generate_filename ,
	generate_tags
)
from utilities.document_scanning import process_handwritten_notes , process_printed_documents
from utilities.format_converting import (
	convert_audio_to_mp3 , convert_document_to_pdf , convert_email_to_pdf ,
	convert_html_to_pdf , convert_image_to_png , convert_png_to_pdf ,
	convert_video_to_mp4 ,
)

_r = tk.Tk( );
_avail = _tkf.families( );
_r.destroy( )
pygame.mixer.init( )
ctk.set_appearance_mode( "Dark" )
ctk.set_default_color_theme( "blue" )

# ═══════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
BG = "#141414";
SIDEBAR = "#1C1C1C";
PREVIEW = "#181818"
CARD = "#222222";
CARD_SEL = "#2A2A2A"
FG = "#EFEFEF";
FG2 = "#909090"
ACCENT = "#E05C52";
ACCENT_L = "#F07068";
ACCENT_D = "#B84840"
GREEN = "#5DBD72";
BLUE = "#4A8ED9";
YELLOW = "#D9A740";
RED = "#CC2222"
FONT = "Segoe UI";
EMOJI_FONT = "Segoe UI Emoji"

# Directories that trigger auto-tagging on confirm
_TAG_DIRS = {
	DIGITAL_ASSET_MANAGEMENT_DIR , FIREFLYIII_DIR , JELLYFIN_DIR ,
	ODOO_MAINTENANCE_DIR , ODOO_PLM_DIR , ODOO_PURCHASE_DIR ,
	ODOO_INVENTORY_DIR , PERFORMANCE_PORTFOLIO_DIR , PERSONAL_ARCHIVE_DIR ,
	MANUALS_ARCHIVE_DIR , SOFTWARE_ARCHIVE_DIR ,
}

# ═══════════════════════════════════════════════════════════════════════════
# BOLD-UNICODE HELPER
# ═══════════════════════════════════════════════════════════════════════════
_BOLD = {
	**{ chr( ord( "A" ) + i ) : chr( 0x1D5D4 + i ) for i in range( 26 ) } ,
	**{ chr( ord( "a" ) + i ) : chr( 0x1D5EE + i ) for i in range( 26 ) } ,
}


def _bold_key_in_label( label: str , key: str ) -> str :
	ku = key.upper( )
	for i , ch in enumerate( label ) :
		if ch.upper( ) == ku :
			return label[ :i ] + _BOLD.get( ch , ch ) + label[ i + 1 : ]
	return f"[{_BOLD.get( key.upper( ) , key.upper( ) )}] {label}"


# ═══════════════════════════════════════════════════════════════════════════
# DESTINATION BUTTON REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DestButton :
	name: str;
	label: str;
	order: int;
	key: str
	targets: list[ Path ] = field( default_factory=list )

	def display_text( self ) -> str :
		return _bold_key_in_label( self.label , self.key )


def _build_dest_registry( ) -> list[ DestButton ] :
	return sorted( [
		DestButton( "AFFINE" , "🧠 AFFINE" , 10 , "a" , [ AFFINE_DIR ] ) ,
		DestButton( "ANKI" , "🃏 ANKI" , 20 , "n" , [ ANKI_DIR ] ) ,
		DestButton( "ARCHIVE" , "📦 ARCHIVE" , 30 , "r" , [ ARCHIVAL_DIR ] ) ,
		DestButton( "BITWARDEN" , "🔐 BITWARDEN" , 40 , "b" , [ BITWARDEN_DIR ] ) ,
		DestButton( "CALIBRE WEB" , "📚 CALIBRE WEB" , 50 , "c" , [ PERSONAL_LIBRARY_DIR ] ) ,
		DestButton( "CRM" , "👥 CRM" , 60 , "m" , [ MONICA_CRM_DIR , ODOO_CRM_DIR ] ) ,
		DestButton( "FIREFLY" , "💰 FIREFLY" , 70 , "f" , [ FIREFLYIII_DIR ] ) ,
		DestButton( "GAMES ARCHIVE" , "🎮 GAMES ARCHIVE" , 75 , "j" , [ GAMES_ARCHIVE_DIR ] ) ,
		DestButton( "GITLAB" , "🦊 GITLAB" , 80 , "g" , [ GITLAB_DIR ] ) ,
		DestButton( "IMMICH" , "📷 IMMICH" , 90 , "i" , [ IMMICH_DIR ] ) ,
		DestButton( "INTERNET ARCHIVE" , "🌍 INTERNET ARCHIVE" , 100 , "t" , [ PERSONAL_ARCHIVE_DIR ] ) ,
		DestButton( "LINKWARDEN" , "🔖 LINKWARDEN" , 120 , "w" , [ LINKWARDEN_DIR ] ) ,
		DestButton( "JELLYFIN" , "🎬 JELLYFIN" , 125 , "l" , [ JELLYFIN_DIR ] ) ,
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


DEST_REGISTRY: list[ DestButton ] = _build_dest_registry( )
DEST_BY_NAME: dict[ str , DestButton ] = { b.name : b for b in DEST_REGISTRY }
DEST_BY_KEY: dict[ str , DestButton ] = { b.key : b for b in DEST_REGISTRY }


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass( frozen=True )
class ConvOption :
	label: str;
	key: str;
	fn: str


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
_ALL_CONV_BY_KEY = { c.key : c for c in _ALL_CONV }

_IMG_EXTS = { e.lower( ) for e in IMAGE_TYPES }
_VID_EXTS = { e.lower( ) for e in VIDEO_TYPES }
_AUD_EXTS = { e.lower( ) for e in AUDIO_TYPES }
_DOC_EXTS = { e.lower( ) for e in DOCUMENT_TYPES }
_EML_EXTS = { e.lower( ) for e in EMAIL_TYPES }
_HTML_EXTS = { "html" , "htm" , "xhtml" }


def _conv_keys_for( file: Path ) -> set[ str ] :
	ext = file.suffix.lower( ).lstrip( "." )
	if ext in _HTML_EXTS : return { "html_pdf" }
	if ext in _IMG_EXTS :  return { "img_png" , "img_pdf" }
	if ext in _VID_EXTS :  return { "vid_mp4" }
	if ext in _AUD_EXTS :  return { "aud_mp3" }
	if ext in _DOC_EXTS :  return { "doc_pdf" }
	if ext in _EML_EXTS :  return { "eml_pdf" }
	return set( )


# ═══════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _unique( directory: Path , file: Path ) -> Path :
	dst = directory / file.name;
	n = 1
	while dst.exists( ) :
		dst = directory / f"{file.stem}_{n}{file.suffix}";
		n += 1
	return dst


def _hover( hex_col: str ) -> str :
	h = hex_col.lstrip( "#" )
	r , g , b = int( h[ 0 :2 ] , 16 ) , int( h[ 2 :4 ] , 16 ) , int( h[ 4 :6 ] , 16 )
	return "#{:02x}{:02x}{:02x}".format(
			min( 255 , int( r * 1.18 ) ) , min( 255 , int( g * 1.18 ) ) , min( 255 , int( b * 1.18 ) ) )


def _tika_metadata( file: Path ) -> str :
	try :
		r = subprocess.run(
				[ str( JAVA_PATH ) , "-jar" , str( TIKA_APP_JAR_PATH ) , "-m" , str( file ) ] ,
				capture_output=True , text=True , timeout=30 )
		return r.stdout.strip( ) or "No metadata returned by Tika."
	except FileNotFoundError : return "Java or Tika JAR not found."
	except subprocess.TimeoutExpired : return "Tika extraction timed out."
	except Exception as e : return f"Tika error: {e}"


def _write_alteration_csv( filename: str , description: str , logger: logging.Logger ) -> None :
	is_new = not ALTERATIONS_CSV.exists( )
	with open( ALTERATIONS_CSV , "a" , newline="" , encoding="utf-8" ) as fh :
		w = csv.writer( fh )
		if is_new : w.writerow( [ "timestamp" , "filename" , "description" ] )
		w.writerow( [ datetime.now( ).isoformat( ) , filename , description ] )
	logger.info( "Alteration logged for '%s': %s" , filename , description[ :80 ] )


def _file_category( file: Path ) -> str :
	ext = file.suffix.lower( ).lstrip( "." )
	if ext == "pdf" :      return "pdf"
	if ext in _IMG_EXTS :  return "image"
	if ext in _VID_EXTS :  return "video"
	if ext in _AUD_EXTS :  return "audio"
	if ext in _HTML_EXTS : return "html"
	if ext in {
		"txt" , "log" , "json" , "csv" , "xml" , "md" , "yaml" , "yml" , "toml" , "ini" ,
		"py" , "js" , "ts" , "css" , "sh" , "bat" ,
	} :
		return "text"
	return "other"


def _is_document_type( file: Path ) -> bool :
	ext = file.suffix.lower( ).lstrip( "." )
	return ext in _DOC_EXTS or ext == "pdf" or ext in _HTML_EXTS or ext in _EML_EXTS or ext in {
		"txt" , "log" , "json" , "csv" , "xml" , "md" , "yaml" , "yml" , "toml" , "ini" ,
	}


def _is_image_type( file: Path ) -> bool :
	return file.suffix.lower( ).lstrip( "." ) in _IMG_EXTS


# ═══════════════════════════════════════════════════════════════════════════
# ROTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _rotate_pdf( file: Path , degrees: int , logger: logging.Logger ) -> None :
	logger.info( "Rotating PDF '%s' by %d°…" , file.name , degrees )
	doc = fitz.open( file )
	for page in doc : page.set_rotation( (page.rotation + degrees) % 360 )
	tmp = file.parent / f"_rot_{file.name}";
	doc.save( tmp );
	doc.close( )
	file.unlink( );
	tmp.rename( file )


def _rotate_image( file: Path , degrees: int , logger: logging.Logger ) -> None :
	img = Image.open( file );
	rotated = img.rotate( -degrees , expand=True )
	rotated.save( file , format="PNG" )


def _rotate_video( file: Path , degrees: int , logger: logging.Logger ) -> None :
	vf = { 90 : "transpose=1" , 180 : "transpose=1,transpose=1" , 270 : "transpose=2" }.get( degrees )
	if not vf : raise ValueError( f"Unsupported rotation angle: {degrees}" )
	tmp = file.parent / f"_rot_{file.name}"
	try :
		subprocess.run( [ "ffmpeg" , "-y" , "-i" , str( file ) , "-vf" , vf ,
											"-c:a" , "copy" , "-metadata:s:v" , "rotate=0" , str( tmp ) ] ,
										capture_output=True , timeout=300 , check=True )
		file.unlink( );
		tmp.rename( file )
	except FileNotFoundError : raise RuntimeError( "ffmpeg not found on PATH" )
	except subprocess.CalledProcessError as e :
		if tmp.exists( ) : tmp.unlink( )
		raise RuntimeError( f"ffmpeg error: {(e.stderr or b'')[ :300 ]}" )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════════════
class FileTriage :
	def __init__( self , source_dir: Path , logger: logging.Logger ) :
		self.source_dir = Path( source_dir )
		self.logger = logger
		self.logger.info( "Initialising FileTriage v6.2 with source: %s" , self.source_dir )

		# ── file queue & history ──────────────────────────────────────────
		self.queue: deque[ Path ] = deque( )
		self.history: list[ list ] = [ ]
		self.current_index: int = -1
		self.current_file: Path | None = None

		# ── UI state ─────────────────────────────────────────────────────
		self.selected_dests: set[ str ] = set( )
		self.selected_conversion: str | None = None

		# ── persistent widget references ─────────────────────────────────
		self._dest_cbs: dict[ str , ctk.CTkCheckBox ] = { }
		self._dest_vars: dict[ str , tk.IntVar ] = { }
		self._conv_btns: dict[ str , ctk.CTkButton ] = { }
		self._hist_widgets: list[ dict ] = [ ]

		# ── batch / flag tracking ────────────────────────────────────────
		self.pending: list[ tuple[ Path , list[ str ] ] ] = [ ]
		self.batch_n: int = 0
		self.flagged: set[ Path ] = set( )

		# ── media playback state ─────────────────────────────────────────
		self.video_cap = None;
		self.video_after_id = None
		self.video_lbl = None;
		self.is_playing = False
		self.zoom_scale = 0.5
		self._vid_size = (640 , 360);
		self._vid_delay = 66

		# ── auto-close state ─────────────────────────────────────────────
		self._closing = False
		self._close_after_id = None

		# ── threading ────────────────────────────────────────────────────
		self._executor = ThreadPoolExecutor( max_workers=4 )

		# ── preview preload cache ────────────────────────────────────────
		self._preload_cache: dict[ str , object ] = { }
		self._preloading: set[ str ] = set( )

		# ── performance tracking ─────────────────────────────────────────
		self._thumb_cache: dict[ str , ctk.CTkImage | None ] = { }
		self.session_start = time.time( )
		self.file_start: float | None = None
		self.file_times: list[ float ] = [ ]

		# ── ensure directories exist ─────────────────────────────────────
		for btn in DEST_REGISTRY :
			for p in btn.targets : p.mkdir( parents=True , exist_ok=True )
		for d in (ALTERATIONS_REQUIRED_DIR , UNESSENTIAL_DIR , DELETE_DIR ,
							UNSUPPORTED_ARTIFACTS_DIR) :
			d.mkdir( parents=True , exist_ok=True )
		self.logger.info( "All destination directories verified/created." )

		self._load_files( )

		# ── build window ─────────────────────────────────────────────────
		self.root = ctk.CTk( )
		self.root.title( "File Triage  v6" )
		self.root.geometry( "1920x1060" )
		self.root.configure( fg_color=BG )
		self._build_ui( )
		self._bind_global_keys( )
		self._tick_timer( )

		if self.queue :
			self.logger.info( "Showing first file…" )
			self.show_next( )
		else :
			self.logger.warning( "Queue is empty at startup." )
		self.logger.info( "Application ready — %d files in queue" , len( self.queue ) )

	# ──────────────────────────────────────────────────────────────────────
	# FILE LOADING
	# ──────────────────────────────────────────────────────────────────────
	def _load_files( self ) :
		primary = [ f for f in self.source_dir.iterdir( )
								if f.is_file( ) and f.parent == self.source_dir ]
		self.queue.extend( primary )
		self.logger.info( "Loaded %d file(s) from source directory: %s" , len( primary ) , self.source_dir )
		for f in primary :
			self.logger.debug( "  queued: %s (%s)" , f.name , f.suffix )
		if UNSUPPORTED_ARTIFACTS_DIR.exists( ) and UNSUPPORTED_ARTIFACTS_DIR != self.source_dir :
			extras = [ f for f in UNSUPPORTED_ARTIFACTS_DIR.iterdir( )
								 if f.is_file( ) and f.parent == UNSUPPORTED_ARTIFACTS_DIR ]
			if extras :
				self.queue.extend( extras )
				self.logger.info( "Appended %d file(s) from UNSUPPORTED_ARTIFACTS: %s" ,
													len( extras ) , UNSUPPORTED_ARTIFACTS_DIR )

	# ══════════════════════════════════════════════════════════════════════
	# UI CONSTRUCTION
	# ══════════════════════════════════════════════════════════════════════
	def _build_ui( self ) :
		main = ctk.CTkFrame( self.root , fg_color=BG )
		main.pack( fill=tk.BOTH , expand=True , padx=10 , pady=10 )

		# ── LEFT: history sidebar ────────────────────────────────────────
		left = ctk.CTkFrame( main , width=290 , fg_color=SIDEBAR , corner_radius=14 )
		left.pack( side=tk.LEFT , fill=tk.Y , padx=(0 , 8) )
		left.pack_propagate( False )
		ctk.CTkLabel( left , text="HISTORY" , font=(FONT , 12 , "bold") ,
									text_color=ACCENT ).pack( pady=(14 , 6) )
		self.hist_frame = CTkScrollableFrame( left , fg_color=SIDEBAR )
		self.hist_frame.pack( fill=tk.BOTH , expand=True , padx=6 , pady=(0 , 8) )

		# ── CENTER column ────────────────────────────────────────────────
		center = ctk.CTkFrame( main , fg_color=BG )
		center.pack( side=tk.LEFT , fill=tk.BOTH , expand=True )

		# ---- info bar ----
		info = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 , height=46 )
		info.pack( fill=tk.X , pady=(0 , 5) );
		info.pack_propagate( False )
		self.file_lbl = ctk.CTkLabel( info , text="—" , font=(FONT , 13 , "bold") , text_color=FG )
		self.file_lbl.pack( side=tk.LEFT , padx=14 )
		self.action_lbl = ctk.CTkLabel( info , text="" , font=(FONT , 11 , "bold") , text_color=ACCENT )
		self.action_lbl.pack( side=tk.LEFT , padx=8 )
		self.queue_lbl = ctk.CTkLabel( info , text="Queue: 0" , font=(FONT , 11) , text_color=FG2 )
		self.queue_lbl.pack( side=tk.RIGHT , padx=14 )

		# ---- preview outer frame ----
		prev_outer = ctk.CTkFrame( center , fg_color=PREVIEW , corner_radius=14 )
		prev_outer.pack( fill=tk.BOTH , expand=True , pady=(0 , 5) )

		# ---- toolbar row 1: rotate, open, zoom controls ----
		tb = ctk.CTkFrame( prev_outer , fg_color=PREVIEW , height=44 )
		tb.pack( fill=tk.X , padx=10 , pady=(8 , 0) );
		tb.pack_propagate( False )

		for txt , deg in [ ("↺ 90°" , 270) , ("180°" , 180) , ("↻ 90°" , 90) ] :
			ctk.CTkButton( tb , text=txt , command=lambda d=deg : self._rotate_current( d ) ,
										 fg_color=CARD , hover_color=CARD_SEL , width=70 , height=34 ,
										 corner_radius=8 , text_color=FG , font=(FONT , 10 , "bold") ,
										 ).pack( side=tk.LEFT , padx=3 )

		ctk.CTkButton( tb , text="🔗 Open External" , command=self._open_external ,
									 fg_color=CARD , hover_color=CARD_SEL , width=135 , height=34 ,
									 corner_radius=8 , text_color=FG , font=(EMOJI_FONT , 10) ,
									 ).pack( side=tk.LEFT , padx=3 )

		# Zoom controls: − label +
		ctk.CTkButton( tb , text="−" , command=self._zoom_out ,
									 fg_color=CARD , hover_color=CARD_SEL , width=34 , height=34 ,
									 corner_radius=8 , text_color=FG , font=(FONT , 14 , "bold") ,
									 ).pack( side=tk.LEFT , padx=(12 , 2) )
		self.zoom_lbl = ctk.CTkLabel( tb , text="🔍 50%" , font=(EMOJI_FONT , 10) , text_color=FG2 )
		self.zoom_lbl.pack( side=tk.LEFT , padx=2 )
		ctk.CTkButton( tb , text="+" , command=self._zoom_in ,
									 fg_color=CARD , hover_color=CARD_SEL , width=34 , height=34 ,
									 corner_radius=8 , text_color=FG , font=(FONT , 14 , "bold") ,
									 ).pack( side=tk.LEFT , padx=2 )

		# ---- toolbar row 2: AI status only ----
		tb2 = ctk.CTkFrame( prev_outer , fg_color=PREVIEW , height=30 )
		tb2.pack( fill=tk.X , padx=10 , pady=(4 , 0) );
		tb2.pack_propagate( False )
		self.ai_status_lbl = ctk.CTkLabel( tb2 , text="" , font=(FONT , 10) , text_color=FG2 )
		self.ai_status_lbl.pack( side=tk.LEFT , padx=12 )

		# ---- scrollable preview area ----
		self.prev_scroll = CTkScrollableFrame( prev_outer , fg_color=PREVIEW )
		self.prev_scroll.pack( fill=tk.BOTH , expand=True , padx=6 , pady=6 )
		self.prev_scroll.bind( "<MouseWheel>" , self._zoom )
		self.prev_scroll._parent_canvas.bind( "<MouseWheel>" , self._zoom )

		# ---- conversion panel ----
		conv_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		conv_panel.pack( fill=tk.X , pady=(0 , 5) )
		conv_hdr = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		conv_hdr.pack( fill=tk.X , padx=14 , pady=(8 , 4) )
		ctk.CTkLabel( conv_hdr ,
									text="CONVERT  (optional · single select · original archived first)" ,
									font=(FONT , 10 , "bold") , text_color=FG2 ).pack( side=tk.LEFT )
		ctk.CTkButton( conv_hdr , text="✕ clear" , command=self._clear_conversion ,
									 fg_color=CARD , hover_color=CARD_SEL , width=70 , height=24 ,
									 corner_radius=6 , text_color=FG2 , font=(FONT , 9) ).pack( side=tk.LEFT , padx=10 )

		self.conv_row = ctk.CTkFrame( conv_panel , fg_color=SIDEBAR )
		self.conv_row.pack( fill=tk.X , padx=10 , pady=(0 , 10) )
		for opt in _ALL_CONV :
			btn = ctk.CTkButton( self.conv_row , text=opt.label ,
													 command=lambda k=opt.key : self._toggle_conversion( k ) ,
													 fg_color=CARD , hover_color=CARD_SEL , text_color=FG2 ,
													 height=34 , corner_radius=8 , font=(EMOJI_FONT , 10) )
			self._conv_btns[ opt.key ] = btn
		self.no_conv_lbl = ctk.CTkLabel( self.conv_row ,
																		 text="No conversions available for this file type" ,
																		 font=(FONT , 10) , text_color=FG2 )

		# ---- TRIAGE ACTIONS panel ----
		dest_panel = ctk.CTkFrame( center , fg_color=SIDEBAR , corner_radius=12 )
		dest_panel.pack( fill=tk.X , pady=(0 , 5) )

		# Header row with clear button
		dest_hdr = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		dest_hdr.pack( fill=tk.X , padx=14 , pady=(8 , 4) )
		ctk.CTkLabel( dest_hdr , text="TRIAGE ACTIONS" ,
									font=(FONT , 10 , "bold") , text_color=FG2 ).pack( side=tk.LEFT )

		# ---- scan / AI action buttons row ----
		action_row = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		action_row.pack( fill=tk.X , padx=10 , pady=(0 , 6) )
		ctk.CTkButton( action_row , text="✅ SCAN HANDWRITING" , command=self._scan_handwriting ,
									 fg_color="#3A506B" , hover_color="#4A6080" , width=180 , height=32 ,
									 corner_radius=8 , text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )
		ctk.CTkButton( action_row , text="✅ SCAN PRINTED" , command=self._scan_printed ,
									 fg_color="#3A506B" , hover_color="#4A6080" , width=160 , height=32 ,
									 corner_radius=8 , text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )
		ctk.CTkButton( action_row , text="🤖 NEW FILENAME" , command=self._ai_rename ,
									 fg_color="#5B4A78" , hover_color="#6B5A88" , width=150 , height=32 ,
									 corner_radius=8 , text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )

		# ---- destination sub-header with clear ----
		dest_sub = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		dest_sub.pack( fill=tk.X , padx=14 , pady=(0 , 4) )
		ctk.CTkLabel( dest_sub , text="Destinations  (click or key · multi-select)" ,
									font=(FONT , 9) , text_color=FG2 ).pack( side=tk.LEFT )
		ctk.CTkButton( dest_sub , text="✕ clear all" , command=self._clear_all_destinations ,
									 fg_color=CARD , hover_color=CARD_SEL , width=80 , height=22 ,
									 corner_radius=6 , text_color=FG2 , font=(FONT , 9) ).pack( side=tk.LEFT , padx=10 )

		# ---- destination grid ----
		self.dest_grid = ctk.CTkFrame( dest_panel , fg_color=SIDEBAR )
		self.dest_grid.pack( fill=tk.X , padx=10 , pady=(0 , 10) )
		cols = 5
		for c in range( cols ) : self.dest_grid.grid_columnconfigure( c , weight=1 )
		for i , db in enumerate( DEST_REGISTRY ) :
			var = tk.IntVar( value=0 )
			cb = ctk.CTkCheckBox(
					self.dest_grid , text=db.display_text( ) , variable=var ,
					command=lambda n=db.name : self._on_dest_toggle( n ) ,
					fg_color=ACCENT , hover_color=ACCENT_L ,
					border_color=FG2 , text_color=FG2 ,
					checkmark_color=FG , corner_radius=6 ,
					font=(EMOJI_FONT , 10) , height=34 ,
			)
			cb.grid( row=i // cols , column=i % cols , padx=4 , pady=4 , sticky="ew" )
			self._dest_cbs[ db.name ] = cb
			self._dest_vars[ db.name ] = var

		# ---- footer ----
		foot = ctk.CTkFrame( center , fg_color=BG , height=52 )
		foot.pack( fill=tk.X );
		foot.pack_propagate( False )

		self.timer_lbl = ctk.CTkLabel( foot , text="⏱ 0s" ,
																	 font=(FONT , 11 , "bold") , text_color=YELLOW )
		self.timer_lbl.pack( side=tk.LEFT , padx=10 )
		self.status_lbl = ctk.CTkLabel( foot , text="" , font=(FONT , 11) , text_color=ACCENT )
		self.status_lbl.pack( side=tk.LEFT , padx=10 )

		nav = ctk.CTkFrame( foot , fg_color=BG );
		nav.pack( side=tk.RIGHT , padx=8 )

		# Action buttons on the right: unessential, flag, delete, prev, confirm
		ctk.CTkButton( nav , text="↓ UNESSENTIAL [↓]" , command=self._send_to_unessential ,
									 fg_color=CARD , hover_color=CARD_SEL , width=150 , height=42 ,
									 corner_radius=10 , text_color=FG2 , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )
		ctk.CTkButton( nav , text="🚩 FLAG [↑]" , command=self._flag_with_dialog ,
									 fg_color=ACCENT_D , hover_color=ACCENT , width=110 , height=42 ,
									 corner_radius=10 , text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )
		ctk.CTkButton( nav , text="🗑 DELETE [Del]" , command=self._delete_file ,
									 fg_color=RED , hover_color=_hover( RED ) , width=120 , height=42 ,
									 corner_radius=10 , text_color=FG , font=(EMOJI_FONT , 10 , "bold") ,
									 ).pack( side=tk.LEFT , padx=3 )

		# Separator
		ctk.CTkFrame( nav , fg_color=FG2 , width=2 , height=32 ).pack( side=tk.LEFT , padx=8 )

		ctk.CTkButton( nav , text="← Prev" , command=self.show_previous , fg_color=CARD ,
									 hover_color=_hover( CARD ) , width=88 , height=42 , corner_radius=10 ,
									 text_color=FG , font=(FONT , 11) ).pack( side=tk.LEFT , padx=3 )
		ctk.CTkButton( nav , text="✓ CONFIRM" , command=self._confirm , fg_color=GREEN ,
									 hover_color=_hover( GREEN ) , width=120 , height=42 , corner_radius=10 ,
									 text_color=FG , font=(FONT , 11 , "bold") ).pack( side=tk.LEFT , padx=3 )

	# ── global key bindings ──────────────────────────────────────────────
	def _bind_global_keys( self ) :
		self.root.bind( "<Left>" , lambda _ : self.show_previous( ) )
		self.root.bind( "<Return>" , lambda _ : self._confirm( ) )
		self.root.bind( "<Delete>" , lambda _ : self._delete_file( ) )
		self.root.bind( "<Up>" , lambda _ : self._flag_with_dialog( ) )
		self.root.bind( "<Down>" , lambda _ : self._send_to_unessential( ) )
		for db in DEST_REGISTRY :
			self.root.bind( db.key , lambda _ , n=db.name : self._on_dest_toggle( n ) )

	# ══════════════════════════════════════════════════════════════════════
	# DESTINATION CHECKBOXES
	# ══════════════════════════════════════════════════════════════════════
	def _on_dest_toggle( self , name: str ) :
		var = self._dest_vars.get( name )
		cb = self._dest_cbs.get( name )
		if not var or not cb : return
		if name not in self.selected_dests :
			var.set( 1 );
			self.selected_dests.add( name )
			cb.configure( text_color=FG )
			self.logger.debug( "Destination selected: %s" , name )
		else :
			var.set( 0 );
			self.selected_dests.discard( name )
			cb.configure( text_color=FG2 )
			self.logger.debug( "Destination deselected: %s" , name )
		self._refresh_action_label( )

	def _reset_dest_checkboxes( self ) :
		for name , var in self._dest_vars.items( ) :
			var.set( 0 )
			self._dest_cbs[ name ].configure( text_color=FG2 )
		self.selected_dests.clear( )

	def _restore_dest_selections( self ) :
		if 0 <= self.current_index < len( self.history ) :
			for name in self.history[ self.current_index ][ 1 ] :
				if name in self._dest_vars :
					self._dest_vars[ name ].set( 1 )
					self._dest_cbs[ name ].configure( text_color=FG )
					self.selected_dests.add( name )

	def _clear_all_destinations( self ) :
		self.logger.info( "Clearing all destination selections." )
		self._reset_dest_checkboxes( )
		self._refresh_action_label( )

	# ══════════════════════════════════════════════════════════════════════
	# CONVERSION PANEL
	# ══════════════════════════════════════════════════════════════════════
	def _refresh_conv_visibility( self ) :
		for btn in self._conv_btns.values( ) : btn.pack_forget( )
		self.no_conv_lbl.pack_forget( )
		self.selected_conversion = None
		if not self.current_file : return
		valid_keys = _conv_keys_for( Path( self.current_file ) )
		if not valid_keys :
			self.no_conv_lbl.pack( padx=10 , pady=4 );
			return
		for key , btn in self._conv_btns.items( ) :
			if key in valid_keys :
				btn.configure( fg_color=CARD , text_color=FG2 )
				btn.pack( side=tk.LEFT , padx=5 , expand=True , fill=tk.X )
		if 0 <= self.current_index < len( self.history ) :
			saved = self.history[ self.current_index ][ 2 ]
			if saved and saved in valid_keys : self._toggle_conversion( saved )

	def _toggle_conversion( self , key: str ) :
		if self.selected_conversion == key :
			self._clear_conversion( );
			return
		if self.selected_conversion and self.selected_conversion in self._conv_btns :
			self._conv_btns[ self.selected_conversion ].configure( fg_color=CARD , text_color=FG2 )
		self.selected_conversion = key
		self.logger.info( "Conversion selected: %s" , key )
		if key in self._conv_btns :
			self._conv_btns[ key ].configure( fg_color=YELLOW , text_color=BG )
		self._refresh_action_label( )

	def _clear_conversion( self ) :
		if self.selected_conversion :
			self.logger.info( "Conversion cleared (was: %s)" , self.selected_conversion )
			if self.selected_conversion in self._conv_btns :
				self._conv_btns[ self.selected_conversion ].configure( fg_color=CARD , text_color=FG2 )
		self.selected_conversion = None
		self._refresh_action_label( )

	def _refresh_action_label( self ) :
		parts = [ ]
		if self.selected_conversion :
			opt = _ALL_CONV_BY_KEY.get( self.selected_conversion )
			parts.append( f"⚡ {opt.label if opt else self.selected_conversion}" )
		if self.selected_dests :
			parts.append( "→ " + ", ".join( sorted( self.selected_dests ) ) )
		self.action_lbl.configure(
				text="  |  ".join( parts ) if parts else "no action selected" ,
				text_color=ACCENT if parts else FG2 )

	# ══════════════════════════════════════════════════════════════════════
	# MEDIA RELEASE
	# ══════════════════════════════════════════════════════════════════════
	def _release_media( self ) :
		self.is_playing = False
		if self.video_after_id :
			self.root.after_cancel( self.video_after_id );
			self.video_after_id = None
		if self.video_cap :
			try : self.video_cap.release( )
			except : pass
			self.video_cap = None
		try : pygame.mixer.music.stop( ); pygame.mixer.music.unload( )
		except : pass

	# ══════════════════════════════════════════════════════════════════════
	# ZOOM
	# ══════════════════════════════════════════════════════════════════════
	def _apply_zoom( self , new_scale: float ) :
		old = self.zoom_scale
		self.zoom_scale = max( 0.2 , min( 4.0 , new_scale ) )
		self.zoom_lbl.configure( text=f"🔍 {int( self.zoom_scale * 100 )}%" )
		if int( old * 100 ) != int( self.zoom_scale * 100 ) :
			self.logger.debug( "Zoom changed: %d%% → %d%%" , int( old * 100 ) , int( self.zoom_scale * 100 ) )
			self._display_file( )

	def _zoom( self , event ) :
		self._apply_zoom( self.zoom_scale + (0.1 if event.delta > 0 else -0.1) )

	def _zoom_in( self ) :
		self._apply_zoom( self.zoom_scale + 0.1 )

	def _zoom_out( self ) :
		self._apply_zoom( self.zoom_scale - 0.1 )

	# ══════════════════════════════════════════════════════════════════════
	# ROTATION
	# ══════════════════════════════════════════════════════════════════════
	def _rotate_current( self , degrees: int ) :
		self._release_media( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		cat = _file_category( Path( self.current_file ) )
		self.logger.info( "Rotating '%s' (%s) by %d°" , Path( self.current_file ).name , cat , degrees )
		self.status_lbl.configure( text="Rotating…" , text_color=YELLOW )
		self.root.update_idletasks( )
		try :
			if cat == "pdf" :     _rotate_pdf( Path( self.current_file ) , degrees , self.logger )
			elif cat == "image" : _rotate_image( Path( self.current_file ) , degrees , self.logger )
			elif cat == "video" : _rotate_video( Path( self.current_file ) , degrees , self.logger )
			else :
				self.logger.warning( "Rotation not supported for category '%s'" , cat )
				self.status_lbl.configure( text="Rotation not supported" , text_color=FG2 );
				return
			self.status_lbl.configure( text=f"✓ Rotated {degrees}°" , text_color=GREEN )
			self._thumb_cache.pop( str( Path( self.current_file ) ) , None )
			self._preload_cache.pop( str( Path( self.current_file ) ) , None )
			self._display_file( )
		except Exception as e :
			self.logger.error( "Rotation failed for '%s': %s" , Path( self.current_file ).name , e )
			self.status_lbl.configure( text="✗ Rotation failed" , text_color=RED )
			messagebox.showerror( "Rotation Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════
	# SCANNING & AI TOOLS
	# ══════════════════════════════════════════════════════════════════════
	def _scan_handwriting( self ) :
		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.warning( "Scan handwriting: no current file or file missing." )
			return
		src = Path( self.current_file )
		out_pdf = src.parent / f"{src.stem}_handwritten.pdf"
		self.logger.info( "Starting handwriting scan: src='%s', out='%s'" , src , out_pdf )
		self.logger.info( "  src exists: %s, src size: %s bytes" , src.exists( ) ,
											src.stat( ).st_size if src.exists( ) else "N/A" )
		self.status_lbl.configure( text="⏳ Scanning handwriting…" , text_color=YELLOW )
		self.root.update_idletasks( )
		self._executor.submit( self._scan_handwriting_worker , src , out_pdf )

	def _scan_handwriting_worker( self , src: Path , out_pdf: Path ) :
		try :
			self.logger.info( "[SCAN-HW] Calling process_handwritten_notes('%s', '%s')" , src , out_pdf )
			result = process_handwritten_notes( str( src ) , str( out_pdf ) )
			self.logger.info( "[SCAN-HW] Success: result='%s'" , result )
			self.root.after( 0 , lambda : self._scan_done(
					f"✓ Handwriting scan → {Path( result ).name}" , Path( result ) ) )
		except Exception as e :
			err_msg = str( e )
			self.logger.error( "[SCAN-HW] Failed for '%s': %s" , src.name , err_msg , exc_info=True )
			self.root.after( 0 , lambda msg=err_msg : self.status_lbl.configure(
					text=f"✗ Scan failed: {msg}" , text_color=RED ) )

	def _scan_printed( self ) :
		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.warning( "Scan printed: no current file or file missing." )
			return
		src = Path( self.current_file )
		out_dir = src.parent / f"{src.stem}_scanned"
		self.logger.info( "Starting printed scan: src='%s', out_dir='%s'" , src , out_dir )
		self.logger.info( "  src exists: %s, src size: %s bytes" , src.exists( ) ,
											src.stat( ).st_size if src.exists( ) else "N/A" )
		self.status_lbl.configure( text="⏳ Scanning printed document…" , text_color=YELLOW )
		self.root.update_idletasks( )
		self._executor.submit( self._scan_printed_worker , src , out_dir )

	def _scan_printed_worker( self , src: Path , out_dir: Path ) :
		try :
			self.logger.info( "[SCAN-PR] Calling process_printed_documents('%s', '%s')" , src , out_dir )
			result = process_printed_documents( str( src ) , str( out_dir ) )
			self.logger.info( "[SCAN-PR] Success: result='%s'" , result )
			self.root.after( 0 , lambda : self._scan_done(
					f"✓ Printed scan → {Path( result ).name}" , None ) )
		except Exception as e :
			err_msg = str( e )
			self.logger.error( "[SCAN-PR] Failed for '%s': %s" , src.name , err_msg , exc_info=True )
			self.root.after( 0 , lambda msg=err_msg : self.status_lbl.configure(
					text=f"✗ Scan failed: {msg}" , text_color=RED ) )

	def _scan_done( self , msg: str , new_file: Path | None ) :
		self.logger.info( "Scan complete: %s" , msg )
		self.status_lbl.configure( text=msg , text_color=GREEN )
		if new_file and new_file.exists( ) :
			self.current_file = new_file
			if 0 <= self.current_index < len( self.history ) :
				self.history[ self.current_index ][ 0 ] = new_file
			self._display_file( )

	def _ai_rename( self ) :
		if not self.current_file or not Path( self.current_file ).exists( ) :
			self.logger.warning( "AI rename: no current file." )
			return
		self.logger.info( "Starting AI rename for '%s'" , Path( self.current_file ).name )
		self.ai_status_lbl.configure( text="⏳ Generating filename…" , text_color=YELLOW )
		self.root.update_idletasks( )
		self._executor.submit( self._ai_rename_worker , Path( self.current_file ) )

	def _ai_rename_worker( self , src: Path ) :
		try :
			if _is_document_type( src ) :
				self.logger.debug( "[AI-RENAME] Extracting document text from '%s'" , src.name )
				content = extract_document_text( self.logger , src )
			elif _is_image_type( src ) :
				self.logger.debug( "[AI-RENAME] Extracting visual description from '%s'" , src.name )
				content = extract_visual_description( self.logger , src )
			else :
				self.logger.debug( "[AI-RENAME] Extracting Tika metadata from '%s'" , src.name )
				content = _tika_metadata( src )
			if not content :
				self.logger.warning( "[AI-RENAME] No content extracted for '%s'" , src.name )
				self.root.after( 0 , lambda : self.ai_status_lbl.configure(
						text="✗ Could not extract content" , text_color=RED ) )
				return
			new_name = generate_filename( self.logger , content )
			if not new_name :
				self.logger.warning( "[AI-RENAME] Filename generation returned empty for '%s'" , src.name )
				self.root.after( 0 , lambda : self.ai_status_lbl.configure(
						text="✗ Filename generation failed" , text_color=RED ) )
				return
			new_path = src.parent / f"{new_name}{src.suffix}"
			new_path = _unique( src.parent , new_path ) if new_path.exists( ) else new_path
			self.logger.info( "[AI-RENAME] Suggested: '%s' → '%s'" , src.name , new_path.name )
			self.root.after( 0 , lambda : self._ai_rename_confirm( src , new_path , new_name ) )
		except Exception as e :
			err_msg = str( e )
			self.logger.error( "[AI-RENAME] Failed for '%s': %s" , src.name , err_msg , exc_info=True )
			self.root.after( 0 , lambda msg=err_msg : self.ai_status_lbl.configure(
					text=f"✗ {msg}" , text_color=RED ) )

	def _ai_rename_confirm( self , src: Path , new_path: Path , new_name: str ) :
		result = simpledialog.askstring(
				"AI Suggested Filename" ,
				f"Current: {src.name}\n\nSuggested name (edit if needed):" ,
				initialvalue=f"{new_name}{src.suffix}" , parent=self.root )
		if result is None :
			self.logger.info( "[AI-RENAME] User cancelled rename for '%s'" , src.name )
			self.ai_status_lbl.configure( text="Cancelled" , text_color=FG2 );
			return
		self._release_media( )
		final_path = src.parent / result
		try :
			src.rename( final_path )
			self.logger.info( "[AI-RENAME] Renamed '%s' → '%s'" , src.name , final_path.name )
			self.current_file = final_path
			if 0 <= self.current_index < len( self.history ) :
				self.history[ self.current_index ][ 0 ] = final_path
			self._thumb_cache.pop( str( src ) , None )
			self._preload_cache.pop( str( src ) , None )
			self.ai_status_lbl.configure( text=f"✓ Renamed → {result}" , text_color=GREEN )
			self.file_lbl.configure( text=f"{result}  ({self.current_index + 1}/{len( self.history ) + len( self.queue )})" )
			self._display_file( )
		except Exception as e :
			self.logger.error( "[AI-RENAME] Rename failed: %s" , e )
			self.ai_status_lbl.configure( text=f"✗ Rename failed: {e}" , text_color=RED )

	# ══════════════════════════════════════════════════════════════════════
	# AUTO-TAGGING PIPELINE
	# ══════════════════════════════════════════════════════════════════════
	def _needs_tagging( self , dest_names: list[ str ] ) -> bool :
		for dname in dest_names :
			db = DEST_BY_NAME.get( dname )
			if db :
				for tgt in db.targets :
					if tgt in _TAG_DIRS : return True
		return False

	def _run_tagging( self , file: Path ) :
		self.logger.info( "[AUTO-TAG] Queuing tagging for '%s'" , file.name )
		self._executor.submit( self._tag_worker , file )

	def _tag_worker( self , file: Path ) :
		try :
			self.logger.info( "[AUTO-TAG] Starting for '%s'" , file.name )
			if _is_document_type( file ) :
				content = extract_document_text( self.logger , file )
			elif _is_image_type( file ) :
				content = extract_visual_description( self.logger , file )
			else :
				content = _tika_metadata( file )
			if not content :
				self.logger.warning( "[AUTO-TAG] No content extracted for '%s'" , file.name )
				return
			tags = generate_tags( self.logger , content )
			if tags :
				tag_file = file.parent / f"{file.stem}.tags.txt"
				tag_file.write_text( tags , encoding="utf-8" )
				self.logger.info( "[AUTO-TAG] Tags saved: %s → %s" , file.name , tags[ :80 ] )
			else :
				self.logger.warning( "[AUTO-TAG] Tag generation failed for '%s'" , file.name )
		except Exception as e :
			self.logger.error( "[AUTO-TAG] Error for '%s': %s" , file.name , e , exc_info=True )

	# ══════════════════════════════════════════════════════════════════════
	# CONFIRM  +  BATCH FLUSH
	# ══════════════════════════════════════════════════════════════════════
	def _confirm( self ) :
		self._release_media( )
		if not self.current_file :
			self.logger.debug( "Confirm called with no current file." )
			return
		self.logger.info( "CONFIRM: file='%s', dests=%s, conv=%s" ,
											Path( self.current_file ).name ,
											list( self.selected_dests ) , self.selected_conversion )

		if 0 <= self.current_index < len( self.history ) :
			self.history[ self.current_index ][ 1 ] = list( self.selected_dests )
			self.history[ self.current_index ][ 2 ] = self.selected_conversion

		if self.selected_conversion :
			self._run_conversion( self.selected_conversion )

		if self.selected_dests :
			dest_list = list( self.selected_dests )
			self.pending.append( (Path( self.current_file ) , dest_list) )
			self.batch_n += 1
			self.logger.info( "Pending batch: %d / %d" , self.batch_n , FILE_TRIAGE_BATCH_SIZE )
			if self._needs_tagging( dest_list ) :
				self._run_tagging( Path( self.current_file ) )
			if self.batch_n >= FILE_TRIAGE_BATCH_SIZE :
				self._flush_silent( )

		self.show_next( )

	def _run_conversion( self , key: str ) :
		self._release_media( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		self.logger.info( "Running conversion '%s' on '%s'" , key , Path( self.current_file ).name )
		try :
			ARCHIVAL_DIR.mkdir( parents=True , exist_ok=True )
			archive_dst = _unique( ARCHIVAL_DIR , Path( self.current_file ) )
			shutil.copy2( str( Path( self.current_file ) ) , str( archive_dst ) )
			self.logger.info( "Archived original to '%s'" , archive_dst )
		except Exception as e :
			self.logger.error( "Archive failed: %s" , e )
			messagebox.showerror( "Archive Error" , f"Could not archive:\n{e}\n\nConversion aborted." )
			return
		self.status_lbl.configure( text="⏳ Converting…" , text_color=YELLOW )
		self.root.update_idletasks( )
		self._executor.submit( self._convert_worker , key , Path( self.current_file ) )

	def _convert_worker( self , key: str , src: Path ) :
		try :
			if key == "img_pdf" :
				png = convert_image_to_png( src=src , logger=self.logger )
				if not png : raise RuntimeError( "image→PNG failed" )
				new = convert_png_to_pdf( src=png , logger=self.logger )
				if not new : raise RuntimeError( "PNG→PDF failed" )
			else :
				fn_map: dict[ str , Callable ] = {
					"img_png" : convert_image_to_png , "vid_mp4" : convert_video_to_mp4 ,
					"aud_mp3" : convert_audio_to_mp3 , "doc_pdf" : convert_document_to_pdf ,
					"eml_pdf" : convert_email_to_pdf , "html_pdf" : convert_html_to_pdf ,
				}
				fn = fn_map.get( key )
				if not fn : return
				new = fn( src=src , logger=self.logger )
			self.logger.info( "Conversion '%s' complete: '%s' → '%s'" , key , src.name , new )
			self.root.after( 0 , lambda : self._convert_done( new ) )
		except Exception as e :
			err_msg = str( e )
			self.logger.error( "Conversion '%s' failed for '%s': %s" , key , src.name , err_msg )
			self.root.after( 0 , lambda msg=err_msg : self._convert_failed( msg ) )

	def _convert_done( self , new_path: Path ) :
		self.current_file = new_path
		if 0 <= self.current_index < len( self.history ) :
			self.history[ self.current_index ][ 0 ] = new_path
		self.status_lbl.configure(
				text=f"✓ Converted → {new_path.suffix.upper( ).lstrip( '.' )} (original archived)" ,
				text_color=GREEN )

	def _convert_failed( self , err: str ) :
		self.status_lbl.configure( text="✗ Conversion failed" , text_color=RED )
		messagebox.showerror( "Conversion Error" , err )

	def _flush_silent( self ) :
		if not self.pending : return
		batch = list( self.pending );
		self.pending.clear( );
		self.batch_n = 0
		self.logger.info( "Flushing batch of %d files…" , len( batch ) )
		self.status_lbl.configure( text=f"⏳ Flushing {len( batch )} files…" , text_color=YELLOW )
		self._executor.submit( self._flush_worker , batch )

	def _flush_worker( self , batch: list[ tuple[ Path , list[ str ] ] ] ) :
		errors: list[ str ] = [ ]
		for file , dest_names in batch :
			if not file.exists( ) :
				self.logger.warning( "Flush: file no longer exists: '%s'" , file )
				continue
			all_targets: list[ tuple[ Path , str ] ] = [ ]
			for dname in dest_names :
				db = DEST_BY_NAME.get( dname )
				if db : all_targets.extend( (tgt , dname) for tgt in db.targets )
			if not all_targets : continue

			self.logger.info( "Flush: moving '%s' → %s" , file.name , [ t[ 1 ] for t in all_targets ] )
			if len( all_targets ) == 1 :
				dst_dir , label = all_targets[ 0 ]
				try :
					dst_dir.mkdir( parents=True , exist_ok=True )
					final = _unique( dst_dir , file )
					shutil.move( str( file ) , str( final ) )
					self.logger.info( "  moved to '%s'" , final )
				except Exception as e :
					errors.append( f"move {file.name}→{label}: {e}" )
					self.logger.error( "  move failed: %s" , e )
					try : shutil.move( str( file ) , str( _unique( UNSUPPORTED_ARTIFACTS_DIR , file ) ) )
					except : pass
			else :
				ok = True
				for dst_dir , label in all_targets :
					try :
						dst_dir.mkdir( parents=True , exist_ok=True )
						final = _unique( dst_dir , file )
						shutil.copy2( str( file ) , str( final ) )
						self.logger.info( "  copied to '%s'" , final )
					except Exception as e :
						errors.append( f"copy {file.name}→{label}: {e}" )
						self.logger.error( "  copy failed: %s" , e )
						ok = False
				if ok and file.exists( ) :
					try :
						file.unlink( )
						self.logger.info( "  original deleted." )
					except Exception as e :
						errors.append( f"delete {file.name}: {e}" )

		count = len( batch )
		if errors :
			self.logger.warning( "Flush completed with %d error(s): %s" , len( errors ) , errors )
		else :
			self.logger.info( "Flush completed: %d files, no errors." , count )
		self.root.after( 0 , lambda : self._flush_done( count , errors ) )

	def _flush_done( self , count: int , errors: list[ str ] ) :
		self.history.clear( );
		self.current_index = -1
		self._thumb_cache.clear( );
		self._preload_cache.clear( )
		self._rebuild_history_sidebar( )
		self.status_lbl.configure( text=f"✓ Flushed {count} files" , text_color=GREEN )

	# ══════════════════════════════════════════════════════════════════════
	# FLAG / UNESSENTIAL / DELETE
	# ══════════════════════════════════════════════════════════════════════
	def _flag_with_dialog( self ) :
		self._release_media( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		self.logger.info( "Flagging file: '%s'" , Path( self.current_file ).name )
		if self.selected_conversion : self._run_conversion( self.selected_conversion )
		desc = simpledialog.askstring( "Flag for Review" ,
																	 f"File: {Path( self.current_file ).name}\n\nNotes:" , parent=self.root )
		if desc is None :
			self.logger.info( "Flag cancelled by user." )
			return
		try :
			dst = _unique( ALTERATIONS_REQUIRED_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.flagged.add( dst )
			_write_alteration_csv( dst.name , desc.strip( ) , self.logger )
			self.logger.info( "Flagged '%s' → '%s' with note: %s" , Path( self.current_file ).name , dst , desc[ :80 ] )
			self.status_lbl.configure( text="🚩 Flagged → ALTERATIONS REQUIRED" , text_color=ACCENT )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Flag move failed: %s" , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	def _send_to_unessential( self ) :
		self._release_media( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		self.logger.info( "Sending to UNESSENTIAL: '%s'" , Path( self.current_file ).name )
		try :
			dst = _unique( UNESSENTIAL_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.logger.info( "Moved '%s' → '%s'" , Path( self.current_file ).name , dst )
			self.status_lbl.configure( text="↓ Moved to UNESSENTIAL" , text_color=FG2 )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Unessential move failed: %s" , e )
			messagebox.showerror( "Error" , f"Move failed: {e}" )

	def _delete_file( self ) :
		self._release_media( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		fname = Path( self.current_file ).name
		if not messagebox.askyesno( "Confirm Delete" , f"Move '{fname}' to DELETE folder?" ) :
			self.logger.info( "Delete cancelled by user for '%s'" , fname )
			return
		try :
			dst = _unique( DELETE_DIR , Path( self.current_file ) )
			shutil.move( str( Path( self.current_file ) ) , str( dst ) )
			self.logger.info( "Deleted '%s' → '%s'" , fname , dst )
			self.status_lbl.configure( text="🗑 Deleted" , text_color=RED )
			self.show_next( )
		except Exception as e :
			self.logger.error( "Delete move failed: %s" , e )
			messagebox.showerror( "Error" , f"Delete failed: {e}" )

	# ══════════════════════════════════════════════════════════════════════
	# NAVIGATION
	# ══════════════════════════════════════════════════════════════════════
	def show_next( self ) :
		if self.file_start :
			elapsed = time.time( ) - self.file_start
			self.file_times.append( elapsed )
			self.logger.debug( "File processing time: %.1fs" , elapsed )
		self._release_media( )

		if self.current_index < len( self.history ) - 1 :
			self.current_index += 1
			self.current_file = self.history[ self.current_index ][ 0 ]
			self.logger.info( "Navigating forward (history) → '%s' [%d]" ,
												Path( self.current_file ).name , self.current_index )
		elif self.queue :
			self.current_file = self.queue.popleft( )
			self.history.append( [ Path( self.current_file ) , [ ] , None ] )
			self.current_index = len( self.history ) - 1
			self.file_start = time.time( )
			self.logger.info( "Dequeued new file → '%s' [%d] (queue remaining: %d)" ,
												Path( self.current_file ).name , self.current_index , len( self.queue ) )
		else :
			self.current_file = None
			self._clear_preview( );
			self._update_info_bar( )
			self._reset_dest_checkboxes( );
			self._refresh_conv_visibility( )
			# Flush any remaining pending files
			if self.pending :
				self.logger.info( "All files reviewed — flushing remaining %d pending files." , len( self.pending ) )
				self._flush_silent( )
			self.logger.info( "All files have been processed. Starting auto-close countdown." )
			self.status_lbl.configure( text="✓ All files reviewed!" , text_color=GREEN )
			self._start_auto_close( )
			return

		self._on_file_changed( )
		self._preload_next( )

	def show_previous( self ) :
		self._release_media( )
		if self.current_index > 0 :
			self.current_index -= 1
			self.current_file = self.history[ self.current_index ][ 0 ]
			self.logger.info( "Navigating back → '%s' [%d]" ,
												Path( self.current_file ).name , self.current_index )
			self._on_file_changed( )

	def _jump_to( self , idx: int ) :
		if 0 <= idx < len( self.history ) :
			self._release_media( )
			self.current_index = idx
			self.current_file = self.history[ idx ][ 0 ]
			self.logger.info( "Jumping to history [%d] → '%s'" , idx , Path( self.current_file ).name )
			self._on_file_changed( )

	def _on_file_changed( self ) :
		self._reset_dest_checkboxes( )
		self._restore_dest_selections( )
		self._refresh_conv_visibility( )
		self._display_file( )
		self._update_info_bar( )
		self._refresh_action_label( )
		self._update_history_highlight( )

	def _update_info_bar( self ) :
		if self.current_file :
			total = len( self.history ) + len( self.queue )
			self.file_lbl.configure(
					text=f"{Path( self.current_file ).name}  ({self.current_index + 1}/{total})" )
		else :
			self.file_lbl.configure( text="—" )
		self.queue_lbl.configure(
				text=f"Queue: {len( self.queue )}  |  Flagged: {len( self.flagged )}" )

	# ══════════════════════════════════════════════════════════════════════
	# AUTO-CLOSE COUNTDOWN
	# ══════════════════════════════════════════════════════════════════════
	def _start_auto_close( self ) :
		self._closing = True
		self._close_remaining = 20
		self._close_popup = ctk.CTkToplevel( self.root )
		self._close_popup.title( "All Files Processed" )
		self._close_popup.geometry( "420x180" )
		self._close_popup.configure( fg_color=SIDEBAR )
		self._close_popup.attributes( "-topmost" , True )
		self._close_popup.resizable( False , False )

		ctk.CTkLabel( self._close_popup , text="✓ All files have been processed!" ,
									font=(FONT , 16 , "bold") , text_color=GREEN ).pack( pady=(24 , 8) )
		self._close_countdown_lbl = ctk.CTkLabel(
				self._close_popup , text=f"Auto-closing in {self._close_remaining}s…" ,
				font=(FONT , 13) , text_color=FG2 )
		self._close_countdown_lbl.pack( pady=(0 , 16) )
		ctk.CTkButton( self._close_popup , text="Cancel — Keep Open" ,
									 command=self._cancel_auto_close ,
									 fg_color=ACCENT , hover_color=ACCENT_L , width=180 , height=38 ,
									 corner_radius=10 , text_color=FG , font=(FONT , 12 , "bold") ,
									 ).pack( )

		self._close_popup.protocol( "WM_DELETE_WINDOW" , self._cancel_auto_close )
		self._tick_auto_close( )

	def _tick_auto_close( self ) :
		if not self._closing : return
		if self._close_remaining <= 0 :
			self.logger.info( "Auto-close countdown reached 0. Shutting down." )
			try : self._close_popup.destroy( )
			except : pass
			self.root.destroy( )
			return
		self._close_countdown_lbl.configure( text=f"Auto-closing in {self._close_remaining}s…" )
		self._close_remaining -= 1
		self._close_after_id = self.root.after( 1000 , self._tick_auto_close )

	def _cancel_auto_close( self ) :
		self.logger.info( "Auto-close cancelled by user." )
		self._closing = False
		if self._close_after_id :
			self.root.after_cancel( self._close_after_id )
			self._close_after_id = None
		try : self._close_popup.destroy( )
		except : pass
		self.status_lbl.configure( text="✓ All files reviewed! (staying open)" , text_color=GREEN )

	# ══════════════════════════════════════════════════════════════════════
	# PRELOADER
	# ══════════════════════════════════════════════════════════════════════
	def _preload_next( self ) :
		next_file = None
		if self.current_index < len( self.history ) - 1 :
			next_file = self.history[ self.current_index + 1 ][ 0 ]
		elif self.queue :
			next_file = self.queue[ 0 ]
		if not next_file or not next_file.exists( ) : return
		key = str( next_file )
		if key in self._preload_cache or key in self._preloading : return
		self._preloading.add( key )
		self._executor.submit( self._preload_worker , next_file , key )

	def _preload_worker( self , file: Path , key: str ) :
		try :
			cat = _file_category( file )
			if cat == "image" :
				img = Image.open( file ).copy( )
				self._preload_cache[ key ] = img
			elif cat == "text" :
				self._preload_cache[ key ] = file.read_text( encoding="utf-8" , errors="ignore" )
			elif cat == "pdf" :
				doc = fitz.open( file )
				pages = [ ]
				for i in range( min( 2 , len( doc ) ) ) :
					pix = doc[ i ].get_pixmap( matrix=fitz.Matrix( 0.5 , 0.5 ) )
					pages.append( Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples ) )
				doc.close( )
				self._preload_cache[ key ] = ("pdf_pages" , pages , len( doc ))
		except Exception as e :
			self.logger.debug( "Preload failed for '%s': %s" , file.name , e )
		finally :
			self._preloading.discard( key )

	# ══════════════════════════════════════════════════════════════════════
	# TIMER
	# ══════════════════════════════════════════════════════════════════════
	def _tick_timer( self ) :
		elapsed = int( time.time( ) - self.session_start )
		avg = sum( self.file_times ) / len( self.file_times ) if self.file_times else 0
		rem = int( len( self.queue ) * avg / 60 ) if avg > 0 else 0
		self.timer_lbl.configure( text=f"⏱ {elapsed}s  ·  avg {avg:.1f}s  ·  ~{rem}m left" )
		self.root.after( 1000 , self._tick_timer )

	# ══════════════════════════════════════════════════════════════════════
	# HISTORY SIDEBAR
	# ══════════════════════════════════════════════════════════════════════
	def _rebuild_history_sidebar( self ) :
		for w in self.hist_frame.winfo_children( ) : w.destroy( )
		self._hist_widgets.clear( )
		for idx in range( len( self.history ) ) :
			self._add_history_card( idx )
		self._update_history_highlight( )

	def _add_history_card( self , idx: int ) :
		file , dests , conv = self.history[ idx ]
		is_flag = file in self.flagged
		frame = ctk.CTkFrame( self.hist_frame , fg_color=CARD , corner_radius=10 ,
													border_width=0 , border_color=CARD )
		frame.pack( fill=tk.X , padx=4 , pady=4 )
		frame.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )
		thumb_lbl = ctk.CTkLabel( frame , text="" , fg_color=CARD , cursor="hand2" )
		thumb_lbl.bind( "<Button-1>" , lambda _ , i=idx : self._jump_to( i ) )
		thumb = self._thumb_cache.get( str( file ) )
		if thumb :
			thumb_lbl.configure( image=thumb )
			thumb_lbl.pack( pady=(6 , 2) , padx=6 )
		else :
			self._executor.submit( self._gen_thumb_bg , file , (260 , 100) , str( file ) )
		name = (file.name[ :29 ] + "…") if len( file.name ) > 30 else file.name
		name_lbl = ctk.CTkLabel( frame ,
														 text=("🚩 " if is_flag else "") + name ,
														 font=(FONT , 10) , text_color=ACCENT if is_flag else FG ,
														 wraplength=252 , justify="left" )
		name_lbl.pack( fill=tk.X , padx=8 , pady=(0 , 2) )
		sub = [ ]
		if conv :
			opt = _ALL_CONV_BY_KEY.get( conv )
			sub.append( f"⚡ {opt.label if opt else conv}" )
		if dests : sub.append( "→ " + ", ".join( dests ) )
		action_lbl = ctk.CTkLabel( frame ,
															 text="  ".join( sub ) if sub else "[no action]" ,
															 font=(FONT , 9) , text_color=FG2 , wraplength=252 )
		action_lbl.pack( fill=tk.X , padx=8 , pady=(0 , 6) )
		self._hist_widgets.append( {
			"frame"    : frame , "thumb_lbl" : thumb_lbl ,
			"name_lbl" : name_lbl , "action_lbl" : action_lbl ,
		} )

	def _ensure_history_card( self , idx: int ) :
		while len( self._hist_widgets ) <= idx :
			self._add_history_card( len( self._hist_widgets ) )

	def _update_history_highlight( self ) :
		for i , hw in enumerate( self._hist_widgets ) :
			is_curr = i == self.current_index
			file = self.history[ i ][ 0 ] if i < len( self.history ) else None
			is_flag = file in self.flagged if file else False
			bg = CARD_SEL if is_curr else CARD
			bdr = ACCENT if is_flag else (BLUE if is_curr else CARD)
			hw[ "frame" ].configure( fg_color=bg ,
															 border_width=2 if (is_curr or is_flag) else 0 , border_color=bdr )
			hw[ "name_lbl" ].configure( font=(FONT , 10 , "bold" if is_curr else "normal") )
			if i < len( self.history ) :
				_ , dests , conv = self.history[ i ]
				sub = [ ]
				if conv :
					opt = _ALL_CONV_BY_KEY.get( conv )
					sub.append( f"⚡ {opt.label if opt else conv}" )
				if dests : sub.append( "→ " + ", ".join( dests ) )
				hw[ "action_lbl" ].configure( text="  ".join( sub ) if sub else "[no action]" )

	def _update_thumb_in_sidebar( self , key: str , photo: ctk.CTkImage ) :
		for i , hw in enumerate( self._hist_widgets ) :
			if i < len( self.history ) and str( self.history[ i ][ 0 ] ) == key :
				hw[ "thumb_lbl" ].configure( image=photo )
				if not hw[ "thumb_lbl" ].winfo_manager( ) :
					hw[ "thumb_lbl" ].pack( pady=(6 , 2) , padx=6 )
				break

	# ══════════════════════════════════════════════════════════════════════
	# PREVIEW DISPATCHER
	# ══════════════════════════════════════════════════════════════════════
	def _clear_preview( self ) :
		for w in self.prev_scroll.winfo_children( ) : w.destroy( )

	def _display_file( self ) :
		self._clear_preview( )
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		cat = _file_category( Path( self.current_file ) )
		self.logger.debug( "Displaying file '%s' (category: %s, zoom: %d%%)" ,
											 Path( self.current_file ).name , cat , int( self.zoom_scale * 100 ) )
		try :
			{ "pdf"   : self._show_pdf , "image" : self._show_image ,
				"video" : self._show_video , "audio" : self._show_audio ,
				"html"  : self._show_html , "text" : self._show_text ,
				}.get( cat , self._show_metadata )( )
		except Exception as e :
			self.logger.error( "Display failed for '%s': %s" , Path( self.current_file ).name , e )
			self._show_metadata( err=str( e ) )
		self._ensure_history_card( self.current_index )

	# ── metadata fallback ────────────────────────────────────────────────
	def _show_metadata( self , err: str = "" ) :
		ctk.CTkLabel( self.prev_scroll , text=f"📋  {Path( self.current_file ).name}" ,
									font=(EMOJI_FONT , 13 , "bold") , text_color=ACCENT ,
									).pack( anchor="w" , padx=16 , pady=(12 , 4) )
		if err :
			ctk.CTkLabel( self.prev_scroll , text=f"⚠ {err}" ,
										font=(FONT , 10) , text_color=YELLOW ).pack( anchor="w" , padx=16 , pady=(0 , 8) )
		meta = _tika_metadata( Path( self.current_file ) )
		box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
													font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=14 , pady=8 )
		box.insert( "1.0" , meta );
		box.configure( state="disabled" )

	# ── HTML ─────────────────────────────────────────────────────────────
	def _show_html( self ) :
		html = Path( self.current_file ).read_text( encoding="utf-8" , errors="ignore" )
		ctk.CTkLabel( self.prev_scroll , text=f"🌐  {Path( self.current_file ).name}" ,
									font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ,
									).pack( anchor="w" , padx=14 , pady=(8 , 4) )
		container = ctk.CTkFrame( self.prev_scroll , fg_color=CARD , corner_radius=10 )
		container.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )
		try :
			hf = HtmlFrame( container , horizontal_scrollbar="auto" )
			hf.pack( fill=tk.BOTH , expand=True )
			hf.load_html( html );
			hf.set_zoom( self.zoom_scale );
			return
		except : pass
		box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
													font=("Consolas" , 10) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=6 )
		box.insert( "1.0" , html );
		box.configure( state="disabled" )

	# ── PDF ──────────────────────────────────────────────────────────────
	def _show_pdf( self ) :
		key = str( Path( self.current_file ) )
		cached = self._preload_cache.pop( key , None )

		doc = fitz.open( Path( self.current_file ) )
		total = len( doc )
		show = min( total , FILE_TRIAGE_MAX_PDF_PG )

		ctk.CTkLabel( self.prev_scroll ,
									text=f"PDF  ·  {total} page(s)" + (f"  ·  showing first {show}" if total > show else "") ,
									font=(FONT , 11 , "bold") , text_color=ACCENT ).pack( pady=(10 , 4) )

		grid = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW )
		grid.pack( fill=tk.BOTH , expand=True , padx=14 )
		cols = min( 4 , show )
		for c in range( cols ) : grid.grid_columnconfigure( c , weight=1 )

		# Base scale 1.0 so that at 50% zoom → 0.5 matrix (readable)
		scale = 1.0 * self.zoom_scale
		preloaded_pages = [ ]
		if cached and isinstance( cached , tuple ) and cached[ 0 ] == "pdf_pages" :
			preloaded_pages = cached[ 1 ]

		def render_page( i , pil_img=None ) :
			frame = ctk.CTkFrame( grid , fg_color=CARD , corner_radius=10 )
			frame.grid( row=i // cols , column=i % cols , padx=8 , pady=8 , sticky="nsew" )
			if pil_img is None :
				pix = doc[ i ].get_pixmap( matrix=fitz.Matrix( scale , scale ) )
				pil_img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
			else :
				nw = int( pil_img.width * self.zoom_scale )
				nh = int( pil_img.height * self.zoom_scale )
				pil_img = pil_img.resize( (max( 1 , nw ) , max( 1 , nh )) , Image.Resampling.LANCZOS )
			ctk_img = ctk.CTkImage( light_image=pil_img , size=(pil_img.width , pil_img.height) )
			lbl = ctk.CTkLabel( frame , text="" , image=ctk_img , fg_color=CARD )
			lbl._ctk_img_ref = ctk_img
			lbl.pack( pady=6 )
			ctk.CTkLabel( frame , text=f"p{i + 1}" , font=(FONT , 9) , text_color=FG2 ).pack( pady=(0 , 6) )

		eager = max( 2 , len( preloaded_pages ) )
		for i in range( min( eager , show ) ) :
			pre_img = preloaded_pages[ i ] if i < len( preloaded_pages ) else None
			render_page( i , pre_img )

		def render_deferred( idx ) :
			if idx >= show : doc.close( ); return
			try : render_page( idx )
			except : pass
			self.root.after( 30 , lambda : render_deferred( idx + 1 ) )

		if show > eager :
			self.root.after( 60 , lambda : render_deferred( eager ) )
		else :
			doc.close( )

	# ── image ────────────────────────────────────────────────────────────
	def _show_image( self ) :
		key = str( Path( self.current_file ) )
		cached = self._preload_cache.pop( key , None )
		if cached and isinstance( cached , Image.Image ) :
			img = cached
		else :
			img = Image.open( Path( self.current_file ) )
		ow , oh = img.size
		ctk.CTkLabel( self.prev_scroll ,
									text=f"{ow}×{oh}  ·  {Path( self.current_file ).suffix.upper( ).lstrip( '.' )}" ,
									font=(FONT , 10) , text_color=FG2 ).pack( anchor="w" , padx=14 , pady=(8 , 2) )
		nw , nh = int( ow * self.zoom_scale ) , int( oh * self.zoom_scale )
		if nw > 1400 : nh = int( nh * 1400 / nw ); nw = 1400
		disp = img.resize( (max( 1 , nw ) , max( 1 , nh )) , Image.Resampling.LANCZOS )
		ctk_img = ctk.CTkImage( light_image=disp , size=(nw , nh) )
		lbl = ctk.CTkLabel( self.prev_scroll , text="" , image=ctk_img , fg_color=PREVIEW )
		lbl._ctk_img_ref = ctk_img
		lbl.pack( pady=10 )

	# ── video ────────────────────────────────────────────────────────────
	def _show_video( self ) :
		self.video_cap = cv2.VideoCapture( str( Path( self.current_file ) ) )
		cap = self.video_cap
		fps = cap.get( cv2.CAP_PROP_FPS ) or 25
		nfrm = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
		dur = nfrm / fps if fps else 0
		vw , vh = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) ) , int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
		pw = min( 720 , int( vw * 0.55 * self.zoom_scale ) )
		ph = int( vh * (pw / vw) ) if vw > 0 else 360
		self._vid_size = (pw , ph)
		self._vid_delay = max( 1 , int( 1000 / min( fps , 15 ) ) )

		ctk.CTkLabel( self.prev_scroll ,
									text=f"🎬  {Path( self.current_file ).name}  ·  {vw}×{vh}  ·  {dur:.1f}s" ,
									font=(EMOJI_FONT , 12 , "bold") , text_color=ACCENT ).pack( pady=(10 , 4) )
		ctk.CTkLabel( self.prev_scroll ,
									text="⚠ Preview only — Open External for full playback" ,
									font=(FONT , 9) , text_color=YELLOW ).pack( )

		ret , frame = cap.read( )
		ctk_img = None
		if ret :
			frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
			frame = cv2.resize( frame , self._vid_size )
			pil = Image.fromarray( frame )
			ctk_img = ctk.CTkImage( light_image=pil , size=self._vid_size )

		self.video_lbl = ctk.CTkLabel( self.prev_scroll , text="" , fg_color="#000000" )
		if ctk_img :
			self.video_lbl.configure( image=ctk_img )
			self.video_lbl._ctk_img_ref = ctk_img
		self.video_lbl.pack( pady=8 )
		cap.set( cv2.CAP_PROP_POS_FRAMES , 0 )

		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW );
		ctrl.pack( )
		self.is_playing = False

		def toggle( ) :
			self.is_playing = not self.is_playing
			play_btn.configure( text="⏸ Pause" if self.is_playing else "▶ Play" )
			if self.is_playing : _next( )

		play_btn = ctk.CTkButton( ctrl , text="▶ Play" , command=toggle ,
															fg_color=BLUE , hover_color=_hover( BLUE ) , width=90 , height=34 ,
															corner_radius=8 , text_color=FG )
		play_btn.pack( side=tk.LEFT , padx=4 )

		def _next( ) :
			if not self.is_playing or not self.video_cap : return
			ret , frame = self.video_cap.read( )
			if ret :
				frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
				frame = cv2.resize( frame , self._vid_size )
				pil = Image.fromarray( frame )
				ci = ctk.CTkImage( light_image=pil , size=self._vid_size )
				self.video_lbl.configure( image=ci )
				self.video_lbl._ctk_img_ref = ci
				self.video_after_id = self.root.after( self._vid_delay , _next )
			else :
				self.video_cap.set( cv2.CAP_PROP_POS_FRAMES , 0 )
				self.is_playing = False;
				play_btn.configure( text="▶ Play" )

		if nfrm > 0 :
			ctk.CTkSlider( self.prev_scroll , from_=0 , to=nfrm - 1 ,
										 number_of_steps=min( nfrm - 1 , 500 ) ,
										 command=lambda v : self._vid_seek( int( v ) ) ,
										 fg_color=CARD , progress_color=ACCENT , button_color=ACCENT_L ,
										 width=400 , height=16 ).pack( pady=(4 , 8) )

	def _vid_seek( self , frame_num: int ) :
		if self.video_cap and self.video_lbl :
			self.video_cap.set( cv2.CAP_PROP_POS_FRAMES , frame_num )
			ret , frame = self.video_cap.read( )
			if ret :
				frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
				frame = cv2.resize( frame , self._vid_size )
				pil = Image.fromarray( frame )
				ci = ctk.CTkImage( light_image=pil , size=self._vid_size )
				self.video_lbl.configure( image=ci )
				self.video_lbl._ctk_img_ref = ci

	# ── audio ────────────────────────────────────────────────────────────
	def _show_audio( self ) :
		ctk.CTkLabel( self.prev_scroll , text=f"🎵  {Path( self.current_file ).name}" ,
									font=(EMOJI_FONT , 14 , "bold") , text_color=ACCENT ).pack( pady=(28 , 8) )
		meta = _tika_metadata( Path( self.current_file ) )
		box = ctk.CTkTextbox( self.prev_scroll , height=120 , fg_color=CARD ,
													text_color=FG2 , font=(FONT , 10) , corner_radius=8 )
		box.pack( fill=tk.X , padx=20 , pady=8 )
		box.insert( "1.0" , meta or "—" );
		box.configure( state="disabled" )
		ctrl = ctk.CTkFrame( self.prev_scroll , fg_color=PREVIEW );
		ctrl.pack( pady=10 )

		def play( ) :
			try : pygame.mixer.music.load( str( Path( self.current_file ) ) ); pygame.mixer.music.play( )
			except Exception as e : messagebox.showerror( "Playback Error" , str( e ) )

		for txt , cmd , col in [ ("▶ Play" , play , GREEN) ,
														 ("⏸ Pause" , pygame.mixer.music.pause , YELLOW) ,
														 ("⏹ Stop" , pygame.mixer.music.stop , ACCENT) ] :
			ctk.CTkButton( ctrl , text=txt , command=cmd , fg_color=col ,
										 hover_color=_hover( col ) , width=90 , height=36 , corner_radius=8 ,
										 text_color=FG , font=(FONT , 11) ).pack( side=tk.LEFT , padx=4 )
		self.root.after( 300 , play )

	# ── text ─────────────────────────────────────────────────────────────
	def _show_text( self ) :
		key = str( Path( self.current_file ) )
		cached = self._preload_cache.pop( key , None )
		content = cached if isinstance( cached , str ) else \
			Path( self.current_file ).read_text( encoding="utf-8" , errors="ignore" )
		lines = len( content.splitlines( ) )
		ctk.CTkLabel( self.prev_scroll ,
									text=f"{lines} lines  ·  {Path( self.current_file ).suffix.upper( ).lstrip( '.' )}" ,
									font=(FONT , 10) , text_color=FG2 ).pack( anchor="w" , padx=14 , pady=(8 , 2) )
		box = ctk.CTkTextbox( self.prev_scroll , fg_color=CARD , text_color=FG ,
													font=("Consolas" , max( 8 , int( 10 * self.zoom_scale ) )) , corner_radius=10 )
		box.pack( fill=tk.BOTH , expand=True , padx=12 , pady=8 )
		box.insert( "1.0" , content );
		box.configure( state="disabled" )

	# ══════════════════════════════════════════════════════════════════════
	# OPEN EXTERNAL
	# ══════════════════════════════════════════════════════════════════════
	def _open_external( self ) :
		if not self.current_file or not Path( self.current_file ).exists( ) : return
		p = Path( self.current_file )
		self.logger.info( "Opening external: '%s'" , p )
		try :
			if os.name == "nt" : os.startfile( p )
			elif os.name == "posix" : subprocess.run( [ "xdg-open" , str( p ) ] )
			else : subprocess.run( [ "open" , str( p ) ] )
		except Exception as e :
			self.logger.error( "Open external failed: %s" , e )
			messagebox.showerror( "Error" , str( e ) )

	# ══════════════════════════════════════════════════════════════════════
	# THUMBNAIL GENERATION
	# ══════════════════════════════════════════════════════════════════════
	def _gen_thumb_bg( self , file: Path , size: tuple , key: str ) :
		if not file.exists( ) : return
		if key in self._thumb_cache and self._thumb_cache[ key ] is not None : return
		self._thumb_cache[ key ] = None
		thumb_img = None;
		ext = file.suffix.lower( )
		try :
			if ext == ".pdf" :
				doc = fitz.open( file )
				if doc :
					pix = doc[ 0 ].get_pixmap( matrix=fitz.Matrix( 0.2 , 0.2 ) )
					img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
					img.thumbnail( size , Image.Resampling.LANCZOS );
					doc.close( )
					thumb_img = img
			elif ext in { ".jpg" , ".jpeg" , ".png" , ".gif" , ".bmp" , ".webp" } :
				img = Image.open( file );
				img.thumbnail( size , Image.Resampling.LANCZOS )
				thumb_img = img
			elif ext in { ".mp4" , ".avi" , ".mkv" , ".mov" } :
				cap = cv2.VideoCapture( str( file ) );
				ret , frame = cap.read( );
				cap.release( )
				if ret :
					img = Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
					img.thumbnail( size , Image.Resampling.LANCZOS );
					thumb_img = img
			elif ext in { ".docx" , ".pptx" , ".xlsx" } :
				with ZipFile( file , "r" ) as z :
					thumbs = [ n for n in z.namelist( ) if "thumbnail" in n.lower( ) ]
					if thumbs and thumbs[ 0 ].lower( ).endswith( (".jpg" , ".jpeg" , ".png") ) :
						img = Image.open( io.BytesIO( z.read( thumbs[ 0 ] ) ) )
						img.thumbnail( size , Image.Resampling.LANCZOS );
						thumb_img = img
		except : return
		if thumb_img :
			self.root.after( 0 , lambda : self._set_thumb( key , thumb_img ) )

	def _set_thumb( self , key: str , pil_img: Image.Image ) :
		try :
			photo = ctk.CTkImage( light_image=pil_img , size=(pil_img.width , pil_img.height) )
			self._thumb_cache[ key ] = photo
			self._update_thumb_in_sidebar( key , photo )
		except : pass

	# ══════════════════════════════════════════════════════════════════════
	# RUN
	# ══════════════════════════════════════════════════════════════════════
	def run( self ) :
		if any( item.is_file( ) for item in self.source_dir.iterdir( ) ) :
			self.logger.info( "Starting mainloop." )
			self.root.mainloop( )
			self.logger.info( "Mainloop exited." )
		else :
			self.logger.warning( "No items found in %s. Not running." , self.source_dir )
