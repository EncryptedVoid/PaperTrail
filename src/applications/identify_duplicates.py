#!/usr/bin/env python3
"""
Duplicate Finder & Reviewer  v2 — Redesigned
─────────────────────────────────────────────
Performance overhaul:
  • LRU preview cache (PIL images, PDF renders, text, metadata)
  • Background preloading of upcoming pairs while user reviews current
  • Stable widget tree — widgets created once, updated via itemconfig
  • All heavy I/O off the main thread
  • Pure CTk where possible; raw tk.Canvas only for pixel rendering
"""

import difflib
import hashlib
import itertools
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
import uuid
from collections import Counter , OrderedDict , defaultdict
from concurrent.futures import Future , ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog , messagebox

import customtkinter as ctk

ctk.set_appearance_mode( "Dark" )
ctk.set_default_color_theme( "blue" )

from PIL import Image , ImageTk
import fitz
import imagehash
import cv2

from config import (
	ARCHIVAL_DIR , DOCUMENT_TYPES , IMAGE_TYPES ,
	JAVA_PATH , TIKA_APP_JAR_PATH ,
)

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SALMON = "#E8735A";
SALMON_HV = "#C95C45"
BG_APP = "#1A1A1A";
BG_PANEL = "#242424"
BG_CARD = "#2E2E2E";
BG_TAB = "#383838"
FG_TEXT = "#E0E0E0";
FG_DIM = "#888888"
FG_GOOD = "#5DBD72";
FG_WARN = "#D9A740"
ACCENT_BLUE = "#3B82F6"
RADIUS = 10
FONT = ("Segoe UI" , 10)
FONT_B = ("Segoe UI" , 10 , "bold")
FONT_S = ("Segoe UI" , 9)
FONT_H = ("Segoe UI" , 11 , "bold")
FONT_MONO = ("Consolas" , 9)

_IMG_EXTS = { f".{e}" for e in IMAGE_TYPES }
_DOC_EXTS = { f".{e}" for e in DOCUMENT_TYPES } | { ".txt" , ".md" , ".csv" , ".log" }
_VID_EXTS = { ".mp4" , ".mov" , ".avi" , ".mkv" , ".webm" , ".m4v" }
_TXT_EXTS = { ".txt" , ".md" , ".csv" , ".log" , ".json" , ".yaml" , ".yml" , ".toml" , ".xml" }
_CAMERA_RAW_EXTS = { ".nef" , ".heic" , ".arw" , ".cr2" }

CACHE_MAX_ITEMS = 64  # max cached preview renders
PREFETCH_AHEAD = 5  # pairs to preload in background
BG_WORKERS = 3  # background thread pool size


# ══════════════════════════════════════════════════════════════════════════════
# THREADED LRU PREVIEW CACHE
# ══════════════════════════════════════════════════════════════════════════════

class PreviewCache :
	"""
    Thread-safe LRU cache for decoded preview data.
    Stores:  (fp, kind) → payload
      kind="image"    → PIL.Image (raw, pre-decode)
      kind="pdf"      → list[PIL.Image] (one per page)
      kind="text"     → str
      kind="meta"     → str
      kind="video"    → PIL.Image (first frame)
    """

	def __init__( self , maxsize: int = CACHE_MAX_ITEMS ) :
		self._data: OrderedDict = OrderedDict( )
		self._maxsize = maxsize
		self._lock = threading.Lock( )
		self._pool = ThreadPoolExecutor( max_workers=BG_WORKERS , thread_name_prefix="pcache" )
		self._pending: dict[ tuple , Future ] = { }
		self._logger: logging.Logger | None = None

	def set_logger( self , logger: logging.Logger ) :
		self._logger = logger

	def get( self , fp: str , kind: str ) :
		key = (fp , kind)
		with self._lock :
			if key in self._data :
				self._data.move_to_end( key )
				return self._data[ key ]
		return None

	def put( self , fp: str , kind: str , value ) :
		key = (fp , kind)
		with self._lock :
			self._data[ key ] = value
			self._data.move_to_end( key )
			while len( self._data ) > self._maxsize :
				self._data.popitem( last=False )

	def prefetch( self , fp: str , kinds: list[ str ] | None = None ) :
		"""Submit background jobs to warm the cache for a file."""
		if not Path( fp ).exists( ) :
			return
		if kinds is None :
			kinds = self._auto_kinds( fp )
		for k in kinds :
			key = (fp , k)
			with self._lock :
				if key in self._data or key in self._pending :
					continue
			fut = self._pool.submit( self._load , fp , k )
			with self._lock :
				self._pending[ key ] = fut

	def prefetch_pair( self , pair: tuple[ str , str ] ) :
		for fp in pair :
			self.prefetch( fp )

	def _auto_kinds( self , fp: str ) -> list[ str ] :
		ext = Path( fp ).suffix.lower( )
		kinds = [ "meta" ]
		if ext in _IMG_EXTS :        kinds.insert( 0 , "image" )
		elif ext == ".pdf" :         kinds.insert( 0 , "pdf" )
		elif ext in _TXT_EXTS :      kinds.insert( 0 , "text" )
		elif ext in _VID_EXTS :      kinds.insert( 0 , "video" )
		return kinds

	def _load( self , fp: str , kind: str ) :
		key = (fp , kind)
		try :
			if kind == "image" :
				val = Image.open( fp )
				val.load( )  # force full decode now
			elif kind == "pdf" :
				val = self._load_pdf( fp )
			elif kind == "text" :
				val = Path( fp ).read_text( errors="ignore" )[ :8000 ]
			elif kind == "video" :
				val = self._load_video_frame( fp )
			elif kind == "meta" :
				val = self._load_meta( fp )
			else :
				return
			self.put( fp , kind , val )
		except Exception as exc :
			if self._logger :
				self._logger.debug( f"PreviewCache._load({Path( fp ).name}, {kind}) failed: {exc}" )
		finally :
			with self._lock :
				self._pending.pop( key , None )

	def _load_pdf( self , fp: str ) -> list[ Image.Image ] :
		doc = fitz.open( fp )
		pages = [ ]
		for page in doc :
			pix = page.get_pixmap( matrix=fitz.Matrix( 1.5 , 1.5 ) )
			img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
			pages.append( img )
		doc.close( )
		return pages

	def _load_video_frame( self , fp: str ) -> Image.Image | None :
		cap = cv2.VideoCapture( fp )
		ret , frame = cap.read( )
		cap.release( )
		if ret :
			return Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
		return None

	def _load_meta( self , fp: str ) -> str :
		stats = _file_stats( fp )
		tika = _tika_meta( fp , self._logger )
		return f"{stats}\n\n── Tika Metadata (--json) ──\n{tika}"

	def shutdown( self ) :
		self._pool.shutdown( wait=False , cancel_futures=True )


# Module-level singleton — created by DuplicateReviewer
_preview_cache: PreviewCache | None = None


# ══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _btn( parent , text , cmd , width=None , color=SALMON , hover=SALMON_HV , **kw ) :
	b = ctk.CTkButton(
			parent , text=text , command=cmd ,
			fg_color=color , hover_color=hover ,
			text_color="white" , corner_radius=6 , font=FONT , **kw ,
	)
	if width :
		b.configure( width=width )
	return b


def _section_label( parent , text ) :
	return ctk.CTkLabel( parent , text=text , font=FONT_H , text_color=SALMON )


def _uuid4_path( dest: Path , src: Path ) -> Path :
	dst = dest / f"{src.stem}__{uuid.uuid4( ).hex[ :8 ]}{src.suffix}"
	while dst.exists( ) :
		dst = dest / f"{src.stem}__{uuid.uuid4( ).hex[ :8 ]}{src.suffix}"
	return dst


def _unique_path( dest: Path , src: Path ) -> Path :
	dst = dest / src.name
	n = 1
	while dst.exists( ) :
		dst = dest / f"{src.stem}_{n}{src.suffix}"
		n += 1
	return dst


def _fmt_size( n: int ) -> str :
	for u in ("B" , "KB" , "MB" , "GB") :
		if n < 1024 :
			return f"{n:.1f} {u}"
		n /= 1024
	return f"{n:.1f} TB"


def sanitize_artifact_name( artifact_name: str ) -> str :
	stem , *ext_parts = artifact_name.rsplit( "." , 1 )
	pattern = r'\s*[-_]?\s*\(?\bcopy\b\)?\s*(\(\d+\))?|\s+\(\d+\)$'
	stem = re.sub( pattern , "" , stem , flags=re.IGNORECASE ).strip( )
	return stem.capitalize( )


def _tika_meta( path: str , logger: logging.Logger | None = None ) -> str :
	file_path = Path( path )
	tika_jar = Path( str( TIKA_APP_JAR_PATH ) )
	if not file_path.exists( ) :
		return f"File not found: {file_path}"
	cmd = [ JAVA_PATH , "-jar" , str( tika_jar ) , "--json" , str( file_path ) ]
	try :
		r = subprocess.run(
				cmd , capture_output=True , text=True ,
				encoding="utf-8" , errors="replace" , timeout=30 ,
		)
		raw = r.stdout.strip( )
		if not raw :
			return r.stderr.strip( ) or "Tika returned no output."
		try :
			data = json.loads( raw )
			if isinstance( data , list ) :
				data = data[ 0 ] if data else { }
		except json.JSONDecodeError :
			return raw
		if not isinstance( data , dict ) :
			return raw

		PRIORITY = [
			"Content-Type" , "File Name" , "File Size" ,
			"Last-Modified" , "Creation-Date" , "Last-Save-Date" ,
			"Author" , "dc:creator" , "meta:author" ,
			"Title" , "dc:title" , "subject" , "dc:subject" ,
			"description" , "dc:description" ,
			"Keywords" , "meta:keyword" , "Language" , "dc:language" ,
			"Producer" , "pdf:PDFVersion" ,
			"xmpTPg:NPages" , "Page-Count" ,
			"Image Width" , "Image Height" ,
			"Compression Type" , "Color Space" ,
			"X Resolution" , "Y Resolution" ,
		]
		seen , lines = set( ) , [ ]
		for k in PRIORITY :
			if k in data :
				lines.append( f"{k:<32} {data[ k ]}" )
				seen.add( k )
		if seen :
			lines.append( "" )
		for k in sorted( data.keys( ) ) :
			if k not in seen and not k.startswith( "X-" ) :
				v = data[ k ]
				if isinstance( v , list ) :
					v = "; ".join( str( i ) for i in v )
				lines.append( f"{k:<32} {v}" )
		return "\n".join( lines ) if lines else "No metadata fields returned."
	except subprocess.TimeoutExpired :
		return "Tika timed out (>30 s)."
	except FileNotFoundError :
		return "java not found on PATH — install a JRE."
	except Exception as e :
		return f"Tika error: {e}"


def _tika_text( path: str ) -> str :
	try :
		r = subprocess.run(
				[ str( JAVA_PATH ) , "-jar" , str( TIKA_APP_JAR_PATH ) , "--text" , path ] ,
				capture_output=True , text=True ,
				encoding="utf-8" , errors="replace" , timeout=30 ,
		)
		return r.stdout.strip( )
	except Exception :
		return ""


def _file_stats( path: str ) -> str :
	p = Path( path )
	if not p.exists( ) :
		return "File not found."
	s = p.stat( )
	return "\n".join( [
		f"Name:     {p.name}" ,
		f"Path:     {p.parent}" ,
		f"Size:     {_fmt_size( s.st_size )} ({s.st_size:,} bytes)" ,
		f"Modified: {time.strftime( '%Y-%m-%d %H:%M:%S' , time.localtime( s.st_mtime ) )}" ,
		f"Created:  {time.strftime( '%Y-%m-%d %H:%M:%S' , time.localtime( s.st_ctime ) )}" ,
		f"Type:     {p.suffix.lower( )}" ,
	] )


def _groups_to_pairs( groups: list ) -> list[ tuple[ str , str ] ] :
	pairs , seen = [ ] , set( )
	for grp in groups :
		for a , b in itertools.combinations( grp , 2 ) :
			key = (min( a , b ) , max( a , b ))
			if key not in seen :
				seen.add( key )
				pairs.append( (a , b) )
	return pairs


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ENGINE  (unchanged logic, cleaned formatting)
# ══════════════════════════════════════════════════════════════════════════════

class DetectionEngine :
	_WEIGHTS = {
		"exact"  : 1.0 , "names" : 0.3 , "names_cross_ext" : 0.3 ,
		"images" : 4.0 , "docs" : 3.0 ,
	}

	def __init__( self , folder , recursive , img_thresh , doc_thresh ,
								do_images , do_docs , do_exact , do_names , do_names_cross_ext ,
								min_kb , on_progress , on_done , on_error ) :
		self.folder = Path( folder )
		self.recursive = recursive
		self.img_thresh = int( img_thresh )
		self.doc_thresh = float( doc_thresh )
		self.do_images = do_images
		self.do_docs = do_docs
		self.do_exact = do_exact
		self.do_names = do_names
		self.do_names_cross_ext = do_names_cross_ext
		self.min_bytes = int( min_kb ) * 1024
		self._cancel = False
		self._progress = on_progress
		self._done = on_done
		self._error = on_error

	def cancel( self ) :
		self._cancel = True

	def _build_phases( self ) :
		enabled = [ ]
		if self.do_exact :           enabled.append( "exact" )
		if self.do_names :           enabled.append( "names" )
		if self.do_names_cross_ext : enabled.append( "names_cross_ext" )
		if self.do_images :          enabled.append( "images" )
		if self.do_docs :            enabled.append( "docs" )
		if not enabled :
			return [ ]
		total = sum( self._WEIGHTS[ p ] for p in enabled )
		phases , cursor = [ ] , 0.02
		for p in enabled :
			w = self._WEIGHTS[ p ] / total * 0.98
			phases.append( (p , cursor , cursor + w) )
			cursor += w
		return phases

	def _phase_progress( self , msg , frac , start , end ) :
		self._progress( msg , start + frac * (end - start) )

	def run( self ) :
		try :
			self._progress( "Collecting files…" , 0.0 )
			files = self._collect( )
			if self._cancel : return
			phases = self._build_phases( )
			pm = { n : (s , e) for n , s , e in phases }
			pairs = [ ]

			if self.do_exact and not self._cancel :
				s , e = pm[ "exact" ]
				self._progress( "Finding exact duplicates…" , s )
				pairs += self._exact( files , s , e )
			if self.do_names and not self._cancel :
				s , e = pm[ "names" ]
				self._progress( "Matching sanitized filenames…" , s )
				pairs += self._names( files , s , e )
			if self.do_names_cross_ext and not self._cancel :
				s , e = pm[ "names_cross_ext" ]
				self._progress( "Matching filenames across extensions…" , s )
				pairs += self._names_cross_ext( files , s , e )
			if self.do_images and not self._cancel :
				s , e = pm[ "images" ]
				pairs += self._images( files , s , e )
			if self.do_docs and not self._cancel :
				s , e = pm[ "docs" ]
				pairs += self._docs( files , s , e )
			if self._cancel : return
			self._done( self._union_find( pairs ) )
		except Exception as e :
			self._error( str( e ) )

	def _collect( self ) :
		it = self.folder.rglob( "*" ) if self.recursive else self.folder.iterdir( )
		out = [ ]
		for p in it :
			try :
				if p.is_file( ) and p.stat( ).st_size >= self.min_bytes :
					out.append( p )
			except OSError :
				pass
		return out

	def _exact( self , files , p_start , p_end ) :
		buckets = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel : return [ ]
			if i % 30 == 0 :
				self._phase_progress( f"Checksumming {i}/{n}…" , i / max( n , 1 ) , p_start , p_end )
			try :
				h = hashlib.sha256( f.read_bytes( ) ).hexdigest( )
				buckets[ h ].append( str( f ) )
			except Exception :
				pass
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	def _names( self , files , p_start , p_end ) :
		buckets = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel : return [ ]
			if f.suffix.lower( ) in _CAMERA_RAW_EXTS : continue
			if i % 50 == 0 :
				self._phase_progress( f"Matching filenames {i}/{n}…" , i / max( n , 1 ) , p_start , p_end )
			buckets[ (sanitize_artifact_name( f.name ) , f.suffix.lower( )) ].append( str( f ) )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	def _names_cross_ext( self , files , p_start , p_end ) :
		buckets = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel : return [ ]
			if f.suffix.lower( ) in _CAMERA_RAW_EXTS : continue
			if i % 50 == 0 :
				self._phase_progress( f"Cross-ext match {i}/{n}…" , i / max( n , 1 ) , p_start , p_end )
			buckets[ sanitize_artifact_name( f.name ) ].append( str( f ) )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	def _images( self , files , p_start , p_end ) :
		img_files = [ f for f in files if f.suffix.lower( ) in _IMG_EXTS ]
		n = len( img_files )
		if n == 0 : return [ ]
		p_mid = p_start + 0.6 * (p_end - p_start)
		hashes = { }
		for i , f in enumerate( img_files ) :
			if self._cancel : return [ ]
			if i % 10 == 0 :
				self._phase_progress( f"Hashing images {i + 1}/{n}…" , i / max( n , 1 ) , p_start , p_mid )
			try :
				img = Image.open( f ).convert( "RGB" )
				hashes[ str( f ) ] = imagehash.phash( img )
			except Exception :
				pass
		items = list( hashes.items( ) )
		m = len( items )
		pairs = [ ]
		for i in range( m ) :
			if self._cancel : return pairs
			if i % 100 == 0 :
				self._phase_progress( f"Comparing images {i}/{m}…" , i / max( m , 1 ) , p_mid , p_end )
			for j in range( i + 1 , m ) :
				if items[ i ][ 1 ] - items[ j ][ 1 ] <= self.img_thresh :
					pairs.append( [ items[ i ][ 0 ] , items[ j ][ 0 ] ] )
		return pairs

	def _docs( self , files , p_start , p_end ) :
		doc_files = [ f for f in files if f.suffix.lower( ) in _DOC_EXTS ]
		n = len( doc_files )
		if n == 0 : return [ ]
		p_mid = p_start + 0.5 * (p_end - p_start)
		texts = { }
		for i , f in enumerate( doc_files ) :
			if self._cancel : return [ ]
			if i % 5 == 0 :
				self._phase_progress( f"Extracting text {i + 1}/{n}…" , i / max( n , 1 ) , p_start , p_mid )
			txt = self._extract_text( f )
			if txt.strip( ) :
				texts[ str( f ) ] = txt
		items = list( texts.items( ) )
		m = len( items )
		total_cmp = max( m * (m - 1) // 2 , 1 )
		pairs , done = [ ] , 0
		for i in range( m ) :
			if self._cancel : return pairs
			for j in range( i + 1 , m ) :
				done += 1
				if done % 20 == 0 :
					self._phase_progress( f"Comparing docs {done}/{total_cmp}…" , done / total_cmp , p_mid , p_end )
				ratio = difflib.SequenceMatcher( None , texts[ items[ i ][ 0 ] ] , texts[ items[ j ][ 0 ] ] ,
																				 autojunk=True ).ratio( )
				if ratio >= self.doc_thresh :
					pairs.append( [ items[ i ][ 0 ] , items[ j ][ 0 ] ] )
		return pairs

	def _extract_text( self , f: Path , max_chars=8000 ) -> str :
		ext = f.suffix.lower( )
		try :
			if ext == ".pdf" :
				doc = fitz.open( f )
				return " ".join( p.get_text( ) for p in doc )[ :max_chars ]
			if ext in { ".txt" , ".md" , ".csv" , ".log" } :
				return f.read_text( errors="ignore" )[ :max_chars ]
			return _tika_text( str( f ) )[ :max_chars ]
		except Exception :
			return ""

	@staticmethod
	def _union_find( pairs ) :
		parent = { }

		def find( x ) :
			parent.setdefault( x , x )
			while parent[ x ] != x :
				parent[ x ] = parent[ parent[ x ] ]
				x = parent[ x ]
			return x

		def union( x , y ) :
			parent[ find( x ) ] = find( y )

		for grp in pairs :
			for a , b in itertools.combinations( grp , 2 ) :
				union( a , b )
		buckets = defaultdict( list )
		for item in list( parent.keys( ) ) :
			buckets[ find( item ) ].append( item )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]


# ══════════════════════════════════════════════════════════════════════════════
# PREVIEW PANE  — stable widget tree, cache-backed, async loading
# ══════════════════════════════════════════════════════════════════════════════

class PreviewPane :
	"""
    Displays a file preview (image / PDF / text / video thumb / metadata).
    Widgets are created ONCE; content is swapped via canvas itemconfig.
    All heavy I/O goes through PreviewCache.
    """

	def __init__( self , parent , side_label: str ) :
		self._fp: str | None = None
		self._zoom = 1.0
		self._fit_scale = 1.0
		self._mode = "preview"
		self._photo_refs: list = [ ]  # prevent GC of PhotoImage
		self._raw_img: Image.Image | None = None
		self._meta_text: str | None = None
		self._loading = False

		self.frame = ctk.CTkFrame( parent , fg_color=BG_CARD , corner_radius=RADIUS )
		self.frame.rowconfigure( 2 , weight=1 )
		self.frame.columnconfigure( 0 , weight=1 )

		# ── header row ──────────────────────────────────────────────────
		hdr = ctk.CTkFrame( self.frame , fg_color=BG_CARD , corner_radius=0 , height=30 )
		hdr.grid( row=0 , column=0 , sticky="ew" , padx=8 , pady=(8 , 2) )
		ctk.CTkLabel( hdr , text=side_label , font=FONT_B , text_color=SALMON ).pack( side="left" , padx=6 )
		self._name_lbl = ctk.CTkLabel( hdr , text="—" , font=FONT_S , text_color=FG_TEXT , wraplength=340 )
		self._name_lbl.pack( side="left" , padx=6 )
		self._size_lbl = ctk.CTkLabel( hdr , text="" , font=FONT_S , text_color=FG_DIM )
		self._size_lbl.pack( side="right" , padx=6 )

		# ── tab bar ─────────────────────────────────────────────────────
		tabs = ctk.CTkFrame( self.frame , fg_color=BG_CARD , corner_radius=0 , height=30 )
		tabs.grid( row=1 , column=0 , sticky="ew" , padx=8 , pady=(0 , 4) )

		self._tb_prev = ctk.CTkButton(
				tabs , text="Preview" , command=self._show_preview ,
				fg_color=SALMON , hover_color=SALMON_HV ,
				text_color="white" , width=88 , height=24 , corner_radius=6 , font=FONT_S )
		self._tb_prev.pack( side="left" , padx=(0 , 4) )

		self._tb_meta = ctk.CTkButton(
				tabs , text="Metadata" , command=self._show_metadata ,
				fg_color=BG_TAB , hover_color=BG_PANEL ,
				text_color=FG_DIM , width=88 , height=24 , corner_radius=6 , font=FONT_S )
		self._tb_meta.pack( side="left" )

		self._zoom_lbl = ctk.CTkLabel( tabs , text="Fit" , font=FONT_S , text_color=FG_DIM , width=40 )
		self._zoom_lbl.pack( side="right" , padx=2 )
		_btn( tabs , "+" , self._zoom_in , width=26 , color=BG_TAB , hover=BG_PANEL ).pack( side="right" , padx=2 )
		_btn( tabs , "−" , self._zoom_out , width=26 , color=BG_TAB , hover=BG_PANEL ).pack( side="right" , padx=2 )
		_btn( tabs , "↗ Open" , self._open_external , width=76 , color=BG_TAB , hover=BG_PANEL ).pack( side="right" ,
																																																	 padx=(8 , 4) )

		# ── canvas (stable — never destroyed) ───────────────────────────
		wrap = tk.Frame( self.frame , bg=BG_APP )
		wrap.grid( row=2 , column=0 , sticky="nsew" , padx=8 , pady=(0 , 8) )
		wrap.rowconfigure( 0 , weight=1 )
		wrap.columnconfigure( 0 , weight=1 )

		self._canvas = tk.Canvas( wrap , bg=BG_APP , highlightthickness=0 , cursor="fleur" )
		vsb = tk.Scrollbar( wrap , orient="vertical" , command=self._canvas.yview )
		hsb = tk.Scrollbar( wrap , orient="horizontal" , command=self._canvas.xview )
		self._canvas.configure( yscrollcommand=vsb.set , xscrollcommand=hsb.set )
		self._canvas.grid( row=0 , column=0 , sticky="nsew" )
		vsb.grid( row=0 , column=1 , sticky="ns" )
		hsb.grid( row=1 , column=0 , sticky="ew" )

		# Create persistent canvas items — we'll update them, not recreate
		self._img_item = self._canvas.create_image( 0 , 0 , anchor="nw" )
		self._txt_item = self._canvas.create_text(
				10 , 10 , anchor="nw" , text="" , fill=FG_TEXT , font=FONT_S , width=430 )
		self._overlay_item = self._canvas.create_text(
				10 , 10 , anchor="nw" , text="" , fill=FG_WARN , font=FONT_S , width=430 )

		# Wheel scroll
		for seq in ("<MouseWheel>" , "<Button-4>" , "<Button-5>") :
			self._canvas.bind( seq , self._on_wheel )

	# ── public API ───────────────────────────────────────────────────────

	def load( self , fp: str | None ) :
		"""Switch to a new file.  Tries cache first, falls back to async."""
		self._fp = fp
		self._zoom = 1.0
		self._fit_scale = 1.0
		self._meta_text = None
		self._raw_img = None
		self._mode = "preview"
		self._update_tabs( )
		self._zoom_lbl.configure( text="Fit" )
		self._clear_canvas( )

		if fp and Path( fp ).exists( ) :
			p = Path( fp )
			self._name_lbl.configure( text=p.name )
			self._size_lbl.configure( text=_fmt_size( p.stat( ).st_size ) )
		else :
			self._name_lbl.configure( text="—" )
			self._size_lbl.configure( text="" )

		self._render( )

	# ── tab switching ────────────────────────────────────────────────────

	def _show_preview( self ) :
		self._mode = "preview"
		self._update_tabs( )
		self._render( )

	def _show_metadata( self ) :
		self._mode = "metadata"
		self._update_tabs( )
		self._render( )

	def _update_tabs( self ) :
		if self._mode == "preview" :
			self._tb_prev.configure( fg_color=SALMON , text_color="white" )
			self._tb_meta.configure( fg_color=BG_TAB , text_color=FG_DIM )
		else :
			self._tb_prev.configure( fg_color=BG_TAB , text_color=FG_DIM )
			self._tb_meta.configure( fg_color=SALMON , text_color="white" )

	# ── canvas helpers ───────────────────────────────────────────────────

	def _clear_canvas( self ) :
		"""Hide all persistent items (cheaper than delete+recreate)."""
		self._canvas.itemconfigure( self._img_item , image="" )
		self._canvas.itemconfigure( self._txt_item , text="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._photo_refs.clear( )
		self._canvas.configure( scrollregion=(0 , 0 , 0 , 0) )

	def _canvas_size( self ) -> tuple[ int , int ] :
		self._canvas.update_idletasks( )
		return max( self._canvas.winfo_width( ) , 100 ) , max( self._canvas.winfo_height( ) , 100 )

	def _show_text( self , text: str , color: str = FG_TEXT , font=None ) :
		"""Display text in the text item."""
		self._canvas.itemconfigure( self._img_item , image="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._canvas.itemconfigure( self._txt_item , text=text , fill=color , font=font or FONT_S )
		self._canvas.configure( scrollregion=(0 , 0 , 460 , text.count( "\n" ) * 15 + 40) )

	def _show_image( self , photo: ImageTk.PhotoImage , w: int , h: int ) :
		"""Display a PhotoImage in the image item."""
		self._photo_refs.append( photo )
		self._canvas.itemconfigure( self._img_item , image=photo )
		self._canvas.coords( self._img_item , 0 , 0 )
		self._canvas.itemconfigure( self._txt_item , text="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._canvas.configure( scrollregion=(0 , 0 , w , h) )

	def _show_loading( self ) :
		self._show_text( "Loading…" , FG_WARN )

	# ── main render dispatch ─────────────────────────────────────────────

	def _render( self ) :
		if not self._fp or not Path( self._fp ).exists( ) :
			self._show_text( "No file loaded." , FG_DIM )
			return

		if self._mode == "metadata" :
			self._render_metadata( )
			return

		ext = Path( self._fp ).suffix.lower( )
		if ext in _IMG_EXTS :       self._render_image( )
		elif ext == ".pdf" :        self._render_pdf( )
		elif ext in _TXT_EXTS :     self._render_text_file( )
		elif ext in _VID_EXTS :     self._render_video( )
		else :                      self._render_metadata( )

	# ── image ────────────────────────────────────────────────────────────

	def _render_image( self ) :
		# Try cache first
		if self._raw_img is None :
			cached = _preview_cache.get( self._fp , "image" ) if _preview_cache else None
			if cached :
				self._raw_img = cached
			else :
				# Load synchronously but it's likely already cached from prefetch
				try :
					self._raw_img = Image.open( self._fp )
					self._raw_img.load( )
					if _preview_cache :
						_preview_cache.put( self._fp , "image" , self._raw_img )
				except Exception as e :
					self._show_text( f"Image error: {e}" , FG_DIM )
					return

		img = self._raw_img
		cw , ch = self._canvas_size( )
		self._fit_scale = min( cw / max( img.width , 1 ) , ch / max( img.height , 1 ) , 1.0 )
		eff = self._fit_scale * self._zoom
		w = max( 1 , int( img.width * eff ) )
		h = max( 1 , int( img.height * eff ) )
		display = img.resize( (w , h) , Image.LANCZOS )
		ph = ImageTk.PhotoImage( display )
		self._show_image( ph , w , h )

	# ── pdf ──────────────────────────────────────────────────────────────

	def _render_pdf( self ) :
		pages = _preview_cache.get( self._fp , "pdf" ) if _preview_cache else None
		if pages is None :
			try :
				doc = fitz.open( self._fp )
				pages = [ ]
				for page in doc :
					pix = page.get_pixmap( matrix=fitz.Matrix( 1.5 , 1.5 ) )
					pages.append( Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples ) )
				doc.close( )
				if _preview_cache :
					_preview_cache.put( self._fp , "pdf" , pages )
			except Exception as e :
				self._show_text( f"PDF error: {e}" , FG_DIM )
				return

		if not pages :
			self._show_text( "Empty PDF." , FG_DIM )
			return

		# Composite all pages into a vertical strip
		cw , _ = self._canvas_size( )
		base_scale = min( cw / max( pages[ 0 ].width , 1 ) , 1.4 )
		scale = base_scale * self._zoom
		gap = 8

		# Clear old images, rebuild composite
		self._clear_canvas( )
		y , max_w = 0 , 0
		for page_img in pages :
			w = max( 1 , int( page_img.width * scale ) )
			h = max( 1 , int( page_img.height * scale ) )
			resized = page_img.resize( (w , h) , Image.LANCZOS )
			ph = ImageTk.PhotoImage( resized )
			self._photo_refs.append( ph )
			self._canvas.create_image( 0 , y , anchor="nw" , image=ph )
			y += h + gap
			max_w = max( max_w , w )
		self._canvas.configure( scrollregion=(0 , 0 , max_w , y) )

	# ── text ─────────────────────────────────────────────────────────────

	def _render_text_file( self ) :
		cached = _preview_cache.get( self._fp , "text" ) if _preview_cache else None
		if cached is None :
			try :
				cached = Path( self._fp ).read_text( errors="ignore" )[ :8000 ]
				if _preview_cache :
					_preview_cache.put( self._fp , "text" , cached )
			except Exception as e :
				self._show_text( f"Read error: {e}" , FG_DIM )
				return
		self._show_text( cached , FG_TEXT , FONT_MONO )

	# ── video ────────────────────────────────────────────────────────────

	def _render_video( self ) :
		frame_img = _preview_cache.get( self._fp , "video" ) if _preview_cache else None
		if frame_img is None :
			try :
				cap = cv2.VideoCapture( self._fp )
				ret , frame = cap.read( )
				cap.release( )
				if ret :
					frame_img = Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
					if _preview_cache :
						_preview_cache.put( self._fp , "video" , frame_img )
				else :
					self._show_text( "Could not decode video frame." , FG_DIM )
					return
			except Exception as e :
				self._show_text( f"Video error: {e}" , FG_DIM )
				return

		cw , ch = self._canvas_size( )
		fit = min( cw / max( frame_img.width , 1 ) , (ch - 30) / max( frame_img.height , 1 ) , 1.0 )
		w = max( 1 , int( frame_img.width * fit * self._zoom ) )
		h = max( 1 , int( frame_img.height * fit * self._zoom ) )
		display = frame_img.resize( (w , h) , Image.LANCZOS )
		ph = ImageTk.PhotoImage( display )
		self._photo_refs.append( ph )
		self._canvas.itemconfigure( self._img_item , image=ph )
		self._canvas.coords( self._img_item , 0 , 0 )
		self._canvas.itemconfigure( self._txt_item , text="" )
		self._canvas.itemconfigure(
				self._overlay_item ,
				text="▶ Video — thumbnail only. Use ↗ Open for playback." ,
				fill=FG_WARN )
		self._canvas.coords( self._overlay_item , 8 , h + 10 )
		self._canvas.configure( scrollregion=(0 , 0 , w , h + 30) )

	# ── metadata ─────────────────────────────────────────────────────────

	def _render_metadata( self ) :
		if self._meta_text is None :
			cached = _preview_cache.get( self._fp , "meta" ) if _preview_cache else None
			if cached :
				self._meta_text = cached
			else :
				self._show_text( "Extracting metadata…" , FG_WARN )
				# Do the extraction in a thread, then call back
				threading.Thread(
						target=self._load_meta_bg , args=(self._fp ,) , daemon=True ,
				).start( )
				return

		self._show_text( self._meta_text , FG_TEXT , FONT_MONO )

	def _load_meta_bg( self , fp: str ) :
		"""Called from a background thread — loads meta then schedules UI update."""
		stats = _file_stats( fp )
		tika = _tika_meta( fp )
		result = f"{stats}\n\n── Tika Metadata (--json) ──\n{tika}"
		if _preview_cache :
			_preview_cache.put( fp , "meta" , result )
		# Schedule UI update on main thread
		try :
			self.frame.after( 0 , self._on_meta_ready , fp , result )
		except Exception :
			pass

	def _on_meta_ready( self , fp: str , text: str ) :
		"""Callback on main thread after metadata extraction completes."""
		if self._fp == fp :  # still viewing the same file
			self._meta_text = text
			if self._mode == "metadata" :
				self._show_text( text , FG_TEXT , FONT_MONO )

	# ── zoom ─────────────────────────────────────────────────────────────

	def _zoom_in( self ) :
		self._zoom = min( self._zoom + 0.25 , 6.0 )
		self._zoom_lbl.configure( text=f"{int( self._fit_scale * self._zoom * 100 )}%" )
		if self._mode == "preview" :
			self._render( )

	def _zoom_out( self ) :
		self._zoom = max( self._zoom - 0.25 , 0.25 )
		self._zoom_lbl.configure( text=f"{int( self._fit_scale * self._zoom * 100 )}%" )
		if self._mode == "preview" :
			self._render( )

	def _on_wheel( self , event ) :
		if event.num == 4 :       self._canvas.yview_scroll( -1 , "units" )
		elif event.num == 5 :     self._canvas.yview_scroll( 1 , "units" )
		else :                    self._canvas.yview_scroll( int( -event.delta / 120 ) , "units" )

	def _open_external( self ) :
		if not self._fp :
			return
		try :
			if os.name == "nt" :  os.startfile( self._fp )
			else :                subprocess.Popen( [ "xdg-open" , self._fp ] )
		except Exception as e :
			messagebox.showerror( "Open Error" , str( e ) )


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FRAME
# ══════════════════════════════════════════════════════════════════════════════

class ScanFrame( ctk.CTkFrame ) :
	def __init__( self , parent , on_groups_ready , default_folder="" ) :
		super( ).__init__( parent , fg_color="transparent" )
		self._on_groups_ready = on_groups_ready
		self._default_folder = default_folder
		self._engine: DetectionEngine | None = None
		self._thread: threading.Thread | None = None
		self._q: queue.Queue = queue.Queue( )
		self._groups = [ ]
		self._build( )

	def _build( self ) :
		outer = ctk.CTkFrame( self , fg_color="transparent" )
		outer.pack( fill="both" , expand=True , padx=12 , pady=8 )
		outer.columnconfigure( 0 , weight=1 )
		row = 0

		# ── Section 1: Folder ───────────────────────────────────────────
		sec1 = ctk.CTkFrame( outer , fg_color=BG_PANEL , corner_radius=RADIUS )
		sec1.grid( row=row , column=0 , sticky="ew" , pady=(0 , 8) );
		row += 1
		_section_label( sec1 , "📁  Source Folder" ).pack( anchor="w" , padx=14 , pady=(12 , 6) )

		fr = ctk.CTkFrame( sec1 , fg_color="transparent" )
		fr.pack( fill="x" , padx=14 , pady=(0 , 12) )
		self._folder_var = tk.StringVar( value=self._default_folder )
		ctk.CTkEntry(
				fr , textvariable=self._folder_var , width=560 , font=FONT ,
				fg_color=BG_CARD , text_color=FG_TEXT , border_color=BG_TAB , corner_radius=6 ,
		).pack( side="left" , padx=(0 , 8) , fill="x" , expand=True )
		_btn( fr , "Browse…" , self._browse , width=90 ).pack( side="left" )
		self._recursive = tk.BooleanVar( value=True )
		ctk.CTkCheckBox( fr , text="Include subfolders" , variable=self._recursive , font=FONT , text_color=FG_TEXT ).pack(
				side="left" , padx=(16 , 0) )

		# ── Section 2: Detection Modes ──────────────────────────────────
		sec2 = ctk.CTkFrame( outer , fg_color=BG_PANEL , corner_radius=RADIUS )
		sec2.grid( row=row , column=0 , sticky="ew" , pady=(0 , 8) );
		row += 1
		_section_label( sec2 , "🔎  Detection Modes" ).pack( anchor="w" , padx=14 , pady=(12 , 8) )

		modes = ctk.CTkFrame( sec2 , fg_color="transparent" )
		modes.pack( fill="x" , padx=14 , pady=(0 , 12) )
		modes.columnconfigure( (0 , 1) , weight=1 )

		self._do_exact = tk.BooleanVar( value=True )
		self._do_names = tk.BooleanVar( value=True )
		self._do_names_cross_ext = tk.BooleanVar( value=False )
		self._do_img = tk.BooleanVar( value=True )
		self._do_doc = tk.BooleanVar( value=True )

		mode_defs = [
			("Exact duplicates" , self._do_exact , "SHA-256 byte match · fast") ,
			("Filename match" , self._do_names , "Sanitized name + same extension") ,
			("Cross-ext filename" , self._do_names_cross_ext , "Sanitized name · any extension") ,
			("Images  (pHash)" , self._do_img , "Perceptual hash · requires imagehash") ,
			("Documents  (text)" , self._do_doc , "Text similarity · PyMuPDF / Tika") ,
		]
		for i , (label , var , hint) in enumerate( mode_defs ) :
			card = ctk.CTkFrame( modes , fg_color=BG_CARD , corner_radius=8 )
			card.grid( row=i // 2 , column=i % 2 , sticky="ew" , padx=4 , pady=3 )
			ctk.CTkCheckBox( card , text=label , variable=var , font=FONT_B , text_color=FG_TEXT ).pack( anchor="w" ,
																																																	 padx=10 ,
																																																	 pady=(8 , 2) )
			ctk.CTkLabel( card , text=hint , font=FONT_S , text_color=FG_DIM ).pack( anchor="w" , padx=30 , pady=(0 , 8) )

		# ── Section 3: Thresholds ───────────────────────────────────────
		sec3 = ctk.CTkFrame( outer , fg_color=BG_PANEL , corner_radius=RADIUS )
		sec3.grid( row=row , column=0 , sticky="ew" , pady=(0 , 8) );
		row += 1
		_section_label( sec3 , "⚙  Thresholds" ).pack( anchor="w" , padx=14 , pady=(12 , 8) )

		thr = ctk.CTkFrame( sec3 , fg_color="transparent" )
		thr.pack( fill="x" , padx=14 , pady=(0 , 12) )
		thr.columnconfigure( (0 , 1 , 2) , weight=1 )

		# Image threshold
		ic = ctk.CTkFrame( thr , fg_color=BG_CARD , corner_radius=8 )
		ic.grid( row=0 , column=0 , sticky="ew" , padx=4 , pady=3 )
		ctk.CTkLabel( ic , text="Image pHash distance" , font=FONT , text_color=FG_TEXT ).pack( anchor="w" , padx=10 ,
																																														pady=(8 , 2) )
		isr = ctk.CTkFrame( ic , fg_color="transparent" )
		isr.pack( fill="x" , padx=10 , pady=(0 , 8) )
		self._img_thresh_val = tk.IntVar( value=10 )
		self._img_thresh_lbl = ctk.CTkLabel( isr , text="10" , font=FONT_B , text_color=SALMON , width=28 )
		self._img_thresh_lbl.pack( side="left" )
		sl_img = ctk.CTkSlider(
				isr , from_=0 , to=20 , number_of_steps=20 , width=140 ,
				command=lambda v : (
					self._img_thresh_val.set( int( round( v ) ) ) ,
					self._img_thresh_lbl.configure( text=str( int( round( v ) ) ) )) )
		sl_img.set( 10 )
		sl_img.pack( side="left" , padx=4 )
		ctk.CTkLabel( isr , text="0 = exact · 20 = loose" , font=FONT_S , text_color=FG_DIM ).pack( side="left" , padx=4 )

		# Doc threshold
		dc = ctk.CTkFrame( thr , fg_color=BG_CARD , corner_radius=8 )
		dc.grid( row=0 , column=1 , sticky="ew" , padx=4 , pady=3 )
		ctk.CTkLabel( dc , text="Document similarity" , font=FONT , text_color=FG_TEXT ).pack( anchor="w" , padx=10 ,
																																													 pady=(8 , 2) )
		dsr = ctk.CTkFrame( dc , fg_color="transparent" )
		dsr.pack( fill="x" , padx=10 , pady=(0 , 8) )
		self._doc_thresh_val = tk.DoubleVar( value=0.85 )
		self._doc_thresh_lbl = ctk.CTkLabel( dsr , text="85 %" , font=FONT_B , text_color=SALMON , width=42 )
		self._doc_thresh_lbl.pack( side="left" )
		sl_doc = ctk.CTkSlider(
				dsr , from_=0.5 , to=1.0 , number_of_steps=50 , width=140 ,
				command=lambda v : (
					self._doc_thresh_val.set( v ) ,
					self._doc_thresh_lbl.configure( text=f"{int( round( v * 100 ) )} %" )) )
		sl_doc.set( 0.85 )
		sl_doc.pack( side="left" , padx=4 )

		# Min size
		mc = ctk.CTkFrame( thr , fg_color=BG_CARD , corner_radius=8 )
		mc.grid( row=0 , column=2 , sticky="ew" , padx=4 , pady=3 )
		ctk.CTkLabel( mc , text="Min file size" , font=FONT , text_color=FG_TEXT ).pack( anchor="w" , padx=10 ,
																																										 pady=(8 , 2) )
		msr = ctk.CTkFrame( mc , fg_color="transparent" )
		msr.pack( fill="x" , padx=10 , pady=(0 , 8) )
		self._min_kb = ctk.CTkEntry(
				msr , width=64 , font=FONT , fg_color=BG_PANEL ,
				text_color=FG_TEXT , border_color=BG_TAB , corner_radius=6 )
		self._min_kb.insert( 0 , "0" )
		self._min_kb.pack( side="left" )
		ctk.CTkLabel( msr , text="KB" , font=FONT , text_color=FG_DIM ).pack( side="left" , padx=6 )

		# ── Section 4: Scan & Results ───────────────────────────────────
		sec4 = ctk.CTkFrame( outer , fg_color=BG_PANEL , corner_radius=RADIUS )
		sec4.grid( row=row , column=0 , sticky="nsew" , pady=(0 , 0) );
		row += 1
		outer.rowconfigure( row - 1 , weight=1 )

		ctrl = ctk.CTkFrame( sec4 , fg_color="transparent" )
		ctrl.pack( fill="x" , padx=14 , pady=(12 , 6) )
		self._scan_btn = _btn( ctrl , "▶  Start Scan" , self._start , width=150 )
		self._scan_btn.pack( side="left" )
		self._cancel_btn = _btn( ctrl , "■  Cancel" , self._cancel , width=100 , color=BG_TAB , hover="#555" )
		self._cancel_btn.pack( side="left" , padx=8 )
		self._cancel_btn.configure( state="disabled" )

		self._status_lbl = ctk.CTkLabel( sec4 , text="Ready — select a folder and press Start." , font=FONT ,
																		 text_color=FG_DIM )
		self._status_lbl.pack( anchor="w" , padx=14 , pady=(0 , 4) )
		self._pbar = ctk.CTkProgressBar( sec4 , height=10 , fg_color=BG_CARD , progress_color=SALMON , corner_radius=5 )
		self._pbar.set( 0 )
		self._pbar.pack( fill="x" , padx=14 , pady=(0 , 8) )

		self._res_lbl = ctk.CTkLabel( sec4 , text="" , font=FONT , text_color=FG_DIM )
		self._res_lbl.pack( padx=14 , pady=(0 , 4) )
		self._proceed_btn = _btn( sec4 , "▶  Proceed to Review →" , self._proceed , width=240 , color=FG_GOOD ,
															hover="#4DAD62" )
		self._proceed_btn.pack( padx=14 , pady=(0 , 14) )
		self._proceed_btn.pack_forget( )

	def _browse( self ) :
		d = filedialog.askdirectory( title="Select folder to scan" )
		if d :
			self._folder_var.set( d )

	def _start( self ) :
		folder = self._folder_var.get( ).strip( )
		if not folder or not Path( folder ).is_dir( ) :
			messagebox.showerror( "Error" , "Please select a valid folder." )
			return
		try :
			min_kb = int( self._min_kb.get( ) or 0 )
		except ValueError :
			min_kb = 0

		self._scan_btn.configure( state="disabled" )
		self._cancel_btn.configure( state="normal" )
		self._pbar.set( 0 )
		self._proceed_btn.pack_forget( )
		self._res_lbl.configure( text="" )
		self._groups = [ ]

		self._engine = DetectionEngine(
				folder=folder , recursive=self._recursive.get( ) ,
				img_thresh=self._img_thresh_val.get( ) , doc_thresh=self._doc_thresh_val.get( ) ,
				do_images=self._do_img.get( ) , do_docs=self._do_doc.get( ) ,
				do_exact=self._do_exact.get( ) , do_names=self._do_names.get( ) ,
				do_names_cross_ext=self._do_names_cross_ext.get( ) , min_kb=min_kb ,
				on_progress=lambda m , p : self._q.put( ("progress" , m , p) ) ,
				on_done=lambda g : self._q.put( ("done" , g) ) ,
				on_error=lambda e : self._q.put( ("error" , e) ) ,
		)
		self._thread = threading.Thread( target=self._engine.run , daemon=True )
		self._thread.start( )
		self._poll( )

	def _cancel( self ) :
		if self._engine :
			self._engine.cancel( )
		self._scan_btn.configure( state="normal" )
		self._cancel_btn.configure( state="disabled" )
		self._status_lbl.configure( text="Cancelled." , text_color=FG_WARN )

	def _poll( self ) :
		try :
			while True :
				item = self._q.get_nowait( )
				kind = item[ 0 ]
				if kind == "progress" :
					_ , msg , pct = item
					self._status_lbl.configure( text=msg , text_color=FG_DIM )
					self._pbar.set( pct )
				elif kind == "done" :
					self._on_done( item[ 1 ] )
					return
				elif kind == "error" :
					self._scan_btn.configure( state="normal" )
					self._cancel_btn.configure( state="disabled" )
					messagebox.showerror( "Scan Error" , item[ 1 ] )
					return
		except queue.Empty :
			pass
		if self._thread and self._thread.is_alive( ) :
			self.after( 100 , self._poll )

	def _on_done( self , groups ) :
		self._groups = groups
		self._scan_btn.configure( state="normal" )
		self._cancel_btn.configure( state="disabled" )
		self._pbar.set( 1.0 )
		n_g = len( groups )
		n_f = sum( len( g ) for g in groups )
		n_p = sum( len( list( itertools.combinations( g , 2 ) ) ) for g in groups )
		if n_g :
			self._status_lbl.configure(
					text=f"✓ Done — {n_g} groups · {n_f} files · {n_p} pairs to review" ,
					text_color=FG_GOOD )
			self._res_lbl.configure(
					text=f"Found {n_g} duplicate group(s) across {n_f} files → {n_p} pair(s) to review." ,
					text_color=FG_TEXT )
			self._proceed_btn.pack( padx=14 , pady=(0 , 14) )
		else :
			self._status_lbl.configure( text="✓ Scan complete — no duplicates found." , text_color=FG_WARN )
			self._res_lbl.configure(
					text="No duplicates found. Try loosening thresholds or enabling more modes." ,
					text_color=FG_WARN )

	def _proceed( self ) :
		if self._groups :
			self._on_groups_ready( self._groups )


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW FRAME
# ══════════════════════════════════════════════════════════════════════════════

class ReviewFrame( ctk.CTkFrame ) :
	def __init__( self , parent , base_dir: Path , logger: logging.Logger ) :
		super( ).__init__( parent , fg_color="transparent" )
		self.base_dir = base_dir
		self.logger = logger
		self._pairs: list[ tuple[ str , str ] ] = [ ]
		self._skipped: list[ tuple[ str , str ] ] = [ ]
		self._idx = 0
		self._log: list = [ ]
		self._stats = Counter( )
		self._lf = self._rf = ""
		self._t0 = time.time( )
		self._build( )

	def load_groups( self , groups ) :
		self._pairs = _groups_to_pairs( groups )
		self._skipped = [ ]
		self._idx , self._log = 0 , [ ]
		self._stats = Counter( loaded=len( self._pairs ) )
		self._t0 = time.time( )
		self._prefetch_upcoming( )
		self._show( )

	def _prefetch_upcoming( self ) :
		"""Warm the preview cache for the next N pairs."""
		if not _preview_cache :
			return
		active = self._active( )
		for i in range( self._idx , min( self._idx + PREFETCH_AHEAD , len( active ) ) ) :
			_preview_cache.prefetch_pair( active[ i ] )

	def _build( self ) :
		# ── top bar ─────────────────────────────────────────────────────
		top = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=46 )
		top.pack( fill="x" , padx=12 , pady=(12 , 6) )
		top.pack_propagate( False )
		_btn( top , "Load JSON manually" , self._load_json , width=160 ).pack( side="left" , padx=12 , pady=8 )
		self._prog_lbl = ctk.CTkLabel( top , text="No pairs loaded." , font=FONT_B , text_color=FG_TEXT )
		self._prog_lbl.pack( side="left" , padx=10 )
		self._stats_lbl = ctk.CTkLabel( top , text="" , font=FONT_S , text_color=FG_DIM )
		self._stats_lbl.pack( side="left" , padx=6 )

		# ── preview panes ───────────────────────────────────────────────
		cmp = ctk.CTkFrame( self , fg_color=BG_APP , corner_radius=0 )
		cmp.pack( fill="both" , expand=True , padx=12 , pady=4 )
		cmp.columnconfigure( 0 , weight=1 )
		cmp.columnconfigure( 1 , weight=1 )
		cmp.rowconfigure( 0 , weight=1 )

		self._lp = PreviewPane( cmp , "← LEFT" )
		self._rp = PreviewPane( cmp , "RIGHT →" )
		self._lp.frame.grid( row=0 , column=0 , sticky="nsew" , padx=(0 , 4) )
		self._rp.frame.grid( row=0 , column=1 , sticky="nsew" , padx=(4 , 0) )

		# ── action bar ──────────────────────────────────────────────────
		acts = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=56 )
		acts.pack( fill="x" , padx=12 , pady=(4 , 12) )
		acts.pack_propagate( False )

		_btn( acts , "←  Keep Left" , self._keep_left , width=210 , color="#2A5A8A" , hover="#3A6A9A" ).pack( side="left" ,
																																																					padx=12 ,
																																																					pady=10 )
		ctk.CTkLabel( acts , text="[←]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )
		_btn( acts , "↑  Keep Both  (UUID4)" , self._keep_both , width=220 , color="#3A7A5A" , hover="#4A8A6A" ).pack(
				side="left" , padx=12 , pady=10 )
		ctk.CTkLabel( acts , text="[↑]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )
		_btn( acts , "↓  Skip" , self._skip , width=100 , color=BG_TAB , hover=BG_PANEL ).pack( side="left" , padx=8 ,
																																														pady=10 )
		ctk.CTkLabel( acts , text="[↓]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )
		ctk.CTkLabel( acts , text="[→]" , font=FONT_S , text_color=FG_DIM ).pack( side="right" , padx=(0 , 8) )
		_btn( acts , "Keep Right  →" , self._keep_right , width=210 , color="#2A5A8A" , hover="#3A6A9A" ).pack(
				side="right" , padx=12 , pady=10 )

		self.after( 200 , self._bind_keys )

	def _bind_keys( self ) :
		root = self.winfo_toplevel( )
		root.bind( "<Left>" , lambda _ : self._keep_left( ) )
		root.bind( "<Right>" , lambda _ : self._keep_right( ) )
		root.bind( "<Up>" , lambda _ : self._keep_both( ) )
		root.bind( "<Down>" , lambda _ : self._skip( ) )

	# ── display ──────────────────────────────────────────────────────────

	def _active( self ) :
		return self._pairs + self._skipped

	def _current( self ) :
		p = self._active( )
		return p[ self._idx ] if self._idx < len( p ) else None

	def _show( self ) :
		while True :
			pair = self._current( )
			if pair is None :
				self._finish( )
				return
			lf , rf = pair
			if Path( lf ).exists( ) and Path( rf ).exists( ) :
				break
			self._idx += 1

		self._lf , self._rf = pair
		total = len( self._active( ) )
		self._prog_lbl.configure( text=f"Pair {self._idx + 1} / {total}" )
		self._stats_lbl.configure( text=(
			f"L:{self._stats.get( 'left' , 0 )}  R:{self._stats.get( 'right' , 0 )}  "
			f"Both:{self._stats.get( 'both' , 0 )}  Skip:{len( self._skipped )}") )
		self._lp.load( self._lf )
		self._rp.load( self._rf )
		self._prefetch_upcoming( )

	# ── decisions ────────────────────────────────────────────────────────

	def _keep_left( self ) :
		if not self._lf : return
		try :
			kept = self._mv( self._lf , self.base_dir )
			archived = self._mv( self._rf , ARCHIVAL_DIR )
			self._rec( { "action" : "keep_left" , "kept" : kept , "archived" : archived } )
			self._stats[ "left" ] += 1
			self.logger.info( f"keep_left | {Path( self._lf ).name}" )
			self._advance( )
		except Exception as e :
			messagebox.showerror( "Error" , str( e ) )

	def _keep_right( self ) :
		if not self._rf : return
		try :
			kept = self._mv( self._rf , self.base_dir )
			archived = self._mv( self._lf , ARCHIVAL_DIR )
			self._rec( { "action" : "keep_right" , "kept" : kept , "archived" : archived } )
			self._stats[ "right" ] += 1
			self.logger.info( f"keep_right | {Path( self._rf ).name}" )
			self._advance( )
		except Exception as e :
			messagebox.showerror( "Error" , str( e ) )

	def _keep_both( self ) :
		if not self._lf or not self._rf : return
		try :
			lp , rp = Path( self._lf ) , Path( self._rf )
			self.base_dir.mkdir( parents=True , exist_ok=True )
			ld = _uuid4_path( self.base_dir , lp )
			rd = _uuid4_path( self.base_dir , rp )
			shutil.move( str( lp ) , str( ld ) )
			shutil.move( str( rp ) , str( rd ) )
			self._rec( {
				"action"         : "keep_both" ,
				"left_original"  : str( lp ) , "left_moved" : str( ld ) ,
				"right_original" : str( rp ) , "right_moved" : str( rd ) ,
			} )
			self._stats[ "both" ] += 1
			self.logger.info( f"keep_both | {ld.name}  +  {rd.name}" )
			self._advance( )
		except Exception as e :
			messagebox.showerror( "Error" , str( e ) )

	def _skip( self ) :
		pair = self._current( )
		if pair is None : return
		if self._idx < len( self._pairs ) :
			self._pairs.pop( self._idx )
			self._skipped.append( pair )
		else :
			self._idx += 1
		self._show( )

	# ── helpers ──────────────────────────────────────────────────────────

	def _mv( self , src , dest ) :
		dest.mkdir( parents=True , exist_ok=True )
		dst = _unique_path( dest , Path( src ) )
		shutil.move( src , dst )
		return str( dst )

	def _rec( self , entry ) :
		entry[ "ts" ] = time.strftime( "%Y-%m-%dT%H:%M:%S" )
		self._log.append( entry )
		self._stats[ "reviewed" ] += 1

	def _advance( self ) :
		self._idx += 1
		self._show( )

	def _load_json( self ) :
		path = filedialog.askopenfilename( title="Load JSON groups/pairs" , filetypes=[ ("JSON" , "*.json") ] )
		if not path : return
		try :
			with open( path ) as f :
				data = json.load( f )
			if not isinstance( data , list ) or not all( isinstance( g , list ) for g in data ) :
				raise ValueError( "JSON must be a list of lists." )
			self.load_groups( data )
		except Exception as e :
			messagebox.showerror( "Load Error" , str( e ) )

	def _finish( self ) :
		log_path = self.base_dir / "decisions.json"
		self.base_dir.mkdir( parents=True , exist_ok=True )
		with open( log_path , "w" ) as f :
			json.dump( self._log , f , indent=2 )
		elapsed = int( time.time( ) - self._t0 )
		self.logger.info(
				f"Review complete | {self._stats[ 'reviewed' ]} pairs | "
				f"L:{self._stats.get( 'left' , 0 )} R:{self._stats.get( 'right' , 0 )} "
				f"Both:{self._stats.get( 'both' , 0 )} | {elapsed}s | log: {log_path}" )
		if _preview_cache :
			_preview_cache.shutdown( )
		self.winfo_toplevel( ).destroy( )


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL APP
# ══════════════════════════════════════════════════════════════════════════════

class DuplicateReviewer :
	def __init__( self , source_dir: Path | None = None , logger: logging.Logger | None = None ) :
		global _preview_cache
		self.logger = logger
		self.source_dir = source_dir

		_preview_cache = PreviewCache( maxsize=CACHE_MAX_ITEMS )
		_preview_cache.set_logger( logger )

		self.root = ctk.CTk( )
		self.root.title( "Duplicate Finder & Reviewer" )
		self.root.geometry( "1440x900" )
		self.root.configure( fg_color=BG_APP )

		self._tabs = ctk.CTkTabview(
				self.root , fg_color=BG_PANEL ,
				segmented_button_fg_color=BG_CARD ,
				segmented_button_selected_color=SALMON ,
				segmented_button_selected_hover_color=SALMON_HV ,
				text_color=FG_TEXT ,
		)
		self._tabs.pack( fill="both" , expand=True , padx=8 , pady=8 )
		self._tabs.add( "🔍  Scan" )
		self._tabs.add( "👁  Review" )

		self._scan = ScanFrame(
				self._tabs.tab( "🔍  Scan" ) ,
				on_groups_ready=self._on_scan_done ,
				default_folder=str( source_dir ) if source_dir else "" ,
		)
		self._scan.pack( fill="both" , expand=True )

		self._review = ReviewFrame(
				self._tabs.tab( "👁  Review" ) ,
				base_dir=source_dir ,
				logger=self.logger ,
		)
		self._review.pack( fill="both" , expand=True )
		self.logger.info( f"App started | source_dir={source_dir}" )

	def _on_scan_done( self , groups ) :
		self._review.load_groups( groups )
		self._tabs.set( "👁  Review" )

	def run( self ) :
		if any( item.is_file( ) for item in self.source_dir.iterdir( ) ) :
			self.root.mainloop( )
		else :
			self.logger.warning( f"No items found in {self.source_dir}. Not running app." )
