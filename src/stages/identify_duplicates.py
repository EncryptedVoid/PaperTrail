#!/usr/bin/env python3
"""
Duplicate Finder & Reviewer  ·  Integrated
────────────────────────────────────────────
Phase 1  SCAN   – detect near-duplicate files in a folder
Phase 2  REVIEW – step through every pair, decide what to keep

Detection
  Images    : perceptual hash (pHash)  ·  Hamming-distance threshold
  Documents : text extraction + difflib SequenceMatcher ratio
  All files : exact SHA-256 match (always runs, free)

Review keys
  ←  Keep Left   →  archive right  to ARCHIVAL_DIR
  →  Keep Right  →  archive left   to ARCHIVAL_DIR
  ↑  Keep Both   →  both moved to base_dir with UUID4 suffix (no collision)
  ↓  Skip        →  defer to end of queue
"""

import difflib
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from collections import Counter , defaultdict
from pathlib import Path
from tkinter import filedialog , messagebox

from config import (ARCHIVAL_DIR , DOCUMENT_TYPES , IMAGE_TYPES , JAVA_PATH , TIKA_APP_JAR_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY ENSURANCE
# ══════════════════════════════════════════════════════════════════════════════

def _pip( *pkgs ) :
	cmd = [ sys.executable , "-m" , "pip" , "install" , "--prefer-binary" , "--quiet" ] + list( pkgs )
	return subprocess.run( cmd ,
												 capture_output=True ,
												 text=True ,
												 encoding="utf-8" ,
												 errors="replace" ,
												 timeout=300 ).returncode == 0


def _installed( pip_name: str , min_ver: str | None = None ) -> bool :
	try :
		ver = importlib.metadata.version( pip_name )
		if min_ver is None :
			return True
		from packaging.version import Version
		return Version( ver ) >= Version( min_ver )
	except Exception :
		return False


def _ensure( pip_name: str , min_ver: str | None = None , label: str | None = None ) -> bool :
	tag = label or pip_name
	spec = f"{pip_name}>={min_ver}" if min_ver else pip_name
	if _installed( pip_name , min_ver ) :
		print( f"✓ {tag} {importlib.metadata.version( pip_name )}" )
		return True
	print( f"  Installing {tag}…" )
	if _pip( spec ) :
		print( f"✓ {tag} installed" )
		return True
	print( f"✗ {tag} failed — some features will be limited" )
	return False


try :
	import customtkinter as ctk

	ctk.set_appearance_mode( "Dark" )
	ctk.set_default_color_theme( "blue" )
except ImportError :
	import tkinter as ctk  # type: ignore

try :
	from PIL import Image , ImageTk

	PIL_AVAILABLE = True
except ImportError :
	PIL_AVAILABLE = False

try :
	import fitz

	FITZ_AVAILABLE = True
except ImportError :
	FITZ_AVAILABLE = False

try :
	import imagehash

	IMAGEHASH_AVAILABLE = True
except ImportError :
	IMAGEHASH_AVAILABLE = False

try :
	import cv2

	CV2_AVAILABLE = True
except ImportError :
	CV2_AVAILABLE = False

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
RADIUS = 10
FONT = ("Segoe UI" , 10)
FONT_B = ("Segoe UI" , 10 , "bold")
FONT_S = ("Segoe UI" , 9)

_IMG_EXTS = { f".{e}" for e in IMAGE_TYPES }
_DOC_EXTS = { f".{e}" for e in DOCUMENT_TYPES } | { ".txt" , ".md" , ".csv" , ".log" }
_VID_EXTS = { ".mp4" , ".mov" , ".avi" , ".mkv" , ".webm" , ".m4v" }


# ══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _btn( parent , text , cmd , width=None , color=SALMON , hover=SALMON_HV , **kw ) :
	b = ctk.CTkButton( parent , text=text , command=cmd ,
										 fg_color=color , hover_color=hover ,
										 text_color="white" , corner_radius=6 , font=FONT , **kw ,
										 )
	if width :
		b.configure( width=width )
	return b


def _uuid4_path( dest: Path , src: Path ) -> Path :
	dst = dest / f"{src.stem}__{uuid.uuid4( ).hex[ :8 ]}{src.suffix}"
	while dst.exists( ) :
		dst = dest / f"{src.stem}__{uuid.uuid4( ).hex[ :8 ]}{src.suffix}"
	return dst


def _unique_path( dest: Path , src: Path ) -> Path :
	dst = dest / src.name
	n = 1
	while dst.exists( ) :
		dst = dest / f"{src.stem}_{n}{src.suffix}";
		n += 1
	return dst


def _fmt_size( n: int ) -> str :
	for u in ("B" , "KB" , "MB" , "GB") :
		if n < 1024 :
			return f"{n:.1f} {u}"
		n /= 1024
	return f"{n:.1f} TB"


def _tika_meta( path: str , logger: logging.Logger | None = None ) -> str :
	"""
    Extract metadata via Apache Tika (--json mode).
    Returns a human-readable string of all key→value pairs.
    Follows the validated call pattern:
        java -jar tika_app.jar --json <file>
    """
	file_path = Path( path )
	tika_jar = Path( str( TIKA_APP_JAR_PATH ) )

	if not file_path.exists( ) :
		msg = f"File not found: {file_path}"
		if logger : logger.error( msg )
		return msg

	file_size = file_path.stat( ).st_size
	if logger : logger.info( f"Tika extraction — {file_path.name}  ({file_size:,} bytes)" )

	if not tika_jar.exists( ) :
		msg = f"Tika JAR not found: {tika_jar}\nSet TIKA_APP_JAR_PATH in config.py"
		if logger : logger.error( msg )
		return msg

	if logger : logger.info( f"Using Tika JAR: {tika_jar}" )

	java = JAVA_PATH or "java"
	cmd = [ java , "-jar" , str( tika_jar ) , "--json" , str( file_path ) ]

	if logger : logger.info( "Executing Tika extraction (--json)" )
	try :
		r = subprocess.run( cmd ,
												capture_output=True ,
												text=True ,
												encoding="utf-8" ,
												errors="replace" ,
												timeout=30 ,
												)
		raw = r.stdout.strip( )
		if not raw :
			return r.stderr.strip( ) or "Tika returned no output."

		# Tika --json returns either a single object {} or an array [{}]
		try :
			data = json.loads( raw )
			if isinstance( data , list ) :
				data = data[ 0 ] if data else { }
		except json.JSONDecodeError :
			return raw  # fallback: return raw text if not valid JSON

		if not isinstance( data , dict ) :
			return raw

		# ── format the JSON fields into aligned key: value lines ──────────
		# Priority fields shown first, then everything else alphabetically
		PRIORITY = [
			"Content-Type" , "File Name" , "File Size" ,
			"Last-Modified" , "Creation-Date" , "Last-Save-Date" ,
			"Author" , "dc:creator" , "meta:author" ,
			"Title" , "dc:title" ,
			"subject" , "dc:subject" ,
			"description" , "dc:description" ,
			"Keywords" , "meta:keyword" ,
			"Language" , "dc:language" ,
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
			lines.append( "" )  # blank separator

		for k in sorted( data.keys( ) ) :
			if k not in seen and not k.startswith( "X-" ) :
				v = data[ k ]
				if isinstance( v , list ) :
					v = "; ".join( str( i ) for i in v )
				lines.append( f"{k:<32} {v}" )

		return "\n".join( lines ) if lines else "No metadata fields returned."

	except subprocess.TimeoutExpired :
		if logger : logger.warning( f"Tika timed out for {file_path.name}" )
		return "Tika timed out (>30 s)."
	except FileNotFoundError :
		return "java not found on PATH — install a JRE."
	except Exception as e :
		if logger : logger.error( f"Tika error: {e}" )
		return f"Tika error: {e}"


def _tika_text( path: str ) -> str :
	jar = Path( str( TIKA_APP_JAR_PATH ) )
	if not jar.exists( ) :
		return ""
	java = JAVA_PATH or "java"
	try :
		r = subprocess.run( [ java , "-jar" , str( jar ) , "--text" , path ] ,
												capture_output=True ,
												text=True ,
												encoding="utf-8" ,
												errors="replace" ,
												timeout=30 ,
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
	] ,
	)


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
# DETECTION ENGINE  (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════

class DetectionEngine :
	def __init__( self , folder , recursive , img_thresh , doc_thresh ,
								do_images , do_docs , do_exact , min_kb ,
								on_progress , on_done , on_error ,
								) :
		self.folder = Path( folder )
		self.recursive = recursive
		self.img_thresh = int( img_thresh )
		self.doc_thresh = float( doc_thresh )
		self.do_images = do_images
		self.do_docs = do_docs
		self.do_exact = do_exact
		self.min_bytes = int( min_kb ) * 1024
		self._cancel = False
		self._progress = on_progress
		self._done = on_done
		self._error = on_error

	def cancel( self ) :
		self._cancel = True

	def run( self ) :
		try :
			self._progress( "Collecting files…" , 0.0 )
			files = self._collect( )
			if self._cancel :
				return

			pairs: list[ list[ str ] ] = [ ]

			# ── exact duplicates (fast, always useful) ─────────────────────
			if self.do_exact :
				self._progress( "Finding exact duplicates…" , 0.02 )
				pairs += self._exact( files )
				if self._cancel :
					return

			# ── perceptual image hashing ────────────────────────────────────
			if self.do_images :
				if IMAGEHASH_AVAILABLE and PIL_AVAILABLE :
					pairs += self._images( files )
				else :
					self._progress( "⚠ imagehash/Pillow unavailable — skipping image scan" , 0.5 )
				if self._cancel :
					return

			# ── document text similarity ────────────────────────────────────
			if self.do_docs :
				pairs += self._docs( files )
				if self._cancel :
					return

			groups = self._union_find( pairs )
			self._done( groups )

		except Exception as e :
			self._error( str( e ) )

	# ── file collection ──────────────────────────────────────────────────────

	def _collect( self ) -> list[ Path ] :
		it = self.folder.rglob( "*" ) if self.recursive else self.folder.iterdir( )
		out = [ ]
		for p in it :
			try :
				if p.is_file( ) and p.stat( ).st_size >= self.min_bytes :
					out.append( p )
			except OSError :
				pass
		return out

	# ── exact SHA-256 ────────────────────────────────────────────────────────

	def _exact( self , files: list[ Path ] ) -> list[ list[ str ] ] :
		buckets: dict[ str , list[ str ] ] = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel :
				return [ ]
			if i % 30 == 0 :
				self._progress( f"Checksumming {i}/{n}…" , 0.02 + 0.13 * i / max( n , 1 ) )
			try :
				h = hashlib.sha256( f.read_bytes( ) ).hexdigest( )
				buckets[ h ].append( str( f ) )
			except Exception :
				pass
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	# ── perceptual image hashing ─────────────────────────────────────────────

	def _images( self , files: list[ Path ] ) -> list[ list[ str ] ] :
		img_files = [ f for f in files if f.suffix.lower( ) in _IMG_EXTS ]
		n = len( img_files )
		if n == 0 :
			return [ ]

		# hash phase
		hashes: dict[ str , any ] = { }
		for i , f in enumerate( img_files ) :
			if self._cancel :
				return [ ]
			if i % 10 == 0 :
				self._progress( f"Hashing images {i + 1}/{n}…" ,
												0.15 + 0.35 * i / max( n , 1 ) ,
												)
			try :
				img = Image.open( f ).convert( "RGB" )
				hashes[ str( f ) ] = imagehash.phash( img )
			except Exception :
				pass

		# comparison phase  O(n²) — fine up to ~3 000 images
		if n > 3000 :
			self._progress( f"⚠ {n} images — comparison may take a few minutes…" , 0.50 )

		items = list( hashes.items( ) )
		m = len( items )
		pairs: list[ list[ str ] ] = [ ]
		for i in range( m ) :
			if self._cancel :
				return pairs
			if i % 100 == 0 :
				self._progress( f"Comparing images {i}/{m}…" ,
												0.50 + 0.20 * i / max( m , 1 ) ,
												)
			for j in range( i + 1 , m ) :
				if items[ i ][ 1 ] - items[ j ][ 1 ] <= self.img_thresh :
					pairs.append( [ items[ i ][ 0 ] , items[ j ][ 0 ] ] )
		return pairs

	# ── document text similarity ─────────────────────────────────────────────

	def _docs( self , files: list[ Path ] ) -> list[ list[ str ] ] :
		doc_files = [ f for f in files if f.suffix.lower( ) in _DOC_EXTS ]
		n = len( doc_files )
		if n == 0 :
			return [ ]

		texts: dict[ str , str ] = { }
		for i , f in enumerate( doc_files ) :
			if self._cancel :
				return [ ]
			if i % 5 == 0 :
				self._progress( f"Extracting text {i + 1}/{n}: {f.name[ :35 ]}…" ,
												0.70 + 0.12 * i / max( n , 1 ) ,
												)
			txt = self._extract_text( f )
			if txt.strip( ) :
				texts[ str( f ) ] = txt

		items = list( texts.items( ) )
		m = len( items )
		pairs: list[ list[ str ] ] = [ ]
		for i in range( m ) :
			if self._cancel :
				return pairs
			for j in range( i + 1 , m ) :
				ratio = difflib.SequenceMatcher(
						None , texts[ items[ i ][ 0 ] ] , texts[ items[ j ][ 0 ] ] , autojunk=True ,
				).ratio( )
				if ratio >= self.doc_thresh :
					pairs.append( [ items[ i ][ 0 ] , items[ j ][ 0 ] ] )
		return pairs

	def _extract_text( self , f: Path , max_chars: int = 8000 ) -> str :
		ext = f.suffix.lower( )
		try :
			if ext == ".pdf" and FITZ_AVAILABLE :
				doc = fitz.open( f )
				return " ".join( p.get_text( ) for p in doc )[ :max_chars ]
			if ext in { ".txt" , ".md" , ".csv" , ".log" } :
				return f.read_text( errors="ignore" )[ :max_chars ]
			return _tika_text( str( f ) )[ :max_chars ]
		except Exception :
			return ""

	# ── union-find grouping ──────────────────────────────────────────────────

	@staticmethod
	def _union_find( pairs: list[ list[ str ] ] ) -> list[ list[ str ] ] :
		parent: dict[ str , str ] = { }

		def find( x: str ) -> str :
			parent.setdefault( x , x )
			while parent[ x ] != x :
				parent[ x ] = parent[ parent[ x ] ]
				x = parent[ x ]
			return x

		def union( x: str , y: str ) :
			parent[ find( x ) ] = find( y )

		for grp in pairs :
			for a , b in itertools.combinations( grp , 2 ) :
				union( a , b )

		buckets: dict[ str , list[ str ] ] = defaultdict( list )
		for item in list( parent.keys( ) ) :
			buckets[ find( item ) ].append( item )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]


# ══════════════════════════════════════════════════════════════════════════════
# PREVIEW PANE  (Preview tab | Metadata tab)
# ══════════════════════════════════════════════════════════════════════════════

class PreviewPane :
	def __init__( self , parent , side_label: str ) :
		self._fp: str | None = None
		self._zoom = 1.0
		self._mode = "preview"
		self._refs = [ ]
		self._meta_cache: str | None = None

		self.frame = ctk.CTkFrame( parent , fg_color=BG_CARD , corner_radius=RADIUS )
		self.frame.rowconfigure( 2 , weight=1 )
		self.frame.columnconfigure( 0 , weight=1 )

		# ── header ──────────────────────────────────────────────────────────
		hdr = ctk.CTkFrame( self.frame , fg_color=BG_CARD , corner_radius=0 , height=30 )
		hdr.grid( row=0 , column=0 , sticky="ew" , padx=8 , pady=(8 , 2) )
		ctk.CTkLabel( hdr , text=side_label , font=FONT_B , text_color=SALMON ).pack( side="left" , padx=6 )
		self._name_lbl = ctk.CTkLabel( hdr , text="—" , font=FONT_S , text_color=FG_TEXT , wraplength=340 )
		self._name_lbl.pack( side="left" , padx=6 )
		self._size_lbl = ctk.CTkLabel( hdr , text="" , font=FONT_S , text_color=FG_DIM )
		self._size_lbl.pack( side="right" , padx=6 )

		# ── tab row ─────────────────────────────────────────────────────────
		tabs = ctk.CTkFrame( self.frame , fg_color=BG_CARD , corner_radius=0 , height=30 )
		tabs.grid( row=1 , column=0 , sticky="ew" , padx=8 , pady=(0 , 4) )
		self._tb_prev = ctk.CTkButton( tabs , text="Preview" , command=self._show_preview ,
																	 fg_color=SALMON , hover_color=SALMON_HV ,
																	 text_color="white" , width=88 , height=24 ,
																	 corner_radius=6 , font=FONT_S ,
																	 )
		self._tb_prev.pack( side="left" , padx=(0 , 4) )
		self._tb_meta = ctk.CTkButton( tabs , text="Metadata" , command=self._show_metadata ,
																	 fg_color=BG_TAB , hover_color=BG_PANEL ,
																	 text_color=FG_DIM , width=88 , height=24 ,
																	 corner_radius=6 , font=FONT_S ,
																	 )
		self._tb_meta.pack( side="left" )
		self._zoom_lbl = ctk.CTkLabel( tabs , text="100%" , font=FONT_S , text_color=FG_DIM , width=40 )
		self._zoom_lbl.pack( side="right" , padx=2 )
		_btn( tabs , "+" , self._zoom_in , width=26 , color=BG_TAB , hover=BG_PANEL ).pack( side="right" , padx=2 )
		_btn( tabs , "−" , self._zoom_out , width=26 , color=BG_TAB , hover=BG_PANEL ).pack( side="right" , padx=2 )
		_btn( tabs , "↗ Open" , self._open_external , width=76 ,
					color=BG_TAB , hover=BG_PANEL ,
					).pack( side="right" , padx=(8 , 4) )

		# ── canvas ──────────────────────────────────────────────────────────
		wrap = tk.Frame( self.frame , bg=BG_APP )
		wrap.grid( row=2 , column=0 , sticky="nsew" , padx=8 , pady=(0 , 8) )
		wrap.rowconfigure( 0 , weight=1 );
		wrap.columnconfigure( 0 , weight=1 )
		self._canvas = tk.Canvas( wrap , bg=BG_APP , highlightthickness=0 , cursor="fleur" )
		vsb = tk.Scrollbar( wrap , orient="vertical" , command=self._canvas.yview )
		hsb = tk.Scrollbar( wrap , orient="horizontal" , command=self._canvas.xview )
		self._canvas.configure( yscrollcommand=vsb.set , xscrollcommand=hsb.set )
		self._canvas.grid( row=0 , column=0 , sticky="nsew" )
		vsb.grid( row=0 , column=1 , sticky="ns" )
		hsb.grid( row=1 , column=0 , sticky="ew" )
		for seq in ("<MouseWheel>" , "<Button-4>" , "<Button-5>") :
			self._canvas.bind( seq , self._on_wheel )

	# ── public ───────────────────────────────────────────────────────────────

	def load( self , fp: str | None ) :
		self._fp , self._zoom , self._meta_cache = fp , 1.0 , None
		self._zoom_lbl.configure( text="100%" )
		self._mode = "preview"
		self._update_tabs( )
		if fp and Path( fp ).exists( ) :
			p = Path( fp )
			self._name_lbl.configure( text=p.name )
			self._size_lbl.configure( text=_fmt_size( p.stat( ).st_size ) )
		else :
			self._name_lbl.configure( text="—" );
			self._size_lbl.configure( text="" )
		self._render( )

	# ── tab control ──────────────────────────────────────────────────────────

	def _show_preview( self ) :
		self._mode = "preview";
		self._update_tabs( );
		self._render( )

	def _show_metadata( self ) :
		self._mode = "metadata";
		self._update_tabs( );
		self._render( )

	def _update_tabs( self ) :
		if self._mode == "preview" :
			self._tb_prev.configure( fg_color=SALMON , text_color="white" )
			self._tb_meta.configure( fg_color=BG_TAB , text_color=FG_DIM )
		else :
			self._tb_prev.configure( fg_color=BG_TAB , text_color=FG_DIM )
			self._tb_meta.configure( fg_color=SALMON , text_color="white" )

	# ── rendering ────────────────────────────────────────────────────────────

	def _render( self ) :
		self._canvas.delete( "all" );
		self._refs.clear( )
		if not self._fp or not Path( self._fp ).exists( ) :
			self._draw_text( "No file loaded." , FG_DIM );
			return
		if self._mode == "metadata" :
			self._render_metadata( );
			return
		ext = Path( self._fp ).suffix.lower( )
		if PIL_AVAILABLE and ext in _IMG_EXTS :           self._render_image( )
		elif FITZ_AVAILABLE and ext == ".pdf" :           self._render_pdf( )
		elif ext in {
			".txt" , ".md" , ".csv" , ".log" , ".json" ,
			".yaml" , ".yml" , ".toml" , ".xml" ,
		} :    self._render_text( )
		elif CV2_AVAILABLE and ext in _VID_EXTS :        self._render_video_thumb( )
		else :                                            self._render_metadata( )

	def _render_image( self ) :
		try :
			img = Image.open( self._fp )
			w = max( 1 , int( img.width * self._zoom ) )
			h = max( 1 , int( img.height * self._zoom ) )
			img = img.resize( (w , h) , Image.LANCZOS )
			ph = ImageTk.PhotoImage( img );
			self._refs.append( ph )
			self._canvas.create_image( 0 , 0 , anchor="nw" , image=ph )
			self._canvas.configure( scrollregion=(0 , 0 , w , h) )
		except Exception as e :
			self._draw_text( f"Image error: {e}" , FG_DIM )

	def _render_pdf( self ) :
		try :
			doc = fitz.open( self._fp )
			y , gap , max_w = 0 , 8 , 0
			for page in doc :
				pix = page.get_pixmap( matrix=fitz.Matrix( self._zoom * 1.4 , self._zoom * 1.4 ) )
				img = Image.frombytes( "RGB" , [ pix.width , pix.height ] , pix.samples )
				ph = ImageTk.PhotoImage( img );
				self._refs.append( ph )
				self._canvas.create_image( 0 , y , anchor="nw" , image=ph )
				y += pix.height + gap;
				max_w = max( max_w , pix.width )
			self._canvas.configure( scrollregion=(0 , 0 , max_w , y) )
			doc.close( )
		except Exception as e :
			self._draw_text( f"PDF error: {e}" , FG_DIM )

	def _render_text( self ) :
		try :
			txt = Path( self._fp ).read_text( errors="ignore" )[ :8000 ]
			self._draw_text( txt , FG_TEXT )
		except Exception as e :
			self._draw_text( f"Read error: {e}" , FG_DIM )

	def _render_video_thumb( self ) :
		try :
			cap = cv2.VideoCapture( self._fp )
			ret , frame = cap.read( );
			cap.release( )
			if ret and PIL_AVAILABLE :
				img = Image.fromarray( cv2.cvtColor( frame , cv2.COLOR_BGR2RGB ) )
				w = max( 1 , int( img.width * self._zoom * 0.6 ) )
				h = max( 1 , int( img.height * self._zoom * 0.6 ) )
				img = img.resize( (w , h) , Image.LANCZOS )
				ph = ImageTk.PhotoImage( img );
				self._refs.append( ph )
				self._canvas.create_image( 0 , 0 , anchor="nw" , image=ph )
				self._canvas.create_text( 8 , h + 10 , anchor="nw" ,
																	text="▶ Video — thumbnail only. Use ↗ Open for full playback." ,
																	fill=FG_WARN , font=FONT_S ,
																	)
				self._canvas.configure( scrollregion=(0 , 0 , w , h + 30) )
			else :
				self._draw_text( "Could not decode video frame." , FG_DIM )
		except Exception as e :
			self._draw_text( f"Video error: {e}" , FG_DIM )

	def set_logger( self , logger: logging.Logger ) :
		self.logger = logger

	def _render_metadata( self ) :
		logger = getattr( self , "_logger" , None )
		# Render file stats immediately, run Tika once and cache
		if self._meta_cache is None :
			stats = _file_stats( self._fp )
			# Immediate partial render so the pane isn't blank during Tika
			self._canvas.create_text( 10 , 10 , anchor="nw" ,
																text="── File Stats ──" , fill=SALMON , font=FONT_B ,
																)
			self._canvas.create_text( 10 , 32 , anchor="nw" ,
																text=stats , fill=FG_TEXT , font=FONT_S , width=430 ,
																)
			self._canvas.create_text( 10 , 160 , anchor="nw" ,
																text="── Tika Metadata  (extracting…) ──" ,
																fill=FG_WARN , font=FONT_B ,
																)
			self._canvas.configure( scrollregion=(0 , 0 , 460 , 185) )
			self._canvas.update_idletasks( )

			tika = _tika_meta( self._fp , logger )
			self._meta_cache = f"{stats}\n\n── Tika Metadata (--json) ──\n{tika}"

		self._canvas.delete( "all" )
		est_h = self._meta_cache.count( "\n" ) * 15 + 40
		self._canvas.create_text( 10 , 10 , anchor="nw" ,
															text=self._meta_cache , fill=FG_TEXT , font=("Consolas" , 9) , width=450 ,
															)
		self._canvas.configure( scrollregion=(0 , 0 , 470 , est_h) )

	def _draw_text( self , text: str , color: str ) :
		self._canvas.create_text( 10 , 10 , anchor="nw" , text=text ,
															fill=color , font=FONT_S , width=430 ,
															)
		self._canvas.configure( scrollregion=(0 , 0 , 450 , text.count( "\n" ) * 15 + 30) )

	# ── controls ─────────────────────────────────────────────────────────────

	def _zoom_in( self ) :
		self._zoom = min( self._zoom + 0.25 , 4.0 )
		self._zoom_lbl.configure( text=f"{int( self._zoom * 100 )}%" )
		if self._mode == "preview" : self._render( )

	def _zoom_out( self ) :
		self._zoom = max( self._zoom - 0.25 , 0.25 )
		self._zoom_lbl.configure( text=f"{int( self._zoom * 100 )}%" )
		if self._mode == "preview" : self._render( )

	def _on_wheel( self , event ) :
		if event.num == 4 :      self._canvas.yview_scroll( -1 , "units" )
		elif event.num == 5 :    self._canvas.yview_scroll( 1 , "units" )
		else :                   self._canvas.yview_scroll( int( -event.delta / 120 ) , "units" )

	def _open_external( self ) :
		if not self._fp : return
		try :
			if os.name == "nt" :   os.startfile( self._fp )
			else :                 subprocess.Popen( [ "xdg-open" , self._fp ] )
		except Exception as e :
			messagebox.showerror( "Open Error" , str( e ) )


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FRAME
# ══════════════════════════════════════════════════════════════════════════════

class ScanFrame( ctk.CTkFrame ) :
	def __init__( self , parent , on_groups_ready , default_folder: str = "" ) :
		super( ).__init__( parent , fg_color="transparent" )
		self._on_groups_ready = on_groups_ready
		self._default_folder = default_folder
		self._engine: DetectionEngine | None = None
		self._thread: threading.Thread | None = None
		self._q: queue.Queue = queue.Queue( )
		self._groups: list[ list[ str ] ] = [ ]
		self._build( )

	def _build( self ) :
		# ── config panel ────────────────────────────────────────────────────
		cfg = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS )
		cfg.pack( fill="x" , padx=12 , pady=(12 , 6) )

		# folder row
		fr = ctk.CTkFrame( cfg , fg_color="transparent" )
		fr.pack( fill="x" , padx=12 , pady=(10 , 4) )
		ctk.CTkLabel( fr , text="Folder:" , font=FONT_B , text_color=FG_TEXT ).pack( side="left" )
		self._folder_var = tk.StringVar( value=self._default_folder )
		ctk.CTkEntry( fr , textvariable=self._folder_var , width=520 ,
									font=FONT , fg_color=BG_CARD , text_color=FG_TEXT ,
									).pack( side="left" , padx=8 )
		_btn( fr , "Browse" , self._browse , width=80 ).pack( side="left" )
		self._recursive = tk.BooleanVar( value=True )
		ctk.CTkCheckBox( fr , text="Recursive" , variable=self._recursive ,
										 font=FONT , text_color=FG_TEXT ,
										 ).pack( side="left" , padx=12 )

		# mode checkboxes
		mr = ctk.CTkFrame( cfg , fg_color="transparent" )
		mr.pack( fill="x" , padx=12 , pady=4 )
		self._do_img = tk.BooleanVar( value=True )
		self._do_doc = tk.BooleanVar( value=True )
		self._do_exact = tk.BooleanVar( value=True )
		for txt , var , tip in [
			("Images  (pHash)" , self._do_img , "requires imagehash + Pillow") ,
			("Documents  (text sim)" , self._do_doc , "uses PyMuPDF / Tika") ,
			("Exact duplicates" , self._do_exact , "SHA-256 — always fast") ,
		] :
			ctk.CTkCheckBox( mr , text=txt , variable=var ,
											 font=FONT , text_color=FG_TEXT ,
											 ).pack( side="left" , padx=10 )
			ctk.CTkLabel( mr , text=f"({tip})" , font=FONT_S ,
										text_color=FG_DIM ,
										).pack( side="left" , padx=(0 , 16) )

		# threshold row
		tr = ctk.CTkFrame( cfg , fg_color="transparent" )
		tr.pack( fill="x" , padx=12 , pady=(4 , 10) )

		# image threshold
		ctk.CTkLabel( tr , text="Image threshold:" , font=FONT ,
									text_color=FG_DIM ,
									).pack( side="left" )
		self._img_thresh_val = tk.IntVar( value=10 )
		self._img_thresh_lbl = ctk.CTkLabel( tr , text="10" , font=FONT_B ,
																				 text_color=SALMON , width=28 ,
																				 )
		self._img_thresh_lbl.pack( side="left" , padx=4 )
		ctk.CTkSlider( tr , from_=0 , to=20 , number_of_steps=20 , width=150 ,
									 command=lambda v : (
										 self._img_thresh_val.set( int( round( v ) ) ) ,
										 self._img_thresh_lbl.configure( text=str( int( round( v ) ) ) )) ,
									 ).set( 10 ); \
				ctk.CTkSlider( tr , from_=0 , to=20 , number_of_steps=20 , width=150 ,
											 command=lambda v : (
												 self._img_thresh_val.set( int( round( v ) ) ) ,
												 self._img_thresh_lbl.configure( text=str( int( round( v ) ) ) )) ,
											 ).pack( side="left" , padx=(0 , 4) )
		ctk.CTkLabel( tr , text="0=exact  20=loose" , font=FONT_S ,
									text_color=FG_DIM ,
									).pack( side="left" , padx=(0 , 20) )

		# doc threshold
		ctk.CTkLabel( tr , text="Doc similarity:" , font=FONT ,
									text_color=FG_DIM ,
									).pack( side="left" )
		self._doc_thresh_val = tk.DoubleVar( value=0.85 )
		self._doc_thresh_lbl = ctk.CTkLabel( tr , text="85 %" , font=FONT_B ,
																				 text_color=SALMON , width=42 ,
																				 )
		self._doc_thresh_lbl.pack( side="left" , padx=4 )
		ctk.CTkSlider( tr , from_=0.5 , to=1.0 , number_of_steps=50 , width=150 ,
									 command=lambda v : (
										 self._doc_thresh_val.set( v ) ,
										 self._doc_thresh_lbl.configure(
												 text=f"{int( round( v * 100 ) )} %" ,
										 )) ,
									 ).set( 0.85 ); \
				ctk.CTkSlider( tr , from_=0.5 , to=1.0 , number_of_steps=50 , width=150 ,
											 command=lambda v : (
												 self._doc_thresh_val.set( v ) ,
												 self._doc_thresh_lbl.configure(
														 text=f"{int( round( v * 100 ) )} %" ,
												 )) ,
											 ).pack( side="left" , padx=(0 , 4) )

		# min size
		ctk.CTkLabel( tr , text="  Min size:" , font=FONT ,
									text_color=FG_DIM ,
									).pack( side="left" , padx=(12 , 0) )
		self._min_kb = ctk.CTkEntry( tr , width=58 , font=FONT ,
																 fg_color=BG_CARD , text_color=FG_TEXT ,
																 )
		self._min_kb.insert( 0 , "0" )
		self._min_kb.pack( side="left" , padx=4 )
		ctk.CTkLabel( tr , text="KB" , font=FONT , text_color=FG_DIM ).pack( side="left" )

		# ── progress panel ───────────────────────────────────────────────────
		prg = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS )
		prg.pack( fill="x" , padx=12 , pady=6 )

		ctrl = ctk.CTkFrame( prg , fg_color="transparent" )
		ctrl.pack( fill="x" , padx=12 , pady=(10 , 4) )
		self._scan_btn = _btn( ctrl , "▶  Start Scan" , self._start , width=140 )
		self._scan_btn.pack( side="left" )
		self._cancel_btn = _btn( ctrl , "■  Cancel" , self._cancel , width=100 ,
														 color=BG_TAB , hover="#555" ,
														 )
		self._cancel_btn.pack( side="left" , padx=8 )
		self._cancel_btn.configure( state="disabled" )

		self._status_lbl = ctk.CTkLabel( prg , text="Ready — select a folder and press Start." ,
																		 font=FONT , text_color=FG_DIM ,
																		 )
		self._status_lbl.pack( anchor="w" , padx=12 , pady=(0 , 4) )
		self._pbar = ctk.CTkProgressBar( prg , height=12 ,
																		 fg_color=BG_CARD , progress_color=SALMON ,
																		 )
		self._pbar.set( 0 )
		self._pbar.pack( fill="x" , padx=12 , pady=(0 , 10) )

		# ── results panel ────────────────────────────────────────────────────
		res = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS )
		res.pack( fill="both" , expand=True , padx=12 , pady=6 )
		self._res_lbl = ctk.CTkLabel( res , text="No scan results yet." ,
																	font=FONT , text_color=FG_DIM ,
																	)
		self._res_lbl.pack( padx=12 , pady=12 )
		self._proceed_btn = _btn( res , "▶  Proceed to Review →" , self._proceed ,
															width=230 , color=FG_GOOD , hover="#4DAD62" ,
															)
		self._proceed_btn.pack( padx=12 , pady=(0 , 12) )
		self._proceed_btn.pack_forget( )

	# ── actions ──────────────────────────────────────────────────────────────

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
		self._groups = [ ]

		self._engine = DetectionEngine(
				folder=folder ,
				recursive=self._recursive.get( ) ,
				img_thresh=self._img_thresh_val.get( ) ,
				doc_thresh=self._doc_thresh_val.get( ) ,
				do_images=self._do_img.get( ) ,
				do_docs=self._do_doc.get( ) ,
				do_exact=self._do_exact.get( ) ,
				min_kb=min_kb ,
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
					_ , groups = item
					self._on_done( groups )
					return
				elif kind == "error" :
					_ , err = item
					self._scan_btn.configure( state="normal" )
					self._cancel_btn.configure( state="disabled" )
					messagebox.showerror( "Scan Error" , err )
					return
		except queue.Empty :
			pass
		if self._thread and self._thread.is_alive( ) :
			self.after( 100 , self._poll )

	def _on_done( self , groups: list[ list[ str ] ] ) :
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
					text_color=FG_GOOD ,
			)
			self._res_lbl.configure(
					text=f"Found {n_g} duplicate group(s) across {n_f} files → {n_p} pair(s) to review." ,
					text_color=FG_TEXT ,
			)
			self._proceed_btn.pack( padx=12 , pady=(0 , 12) )
		else :
			self._status_lbl.configure( text="✓ Scan complete — no duplicates found." , text_color=FG_WARN )
			self._res_lbl.configure(
					text="No duplicates found with the current settings.\n"
							 "Try lowering the image threshold, lowering the doc threshold, or enabling more modes." ,
					text_color=FG_WARN ,
			)

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

	# ── load ─────────────────────────────────────────────────────────────────

	def load_groups( self , groups: list[ list[ str ] ] ) :
		self._pairs = _groups_to_pairs( groups )
		self._skipped = [ ]
		self._idx , self._log = 0 , [ ]
		self._stats = Counter( loaded=len( self._pairs ) )
		self._t0 = time.time( )
		self._show( )

	# ── build ────────────────────────────────────────────────────────────────

	def _build( self ) :
		# top bar
		top = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=46 )
		top.pack( fill="x" , padx=12 , pady=(12 , 6) )
		top.pack_propagate( False )
		_btn( top , "Load JSON manually" , self._load_json , width=160 ).pack( side="left" , padx=12 , pady=8 )
		self._prog_lbl = ctk.CTkLabel( top , text="No pairs loaded." , font=FONT_B , text_color=FG_TEXT )
		self._prog_lbl.pack( side="left" , padx=10 )
		self._stats_lbl = ctk.CTkLabel( top , text="" , font=FONT_S , text_color=FG_DIM )
		self._stats_lbl.pack( side="left" , padx=6 )

		# previews
		cmp = ctk.CTkFrame( self , fg_color=BG_APP , corner_radius=0 )
		cmp.pack( fill="both" , expand=True , padx=12 , pady=4 )
		cmp.columnconfigure( 0 , weight=1 );
		cmp.columnconfigure( 1 , weight=1 )
		cmp.rowconfigure( 0 , weight=1 )
		self._lp = PreviewPane( cmp , "← LEFT" )
		self._rp = PreviewPane( cmp , "RIGHT →" )
		self._lp.frame.grid( row=0 , column=0 , sticky="nsew" , padx=(0 , 4) )
		self._rp.frame.grid( row=0 , column=1 , sticky="nsew" , padx=(4 , 0) )

		# action bar
		acts = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=56 )
		acts.pack( fill="x" , padx=12 , pady=(4 , 12) )
		acts.pack_propagate( False )

		_btn( acts , "←  Keep Left" , self._keep_left , width=210 ,
					color="#2A5A8A" , hover="#3A6A9A" ,
					).pack( side="left" , padx=12 , pady=10 )
		ctk.CTkLabel( acts , text="[←]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "↑  Keep Both  (UUID4)" , self._keep_both , width=220 ,
					color="#3A7A5A" , hover="#4A8A6A" ,
					).pack( side="left" , padx=12 , pady=10 )
		ctk.CTkLabel( acts , text="[↑]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "↓  Skip" , self._skip , width=100 ,
					color=BG_TAB , hover=BG_PANEL ,
					).pack( side="left" , padx=8 , pady=10 )
		ctk.CTkLabel( acts , text="[↓]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		ctk.CTkLabel( acts , text="[→]" , font=FONT_S , text_color=FG_DIM ).pack( side="right" , padx=(0 , 8) )
		_btn( acts , "Keep Right  →" , self._keep_right , width=210 ,
					color="#2A5A8A" , hover="#3A6A9A" ,
					).pack( side="right" , padx=12 , pady=10 )

		# Defer key bindings until the widget is fully attached to a window
		self.after( 200 , self._bind_keys )

	def _bind_keys( self ) :
		root = self.winfo_toplevel( )
		root.bind( "<Left>" , lambda _ : self._keep_left( ) )
		root.bind( "<Right>" , lambda _ : self._keep_right( ) )
		root.bind( "<Up>" , lambda _ : self._keep_both( ) )
		root.bind( "<Down>" , lambda _ : self._skip( ) )

	# ── display ──────────────────────────────────────────────────────────────

	def _active( self ) -> list[ tuple[ str , str ] ] :
		return self._pairs + self._skipped

	def _current( self ) -> tuple[ str , str ] | None :
		p = self._active( )
		return p[ self._idx ] if self._idx < len( p ) else None

	def _show( self ) :
		pair = self._current( )
		if pair is None :
			self._finish( );
			return
		self._lf , self._rf = pair
		total = len( self._active( ) )
		self._prog_lbl.configure( text=f"Pair {self._idx + 1} / {total}" )
		self._stats_lbl.configure( text=(
			f"L:{self._stats.get( 'left' , 0 )}  R:{self._stats.get( 'right' , 0 )}  "
			f"Both:{self._stats.get( 'both' , 0 )}  Skip:{len( self._skipped )}"
		) ,
		)
		self._lp.load( self._lf )
		self._rp.load( self._rf )

	# ── decisions ────────────────────────────────────────────────────────────

	def _keep_left( self ) :
		if not self._lf : return
		if not messagebox.askyesno( "Confirm" ,
																f"Archive RIGHT and keep LEFT?\n\n{Path( self._rf ).name}" ,
																) : return
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
		if not messagebox.askyesno( "Confirm" ,
																f"Archive LEFT and keep RIGHT?\n\n{Path( self._lf ).name}" ,
																) : return
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
			ld , rd = _uuid4_path( self.base_dir , lp ) , _uuid4_path( self.base_dir , rp )
			shutil.move( str( lp ) , str( ld ) )
			shutil.move( str( rp ) , str( rd ) )
			self._rec( { "action"         : "keep_both" ,
									 "left_original"  : str( lp ) , "left_moved" : str( ld ) ,
									 "right_original" : str( rp ) , "right_moved" : str( rd ) ,
									 } ,
								 )
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

	# ── helpers ───────────────────────────────────────────────────────────────

	def _mv( self , src: str , dest: Path ) -> str :
		dest.mkdir( parents=True , exist_ok=True )
		dst = _unique_path( dest , Path( src ) )
		shutil.move( src , dst )
		return str( dst )

	def _rec( self , entry: dict ) :
		entry[ "ts" ] = time.strftime( "%Y-%m-%dT%H:%M:%S" )
		self._log.append( entry )
		self._stats[ "reviewed" ] += 1

	def _advance( self ) :
		self._idx += 1;
		self._show( )

	def _load_json( self ) :
		path = filedialog.askopenfilename( title="Load JSON groups/pairs" ,
																			 filetypes=[ ("JSON" , "*.json") ] ,
																			 )
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
		messagebox.showinfo( "All Done" ,
												 f"All {self._stats[ 'reviewed' ]} pairs reviewed!\n\n"
												 f"Keep Left:  {self._stats[ 'left' ]}\n"
												 f"Keep Right: {self._stats[ 'right' ]}\n"
												 f"Keep Both:  {self._stats[ 'both' ]}\n"
												 f"Elapsed:    {elapsed}s\n\n"
												 f"Decision log:\n{log_path}" ,
												 )


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL APP
# ══════════════════════════════════════════════════════════════════════════════

class DuplicateReviewer :
	def __init__(
			self ,
			source_dir: Path | None = None ,
			logger: logging.Logger | None = None ,
	) :
		self.logger = logger
		self.source_dir = source_dir
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
		self.logger.info(
				f"App started | source_dir={source_dir} | base_dir={source_dir}" ,
		)

	def _on_scan_done( self , groups: list[ list[ str ] ] ) :
		"""Called by ScanFrame when the user clicks Proceed."""
		self._review.load_groups( groups )
		self._tabs.set( "👁  Review" )

	def run( self ) :
		if any( item.is_file( ) for item in self.source_dir.iterdir( ) ) :
			self.root.mainloop( )
		else :
			self.logger.warning( f"No items found in {self.source_dir}. Not running \"DuplicateReviewer\" application" )
