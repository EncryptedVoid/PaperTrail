#!/usr/bin/env python3
"""
Duplicate Finder & Reviewer  v3 — Per-category streaming
─────────────────────────────────────────────────────────
v3 changes:
  • Detection runs per file-type category (Media → Images → Documents → Other)
  • Results stream to the reviewer as each category finishes
  • User can review early results while later categories still scan
  • "Needs Alteration" action moves both files to ALTERATIONS_REQUIRED_DIR

v2 performance (preserved):
  • LRU preview cache, background preloading, stable widget tree
  • Size-bucketing → parallel SHA-256, BK-tree pHash, doc prefilters
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
from concurrent.futures import Future , ThreadPoolExecutor , as_completed
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
	ALTERATIONS_REQUIRED_DIR ,
	ARCHIVAL_DIR ,
	DELETE_DIR ,
	DOCUMENT_TYPES ,
	IMAGE_TYPES ,
	JAVA_PATH ,
	TIKA_APP_JAR_PATH ,
)

fitz.TOOLS.mupdf_display_errors( False )

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
_AUDIO_EXTS = { ".mp3" , ".wav" , ".flac" , ".aac" , ".ogg" , ".wma" , ".m4a" , ".opus" }
_TXT_EXTS = { ".txt" , ".md" , ".csv" , ".log" , ".json" , ".yaml" , ".yml" , ".toml" , ".xml" }
_CAMERA_RAW_EXTS = { ".nef" , ".heic" , ".arw" , ".cr2" }

CACHE_MAX_ITEMS = 64
PREFETCH_AHEAD = 5
BG_WORKERS = 3

HASH_WORKERS = 8
HASH_CHUNK_SIZE = 1 << 16
IMG_HASH_WORKERS = 6
DOC_SIZE_RATIO = 0.3
DOC_LEN_RATIO = 0.4

# ── File-type categories (scanned in this order) ────────────────────────
# Media is fastest (exact+names only), images medium (pHash), docs slowest.
CATEGORY_ORDER = [ "Media" , "Images" , "Documents" , "Other" ]
CATEGORY_EXTS = {
	"Images"    : _IMG_EXTS ,
	"Documents" : _DOC_EXTS | { ".pdf" } ,
	"Media"     : _VID_EXTS | _AUDIO_EXTS ,
	# "Other" is implicit — anything not in the above
}


def _categorize_files( files: list[ Path ] ) -> dict[ str , list[ Path ] ] :
	buckets: dict[ str , list[ Path ] ] = { c : [ ] for c in CATEGORY_ORDER }
	assigned = set( )
	for cat in ("Images" , "Documents" , "Media") :
		exts = CATEGORY_EXTS[ cat ]
		for f in files :
			if f.suffix.lower( ) in exts :
				buckets[ cat ].append( f )
				assigned.add( id( f ) )
	for f in files :
		if id( f ) not in assigned :
			buckets[ "Other" ].append( f )
	return { k : v for k , v in buckets.items( ) if v }


# ══════════════════════════════════════════════════════════════════════════════
# BK-TREE
# ══════════════════════════════════════════════════════════════════════════════

class BKTree :
	def __init__( self , distance_fn ) :
		self._dist = distance_fn
		self._root = None

	def insert( self , item ) :
		if self._root is None :
			self._root = (item , { })
			return
		node = self._root
		while True :
			d = self._dist( node[ 0 ] , item )
			if d in node[ 1 ] :
				node = node[ 1 ][ d ]
			else :
				node[ 1 ][ d ] = (item , { })
				return

	def find( self , item , threshold ) -> list :
		if self._root is None :
			return [ ]
		results = [ ]
		candidates = [ self._root ]
		while candidates :
			node = candidates.pop( )
			d = self._dist( node[ 0 ] , item )
			if d <= threshold :
				results.append( node[ 0 ] )
			for child_d , child_node in node[ 1 ].items( ) :
				if d - threshold <= child_d <= d + threshold :
					candidates.append( child_node )
		return results


# ══════════════════════════════════════════════════════════════════════════════
# THREADED LRU PREVIEW CACHE
# ══════════════════════════════════════════════════════════════════════════════

class PreviewCache :
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
				val = Image.open( fp );
				val.load( )
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
# FAST HASHING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sha256_chunked( filepath: Path ) -> str | None :
	try :
		h = hashlib.sha256( )
		with open( filepath , "rb" ) as f :
			while True :
				chunk = f.read( HASH_CHUNK_SIZE )
				if not chunk :
					break
				h.update( chunk )
		return h.hexdigest( )
	except Exception :
		return None


def _phash_one( filepath: Path ) -> tuple[ str , object ] | None :
	try :
		img = Image.open( filepath ).convert( "RGB" )
		return (str( filepath ) , imagehash.phash( img ))
	except Exception :
		return None


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ENGINE  (v3 — accepts pre-collected file list per category)
# ══════════════════════════════════════════════════════════════════════════════

class DetectionEngine :
	_WEIGHTS = {
		"exact"  : 1.0 , "names" : 0.3 , "names_cross_ext" : 0.3 ,
		"images" : 4.0 , "docs" : 3.0 ,
	}

	def __init__( self , folder=None , files=None , recursive=True ,
								img_thresh=10 , doc_thresh=0.85 ,
								do_images=True , do_docs=True , do_exact=True ,
								do_names=True , do_names_cross_ext=False ,
								min_kb=0 , on_progress=None , on_done=None , on_error=None ) :
		self.folder = Path( folder ) if folder else None
		self._provided_files = files
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
		self._progress = on_progress or (lambda m , p : None)
		self._done = on_done or (lambda g : None)
		self._error = on_error or (lambda e : None)

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

	def run( self ) -> list :
		"""Run detection. Returns groups list. Also calls on_done callback."""
		try :
			self._progress( "Collecting files…" , 0.0 )
			files = self._collect( )
			if self._cancel :
				return [ ]
			self._progress( f"Found {len( files )} files. Building index…" , 0.01 )

			self._size_map: dict[ str , int ] = { }
			for f in files :
				try :
					self._size_map[ str( f ) ] = f.stat( ).st_size
				except OSError :
					pass

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
			if self._cancel :
				return [ ]

			groups = self._union_find( pairs )
			self._done( groups )
			return groups
		except Exception as e :
			self._error( str( e ) )
			return [ ]

	def _collect( self ) -> list[ Path ] :
		if self._provided_files is not None :
			return [
				f for f in self._provided_files
				if f.is_file( ) and f.stat( ).st_size >= self.min_bytes
			]
		it = self.folder.rglob( "*" ) if self.recursive else self.folder.iterdir( )
		out = [ ]
		for p in it :
			try :
				if p.is_file( ) and p.stat( ).st_size >= self.min_bytes :
					out.append( p )
			except OSError :
				pass
		return out

	# ── EXACT DUPES ──────────────────────────────────────────────────────

	def _exact( self , files , p_start , p_end ) :
		size_buckets: dict[ int , list[ Path ] ] = defaultdict( list )
		for f in files :
			sz = self._size_map.get( str( f ) )
			if sz is not None :
				size_buckets[ sz ].append( f )
		to_hash = [ ]
		for sz , group in size_buckets.items( ) :
			if len( group ) >= 2 :
				to_hash.extend( group )
		n_skip = len( files ) - len( to_hash )
		self._phase_progress(
				f"Size prefilter: {len( to_hash )} candidates ({n_skip} unique sizes skipped)" ,
				0.05 , p_start , p_end )
		if not to_hash :
			return [ ]
		hash_buckets: dict[ str , list[ str ] ] = defaultdict( list )
		total = len( to_hash )
		done = 0
		with ThreadPoolExecutor( max_workers=HASH_WORKERS ) as pool :
			future_map = { pool.submit( _sha256_chunked , f ) : f for f in to_hash }
			for future in as_completed( future_map ) :
				if self._cancel :
					pool.shutdown( wait=False , cancel_futures=True )
					return [ ]
				done += 1
				if done % 100 == 0 :
					self._phase_progress( f"Hashing {done}/{total}…" ,
																0.05 + 0.95 * (done / total) , p_start , p_end )
				f = future_map[ future ]
				h = future.result( )
				if h is not None :
					hash_buckets[ h ].append( str( f ) )
		self._phase_progress( f"Hashing done ({total} files)" , 1.0 , p_start , p_end )
		return [ v for v in hash_buckets.values( ) if len( v ) >= 2 ]

	# ── FILENAME MATCH ───────────────────────────────────────────────────

	def _names( self , files , p_start , p_end ) :
		buckets = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel : return [ ]
			if f.suffix.lower( ) in _CAMERA_RAW_EXTS : continue
			if i % 200 == 0 :
				self._phase_progress( f"Matching filenames {i}/{n}…" , i / max( n , 1 ) , p_start , p_end )
			buckets[ (sanitize_artifact_name( f.name ) , f.suffix.lower( )) ].append( str( f ) )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	def _names_cross_ext( self , files , p_start , p_end ) :
		buckets = defaultdict( list )
		n = len( files )
		for i , f in enumerate( files ) :
			if self._cancel : return [ ]
			if f.suffix.lower( ) in _CAMERA_RAW_EXTS : continue
			if i % 200 == 0 :
				self._phase_progress( f"Cross-ext match {i}/{n}…" , i / max( n , 1 ) , p_start , p_end )
			buckets[ sanitize_artifact_name( f.name ) ].append( str( f ) )
		return [ v for v in buckets.values( ) if len( v ) >= 2 ]

	# ── IMAGES ───────────────────────────────────────────────────────────

	def _images( self , files , p_start , p_end ) :
		img_files = [ f for f in files if f.suffix.lower( ) in _IMG_EXTS ]
		n = len( img_files )
		if n == 0 :
			return [ ]
		p_hash_end = p_start + 0.5 * (p_end - p_start)
		p_tree_end = p_start + 0.7 * (p_end - p_start)
		self._phase_progress( f"Hashing {n} images ({IMG_HASH_WORKERS} threads)…" , 0.0 , p_start , p_hash_end )
		hashes: list[ tuple[ str , object ] ] = [ ]
		done = 0
		with ThreadPoolExecutor( max_workers=IMG_HASH_WORKERS ) as pool :
			future_map = { pool.submit( _phash_one , f ) : f for f in img_files }
			for future in as_completed( future_map ) :
				if self._cancel :
					pool.shutdown( wait=False , cancel_futures=True )
					return [ ]
				done += 1
				if done % 50 == 0 :
					self._phase_progress( f"pHash {done}/{n}…" , done / max( n , 1 ) , p_start , p_hash_end )
				result = future.result( )
				if result is not None :
					hashes.append( result )
		self._phase_progress( f"Hashed {len( hashes )} images. Building BK-tree…" , 0.0 , p_hash_end , p_tree_end )
		if len( hashes ) < 2 :
			return [ ]

		def hamming_dist( a , b ) :
			return a[ 1 ] - b[ 1 ]

		tree = BKTree( hamming_dist )
		for item in hashes :
			tree.insert( item )
		self._phase_progress( "BK-tree built. Querying…" , 0.0 , p_tree_end , p_end )
		pairs: list[ list[ str ] ] = [ ]
		seen_pairs: set[ tuple[ str , str ] ] = set( )
		m = len( hashes )
		for i , item in enumerate( hashes ) :
			if self._cancel :
				return pairs
			if i % 200 == 0 :
				self._phase_progress( f"BK-tree query {i}/{m}…" , i / max( m , 1 ) , p_tree_end , p_end )
			neighbors = tree.find( item , self.img_thresh )
			for nb in neighbors :
				if nb[ 0 ] == item[ 0 ] :
					continue
				key = (min( item[ 0 ] , nb[ 0 ] ) , max( item[ 0 ] , nb[ 0 ] ))
				if key not in seen_pairs :
					seen_pairs.add( key )
					pairs.append( [ item[ 0 ] , nb[ 0 ] ] )
		self._phase_progress( f"Image comparison done ({len( pairs )} pairs)" , 1.0 , p_tree_end , p_end )
		return pairs

	# ── DOCUMENTS ────────────────────────────────────────────────────────

	def _docs( self , files , p_start , p_end ) :
		doc_files = [ f for f in files if f.suffix.lower( ) in _DOC_EXTS ]
		n = len( doc_files )
		if n == 0 :
			return [ ]
		p_mid = p_start + 0.5 * (p_end - p_start)
		texts: dict[ str , str ] = { }
		for i , f in enumerate( doc_files ) :
			if self._cancel : return [ ]
			if i % 5 == 0 :
				self._phase_progress( f"Extracting text {i + 1}/{n}…" , i / max( n , 1 ) , p_start , p_mid )
			txt = self._extract_text( f )
			if txt.strip( ) :
				texts[ str( f ) ] = txt
		items = list( texts.items( ) )
		m = len( items )
		lengths = { k : len( v ) for k , v in items }
		file_sizes = { }
		for k , _ in items :
			file_sizes[ k ] = self._size_map.get( k , 0 )
		total_cmp = max( m * (m - 1) // 2 , 1 )
		pairs , done , skipped = [ ] , 0 , 0
		for i in range( m ) :
			if self._cancel : return pairs
			fp_i , txt_i = items[ i ]
			len_i = lengths[ fp_i ]
			sz_i = file_sizes[ fp_i ]
			for j in range( i + 1 , m ) :
				done += 1
				fp_j = items[ j ][ 0 ]
				len_j = lengths[ fp_j ]
				sz_j = file_sizes[ fp_j ]
				if sz_i and sz_j :
					ratio = min( sz_i , sz_j ) / max( sz_i , sz_j )
					if ratio < DOC_SIZE_RATIO :
						skipped += 1;
						continue
				if len_i and len_j :
					ratio = min( len_i , len_j ) / max( len_i , len_j )
					if ratio < DOC_LEN_RATIO :
						skipped += 1;
						continue
				if done % 50 == 0 :
					self._phase_progress(
							f"Comparing docs {done}/{total_cmp} ({skipped} skipped)…" ,
							done / total_cmp , p_mid , p_end )
				sim = difflib.SequenceMatcher( None , txt_i , texts[ fp_j ] , autojunk=True ).ratio( )
				if sim >= self.doc_thresh :
					pairs.append( [ fp_i , fp_j ] )
		self._phase_progress( f"Document comparison done ({skipped} skipped by prefilter)" , 1.0 , p_mid , p_end )
		return pairs

	def _extract_text( self , f: Path , max_chars=8000 ) -> str :
		ext = f.suffix.lower( )
		try :
			if ext == ".pdf" :
				doc = fitz.open( f )
				txt = " ".join( p.get_text( ) for p in doc )[ :max_chars ]
				doc.close( )
				return txt
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
# PREVIEW PANE
# ══════════════════════════════════════════════════════════════════════════════

class PreviewPane :
	def __init__( self , parent , side_label: str ) :
		self._fp: str | None = None
		self._zoom = 1.0
		self._fit_scale = 1.0
		self._mode = "preview"
		self._photo_refs: list = [ ]
		self._raw_img: Image.Image | None = None
		self._meta_text: str | None = None

		self.frame = ctk.CTkFrame( parent , fg_color=BG_CARD , corner_radius=RADIUS )
		self.frame.rowconfigure( 2 , weight=1 )
		self.frame.columnconfigure( 0 , weight=1 )

		hdr = ctk.CTkFrame( self.frame , fg_color=BG_CARD , corner_radius=0 , height=30 )
		hdr.grid( row=0 , column=0 , sticky="ew" , padx=8 , pady=(8 , 2) )
		ctk.CTkLabel( hdr , text=side_label , font=FONT_B , text_color=SALMON ).pack( side="left" , padx=6 )
		self._name_lbl = ctk.CTkLabel( hdr , text="—" , font=FONT_S , text_color=FG_TEXT , wraplength=340 )
		self._name_lbl.pack( side="left" , padx=6 )
		self._size_lbl = ctk.CTkLabel( hdr , text="" , font=FONT_S , text_color=FG_DIM )
		self._size_lbl.pack( side="right" , padx=6 )

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

		self._img_item = self._canvas.create_image( 0 , 0 , anchor="nw" )
		self._txt_item = self._canvas.create_text( 10 , 10 , anchor="nw" , text="" , fill=FG_TEXT , font=FONT_S ,
																							 width=430 )
		self._overlay_item = self._canvas.create_text( 10 , 10 , anchor="nw" , text="" , fill=FG_WARN , font=FONT_S ,
																									 width=430 )

		for seq in ("<MouseWheel>" , "<Button-4>" , "<Button-5>") :
			self._canvas.bind( seq , self._on_wheel )

	def load( self , fp: str | None ) :
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

	def _clear_canvas( self ) :
		self._canvas.itemconfigure( self._img_item , image="" )
		self._canvas.itemconfigure( self._txt_item , text="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._photo_refs.clear( )
		self._canvas.configure( scrollregion=(0 , 0 , 0 , 0) )

	def _canvas_size( self ) -> tuple[ int , int ] :
		self._canvas.update_idletasks( )
		return max( self._canvas.winfo_width( ) , 100 ) , max( self._canvas.winfo_height( ) , 100 )

	def _show_text( self , text: str , color: str = FG_TEXT , font=None ) :
		self._canvas.itemconfigure( self._img_item , image="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._canvas.itemconfigure( self._txt_item , text=text , fill=color , font=font or FONT_S )
		self._canvas.configure( scrollregion=(0 , 0 , 460 , text.count( "\n" ) * 15 + 40) )

	def _show_image( self , photo: ImageTk.PhotoImage , w: int , h: int ) :
		self._photo_refs.append( photo )
		self._canvas.itemconfigure( self._img_item , image=photo )
		self._canvas.coords( self._img_item , 0 , 0 )
		self._canvas.itemconfigure( self._txt_item , text="" )
		self._canvas.itemconfigure( self._overlay_item , text="" )
		self._canvas.configure( scrollregion=(0 , 0 , w , h) )

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

	def _render_image( self ) :
		if self._raw_img is None :
			cached = _preview_cache.get( self._fp , "image" ) if _preview_cache else None
			if cached :
				self._raw_img = cached
			else :
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
		cw , _ = self._canvas_size( )
		base_scale = min( cw / max( pages[ 0 ].width , 1 ) , 1.4 )
		scale = base_scale * self._zoom
		self._clear_canvas( )
		y , max_w , gap = 0 , 0 , 8
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
		self._canvas.itemconfigure( self._overlay_item ,
																text="▶ Video — thumbnail only. Use ↗ Open for playback." , fill=FG_WARN )
		self._canvas.coords( self._overlay_item , 8 , h + 10 )
		self._canvas.configure( scrollregion=(0 , 0 , w , h + 30) )

	def _render_metadata( self ) :
		if self._meta_text is None :
			cached = _preview_cache.get( self._fp , "meta" ) if _preview_cache else None
			if cached :
				self._meta_text = cached
			else :
				self._show_text( "Extracting metadata…" , FG_WARN )
				threading.Thread( target=self._load_meta_bg , args=(self._fp ,) , daemon=True ).start( )
				return
		self._show_text( self._meta_text , FG_TEXT , FONT_MONO )

	def _load_meta_bg( self , fp: str ) :
		stats = _file_stats( fp )
		tika = _tika_meta( fp )
		result = f"{stats}\n\n── Tika Metadata (--json) ──\n{tika}"
		if _preview_cache :
			_preview_cache.put( fp , "meta" , result )
		try :
			self.frame.after( 0 , self._on_meta_ready , fp , result )
		except Exception :
			pass

	def _on_meta_ready( self , fp: str , text: str ) :
		if self._fp == fp :
			self._meta_text = text
			if self._mode == "metadata" :
				self._show_text( text , FG_TEXT , FONT_MONO )

	def _zoom_in( self ) :
		self._zoom = min( self._zoom + 0.25 , 6.0 )
		self._zoom_lbl.configure( text=f"{int( self._fit_scale * self._zoom * 100 )}%" )
		if self._mode == "preview" : self._render( )

	def _zoom_out( self ) :
		self._zoom = max( self._zoom - 0.25 , 0.25 )
		self._zoom_lbl.configure( text=f"{int( self._fit_scale * self._zoom * 100 )}%" )
		if self._mode == "preview" : self._render( )

	def _on_wheel( self , event ) :
		if event.num == 4 :       self._canvas.yview_scroll( -1 , "units" )
		elif event.num == 5 :     self._canvas.yview_scroll( 1 , "units" )
		else :                    self._canvas.yview_scroll( int( -event.delta / 120 ) , "units" )

	def _open_external( self ) :
		if not self._fp : return
		try :
			if os.name == "nt" :  os.startfile( self._fp )
			else :                subprocess.Popen( [ "xdg-open" , self._fp ] )
		except Exception as e :
			messagebox.showerror( "Open Error" , str( e ) )


# ══════════════════════════════════════════════════════════════════════════════
# SCAN FRAME  (v3 — per-category streaming)
# ══════════════════════════════════════════════════════════════════════════════

class ScanFrame( ctk.CTkFrame ) :
	def __init__( self , parent , on_category_done , on_all_done , default_folder="" ) :
		super( ).__init__( parent , fg_color="transparent" )
		self._on_category_done = on_category_done
		self._on_all_done = on_all_done
		self._default_folder = default_folder
		self._cancel_flag = False
		self._active_engine: DetectionEngine | None = None
		self._thread: threading.Thread | None = None
		self._q: queue.Queue = queue.Queue( )
		self._build( )

	def _build( self ) :
		outer = ctk.CTkFrame( self , fg_color="transparent" )
		outer.pack( fill="both" , expand=True , padx=12 , pady=8 )
		outer.columnconfigure( 0 , weight=1 )
		row = 0

		# ── Section 1: Folder ────────────────────────────────────────
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

		# ── Section 2: Detection Modes ───────────────────────────────
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
			("Exact duplicates" , self._do_exact , "SHA-256 byte match · size prefilter · parallel") ,
			("Filename match" , self._do_names , "Sanitized name + same extension") ,
			("Cross-ext filename" , self._do_names_cross_ext , "Sanitized name · any extension") ,
			("Images  (pHash)" , self._do_img , "Parallel pHash + BK-tree · O(n log n)") ,
			("Documents  (text)" , self._do_doc , "Text similarity · size + length prefilter") ,
		]
		for i , (label , var , hint) in enumerate( mode_defs ) :
			card = ctk.CTkFrame( modes , fg_color=BG_CARD , corner_radius=8 )
			card.grid( row=i // 2 , column=i % 2 , sticky="ew" , padx=4 , pady=3 )
			ctk.CTkCheckBox( card , text=label , variable=var , font=FONT_B , text_color=FG_TEXT ).pack(
					anchor="w" , padx=10 , pady=(8 , 2) )
			ctk.CTkLabel( card , text=hint , font=FONT_S , text_color=FG_DIM ).pack( anchor="w" , padx=30 , pady=(0 , 8) )

		# ── Section 3: Thresholds ────────────────────────────────────
		sec3 = ctk.CTkFrame( outer , fg_color=BG_PANEL , corner_radius=RADIUS )
		sec3.grid( row=row , column=0 , sticky="ew" , pady=(0 , 8) );
		row += 1
		_section_label( sec3 , "⚙  Thresholds" ).pack( anchor="w" , padx=14 , pady=(12 , 8) )
		thr = ctk.CTkFrame( sec3 , fg_color="transparent" )
		thr.pack( fill="x" , padx=14 , pady=(0 , 12) )
		thr.columnconfigure( (0 , 1 , 2) , weight=1 )

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

		# ── Section 4: Scan & Category Status ────────────────────────
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

		# Per-category status labels
		self._cat_frame = ctk.CTkFrame( sec4 , fg_color="transparent" )
		self._cat_frame.pack( fill="x" , padx=14 , pady=(0 , 8) )
		self._cat_labels: dict[ str , ctk.CTkLabel ] = { }
		for cat in CATEGORY_ORDER :
			lbl = ctk.CTkLabel( self._cat_frame , text=f"⬚ {cat}" , font=FONT_S , text_color=FG_DIM )
			lbl.pack( side="left" , padx=(0 , 16) )
			self._cat_labels[ cat ] = lbl

		self._res_lbl = ctk.CTkLabel( sec4 , text="" , font=FONT , text_color=FG_DIM )
		self._res_lbl.pack( padx=14 , pady=(0 , 14) )

	def _browse( self ) :
		d = filedialog.askdirectory( title="Select folder to scan" )
		if d : self._folder_var.set( d )

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
		self._res_lbl.configure( text="" )
		self._cancel_flag = False

		# Reset category labels
		for cat in CATEGORY_ORDER :
			self._cat_labels[ cat ].configure( text=f"⬚ {cat}" , text_color=FG_DIM )

		# Capture settings for the background thread
		settings = dict(
				folder=folder ,
				recursive=self._recursive.get( ) ,
				img_thresh=self._img_thresh_val.get( ) ,
				doc_thresh=self._doc_thresh_val.get( ) ,
				do_images=self._do_img.get( ) ,
				do_docs=self._do_doc.get( ) ,
				do_exact=self._do_exact.get( ) ,
				do_names=self._do_names.get( ) ,
				do_names_cross_ext=self._do_names_cross_ext.get( ) ,
				min_kb=min_kb ,
		)

		self._thread = threading.Thread(
				target=self._run_categories , args=(settings ,) , daemon=True )
		self._thread.start( )
		self._poll( )

	def _run_categories( self , s: dict ) :
		"""Background thread: collect files, bucket by category, scan each."""
		try :
			self._q.put( ("progress" , "Collecting files…" , 0.0) )
			folder = Path( s[ "folder" ] )
			it = folder.rglob( "*" ) if s[ "recursive" ] else folder.iterdir( )
			min_bytes = s[ "min_kb" ] * 1024
			all_files = [ ]
			for p in it :
				if self._cancel_flag :
					return
				try :
					if p.is_file( ) and p.stat( ).st_size >= min_bytes :
						all_files.append( p )
				except OSError :
					pass

			buckets = _categorize_files( all_files )
			cat_names = [ c for c in CATEGORY_ORDER if c in buckets ]
			total_files = sum( len( v ) for v in buckets.values( ) )
			self._q.put( ("progress" , f"Found {total_files} files in {len( cat_names )} categories." , 0.01) )

			total_cats = len( cat_names )
			total_pairs = 0

			for ci , cat in enumerate( cat_names ) :
				if self._cancel_flag :
					return
				cat_files = buckets[ cat ]
				self._q.put( ("cat_start" , cat , len( cat_files )) )

				# Determine which content-similarity modes apply
				cat_do_images = s[ "do_images" ] and cat == "Images"
				cat_do_docs = s[ "do_docs" ] and cat == "Documents"

				# Progress for this category maps to a slice of the bar
				cat_start = ci / total_cats
				cat_end = (ci + 1) / total_cats

				engine = DetectionEngine(
						files=cat_files ,
						img_thresh=s[ "img_thresh" ] ,
						doc_thresh=s[ "doc_thresh" ] ,
						do_images=cat_do_images ,
						do_docs=cat_do_docs ,
						do_exact=s[ "do_exact" ] ,
						do_names=s[ "do_names" ] ,
						do_names_cross_ext=s[ "do_names_cross_ext" ] ,
						min_kb=0 ,  # already filtered
						on_progress=lambda m , p , _s=cat_start , _e=cat_end , _c=cat :
						self._q.put( ("progress" , f"[{_c}] {m}" , _s + p * (_e - _s)) ) ,
						on_done=lambda g : None ,
						on_error=lambda e : self._q.put( ("error" , e) ) ,
				)
				self._active_engine = engine
				groups = engine.run( )
				self._active_engine = None

				if self._cancel_flag :
					return

				n_pairs = sum( len( list( itertools.combinations( g , 2 ) ) ) for g in groups ) if groups else 0
				total_pairs += n_pairs
				self._q.put( ("cat_done" , cat , groups , n_pairs) )

			self._q.put( ("all_done" , total_pairs) )

		except Exception as e :
			self._q.put( ("error" , str( e )) )

	def _cancel( self ) :
		self._cancel_flag = True
		if self._active_engine :
			self._active_engine.cancel( )
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
					self._pbar.set( max( 0.0 , min( 1.0 , pct ) ) )

				elif kind == "cat_start" :
					_ , cat , n_files = item
					self._cat_labels[ cat ].configure(
							text=f"⏳ {cat} ({n_files})" , text_color=FG_WARN )

				elif kind == "cat_done" :
					_ , cat , groups , n_pairs = item
					if n_pairs > 0 :
						self._cat_labels[ cat ].configure(
								text=f"✓ {cat} ({n_pairs} pairs)" , text_color=FG_GOOD )
					else :
						self._cat_labels[ cat ].configure(
								text=f"✓ {cat} (clean)" , text_color=FG_DIM )
					if groups :
						self._on_category_done( cat , groups )

				elif kind == "all_done" :
					_ , total_pairs = item
					self._scan_btn.configure( state="normal" )
					self._cancel_btn.configure( state="disabled" )
					self._pbar.set( 1.0 )
					if total_pairs > 0 :
						self._status_lbl.configure(
								text=f"✓ All categories scanned — {total_pairs} total pair(s)." ,
								text_color=FG_GOOD )
						self._res_lbl.configure(
								text=f"Scanning complete. {total_pairs} pair(s) queued for review." ,
								text_color=FG_TEXT )
					else :
						self._status_lbl.configure(
								text="✓ All categories scanned — no duplicates found." ,
								text_color=FG_WARN )
						self._res_lbl.configure(
								text="No duplicates found. Try loosening thresholds or enabling more modes." ,
								text_color=FG_WARN )
					self._on_all_done( )
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


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW FRAME  (v3 — incremental loading + alterations button)
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
		self._scan_complete = False
		self._known_pair_keys: set[ tuple[ str , str ] ] = set( )
		self._build( )

	def reset( self ) :
		"""Reset state for a new scan."""
		self._pairs.clear( )
		self._skipped.clear( )
		self._known_pair_keys.clear( )
		self._idx = 0
		self._log.clear( )
		self._stats = Counter( )
		self._t0 = time.time( )
		self._scan_complete = False
		self._lf = self._rf = ""
		self._update_header( )
		self._lp.load( None )
		self._rp.load( None )

	def add_groups( self , groups ) :
		"""Append newly discovered duplicate groups (called per category)."""
		new_pairs = _groups_to_pairs( groups )
		added = 0
		for p in new_pairs :
			key = (min( p[ 0 ] , p[ 1 ] ) , max( p[ 0 ] , p[ 1 ] ))
			if key not in self._known_pair_keys :
				self._known_pair_keys.add( key )
				self._pairs.append( p )
				added += 1
		self._stats[ "loaded" ] = len( self._pairs ) + len( self._skipped )
		self.logger.info( f"ReviewFrame.add_groups: +{added} pairs (total {len( self._pairs )})" )
		self._prefetch_upcoming( )

		# If we're sitting at "no pairs" or past the end, show the first new pair
		if self._lf == "" and self._rf == "" and self._pairs :
			self._idx = 0
			self._show( )
		else :
			self._update_header( )

	def mark_scan_complete( self ) :
		self._scan_complete = True
		self._update_header( )

	def _prefetch_upcoming( self ) :
		if not _preview_cache : return
		active = self._active( )
		for i in range( self._idx , min( self._idx + PREFETCH_AHEAD , len( active ) ) ) :
			_preview_cache.prefetch_pair( active[ i ] )

	def _build( self ) :
		# ── Top bar ──────────────────────────────────────────────────
		top = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=46 )
		top.pack( fill="x" , padx=12 , pady=(12 , 6) )
		top.pack_propagate( False )
		_btn( top , "Load JSON manually" , self._load_json , width=160 ).pack( side="left" , padx=12 , pady=8 )
		self._prog_lbl = ctk.CTkLabel( top , text="No pairs loaded." , font=FONT_B , text_color=FG_TEXT )
		self._prog_lbl.pack( side="left" , padx=10 )
		self._stats_lbl = ctk.CTkLabel( top , text="" , font=FONT_S , text_color=FG_DIM )
		self._stats_lbl.pack( side="left" , padx=6 )
		self._scan_status_lbl = ctk.CTkLabel( top , text="" , font=FONT_S , text_color=FG_WARN )
		self._scan_status_lbl.pack( side="right" , padx=12 )

		# ── Preview panes ────────────────────────────────────────────
		cmp = ctk.CTkFrame( self , fg_color=BG_APP , corner_radius=0 )
		cmp.pack( fill="both" , expand=True , padx=12 , pady=4 )
		cmp.columnconfigure( 0 , weight=1 )
		cmp.columnconfigure( 1 , weight=1 )
		cmp.rowconfigure( 0 , weight=1 )
		self._lp = PreviewPane( cmp , "← LEFT" )
		self._rp = PreviewPane( cmp , "RIGHT →" )
		self._lp.frame.grid( row=0 , column=0 , sticky="nsew" , padx=(0 , 4) )
		self._rp.frame.grid( row=0 , column=1 , sticky="nsew" , padx=(4 , 0) )

		# ── Action bar ───────────────────────────────────────────────
		acts = ctk.CTkFrame( self , fg_color=BG_PANEL , corner_radius=RADIUS , height=56 )
		acts.pack( fill="x" , padx=12 , pady=(4 , 12) )
		acts.pack_propagate( False )

		_btn( acts , "←  Keep Left" , self._keep_left , width=180 ,
					color="#2A5A8A" , hover="#3A6A9A" ).pack( side="left" , padx=(12 , 4) , pady=10 )
		ctk.CTkLabel( acts , text="[←]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "↑  Keep Both  (UUID4)" , self._keep_both , width=190 ,
					color="#3A7A5A" , hover="#4A8A6A" ).pack( side="left" , padx=(8 , 4) , pady=10 )
		ctk.CTkLabel( acts , text="[↑]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "↓  Skip" , self._skip , width=80 ,
					color=BG_TAB , hover=BG_PANEL ).pack( side="left" , padx=(8 , 4) , pady=10 )
		ctk.CTkLabel( acts , text="[↓]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "⚠  Needs Alteration" , self._needs_alteration , width=170 ,
					color="#8B6914" , hover="#A07A1E" ).pack( side="left" , padx=(8 , 4) , pady=10 )
		ctk.CTkLabel( acts , text="[A]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		_btn( acts , "🗑  Delete Both" , self._delete_both , width=140 ,
					color="#8B2020" , hover="#A03030" ).pack( side="left" , padx=(8 , 4) , pady=10 )
		ctk.CTkLabel( acts , text="[Del]" , font=FONT_S , text_color=FG_DIM ).pack( side="left" )

		ctk.CTkLabel( acts , text="[→]" , font=FONT_S , text_color=FG_DIM ).pack( side="right" , padx=(0 , 8) )
		_btn( acts , "Keep Right  →" , self._keep_right , width=180 ,
					color="#2A5A8A" , hover="#3A6A9A" ).pack( side="right" , padx=(4 , 12) , pady=10 )

		self.after( 200 , self._bind_keys )

	def _bind_keys( self ) :
		root = self.winfo_toplevel( )
		root.bind( "<Left>" , lambda _ : self._keep_left( ) )
		root.bind( "<Right>" , lambda _ : self._keep_right( ) )
		root.bind( "<Up>" , lambda _ : self._keep_both( ) )
		root.bind( "<Down>" , lambda _ : self._skip( ) )
		root.bind( "<Delete>" , lambda _ : self._delete_both( ) )
		root.bind( "a" , lambda _ : self._needs_alteration( ) )
		root.bind( "A" , lambda _ : self._needs_alteration( ) )

	def _active( self ) :
		return self._pairs + self._skipped

	def _current( self ) :
		p = self._active( )
		return p[ self._idx ] if self._idx < len( p ) else None

	def _update_header( self ) :
		total = len( self._active( ) )
		if total == 0 :
			self._prog_lbl.configure( text="No pairs loaded." )
			self._stats_lbl.configure( text="" )
		else :
			pos = min( self._idx + 1 , total )
			self._prog_lbl.configure( text=f"Pair {pos} / {total}" )
			self._stats_lbl.configure( text=(
				f"L:{self._stats.get( 'left' , 0 )}  R:{self._stats.get( 'right' , 0 )}  "
				f"Both:{self._stats.get( 'both' , 0 )}  Alt:{self._stats.get( 'alteration' , 0 )}  "
				f"Del:{self._stats.get( 'deleted' , 0 )}  Skip:{len( self._skipped )}") )
		if self._scan_complete :
			self._scan_status_lbl.configure( text="Scan complete" , text_color=FG_GOOD )
		else :
			self._scan_status_lbl.configure( text="⏳ Scanning…" , text_color=FG_WARN )

	def _show( self ) :
		while True :
			pair = self._current( )
			if pair is None :
				self._lf = self._rf = ""
				self._lp.load( None )
				self._rp.load( None )
				if self._scan_complete :
					self._finish( )
				else :
					self._update_header( )
					self._prog_lbl.configure( text="Waiting for more results…" )
				return
			lf , rf = pair
			if Path( lf ).exists( ) and Path( rf ).exists( ) :
				break
			self._idx += 1
		self._lf , self._rf = pair
		self._update_header( )
		self._lp.load( self._lf )
		self._rp.load( self._rf )
		self._prefetch_upcoming( )

	# ── Actions ──────────────────────────────────────────────────────

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

	def _delete_both( self ) :
		if not self._lf or not self._rf : return
		try :
			del_l = self._mv( self._lf , DELETE_DIR )
			del_r = self._mv( self._rf , DELETE_DIR )
			self._rec( { "action" : "delete_both" , "deleted_left" : del_l , "deleted_right" : del_r } )
			self._stats[ "deleted" ] += 1
			self.logger.info( f"delete_both | {Path( self._lf ).name}  +  {Path( self._rf ).name}" )
			self._advance( )
		except Exception as e :
			messagebox.showerror( "Error" , str( e ) )

	def _needs_alteration( self ) :
		if not self._lf or not self._rf : return
		try :
			alt_l = self._mv( self._lf , ALTERATIONS_REQUIRED_DIR )
			alt_r = self._mv( self._rf , ALTERATIONS_REQUIRED_DIR )
			self._rec( {
				"action"      : "needs_alteration" ,
				"moved_left"  : alt_l ,
				"moved_right" : alt_r ,
			} )
			self._stats[ "alteration" ] += 1
			self.logger.info(
					f"needs_alteration | {Path( self._lf ).name}  +  {Path( self._rf ).name} "
					f"→ {ALTERATIONS_REQUIRED_DIR}" )
			self._advance( )
		except Exception as e :
			messagebox.showerror( "Error" , str( e ) )

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
			self.add_groups( data )
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
				f"Both:{self._stats.get( 'both' , 0 )} Alt:{self._stats.get( 'alteration' , 0 )} "
				f"Del:{self._stats.get( 'deleted' , 0 )} | {elapsed}s | log: {log_path}" )
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
				on_category_done=self._on_category_done ,
				on_all_done=self._on_all_done ,
				default_folder=str( source_dir ) if source_dir else "" ,
		)
		self._scan.pack( fill="both" , expand=True )

		self._review = ReviewFrame(
				self._tabs.tab( "👁  Review" ) ,
				base_dir=source_dir ,
				logger=self.logger ,
		)
		self._review.pack( fill="both" , expand=True )

		self._first_results = True
		self.logger.info( f"App started | source_dir={source_dir}" )

	def _on_category_done( self , cat_name: str , groups: list ) :
		"""Called per category as results stream in."""
		self.logger.info( f"Category '{cat_name}' done — {len( groups )} group(s)" )
		self._review.add_groups( groups )
		# Switch to Review tab on first results so user can start immediately
		if self._first_results :
			self._first_results = False
			self._tabs.set( "👁  Review" )

	def _on_all_done( self ) :
		"""Called when all categories have finished scanning."""
		self._review.mark_scan_complete( )

	def run( self ) :
		if any( item.is_file( ) for item in self.source_dir.iterdir( ) ) :
			self.root.mainloop( )
		else :
			self.logger.warning( f"No items found in {self.source_dir}. Not running app." )
