"""
Folder Manager – Tkinter GUI for reviewing, zipping/unzipping, and moving
folders and archives from a source directory into a destination directory.

Dependencies:
    pip install py7zr rarfile

External tool (optional):
    UnRAR – https://www.rarlab.com/rar_addfiles.htm  (needed by rarfile)
"""

import logging
import os
import shutil
import tarfile
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import messagebox

import py7zr
import rarfile

from config import ARCHIVE_TYPES

# ── Theme ────────────────────────────────────────────────────────────
BG = "#1a1a1a"
BG_CARD = "#242424"
BG_HOVER = "#2e2e2e"
FG = "#e0e0e0"
FG_DIM = "#888888"
ACCENT = "#d63031"
ACCENT_HOVER = "#ff4444"
ACCENT_DIM = "#8b1a1a"
FONT = ("Segoe UI" , 11)
FONT_BOLD = ("Segoe UI" , 11 , "bold")
FONT_SM = ("Segoe UI" , 9)
FONT_LG = ("Segoe UI" , 14 , "bold")
FONT_TITLE = ("Segoe UI" , 18 , "bold")


# ── Helpers ──────────────────────────────────────────────────────────

def is_archive( path: Path ) -> bool :
	name = path.name.lower( )
	for ext in ARCHIVE_TYPES :
		if name.endswith( ext ) :
			return True
	return False


def archive_ext( path: Path ) -> str :
	"""Return the full archive extension (handles .tar.gz etc.)."""
	name = path.name.lower( )
	for ext in sorted( ARCHIVE_TYPES , key=len , reverse=True ) :
		if name.endswith( ext ) :
			return ext
	return path.suffix.lower( )


def archive_stem( path: Path ) -> str :
	"""Return the name without any archive extension (handles .tar.gz etc.)."""
	name = path.name
	lower = name.lower( )
	for ext in sorted( ARCHIVE_TYPES , key=len , reverse=True ) :
		if lower.endswith( ext ) :
			return name[ : len( name ) - len( ext ) ]
	return path.stem


def extract_archive( archive_path: Path , dest: Path ) -> None :
	"""Extract any supported archive into *dest*."""
	ext = archive_ext( archive_path )
	dest.mkdir( parents=True , exist_ok=True )

	if ext == ".zip" :
		with zipfile.ZipFile( archive_path , "r" ) as zf :
			zf.extractall( dest )

	elif ext == ".7z" :
		with py7zr.SevenZipFile( archive_path , "r" ) as sz :
			sz.extractall( dest )

	elif ext == ".rar" :
		with rarfile.RarFile( archive_path , "r" ) as rf :
			rf.extractall( dest )

	elif ext in { ".tar" , ".tgz" , ".tbz2" , ".txz" , ".tar.gz" , ".tar.bz2" , ".tar.xz" , ".gz" , ".bz2" , ".xz" } :
		with tarfile.open( archive_path , "r:*" ) as tf :
			tf.extractall( dest )
	else :
		raise RuntimeError( f"Unsupported archive format: {ext}" )


def zip_folder( folder_path: Path , dest_zip: Path ) -> None :
	"""Compress *folder_path* into a .zip at *dest_zip*."""
	with zipfile.ZipFile( dest_zip , "w" , zipfile.ZIP_DEFLATED ) as zf :
		for root , _ , files in os.walk( folder_path ) :
			for f in files :
				full = Path( root ) / f
				zf.write( full , full.relative_to( folder_path ) )


# ── Canvas icon drawing ──────────────────────────────────────────────

def draw_folder_icon( canvas: tk.Canvas , x: int , y: int , size: int = 64 , color: str = ACCENT ) -> None :
	hs = size // 2
	canvas.create_polygon(
			x - hs , y - hs + 4 ,
			x - hs , y - hs - 2 ,
			x - hs + size * 0.35 , y - hs - 2 ,
			x - hs + size * 0.45 , y - hs + 4 ,
			fill=color , outline="" ,
	)
	canvas.create_rectangle( x - hs , y - hs + 4 , x + hs , y + hs , fill=color , outline="" )
	canvas.create_line( x - hs + 2 , y - hs + 10 , x + hs - 2 , y - hs + 10 , fill="#ffffff" , width=1 )


def draw_archive_icon( canvas: tk.Canvas , x: int , y: int , size: int = 64 , color: str = ACCENT ) -> None :
	hs = size // 2
	canvas.create_rectangle( x - hs + 8 , y - hs , x + hs - 8 , y + hs , fill=color , outline="" )
	stripe_color = "#1a1a1a"
	for i in range( 5 ) :
		sy = y - hs + 6 + i * (size // 6)
		offset = 4 if i % 2 == 0 else -4
		canvas.create_rectangle( x + offset - 5 , sy , x + offset + 5 , sy + 4 , fill=stripe_color , outline="" )


# ── Item kind ────────────────────────────────────────────────────────

class _ItemKind :
	FOLDER = "folder"
	ARCHIVE = "archive"


# ── Main Application ─────────────────────────────────────────────────

class FolderManagerApp( tk.Tk ) :
	"""
	Tkinter GUI that walks through every folder / archive inside *source_dir*
	and lets the user choose to zip, unzip, or move each item into *dest_dir*.

	Parameters
	----------
	source_dir : Path
			Directory to scan. Must exist and contain at least one folder or archive.
	dest_dir : Path
			Directory where processed items are moved to. Created if it doesn't exist.
	logger : logging.Logger
			Pre-configured logger instance — the app will NOT create its own.
	"""

	def __init__( self , source_dir: Path , dest_dir: Path , logger: logging.Logger ) -> None :
		# ── Validate before touching Tk ──────────────────────────────
		self._log = logger
		self._source_root = source_dir
		self._dest_dir = dest_dir

		if not self._validate_startup( ) :
			return  # Do NOT call super().__init__(); the app never starts.

		super( ).__init__( )
		self.title( "Folder Manager" )
		self.configure( bg=BG )
		self.geometry( "900x700" )
		self.minsize( 750 , 550 )

		self._current_source: Path = self._source_root
		self._dir_stack: list[ Path ] = [ ]
		self._items: list[ dict ] = [ ]

		self._build_ui( )
		self._scan( )

	# ── Startup validation ───────────────────────────────────────────

	def _validate_startup( self ) -> bool :
		"""Return True if the app is safe to launch, False otherwise."""
		if not self._source_root.exists( ) :
			self._log.error(
					"Source directory does not exist: %s — aborting startup." , self._source_root ,
			)
			return False

		if not self._source_root.is_dir( ) :
			self._log.error(
					"Source path is not a directory: %s — aborting startup." , self._source_root ,
			)
			return False

		# Check for at least one folder or archive
		has_items = any(
				entry.is_dir( ) or (entry.is_file( ) and is_archive( entry ))
				for entry in self._source_root.iterdir( ) ,
		)
		if not has_items :
			self._log.error(
					"Source directory is empty (no folders or archives found): %s — aborting startup." ,
					self._source_root ,
			)
			return False

		self._log.info( "Source validated: %s" , self._source_root )
		self._log.info( "Destination set:  %s" , self._dest_dir )
		return True

	# ── UI construction ──────────────────────────────────────────────

	def _build_ui( self ) -> None :
		# Top bar
		top = tk.Frame( self , bg=BG )
		top.pack( fill="x" , padx=20 , pady=(18 , 6) )

		tk.Label( top , text="Folder Manager" , font=FONT_TITLE , bg=BG , fg=ACCENT ).pack( side="left" )

		self._btn_refresh = self._make_btn( top , "⟳  Refresh" , self._scan )
		self._btn_refresh.pack( side="right" )

		# Path labels
		path_frame = tk.Frame( self , bg=BG )
		path_frame.pack( fill="x" , padx=20 , pady=(0 , 4) )
		tk.Label(
				path_frame , text=f"Source: {self._source_root}" , font=FONT_SM , bg=BG , fg=FG_DIM , anchor="w" ,
		).pack( fill="x" )
		tk.Label(
				path_frame , text=f"Dest:   {self._dest_dir}" , font=FONT_SM , bg=BG , fg=FG_DIM , anchor="w" ,
		).pack( fill="x" )

		self._lbl_current = tk.Label(
				path_frame , text="" , font=FONT_SM , bg=BG , fg=ACCENT , anchor="w" ,
		)
		self._lbl_current.pack( fill="x" )

		# Separator
		tk.Frame( self , bg=ACCENT_DIM , height=2 ).pack( fill="x" , padx=20 , pady=(6 , 10) )

		# Scrollable card grid
		container = tk.Frame( self , bg=BG )
		container.pack( fill="both" , expand=True , padx=20 , pady=(0 , 12) )

		self._canvas = tk.Canvas( container , bg=BG , highlightthickness=0 )
		scrollbar = tk.Scrollbar(
				container , orient="vertical" , command=self._canvas.yview ,
				bg=BG , troughcolor=BG_CARD , activebackground=ACCENT ,
		)
		self._scroll_frame = tk.Frame( self._canvas , bg=BG )
		self._scroll_frame.bind(
				"<Configure>" , lambda _ : self._canvas.configure( scrollregion=self._canvas.bbox( "all" ) ) ,
		)
		self._canvas.create_window( (0 , 0) , window=self._scroll_frame , anchor="nw" )
		self._canvas.configure( yscrollcommand=scrollbar.set )
		self._canvas.pack( side="left" , fill="both" , expand=True )
		scrollbar.pack( side="right" , fill="y" )
		self._canvas.bind_all(
				"<MouseWheel>" , lambda e : self._canvas.yview_scroll( int( -1 * (e.delta / 120) ) , "units" ) ,
		)

		# Status bar
		self._status_var = tk.StringVar( value="Ready." )
		tk.Label(
				self , textvariable=self._status_var , font=FONT_SM , bg=BG_CARD , fg=FG_DIM ,
				anchor="w" , padx=12 , pady=6 ,
		).pack( fill="x" , side="bottom" )

	# ── Reusable button factory ──────────────────────────────────────

	def _make_btn( self , parent: tk.Widget , text: str , command , accent: bool = False ) -> tk.Label :
		bg_c = ACCENT if accent else BG_CARD
		fg_c = "#ffffff" if accent else FG
		hover = ACCENT_HOVER if accent else BG_HOVER
		btn = tk.Label( parent , text=text , font=FONT_BOLD , bg=bg_c , fg=fg_c , padx=14 , pady=6 , cursor="hand2" )
		btn.bind( "<Button-1>" , lambda _ : command( ) )
		btn.bind( "<Enter>" , lambda _ : btn.configure( bg=hover ) )
		btn.bind( "<Leave>" , lambda _ : btn.configure( bg=bg_c ) )
		return btn

	# ── Scan current source ──────────────────────────────────────────

	def _scan( self ) -> None :
		self._items.clear( )

		if not self._current_source.is_dir( ) :
			self._log.warning( "Current source no longer exists: %s" , self._current_source )
			self._status_var.set( "⚠  Current directory no longer exists." )
			self._render_grid( )
			return

		for entry in sorted( self._current_source.iterdir( ) ) :
			if entry.is_dir( ) :
				self._items.append( { "path" : entry , "kind" : _ItemKind.FOLDER , "name" : entry.name } )
			elif entry.is_file( ) and is_archive( entry ) :
				self._items.append( { "path" : entry , "kind" : _ItemKind.ARCHIVE , "name" : entry.name } )

		self._log.info( "Scanned %s — found %d item(s)." , self._current_source , len( self._items ) )

		# Update current-path label
		if self._current_source == self._source_root :
			self._lbl_current.configure( text="" )
		else :
			rel = self._current_source.relative_to( self._source_root )
			self._lbl_current.configure( text=f"📂  …/{rel}" )

		self._render_grid( )
		self._status_var.set( f"Found {len( self._items )} item(s)." )

	# ── Render card grid ─────────────────────────────────────────────

	def _render_grid( self ) -> None :
		for w in self._scroll_frame.winfo_children( ) :
			w.destroy( )

		# Back button if we've drilled in
		if self._dir_stack :
			back = self._make_btn( self._scroll_frame , "←  Back to parent" , self._pop_folder )
			back.grid( row=0 , column=0 , columnspan=4 , pady=(0 , 12) , sticky="w" )
			row_offset = 1
		else :
			row_offset = 0

		if not self._items :
			tk.Label(
					self._scroll_frame , text="No folders or archives found." ,
					font=FONT , bg=BG , fg=FG_DIM ,
			).grid( row=row_offset , column=0 , columnspan=4 , pady=40 )
			return

		cols = 4
		for idx , item in enumerate( self._items ) :
			r , c = divmod( idx , cols )
			card = self._make_card( self._scroll_frame , item )
			card.grid( row=r + row_offset , column=c , padx=10 , pady=10 , sticky="n" )

		for c in range( cols ) :
			self._scroll_frame.columnconfigure( c , weight=1 )

	# ── Single card ──────────────────────────────────────────────────

	def _make_card( self , parent: tk.Widget , item: dict ) -> tk.Frame :
		card = tk.Frame(
				parent , bg=BG_CARD , padx=12 , pady=12 ,
				highlightbackground=ACCENT_DIM , highlightthickness=1 ,
		)

		# Icon
		icon = tk.Canvas( card , width=80 , height=80 , bg=BG_CARD , highlightthickness=0 )
		icon.pack( pady=(4 , 6) )
		if item[ "kind" ] == _ItemKind.ARCHIVE :
			draw_archive_icon( icon , 40 , 40 , 64 , ACCENT )
		else :
			draw_folder_icon( icon , 40 , 40 , 64 , ACCENT )

		# Kind badge
		kind_text = archive_ext( item[ "path" ] ).upper( ) if item[ "kind" ] == _ItemKind.ARCHIVE else "FOLDER"
		tk.Label( card , text=kind_text , font=FONT_SM , bg=BG_CARD , fg=ACCENT ).pack( )

		# Name (truncated)
		display = item[ "name" ] if len( item[ "name" ] ) <= 22 else item[ "name" ][ :19 ] + "…"
		tk.Label( card , text=display , font=FONT_BOLD , bg=BG_CARD , fg=FG , wraplength=160 ).pack( pady=(2 , 8) )

		# Action buttons
		bf = tk.Frame( card , bg=BG_CARD )
		bf.pack( )

		if item[ "kind" ] == _ItemKind.ARCHIVE :
			self._make_btn( bf , "📦 Unzip" , lambda i=item : self._action_unzip( i ) , accent=True ).pack(
					side="left" , padx=(0 , 4) ,
			)
			self._make_btn( bf , "➜ Move" , lambda i=item : self._action_move_archive( i ) ).pack( side="left" )
		else :
			self._make_btn( bf , "🗜 Zip" , lambda i=item : self._action_zip( i ) , accent=True ).pack(
					side="left" , padx=(0 , 4) ,
			)
			self._make_btn( bf , "➜ Move" , lambda i=item : self._action_move_folder( i ) ).pack( side="left" )

		return card

	# ── Ensure dest exists ───────────────────────────────────────────

	def _ensure_dest( self ) -> bool :
		try :
			self._dest_dir.mkdir( parents=True , exist_ok=True )
			return True
		except OSError as exc :
			self._log.error( "Cannot create destination directory: %s" , exc )
			messagebox.showerror( "Destination Error" , str( exc ) )
			return False

	# ── Actions ──────────────────────────────────────────────────────

	def _action_unzip( self , item: dict ) -> None :
		if not self._ensure_dest( ) :
			return

		archive_path: Path = item[ "path" ]
		stem = archive_stem( archive_path )
		extract_to = self._current_source / f"_extracted_{stem}"

		self._log.info( "Extracting archive: %s" , archive_path.name )
		self._status_var.set( f"Extracting {archive_path.name}…" )
		self.update_idletasks( )

		try :
			extract_archive( archive_path , extract_to )
		except Exception as exc :
			self._log.error( "Extraction failed for %s: %s" , archive_path.name , exc )
			messagebox.showerror( "Extraction Error" , str( exc ) )
			self._status_var.set( "Extraction failed." )
			return

		self._log.info( "Extraction complete: %s" , archive_path.name )

		# Inspect contents
		children = list( extract_to.iterdir( ) )
		sub_dirs = [ c for c in children if c.is_dir( ) ]
		sub_archives = [ c for c in children if c.is_file( ) and is_archive( c ) ]

		if sub_dirs or sub_archives :
			# Nested content — drill in so the user can review
			archive_path.unlink( missing_ok=True )
			final_folder = self._current_source / stem
			if final_folder.exists( ) :
				shutil.rmtree( final_folder )
			extract_to.rename( final_folder )
			self._log.info( "Drilling into nested content: %s" , final_folder )
			self._push_into_folder( final_folder )
		else :
			# Flat content — move everything straight to dest
			self._move_contents( extract_to , self._dest_dir )
			shutil.rmtree( extract_to , ignore_errors=True )
			archive_path.unlink( missing_ok=True )
			self._log.info( "Moved flat contents of %s → %s" , stem , self._dest_dir )
			self._status_var.set( f"Moved contents of {stem} → destination." )
			self._scan( )

	def _action_move_archive( self , item: dict ) -> None :
		if not self._ensure_dest( ) :
			return
		dest = self._dest_dir / item[ "path" ].name
		shutil.move( str( item[ "path" ] ) , str( dest ) )
		self._log.info( "Moved archive (zipped): %s → %s" , item[ "name" ] , dest )
		self._status_var.set( f"Moved {item[ 'name' ]} → destination (kept zipped)." )
		self._scan( )

	def _action_zip( self , item: dict ) -> None :
		if not self._ensure_dest( ) :
			return
		folder: Path = item[ "path" ]
		zip_path = self._dest_dir / f"{folder.name}.zip"

		self._log.info( "Zipping folder: %s" , folder.name )
		self._status_var.set( f"Zipping {folder.name}…" )
		self.update_idletasks( )

		try :
			zip_folder( folder , zip_path )
			shutil.rmtree( folder )
			self._log.info( "Zipped & moved: %s → %s" , folder.name , zip_path )
			self._status_var.set( f"Zipped & moved {folder.name}.zip → destination." )
		except Exception as exc :
			self._log.error( "Zip failed for %s: %s" , folder.name , exc )
			messagebox.showerror( "Zip Error" , str( exc ) )
			self._status_var.set( "Zip failed." )
			return
		self._scan( )

	def _action_move_folder( self , item: dict ) -> None :
		if not self._ensure_dest( ) :
			return
		dest = self._dest_dir / item[ "path" ].name
		shutil.move( str( item[ "path" ] ) , str( dest ) )
		self._log.info( "Moved folder: %s → %s" , item[ "name" ] , dest )
		self._status_var.set( f"Moved folder {item[ 'name' ]} → destination." )
		self._scan( )

	# ── Directory stack navigation ───────────────────────────────────

	def _push_into_folder( self , folder: Path ) -> None :
		self._dir_stack.append( self._current_source )
		self._current_source = folder
		self._scan( )

	def _pop_folder( self ) -> None :
		if self._dir_stack :
			self._current_source = self._dir_stack.pop( )
			self._scan( )

	# ── File-move helper ─────────────────────────────────────────────

	def _move_contents( self , src_folder: Path , dest_folder: Path ) -> None :
		dest_folder.mkdir( parents=True , exist_ok=True )
		for child in src_folder.iterdir( ) :
			target = dest_folder / child.name
			if target.exists( ) :
				base , ext = target.stem , target.suffix
				i = 1
				while target.exists( ) :
					target = dest_folder / f"{base}_{i}{ext}"
					i += 1
			shutil.move( str( child ) , str( target ) )
			self._log.debug( "Moved %s → %s" , child.name , target )
