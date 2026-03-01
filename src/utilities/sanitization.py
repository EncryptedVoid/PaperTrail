"""
Sanitization Utility Functions

This module provides utility functions for validating and checking file integrity,
including password protection detection, corruption checking, empty file detection,
and file type validation.

Key functionality:
- Password protection detection for ZIP, PDF, Office documents (DOCX/XLSX/PPTX), RAR, and 7Z files
- File corruption detection for images, PDFs, ZIP archives, and generic binary files
- Empty file detection (zero-byte files)
- File type validation using Apache Tika content-based detection

These utilities are used by the sanitization pipeline to ensure only valid,
processable files are accepted for further processing.
"""
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Set

import msoffcrypto
import py7zr
import pymupdf
import rarfile

from config import (
	ANKI_EXTENSIONS ,
	ARCHIVE_TYPES ,
	AUDIO_TYPES ,
	CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS ,
	IMAGE_TYPES ,
	MICROSOFT_FILE_TYPES ,
	MIME_TO_EXT_MAP ,
	TEXT_TYPES ,
	TIKA_APP_JAR_PATH ,
	VIDEO_TYPES
)


def is_password_protected( artifact_location: Path ) -> bool :
	"""
	Check if a file is password-protected or encrypted.
	Supported formats: ZIP, PDF, DOCX, XLSX, PPTX, RAR, 7Z

	Args:
		artifact_location: Path object pointing to the file to check

	Returns:
		bool: True if the file is password-protected or encrypted, False otherwise

	Raises:
		FileNotFoundError: If the specified file does not exist
	"""
	# Verify the file exists before attempting to check protection status
	# Path.exists() returns True if the artifact_location points to an existing file or directory
	if not artifact_location.exists( ) :
		raise FileNotFoundError( f"File not found: {artifact_location}" )

	# Extract file extension and convert to lowercase for case-insensitive comparison
	# Path.suffix returns the file extension including the dot (e.g., '.zip')
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )

	# Check ZIP files by examining the flag_bits in each file entry
	# The 0x1 bit indicates encryption in ZIP format
	if artifact_ext == "zip" :
		try :
			# zipfile.ZipFile opens ZIP archives for reading
			# Context manager (with statement) ensures proper file closure
			with zipfile.ZipFile( artifact_location ) as zf :
				# zf.infolist() returns a list of ZipInfo objects for each file in archive
				for info in zf.infolist( ) :
					# Bitwise AND operation checks if encryption flag is set
					if info.flag_bits & 0x1 :
						return True
			return False
		except Exception :
			# If ZIP file is malformed or unreadable, assume not protected
			return False

	# Check PDF files using PyMuPDF (fitz) library
	# PDFs can be password-protected or encrypted
	elif artifact_ext == "pdf" :
		doc = pymupdf.open( artifact_location )
		# needs_pass indicates if a password is required to open
		# is_encrypted indicates if the document has any encryption
		is_protected = doc.needs_pass or doc.is_encrypted
		# Always close the document to free resources
		doc.close( )
		return is_protected

	# Check Microsoft Office files (modern XML-based formats)
	# These formats are actually ZIP archives containing XML files
	elif artifact_ext in MICROSOFT_FILE_TYPES :
		try :
			# Open file in binary mode ('rb') for msoffcrypto processing
			with open( artifact_location , "rb" ) as f :
				# OfficeFile class parses the Office document structure
				office_file = msoffcrypto.OfficeFile( f )
				# is_encrypted() checks for document-level encryption
				return office_file.is_encrypted( )
		except Exception :
			# File may be corrupted or in legacy format
			return False

	# Check RAR archive files
	# RAR format supports per-file password protection
	elif artifact_ext == "rar" :
		try :
			# RarFile class opens RAR archives
			with rarfile.RarFile( artifact_location ) as rf :
				# Check if any file in the archive requires a password
				# any() returns True if at least one element is True
				return any( info.needs_password( ) for info in rf.infolist( ) )
		except Exception :
			# RAR file may be corrupted or in unsupported version
			return False

	# Check 7-Zip archive files
	# 7Z format supports archive-level encryption
	elif artifact_ext == "7z" :
		try :
			# Open archive in read mode ('r')
			with py7zr.SevenZipFile( artifact_location , "r" ) as archive :
				# needs_password() checks if password is required for extraction
				return archive.needs_password( )
		except Exception :
			# 7Z file may be corrupted
			return False

	# File format is not supported for password protection detection
	return False


def is_corrupted( artifact_location: Path ) -> bool :
	"""
	Returns True if the file is detectably corrupted, False if valid or type unknown.

	Args:
		artifact_location: Path to the file to check.
	"""

	# Basic sanity checks
	if not artifact_location.exists( ) :
		return True
	if artifact_location.stat( ).st_size == 0 :
		return True  # zero-byte is always corrupted

	if "._" in artifact_location.stem.lower( ).strip( ) :
		return True

	# Execute Apache Tika as a subprocess using Java
	# subprocess.run() launches an external process and waits for completion
	result = subprocess.run(
			[
				"java" ,  # Java Runtime Environment command
				"-jar" ,  # Run JAR file
				str( TIKA_APP_JAR_PATH ) ,  # Path to Tika JAR
				"--detect" ,  # Tika command to detect MIME type
				str( artifact_location ) ,  # File to analyze
			] ,
			capture_output=True ,  # Capture stdout and stderr
			text=True ,  # Return output as string (not bytes)
			timeout=30 ,  # Timeout after 30 seconds
	)

	# Extract MIME type from Tika's stdout and remove whitespace
	# MIME type format: type/subtype (e.g., "application/pdf")
	detected_mime_type = result.stdout.strip( )

	if not detected_mime_type in MIME_TO_EXT_MAP :
		raise RuntimeError( f"Mime extension mapping not found for mime [{detected_mime_type}]" )

	mapped_mime_ext: Set[ str ] = MIME_TO_EXT_MAP[ detected_mime_type ]

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
	if artifact_ext == "jpeg" or artifact_ext == "cr2" or artifact_ext == "arw" or artifact_ext == "nef" :
		artifact_ext = "jpg"
	elif artifact_ext in ANKI_EXTENSIONS :
		artifact_ext = "zip"

	return any(
			((artifact_ext == extension.lower( ).strip( ).strip( '.' ))
			 for extension in mapped_mime_ext) ,
	)


def is_supported_type( artifact_location: Path ) -> bool :
	"""
	Check if a file's type is supported for processing.

	Args:
		artifact_location: Path object pointing to the file to check

	Returns:
		bool: True if file type is supported, False otherwise
	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )

	return (
			artifact_ext in EMAIL_TYPES
			or artifact_ext in MICROSOFT_FILE_TYPES
			or artifact_ext in DOCUMENT_TYPES
			or artifact_ext in IMAGE_TYPES
			or artifact_ext in VIDEO_TYPES
			or artifact_ext in AUDIO_TYPES
			or artifact_ext in TEXT_TYPES
			or artifact_ext in ARCHIVE_TYPES
			or artifact_ext in CODE_EXTENSIONS
			or artifact_ext in CAD_FILES
			or artifact_ext in ANKI_EXTENSIONS
			or artifact_ext in EXECUTABLE_EXTENSIONS
			or artifact_ext in DIGITAL_CONTACT_EXTENSIONS
	)


def sanitize_artifact_name( artifact_name: str ) -> str :
	stem , *ext_parts = artifact_name.rsplit( "." , 1 )

	pattern = r'\s*[-_]?\s*\(?\bcopy\b\)?\s*(\(\d+\))?|\s+\(\d+\)$'
	stem = re.sub( pattern , "" , stem , flags=re.IGNORECASE ).strip( )

	return stem.capitalize( )
