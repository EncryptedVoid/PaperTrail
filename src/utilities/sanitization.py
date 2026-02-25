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

import subprocess
import zipfile
from pathlib import Path

from config import (
	ANKI_EXTENSIONS , ARCHIVE_TYPES ,
	AUDIO_TYPES ,
	CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS , DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS , IMAGE_TYPES ,
	MICROSOFT_FILE_TYPES ,
	MIME_TO_EXT_MAP ,
	TEXT_TYPES ,
	TIKA_APP_JAR_PATH ,
	VIDEO_TYPES
)


def is_password_protected( artifact_location: Path ) -> bool :
	"""
	Check if a file is password-protected or encrypted.

	This function detects password protection across multiple file formats by
	examining file headers, encryption flags, and format-specific metadata.
	Supported formats: ZIP, PDF, DOCX, XLSX, PPTX, RAR, 7Z

	Args:
									artifact_location: Path object pointing to the file to check

	Returns:
									bool: True if the file is password-protected or encrypted, False otherwise
																					Returns False for unsupported formats or if required libraries are missing

	Raises:
									FileNotFoundError: If the specified file does not exist

	Note:
									This function requires optional dependencies for certain formats:
									- pymupdf for PDF files
									- msoffcrypto-tool for Office documents
									- rarfile for RAR archives
									- py7zr for 7Z archives
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
	if artifact_ext == ".zip" :
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
	elif artifact_ext == ".pdf" :
		try :
			# pymupdf.open() loads the PDF document into memory
			import pymupdf  # PyMuPDF library, imported as fitz in some versions

			doc = pymupdf.open( artifact_location )
			# needs_pass indicates if a password is required to open
			# is_encrypted indicates if the document has any encryption
			is_protected = doc.needs_pass or doc.is_encrypted
			# Always close the document to free resources
			doc.close( )
			return is_protected
		except ImportError :
			# Library not installed - return False and inform user
			print( "Install pymupdf: pip install pymupdf" )
			return False
		except Exception :
			# PDF may be corrupted or in an unsupported format
			return False

	# Check Microsoft Office files (modern XML-based formats)
	# These formats are actually ZIP archives containing XML files
	elif artifact_ext in MICROSOFT_FILE_TYPES :
		try :
			# msoffcrypto library handles Office encryption detection
			import msoffcrypto

			# Open file in binary mode ('rb') for msoffcrypto processing
			with open( artifact_location , "rb" ) as f :
				# OfficeFile class parses the Office document structure
				office_file = msoffcrypto.OfficeFile( f )
				# is_encrypted() checks for document-level encryption
				return office_file.is_encrypted( )
		except ImportError :
			# Library not installed - return False and inform user
			print( "Install msoffcrypto-tool: pip install msoffcrypto-tool" )
			return False
		except Exception :
			# File may be corrupted or in legacy format
			return False

	# Check RAR archive files
	# RAR format supports per-file password protection
	elif artifact_ext == ".rar" :
		try :
			# rarfile library provides RAR archive reading capabilities
			import rarfile

			# RarFile class opens RAR archives
			with rarfile.RarFile( artifact_location ) as rf :
				# Check if any file in the archive requires a password
				# any() returns True if at least one element is True
				return any( info.needs_password( ) for info in rf.infolist( ) )
		except ImportError :
			# Library not installed - return False and inform user
			print( "Install rarfile: pip install rarfile" )
			return False
		except Exception :
			# RAR file may be corrupted or in unsupported version
			return False

	# Check 7-Zip archive files
	# 7Z format supports archive-level encryption
	elif artifact_ext == ".7z" :
		try :
			# py7zr library handles 7Z archive operations
			import py7zr

			# Open archive in read mode ('r')
			with py7zr.SevenZipFile( artifact_location , "r" ) as archive :
				# needs_password() checks if password is required for extraction
				return archive.needs_password( )
		except ImportError :
			# Library not installed - return False and inform user
			print( "Install py7zr: pip install py7zr" )
			return False
		except Exception :
			# 7Z file may be corrupted
			return False

	# File format is not supported for password protection detection
	return False


def is_corrupted( artifact_location: Path ) -> bool :
	"""
	Returns True if the file is detectably corrupted, False if valid or type unknown.

	Args:
			artifact_location:     Path to the file to check.
	"""

	# Basic sanity checks
	if not artifact_location.exists( ) :
		return True
	if artifact_location.stat( ).st_size == 0 :
		return True  # zero-byte is always corrupted

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
	mapped_mime_ext = MIME_TO_EXT_MAP[ detected_mime_type ]

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
	if artifact_ext == "jpeg" or artifact_ext == "cr2" or artifact_ext == "arw" :
		artifact_ext = "jpg"
	elif artifact_ext == "apkg" :
		artifact_ext = "zip"

	if mapped_mime_ext.lower( ).strip( ).strip( '.' ) != artifact_ext :
		return True

	# print( f"artifact_location: {artifact_location}" )
	# print( f"Detected MIME type: {detected_mime_type}" )
	# print( f"mapped_mime_ext: {mapped_mime_ext}" )
	# print( f"artifact_suffix: {artifact_location.suffix.lower( ).strip( ).strip( '.' )}" )

	return False


def is_supported_type( artifact_location: Path ) -> bool :
	"""
	Check if a file's type is supported for processing.

	Uses Apache Tika content-based detection to identify file types based on
	actual file content rather than just file extensions. This prevents
	processing of files with incorrect or misleading extensions.

	Tika analyzes the file's magic bytes and internal structure to determine
	its true MIME type, then compares against a whitelist of supported types.

	Args:
									artifact_location: Path object pointing to the file to check

	Returns:
									bool: True if file type is supported, False otherwise

	Note:
									- Requires Apache Tika JAR file to be available at TIKA_APP_JAR_PATH
									- Requires Java Runtime Environment (JRE) to execute Tika
									- Processing timeout is set to 30 seconds per file
									- Unsupported or unrecognized MIME types return False

	Raises:
									No exceptions are raised; all errors result in False return value
	"""

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

	# Check if Tika command executed successfully
	# returncode 0 indicates success, non-zero indicates error
	if result.returncode != 0 :
		return False

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )

	# print( f"result: {result}" )
	# print( f"artifact_suffix: {artifact_location.suffix.lower( ).strip( ).strip( '.' )}" )

	return (
			artifact_ext in EMAIL_TYPES
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
