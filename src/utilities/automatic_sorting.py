"""
Automatic file type detection and classification utilities.

This module provides functions for detecting and classifying various file types
based on content analysis, metadata inspection, and pattern matching. Supports
detection of bookmarks, Anki decks, backup codes, books, financial documents,
and source code files.

Key Features:
- Content-based file type detection
- Metadata analysis for classification
- Pattern matching for specific file formats
- Support for multiple file formats and encodings
"""

import email
import json
import logging
import time
from pathlib import Path

from config import (
	ANKI_EXTENSIONS ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS ,
	PROFILE_PREFIX ,
	TEXT_TYPES ,
	VIDEO_TYPES ,
)
from utilities.visual_processor import VisualProcessor


def _extract_from_email( artifact_location: Path , logger: logging.Logger ) -> str :
	"""
	Extract text content from .eml email file.

	Parses RFC 822 email format and extracts subject and body content.
	Handles both plain text and multipart messages.

	Args:
									artifact_location: Path object pointing to .eml file
									logger: Logger instance for tracking extraction

	Returns:
									str: Combined email subject and body text
	"""
	try :
		# Open and parse email file using Python's email library
		with open( artifact_location , "r" , encoding="utf-8" , errors="ignore" ) as f :
			# email.message_from_file() parses RFC 822 format
			msg = email.message_from_file( f )

		# Extract subject line from email headers
		# get() returns empty string if header not found
		subject = msg.get( "Subject" , "" )

		# Extract body content - handle multipart messages
		body = ""
		# is_multipart() checks if email has multiple parts (HTML, plain text, attachments)
		if msg.is_multipart( ) :
			# walk() iterates through all message parts recursively
			for part in msg.walk( ) :
				# get_content_type() returns MIME type (e.g., 'text/plain')
				content_type = part.get_content_type( )
				# Only process plain text parts, skip HTML and attachments
				if content_type == "text/plain" :
					try :
						# get_payload(decode=True) decodes base64/quoted-printable
						body += part.get_payload( decode=True ).decode(
								"utf-8" , errors="ignore" ,
						)
					except :
						# Silently skip parts that fail to decode
						pass
		else :
			# Simple message with single part
			try :
				body = msg.get_payload( decode=True ).decode( "utf-8" , errors="ignore" )
			except :
				# Fallback to string representation if decoding fails
				body = str( msg.get_payload( ) )

		# Combine subject and body with separator
		text = f"{subject}\n\n{body}"
		logger.debug( f"Extracted {len( text )} characters from email" )
		return text

	except Exception as e :
		logger.error( f"Failed to parse email: {e}" )
		return ""


def is_bookmark_file( artifact_location: Path ) -> bool :
	"""
	Detect if an HTML file is a browser bookmark export .

	Args:
		artifact_location: Path object pointing to the HTML file to analyze

	Returns:
		bool: True if file matches bookmark export format, False otherwise

	"""

	# try :
	# 	# Read the HTML file content with UTF-8 encoding
	# 	# Using 'r' mode for text files, errors='ignore' handles encoding issues
	# 	with open( artifact_location , "r" , encoding="utf-8" , errors="ignore" ) as f :
	# 		html_content = f.read( )
	# except Exception as e :
	# 	# Return False if file cannot be read (permissions, encoding errors, etc.)
	# 	return False
	#
	# # Check for required Netscape bookmark DOCTYPE declaration
	# # re.IGNORECASE flag makes the search case-insensitive
	# has_netscape_doctype = bool(
	# 		re.search( r"<!DOCTYPE\s+NETSCAPE-Bookmark-file-1>" , html_content , re.IGNORECASE ) ,
	# )
	#
	# # Check for definition list (DL) structure used in bookmark hierarchy
	# # DL tags contain bookmark folders and items
	# has_dl_structure = bool( re.search( r"<DL\s*>" , html_content , re.IGNORECASE ) )
	#
	# # Check for DT (definition term) entries with anchor tags
	# # DT tags mark individual bookmarks, A tags contain the actual links
	# has_dt_anchors = bool(
	# 		re.search( r"<DT>\s*<A\s+[^>]*HREF\s*=" , html_content , re.IGNORECASE ) ,
	# )
	#
	# # Check for ADD_DATE attribute specific to bookmark exports
	# # ADD_DATE contains Unix timestamp when bookmark was created
	# has_add_date = bool(
	# 		re.search( r'ADD_DATE\s*=\s*["\']?\d+["\']?' , html_content , re.IGNORECASE ) ,
	# )
	#
	# # All conditions must be met for absolute certainty
	# # Boolean AND operation ensures strict matching
	# return has_netscape_doctype and has_dl_structure and has_dt_anchors and has_add_date

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	# with open( profile_path , "r" , encoding="utf-8" ) as f :
	# 	profile_data = json.load( f )  # returns a plain Python dict
	#
	# artifact_label = profile_data[ "original_name" ]

	artifact_label = artifact_location.stem.lower( ).strip( )

	if artifact_ext != ".html" :
		return False

	return "bookmark" in artifact_label


def is_anki_deck( artifact_location: Path ) -> bool :
	"""
	Detect if a file is an Anki flashcard deck with high accuracy.

	Args:
		artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if file is detected as Anki deck, False otherwise

	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	# with open( profile_path , "r" , encoding="utf-8" ) as f :
	# 	profile_data = json.load( f )  # returns a plain Python dict
	#
	# artifact_label = profile_data[ "original_name" ]

	artifact_label = artifact_location.stem.lower( ).strip( )

	if artifact_ext in ANKI_EXTENSIONS :
		return True

	if "anki" in artifact_label :
		return True

	if artifact_ext in TEXT_TYPES :
		with open( artifact_location , "r" , encoding="utf-8" , errors="replace" ) as f :
			first_line = f.readline( ).lower( )

		if ((first_line.find( "question" ) != -1)
				and (first_line.find( "answer" ) != -1)
				and (first_line.find( "question" ) < first_line.find( "answer" ))
		) :
			return True

		if ((first_line.find( "front" ) != -1)
				and (first_line.find( "back" ) != -1)
				and (first_line.find( "front" ) < first_line.find( "back" ))
		) :
			return True

	return False


def is_bitwarden_related( artifact_location: Path ) -> bool :
	"""
	Detect if a file contains 2FA backup/recovery codes.

	Args:
	artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if backup codes detected, False otherwise
	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	# with open( profile_path , "r" , encoding="utf-8" ) as f :
	# 	profile_data = json.load( f )  # returns a plain Python dict
	#
	# artifact_label = profile_data[ "original_name" ]

	artifact_label = artifact_location.stem.lower( ).strip( )

	if (artifact_ext not in TEXT_TYPES
			or artifact_ext not in DOCUMENT_TYPES
			or artifact_ext not in EMAIL_TYPES
	) :
		return False

	# any() returns True if at least one keyword is found in filename
	if any(
			(keyword in artifact_label
			 for keyword in [ "backup" , "recovery" , "2fa" , "mfa" , "dashlane" , "bitwarden" ]) ,
	) :
		return True

	return False


def is_financial_document(
		artifact_location: Path ,
		visual_processor: VisualProcessor ,
		logger: logging.Logger ,
) -> bool :
	"""
	Determine if a document contains financial, invoice, or purchase-related data.

	Args:
		artifact_location: Path object pointing to the document to analyze
		visual_processor: VisualProcessor instance for OCR/text extraction
		logger: Logger instance for tracking analysis progress

	Returns:
		bool: True if document contains financial data, False otherwise

	"""

	# Record start time for performance tracking
	start_time = time.time( )
	logger.info( f"Analyzing document for financial content: {artifact_location.name}" )

	try :
		# Extract text based on file type
		# Different file types require different extraction methods
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
		# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

		# with open( profile_path , "r" , encoding="utf-8" ) as f :
		# 	profile_data = json.load( f )  # returns a plain Python dict
		#
		# artifact_label = profile_data[ "original_name" ]

		artifact_label = artifact_location.stem.lower( ).strip( )

		if (artifact_ext not in DOCUMENT_TYPES
				or artifact_ext not in EMAIL_TYPES
				or artifact_ext not in TEXT_TYPES
		) :
			return False

		if any( keyword in artifact_label for keyword in [ "paystub" , " t4 " , "invoice" , "cheque" ] ) :
			return True

		return False

	except Exception as e :
		# Log error with full exception details using exc_info
		logger.error( f"Error analyzing {artifact_location.name}: {e}" , exc_info=True )
		return False


def is_book( artifact_location: Path ) -> bool :
	"""
	Detect whether a document is a book type.

	Args:
		artifact_location: Path object pointing to document file

	Returns:
		bool: True if document is likely a book, False otherwise

	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	# with open( profile_path , "r" , encoding="utf-8" ) as f :
	# 	profile_data = json.load( f )  # returns a plain Python dict
	#
	# artifact_label = profile_data[ "original_name" ]

	artifact_label = artifact_location.stem.lower( ).strip(" " ).strip(".").strip("_")

	if artifact_ext in [ "epub" , "cbr" , "djvu" ] :
		return True

	if "solution" in artifact_label and "manual" in artifact_label :
		return True

	if any( title_keyword in artifact_label for title_keyword in [ "edition" , "book" , "libgen"] ) :
		return True

	return False


def is_code( artifact_location: Path ) -> bool :
	"""
	Check if file is source code based on extension.

	Args:
		artifact_location: Path object to check

	Returns:
		bool: True if file is code, False otherwise
	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	# artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	# profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	# with open( profile_path , "r" , encoding="utf-8" ) as f :
	# 	profile_data = json.load( f )  # returns a plain Python dict
	#
	# artifact_label = profile_data[ "original_name" ]

	artifact_label = artifact_location.stem.lower( ).strip( ).strip(".").strip("_")

	return "README" in artifact_label or artifact_ext in CODE_EXTENSIONS


def is_executable( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in EXECUTABLE_EXTENSIONS


def is_3d_file( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in CAD_FILES


def is_digital_contact_file( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in DIGITAL_CONTACT_EXTENSIONS


def is_video_course( artifact_location: Path ) -> bool :
	if artifact_location.suffix.lower( ).strip( ).strip( "." ) not in VIDEO_TYPES :
		return False

	artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"
	
	if not Path( profile_path ).exists( ) :
		return False

	with open( profile_path , "r" , encoding="utf-8" ) as f :
		profile_data = json.load( f )  # returns a plain Python dict

	print( f"profile_data: {profile_data}" )

	if "youtube.com" in profile_data[ "metadata" ][ "format" ][ "PURL" ].lower( ) :
		return True

	return False
