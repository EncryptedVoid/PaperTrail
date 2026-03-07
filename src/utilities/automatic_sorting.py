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
from pathlib import Path

from config import (
	ANKI_EXTENSIONS ,
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	AUDIO_TYPES , CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS ,
	DOCUMENT_TYPES ,
	EMAIL_TYPES ,
	EXECUTABLE_EXTENSIONS ,
	PROFILE_PREFIX ,
	TEXT_TYPES ,
	VIDEO_TYPES ,
)
from utilities.ai_processing import (detect_academic_theme , detect_book_theme , detect_document_scan ,
																		 detect_financial_theme ,
																		 detect_immigration_theme ,
																		 detect_instruction_manual_theme ,
																		 detect_legal_theme , detect_professional_theme , detect_textbook_theme ,
																		 detect_video_course_theme)


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
	logger.debug( f"[EMAIL_EXTRACT] Attempting to extract text from '{artifact_location.name}'" )
	try :
		with open( artifact_location , "r" , encoding="utf-8" , errors="ignore" ) as f :
			msg = email.message_from_file( f )

		subject = msg.get( "Subject" , "" )

		body = ""
		if msg.is_multipart( ) :
			logger.debug( f"[EMAIL_EXTRACT] '{artifact_location.name}' is multipart, walking message parts" )
			for part in msg.walk( ) :
				content_type = part.get_content_type( )
				if content_type == "text/plain" :
					try :
						body += part.get_payload( decode=True ).decode( "utf-8" , errors="ignore" )
					except :
						logger.debug(
								f"[EMAIL_EXTRACT] Failed to decode a text/plain part in '{artifact_location.name}', skipping" )
		else :
			try :
				body = msg.get_payload( decode=True ).decode( "utf-8" , errors="ignore" )
			except :
				body = str( msg.get_payload( ) )
				logger.debug(
						f"[EMAIL_EXTRACT] Payload decode failed for '{artifact_location.name}', fell back to string representation" )

		text = f"{subject}\n\n{body}"
		logger.debug( f"[EMAIL_EXTRACT] Extracted {len( text )} characters from '{artifact_location.name}'" )
		return text

	except Exception as e :
		logger.error( f"[EMAIL_EXTRACT] Failed to parse '{artifact_location.name}': {e}" , exc_info=True )
		return ""


def is_bookmark_file( artifact_location: Path ) -> bool :
	"""
	Detect if an HTML file is a browser bookmark export.

	Args:
		artifact_location: Path object pointing to the HTML file to analyze

	Returns:
		bool: True if file matches bookmark export format, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( )

	if artifact_ext != ".html" :
		return False

	if "bookmark" in artifact_label :
		return True

	return False


def is_anki_deck( artifact_location: Path ) -> bool :
	"""
	Detect if a file is an Anki flashcard deck.

	Args:
		artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if file is detected as Anki deck, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
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


def is_personal_security_item( artifact_location: Path ) -> bool :
	"""
	Detect if a file contains 2FA backup/recovery codes.

	Args:
		artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if backup codes detected, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( ).strip( "." ).strip( "_" )

	if (artifact_ext not in TEXT_TYPES
			and artifact_ext not in DOCUMENT_TYPES
			and artifact_ext not in EMAIL_TYPES
	) :
		return False

	if any(
			(keyword in artifact_label
			 for keyword in [ "backup" , "recovery" , "2fa" , "mfa" , "dashlane" , "bitwarden" ]) ,
	) :
		return True

	return False


def is_code( artifact_location: Path ) -> bool :
	"""
	Check if file is source code based on extension or filename.

	Args:
		artifact_location: Path object to check

	Returns:
		bool: True if file is code, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( " " ).strip( "." ).strip( "_" )

	return "readme" in artifact_label or artifact_ext in CODE_EXTENSIONS


def is_executable( artifact_location: Path ) -> bool :
	"""
	Check if file is an executable based on extension.

	Args:
		artifact_location: Path object to check

	Returns:
		bool: True if file is an executable, False otherwise
	"""
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in EXECUTABLE_EXTENSIONS


def is_3d_file( artifact_location: Path ) -> bool :
	"""
	Check if file is a 3D model / CAD file based on extension.

	Args:
		artifact_location: Path object to check

	Returns:
		bool: True if file is a 3D/CAD file, False otherwise
	"""
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in CAD_FILES


def is_digital_contact_file( artifact_location: Path ) -> bool :
	"""
	Check if file is a digital contact (vCard, etc.) based on extension.

	Args:
		artifact_location: Path object to check

	Returns:
		bool: True if file is a digital contact file, False otherwise
	"""
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in DIGITAL_CONTACT_EXTENSIONS


def is_video_course( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect if a video file is a course/educational video.

	Checks the artifact profile metadata for YouTube origin and falls back
	to AI-based theme detection.

	Args:
		artifact_location: Path object pointing to the video file
		logger: Logger instance for tracking detection

	Returns:
		bool: True if video is detected as a course, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( " " ).strip( "." ).strip( "_" )

	if artifact_ext not in VIDEO_TYPES :
		logger.debug( f"[VIDEO_COURSE] '{artifact_location.name}' is not a video type (ext='{artifact_ext}'), skipping" )
		return False

	artifact_uuid = artifact_location.stem[ len( ARTIFACT_PREFIX ) + 1 : ]
	profile_path = f"{ARTIFACT_PROFILES_DIR}\\{PROFILE_PREFIX}-{artifact_uuid}.json"

	if not Path( profile_path ).exists( ) :
		logger.debug( f"[VIDEO_COURSE] No profile found for '{artifact_location.name}' at {profile_path}, skipping" )
		return False

	with open( profile_path , "r" , encoding="utf-8" ) as f :
		profile_data = json.load( f )

	logger.debug( f"[VIDEO_COURSE] Loaded profile for '{artifact_location.name}': {profile_data}" )

	if "youtube.com" in profile_data[ "metadata" ][ "format" ][ "PURL" ].lower( ) :
		logger.info( f"[VIDEO_COURSE] '{artifact_location.name}' matched YouTube origin via profile metadata" )
		return True

	logger.debug( f"[VIDEO_COURSE] Running AI theme detection on '{artifact_location.name}'" )
	if detect_video_course_theme( artifact_location=artifact_location , logger=logger ) :
		logger.info( f"[VIDEO_COURSE] '{artifact_location.name}' matched via AI theme detection" )
		return True

	logger.debug( f"[VIDEO_COURSE] '{artifact_location.name}' did not match any video course criteria" )
	return False


def is_book( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a book.

	Args:
		artifact_location: Path object pointing to document file
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is likely a book, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( " " ).strip( "." ).strip( "_" )

	if artifact_ext in [ "epub" , "cbr" , "djvu" ] :
		logger.info( f"[BOOK] '{artifact_location.name}' matched book extension '{artifact_ext}'" )
		return True

	if "solution" in artifact_label and "manual" in artifact_label :
		logger.info( f"[BOOK] '{artifact_location.name}' matched 'solution manual' keyword pattern in filename" )
		return True

	if any( keyword in artifact_label for keyword in [ "edition" , "book" , "libgen" ] ) :
		logger.info( f"[BOOK] '{artifact_location.name}' matched book keyword in filename" )
		return True

	logger.debug( f"[BOOK] Running AI theme detection on '{artifact_location.name}'" )
	if detect_book_theme( logger=logger , content=content ) :
		logger.info( f"[BOOK] '{artifact_location.name}' matched via AI theme detection" )
		return True

	logger.debug( f"[BOOK] '{artifact_location.name}' did not match any book criteria" )
	return False


def is_textbook( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a textbook.

	Args:
		artifact_location: Path object pointing to document file
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is likely a textbook, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( " " ).strip( "." ).strip( "_" )

	if artifact_ext in [ "epub" , "cbr" , "djvu" ] :
		logger.info( f"[TEXTBOOK] '{artifact_location.name}' matched textbook extension '{artifact_ext}'" )
		return True

	if "solution" in artifact_label and "manual" in artifact_label :
		logger.info( f"[TEXTBOOK] '{artifact_location.name}' matched 'solution manual' keyword pattern in filename" )
		return True

	if any( keyword in artifact_label for keyword in [ "edition" , "book" , "libgen" ] ) :
		logger.info( f"[TEXTBOOK] '{artifact_location.name}' matched textbook keyword in filename" )
		return True

	logger.debug( f"[TEXTBOOK] Running AI theme detection on '{artifact_location.name}'" )
	if detect_textbook_theme( logger=logger , content=content ) :
		logger.info( f"[TEXTBOOK] '{artifact_location.name}' matched via AI theme detection" )
		return True

	logger.debug( f"[TEXTBOOK] '{artifact_location.name}' did not match any textbook criteria" )
	return False


def is_professional( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a professional/resume document.

	Args:
		artifact_location: Path object pointing to document file
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is likely a professional document, False otherwise
	"""
	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( ).strip( ).strip( "." ).strip( "_" )

	if "resume" in artifact_label or "certificate" in artifact_label :
		logger.info( f"[PROFESSIONAL] '{artifact_location.name}' matched professional keyword in filename" )
		return True

	logger.debug( f"[PROFESSIONAL] Running AI theme detection on '{artifact_location.name}'" )
	if detect_professional_theme( logger=logger , content=content ) :
		logger.info( f"[PROFESSIONAL] '{artifact_location.name}' matched via AI theme detection" )
		return True

	logger.debug( f"[PROFESSIONAL] '{artifact_location.name}' did not match any professional criteria" )
	return False


def is_financial_document( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document contains financial, invoice, or purchase-related data.

	Args:
		artifact_location: Path object pointing to the document to analyze
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document contains financial data, False otherwise
	"""
	try :
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		artifact_label = artifact_location.stem.lower( ).strip( )

		if (artifact_ext not in DOCUMENT_TYPES
				or artifact_ext not in EMAIL_TYPES
				or artifact_ext not in TEXT_TYPES
		) :
			logger.debug( f"[FINANCIAL] '{artifact_location.name}' is not a supported type (ext='{artifact_ext}'), skipping" )
			return False

		if any( keyword in artifact_label for keyword in [ "paystub" , " t4 " , "invoice" , "cheque" ] ) :
			logger.info( f"[FINANCIAL] '{artifact_location.name}' matched financial keyword in filename" )
			return True

		logger.debug( f"[FINANCIAL] Running AI theme detection on '{artifact_location.name}'" )
		if detect_financial_theme( logger=logger , content=content ) :
			logger.info( f"[FINANCIAL] '{artifact_location.name}' matched via AI theme detection" )
			return True

		logger.debug( f"[FINANCIAL] '{artifact_location.name}' did not match any financial criteria" )
		return False

	except Exception as e :
		logger.error( f"[FINANCIAL] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False


def is_immigration( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an immigration-related document.

	Args:
		artifact_location: Path object pointing to the document to analyze
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is immigration-related, False otherwise
	"""
	try :
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		artifact_label = artifact_location.stem.lower( ).strip( )

		if artifact_ext in AUDIO_TYPES or artifact_ext in VIDEO_TYPES :
			logger.debug( f"[IMMIGRATION] '{artifact_location.name}' is audio/video (ext='{artifact_ext}'), skipping" )
			return False

		if any( keyword in artifact_label for keyword in [ "immigration" , "refugee" , "passport" , "work permit" ] ) :
			logger.info( f"[IMMIGRATION] '{artifact_location.name}' matched immigration keyword in filename" )
			return True

		logger.debug( f"[IMMIGRATION] Running AI theme detection on '{artifact_location.name}'" )
		if detect_immigration_theme( logger=logger , content=content ) :
			logger.info( f"[IMMIGRATION] '{artifact_location.name}' matched via AI theme detection" )
			return True

		logger.debug( f"[IMMIGRATION] '{artifact_location.name}' did not match any immigration criteria" )
		return False

	except Exception as e :
		logger.error( f"[IMMIGRATION] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False


def is_legal( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a legal document.

	Args:
		artifact_location: Path object pointing to the document to analyze
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is a legal document, False otherwise
	"""
	try :
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		artifact_label = artifact_location.stem.lower( ).strip( )

		if artifact_ext in AUDIO_TYPES or artifact_ext in VIDEO_TYPES :
			logger.debug( f"[LEGAL] '{artifact_location.name}' is audio/video (ext='{artifact_ext}'), skipping" )
			return False

		logger.debug( f"[LEGAL] Running AI theme detection on '{artifact_location.name}'" )
		if detect_legal_theme( logger=logger , content=content ) :
			logger.info( f"[LEGAL] '{artifact_location.name}' matched via AI theme detection" )
			return True

		logger.debug( f"[LEGAL] '{artifact_location.name}' did not match any legal criteria" )
		return False

	except Exception as e :
		logger.error( f"[LEGAL] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False


def is_academic( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an academic/educational document.

	Args:
		artifact_location: Path object pointing to the document to analyze
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is academic, False otherwise
	"""
	try :
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		artifact_label = artifact_location.stem.lower( ).strip( )

		if artifact_ext in AUDIO_TYPES or artifact_ext in VIDEO_TYPES :
			logger.debug( f"[ACADEMIC] '{artifact_location.name}' is audio/video (ext='{artifact_ext}'), skipping" )
			return False

		if any( keyword in artifact_label for keyword in [ "syllabus" , "midterm" , "lecture" , "final exam" ] ) :
			logger.info( f"[ACADEMIC] '{artifact_location.name}' matched academic keyword in filename" )
			return True

		logger.debug( f"[ACADEMIC] Running AI theme detection on '{artifact_location.name}'" )
		if detect_academic_theme( logger=logger , content=content ) :
			logger.info( f"[ACADEMIC] '{artifact_location.name}' matched via AI theme detection" )
			return True

		logger.debug( f"[ACADEMIC] '{artifact_location.name}' did not match any academic criteria" )
		return False

	except Exception as e :
		logger.error( f"[ACADEMIC] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False


def is_instruction_manual( artifact_location: Path , logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an instruction manual.

	Args:
		artifact_location: Path object pointing to the document to analyze
		logger: Logger instance for tracking detection
		content: Pre-extracted text content of the document

	Returns:
		bool: True if document is an instruction manual, False otherwise
	"""
	try :
		artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( '.' )
		artifact_label = artifact_location.stem.lower( ).strip( )

		if artifact_ext in AUDIO_TYPES or artifact_ext in VIDEO_TYPES or artifact_ext not in DOCUMENT_TYPES :
			logger.debug(
					f"[INSTRUCTION_MANUAL] '{artifact_location.name}' is not a valid document type (ext='{artifact_ext}'), skipping" )
			return False

		if "solutions" not in artifact_label and "manual" in artifact_label :
			logger.info(
					f"[INSTRUCTION_MANUAL] '{artifact_location.name}' matched 'manual' keyword in filename (excluding 'solutions manual')" )
			return True

		logger.debug( f"[INSTRUCTION_MANUAL] Running AI theme detection on '{artifact_location.name}'" )
		if detect_instruction_manual_theme( logger=logger , content=content ) :
			logger.info( f"[INSTRUCTION_MANUAL] '{artifact_location.name}' matched via AI theme detection" )
			return True

		logger.debug( f"[INSTRUCTION_MANUAL] '{artifact_location.name}' did not match any instruction manual criteria" )
		return False

	except Exception as e :
		logger.error( f"[INSTRUCTION_MANUAL] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False


def is_unscanned_document( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether an image is a photograph of a physical document that needs scanning.

	Args:
		artifact_location: Path object pointing to the image to analyze
		logger: Logger instance for tracking detection

	Returns:
		bool: True if image appears to be an unscanned document, False otherwise
	"""
	try :
		logger.debug( f"[UNSCANNED_DOCUMENT] Running document scan detection on '{artifact_location.name}'" )
		if detect_document_scan( artifact_location=artifact_location , logger=logger ) :
			logger.info( f"[UNSCANNED_DOCUMENT] '{artifact_location.name}' detected as unscanned physical document" )
			return True

		logger.debug( f"[UNSCANNED_DOCUMENT] '{artifact_location.name}' does not appear to be an unscanned document" )
		return False

	except Exception as e :
		logger.error( f"[UNSCANNED_DOCUMENT] Error analyzing '{artifact_location.name}': {e}" , exc_info=True )
		return False
