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
import logging
import re
import time
from pathlib import Path

from config import (
	CAD_FILES ,
	CODE_EXTENSIONS ,
	DIGITAL_CONTACT_EXTENSIONS ,
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


def _contains_financial_content( text: str , logger: logging.Logger ) -> bool :
	"""
	Analyze text to determine if it contains financial/invoice/purchase content.

	Args:
		text: Text content to analyze
		logger: Logger instance for tracking analysis

	Returns:
		bool: True if financial content detected based on scoring thresholds
	"""
	# Convert to lowercase for case-insensitive matching
	text_lower = text.lower( )

	# Financial keywords grouped by category for granular scoring
	# Invoice keywords are strongest indicators
	invoice_keywords = [ "invoice" , "receipt" , "bill" , "statement" , "quote" , "quotation" ]

	# Payment keywords indicate financial transactions
	payment_keywords = [
		"payment" ,
		"paid" ,
		"due" ,
		"owing" ,
		"balance" ,
		"transaction" ,
		"purchase" ,
		"order" ,
		"sale" ,
		"refund" ,
	]

	# Financial terms commonly appear in monetary documents
	financial_terms = [
		"total" ,
		"subtotal" ,
		"amount" ,
		"price" ,
		"cost" ,
		"fee" ,
		"charge" ,
		"tax" ,
		"vat" ,
		"gst" ,
		"discount" ,
		"credit" ,
		"debit" ,
	]

	# Business terms provide commercial context
	business_terms = [
		"customer" ,
		"vendor" ,
		"supplier" ,
		"billing" ,
		"account number" ,
		"reference number" ,
		"po number" ,
		"order number" ,
	]

	# Count keyword matches using sum with generator expression
	invoice_score = sum( 1 for kw in invoice_keywords if kw in text_lower )
	payment_score = sum( 1 for kw in payment_keywords if kw in text_lower )
	financial_score = sum( 1 for kw in financial_terms if kw in text_lower )
	business_score = sum( 1 for kw in business_terms if kw in text_lower )

	# Calculate total keyword score across all categories
	total_keyword_score = (
			invoice_score + payment_score + financial_score + business_score
	)

	logger.debug(
			f"Keyword scores - Invoice: {invoice_score}, Payment: {payment_score}, "
			f"Financial: {financial_score}, Business: {business_score}" ,
	)

	# Pattern detection for common financial formats
	pattern_score = 0

	# Currency symbols with amounts (e.g., $123.45, €50.00, £99)
	# \s* allows optional whitespace, \d+ matches one or more digits
	# (?:,\d{3})* matches optional thousands separators
	# (?:\.\d{2})? matches optional decimal part
	currency_pattern = r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?"
	if re.search( currency_pattern , text ) :
		pattern_score += 2
		logger.debug( "Currency pattern detected" )

	# Amount patterns (e.g., "Total: 123.45", "Amount: $50")
	# (?:...) creates non-capturing group
	amount_pattern = r"(?:total|amount|price|cost|subtotal|balance)[\s:]+\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?"
	if re.search( amount_pattern , text_lower ) :
		pattern_score += 2
		logger.debug( "Amount pattern detected" )

	# Date patterns common in invoices (MM/DD/YYYY or YYYY-MM-DD)
	date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2}"
	if re.search( date_pattern , text ) :
		pattern_score += 1
		logger.debug( "Date pattern detected" )

	# Invoice number patterns (e.g., "INV-12345", "Invoice #123")
	# \w+ matches alphanumeric characters, \d+ ensures numeric component
	invoice_number_pattern = r"(?:invoice|receipt|bill|ref(?:erence)?)[#\s:-]*\w+\d+"
	if re.search( invoice_number_pattern , text_lower ) :
		pattern_score += 2
		logger.debug( "Invoice number pattern detected" )

	# Calculate total score combining keywords and patterns
	total_score = total_keyword_score + pattern_score

	logger.info(
			f"Financial analysis - Keyword score: {total_keyword_score}, "
			f"Pattern score: {pattern_score}, Total: {total_score}" ,
	)

	# Decision threshold: need strong evidence across multiple dimensions
	# At least 1 invoice keyword + 4 total score indicates financial doc
	if invoice_score >= 1 and total_score >= 4 :
		return True
	# Or 3+ keywords + 2+ patterns (strong combined evidence)
	elif total_keyword_score >= 3 and pattern_score >= 2 :
		return True
	# Or 5+ keywords regardless of patterns (keyword density)
	elif total_keyword_score >= 5 :
		return True

	return False


def is_bookmark_file( artifact_location: Path ) -> bool :
	"""
	Detect if an HTML file is a browser bookmark export with high accuracy.

	Analyzes HTML structure for Netscape bookmark file format markers including
	DOCTYPE declaration, definition list structure, anchor tags, and timestamp
	attributes. All conditions must be met for positive detection.

	Args:
		artifact_location: Path object pointing to the HTML file to analyze

	Returns:
		bool: True if file matches bookmark export format, False otherwise

	"""

	try :
		# Read the HTML file content with UTF-8 encoding
		# Using 'r' mode for text files, errors='ignore' handles encoding issues
		with open( artifact_location , "r" , encoding="utf-8" , errors="ignore" ) as f :
			html_content = f.read( )
	except Exception as e :
		# Return False if file cannot be read (permissions, encoding errors, etc.)
		return False

	# Check for required Netscape bookmark DOCTYPE declaration
	# re.IGNORECASE flag makes the search case-insensitive
	has_netscape_doctype = bool(
			re.search( r"<!DOCTYPE\s+NETSCAPE-Bookmark-file-1>" , html_content , re.IGNORECASE ) ,
	)

	# Check for definition list (DL) structure used in bookmark hierarchy
	# DL tags contain bookmark folders and items
	has_dl_structure = bool( re.search( r"<DL\s*>" , html_content , re.IGNORECASE ) )

	# Check for DT (definition term) entries with anchor tags
	# DT tags mark individual bookmarks, A tags contain the actual links
	has_dt_anchors = bool(
			re.search( r"<DT>\s*<A\s+[^>]*HREF\s*=" , html_content , re.IGNORECASE ) ,
	)

	# Check for ADD_DATE attribute specific to bookmark exports
	# ADD_DATE contains Unix timestamp when bookmark was created
	has_add_date = bool(
			re.search( r'ADD_DATE\s*=\s*["\']?\d+["\']?' , html_content , re.IGNORECASE ) ,
	)

	# All conditions must be met for absolute certainty
	# Boolean AND operation ensures strict matching
	return has_netscape_doctype and has_dl_structure and has_dt_anchors and has_add_date


def is_anki_deck( artifact_location: Path ) -> bool :
	"""
	Detect if a file is an Anki flashcard deck with high accuracy.

	Args:
		artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if file is detected as Anki deck, False otherwise

	"""

	artifact_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	artifact_label = artifact_location.stem.lower( )

	if artifact_ext in [ "html" ] :
		return False

	if "bookmark" in artifact_label :
		return True

	if artifact_ext in [ "apkg" ] :
		return True

	if "anki" in artifact_label :
		return True

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


def is_backup_codes_file( artifact_location: Path ) -> bool :
	"""
	Detect if a file contains 2FA backup/recovery codes.

	Args:
	artifact_location: Path object pointing to the file to analyze

	Returns:
		bool: True if backup codes detected, False otherwise
	"""

	# Quick check: filename contains backup code keywords
	# os.path.basename() extracts filename from full path
	filename = artifact_location.stem.lower( ).strip( )

	# any() returns True if at least one keyword is found in filename
	if any( keyword in filename for keyword in [ "backup" , "recovery" , "2fa" , "mfa" ] ) :
		return True

	return False


def is_financial_document(
		artifact_location: Path , visual_processor: VisualProcessor , logger: logging.Logger ,
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

		if artifact_ext in [ "txt" , "csv" , "json" , "xml" , "log" ] :
			logger.debug( f"Reading plain text file: {artifact_ext}" )
			try :
				# errors='ignore' handles invalid UTF-8 sequences gracefully
				with open( artifact_location , "r" , encoding="utf-8" , errors="ignore" ) as f :
					text = f.read( )
				logger.debug( f"Extracted {len( text )} characters from plain text file" )
			except Exception as e :
				logger.error( f"Failed to read text file: {e}" )
				text = ""

		# Email files - parse email format
		# .eml files use RFC 822 message format
		elif artifact_ext == ".eml" :
			logger.debug( "Parsing email file" )
			text = _extract_from_email( artifact_location , logger )

		# Images and PDFs - use VisualProcessor
		# VisualProcessor performs OCR (Optical Character Recognition)
		elif artifact_ext in [ ".pdf" , ".jpg" , ".jpeg" , ".png" , ".bmp" , ".tiff" , ".webp" ] :
			logger.debug( f"Using VisualProcessor for {artifact_ext} file" )
			try :
				# VisualProcessor.extract_text() performs OCR on images/PDFs
				text = visual_processor.extract_text( artifact_location )
				logger.debug( f"Extracted {len( text )} characters via OCR" )
			except Exception as e :
				logger.error( f"VisualProcessor extraction failed: {e}" )
				return False

		# Validate extracted text has meaningful content
		# Minimum 10 characters required for analysis
		if len( text.strip( ) ) < 10 :
			logger.warning( f"No meaningful text extracted from {artifact_location.name}" )
			return False

		# Analyze for financial content using keyword and pattern matching
		is_financial = _contains_financial_content( text , logger )

		# Calculate processing time
		elapsed_time = time.time( ) - start_time
		logger.info( f"Financial content detected: {is_financial}" )
		logger.info( f"Analysis completed in {elapsed_time:.2f} seconds" )

		return is_financial

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
	artifact_label = artifact_location.stem.lower( )

	if artifact_ext in [ "epub" , "cbr" , "djvu" ] :
		return True

	if any( title_keyword in artifact_label for title_keyword in [ "edition" , "textbook" , "coursebook" ] ) :
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

	return (artifact_ext in [ "html" ]) and (artifact_ext in CODE_EXTENSIONS)


def is_executable( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in [ "exe" ]


def is_3d_file( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in CAD_FILES


def is_digital_contact_file( artifact_location: Path ) -> bool :
	return artifact_location.suffix.lower( ).strip( ).strip( '.' ) in DIGITAL_CONTACT_EXTENSIONS
