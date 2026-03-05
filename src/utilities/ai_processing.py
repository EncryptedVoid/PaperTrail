"""
LLM-Based Semantic Extraction Services

Provides focused functions for document analysis using local OLLAMA models:
- generate_filename: Produces a clean, descriptive filename from content
- extract_document_text: Pulls text from PDFs/documents via Apache Tika
- extract_visual_description: Describes images/documents visually via vision model
- generate_tags: Creates comma-separated categorical tags from content
- detect_*_theme: Content-based thematic classifiers for file sorting

Optimized for 12GB VRAM (RTX 3060) using:
- qwen2.5:7b for text tasks (fast, strong instruction-following)
- minicpm-v:8b for vision tasks (multimodal, VRAM-efficient)

Author: Ashiq Gazi
"""

import base64
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import List , Optional

import fitz  # pymupdf — pip install pymupdf
import ollama
import pypdf

from config import (
	ARTIFACT_SCANNING_PROMPT ,
	FILENAME_PROMPT_TEMPLATE ,
	FILENAME_SYSTEM_PROMPT ,
	JAVA_PATH ,
	ORGANISATIONAL_TAGS_PROMPT_TEMPLATE ,
	PDF_SUBSET_SIZE ,
	SCAN_SYSTEM_PROMPT ,
	TAGS_SYSTEM_PROMPT ,
	TEMP_DIR ,
	TEXT_MODEL ,
	TIKA_APP_JAR_PATH ,
	VISION_MODEL ,
	VISUAL_DESC_PROMPT ,
	VISUAL_DESC_SYSTEM_PROMPT ,
)
from utilities.artifact_data_manipulation import get_metadata

# ──────────────────────────────────────────────────────────────────────────────
# Detection system prompt & prompt templates
# ──────────────────────────────────────────────────────────────────────────────

_DETECTION_SYSTEM_PROMPT = (
	"You are a file classification assistant. You will be given content extracted "
	"from a file and a question about what type of document it is. Analyze the "
	"content carefully and respond with ONLY the word 'True' or 'False'. "
	"Do not include any explanation, punctuation, or additional text."
)

_VIDEO_COURSE_PROMPT = (
	"Below is the metadata of a video file. Determine whether this video is part "
	"of an educational course, tutorial series, lecture recording, or instructional "
	"video (e.g. from platforms like Udemy, Coursera, YouTube educational channels, "
	"university recordings, coding bootcamps, etc.).\n\n"
	"Look for indicators such as: numbered episodes/sections, course titles, "
	"instructor names, educational platform references, lecture/lesson keywords, "
	"chapter structures, or systematic educational naming conventions.\n\n"
	"Metadata:\n{content}\n\n"
	"Is this a video course file? Respond ONLY with True or False."
)

_BOOK_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"an officially published book. This includes novels, short story collections, "
	"memoirs, biographies, self-help books, poetry anthologies, reference books, "
	"cookbooks, or any other commercially published literary work.\n\n"
	"Look for indicators such as: ISBN numbers, publisher information, copyright "
	"pages, table of contents with chapters, author attribution, dedication pages, "
	"forewords, acknowledgments, or literary narrative structure.\n\n"
	"IMPORTANT: This must NOT be a textbook (educational material with exercises, "
	"problem sets, edition numbers tied to academic subjects, or classroom-oriented "
	"content). Textbooks are classified separately.\n\n"
	"Content:\n{content}\n\n"
	"Is this an officially published book (NOT a textbook)? Respond ONLY with True or False."
)

_TEXTBOOK_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"an educational textbook — a formal, information-dense book designed for "
	"structured learning in an academic subject.\n\n"
	"Look for indicators such as: edition numbers (e.g. '4th Edition'), subject "
	"indicators (e.g. 'Introduction to Physics'), chapter summaries, review "
	"questions, exercises and problem sets, learning objectives, academic publisher "
	"names (Pearson, McGraw-Hill, Wiley, O'Reilly, etc.), index sections, "
	"glossaries, bibliographies, or references to curricula and courses.\n\n"
	"Textbooks are typically longer, more detailed, and more information-dense than "
	"general-purpose books and are oriented toward classroom or self-study use.\n\n"
	"Content:\n{content}\n\n"
	"Is this an educational textbook? Respond ONLY with True or False."
)

_PROFESSIONAL_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"directly related to professional employment or career credentials.\n\n"
	"This includes: resumes, CVs, cover letters, job offer letters, employment "
	"contracts, termination letters, performance reviews, professional certificates "
	"(course completion, professional designations, licenses), letters of "
	"recommendation, pay negotiation letters, or non-disclosure/non-compete "
	"agreements tied to employment.\n\n"
	"This does NOT include: books about careers, job-seeking guides, general "
	"business documents, or educational materials about professions. The document "
	"must be an actual professional artifact belonging to a specific person.\n\n"
	"Content:\n{content}\n\n"
	"Is this a professional/employment document? Respond ONLY with True or False."
)

_FINANCIAL_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"related to finances, monetary transactions, or financial products.\n\n"
	"This includes: invoices, receipts, purchase confirmations, order tracking "
	"details, bank statements, tax forms (T4, W-2, 1099, etc.), pay stubs, "
	"credit card statements, credit card offers, loan documents, insurance "
	"policies, investment statements, budget spreadsheets, expense reports, "
	"financial advisories, billing notices, subscription confirmations, refund "
	"notices, or any other document involving money, payments, or financial "
	"accounts.\n\n"
	"Content:\n{content}\n\n"
	"Is this a financial document? Respond ONLY with True or False."
)

_IMMIGRATION_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"related to immigration, travel authorization, or border services.\n\n"
	"This includes: documents from governmental immigration bodies (Canada's IRCC, "
	"CBSA, IRB, IRD; America's USCIS, ICE, CBP; UK's Home Office; or equivalents "
	"in any country), visa applications or approvals, work permits, study permits, "
	"permanent residency cards/documents, citizenship certificates, refugee claims, "
	"travel visas, passport applications, NEXUS/Global Entry documents, port of "
	"entry records, removal orders, immigration hearing notices, flight tickets, "
	"boarding passes, Airbnb/hotel bookings for travel, or any official document "
	"from a governmental body in charge of immigration and border control.\n\n"
	"Content:\n{content}\n\n"
	"Is this an immigration or travel-related document? Respond ONLY with True or False."
)

_LEGAL_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"a legal document or legally binding material.\n\n"
	"This includes: contracts (service agreements, rental/lease agreements, account "
	"terms), bank account contract documents, documents from lawyers or law firms, "
	"wills and testaments, testimonies, affidavits, court filings, terms and "
	"conditions, privacy policies, cease and desist letters, legal notices, "
	"power of attorney documents, notarized documents, settlement agreements, "
	"NDAs, liability waivers, regulatory filings, or any document that is legally "
	"binding, law-related, or produced by/for legal proceedings.\n\n"
	"Content:\n{content}\n\n"
	"Is this a legal document? Respond ONLY with True or False."
)

_ACADEMIC_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"related to academia or school work at any level (elementary, middle school, "
	"high school, or university).\n\n"
	"This includes: research papers, journal articles, theses, dissertations, "
	"grade reports, transcripts, rubrics, assignments, homework, lab reports, "
	"lecture notes, presentation slides, course syllabi, study guides, exam papers "
	"(midterms, finals, quizzes), project proposals, school newsletters, academic "
	"conference papers, peer reviews, or any document produced for or by an "
	"educational institution as part of the learning process.\n\n"
	"This does NOT include textbooks (those are classified separately).\n\n"
	"Content:\n{content}\n\n"
	"Is this an academic/school document? Respond ONLY with True or False."
)

_INSTRUCTION_MANUAL_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"an instruction manual, user guide, or product documentation.\n\n"
	"This includes: product manuals, user guides, setup instructions, assembly "
	"instructions, quick-start guides, installation guides, troubleshooting guides, "
	"owner's manuals (cars, appliances, electronics), safety data sheets, "
	"maintenance guides, configuration documentation, or any document whose primary "
	"purpose is to instruct a user on how to use, build, set up, or maintain a "
	"product or system.\n\n"
	"Content:\n{content}\n\n"
	"Is this an instruction manual or product guide? Respond ONLY with True or False."
)

_DOCUMENT_SCAN_PROMPT = (
	"Look at this image carefully. Determine whether this image is a scan or "
	"photograph of a full page of text — meaning a document that was intended to "
	"be a PDF or text file but ended up as an image instead.\n\n"
	"Indicators of a document scan: large blocks of readable text filling most of "
	"the image, paragraph structure, consistent font usage, margins resembling a "
	"printed page, letterhead, form fields, or any layout typical of a printed or "
	"typed document.\n\n"
	"This is NOT a document scan if: the image contains only a small amount of text "
	"(like a sign, label, meme, or caption), or is primarily a photograph, "
	"illustration, screenshot of a UI, or artistic image with incidental text.\n\n"
	"Is this image a scan/photo of a full-page text document? Respond ONLY with True or False."
)

_DOCUMENT_SCAN_SYSTEM_PROMPT = (
	"You are a document scan detection assistant. Analyze images to determine if "
	"they are scans or photographs of text documents. Respond with ONLY 'True' or "
	"'False'. Do not include any explanation or additional text."
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _llm_generate(
		logger: logging.Logger ,
		prompt: str ,
		system: str ,
		model: str = TEXT_MODEL ,
) -> Optional[ str ] :
	"""Send a text prompt to the LLM and return the response string."""
	logger.debug( f"[LLM] Sending prompt to {model} ({len( prompt )} chars)" )
	start = time.perf_counter( )

	try :
		client = ollama.Client( )
		response = client.generate(
				model=model ,
				system=system ,
				prompt=prompt ,
				stream=False ,
		)
		elapsed = time.perf_counter( ) - start
		result = response[ "response" ].strip( )
		logger.debug( f"[LLM] Response received from {model} in {elapsed:.2f}s ({len( result )} chars)" )
		return result

	except Exception as e :
		elapsed = time.perf_counter( ) - start
		logger.error( f"[LLM] Generation failed on {model} after {elapsed:.2f}s: {type( e ).__name__}: {e}" ,
									exc_info=True )
		return None


def _parse_bool_response( response: Optional[ str ] ) -> bool :
	"""
	Parse an LLM response that should be 'True' or 'False'.

	Handles minor formatting variations (extra whitespace, quotes, punctuation).
	Returns False if the response is None or unparseable.
	"""
	if response is None :
		return False
	cleaned = response.strip( ).strip( "\"'`.,!" ).lower( )
	return cleaned == "true"


def extract_text_for_detection(
		logger: logging.Logger ,
		artifact_location: Path ,
		max_content_chars: int = 9000 ,
) -> Optional[ str ] :
	"""
	Extract text content from a document for thematic detection.

	For PDFs, subsets the document first via compile_doc_subset to keep
	token usage manageable. Then extracts text via extract_document_text.
	Returns truncated content string or None on failure.
	"""
	logger.info( f"[DETECT] Extracting text for detection from '{artifact_location.name}'" )
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	source_path = artifact_location

	# Subset large PDFs before extraction
	if file_ext == "pdf" :
		logger.debug( f"[DETECT] '{artifact_location.name}' is a PDF, attempting page subset" )
		try :
			source_path = compile_doc_subset( artifact_location , logger )
			if source_path != artifact_location :
				logger.debug( f"[DETECT] Subsetted '{artifact_location.name}' -> '{source_path.name}'" )
			else :
				logger.debug( f"[DETECT] '{artifact_location.name}' has {PDF_SUBSET_SIZE} or fewer pages, no subset needed" )
		except Exception as e :
			logger.warning(
					f"[DETECT] PDF subset failed for '{artifact_location.name}', using original: {type( e ).__name__}: {e}" )
			source_path = artifact_location

	content = extract_document_text( logger , source_path )

	# Clean up temporary subset file
	if source_path != artifact_location and source_path.exists( ) :
		try :
			source_path.unlink( )
			logger.debug( f"[DETECT] Cleaned up temporary subset file '{source_path.name}'" )
		except OSError as e :
			logger.warning( f"[DETECT] Failed to clean up temporary subset file '{source_path.name}': {e}" )

	if not content :
		logger.warning( f"[DETECT] No content extracted from '{artifact_location.name}'" )
		return None

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug(
				f"[DETECT] Content truncated from {len( content )} to {max_content_chars} chars for '{artifact_location.name}'" )

	logger.info( f"[DETECT] Extracted {len( truncated )} chars from '{artifact_location.name}'" )
	return truncated


# ──────────────────────────────────────────────────────────────────────────────
# Public API — Generation functions
# ──────────────────────────────────────────────────────────────────────────────

def generate_filename(
		logger: logging.Logger ,
		content: str ,
		max_content_chars: int = 9000 ,
) -> Optional[ str ] :
	"""
	Generate a clean, descriptive filename from document content.

	Works for any file whose content has already been extracted (text or visual description).
	Returns a lowercase underscore-separated name without extension, or None on failure.
	"""
	logger.info( f"[FILENAME] Generating descriptive filename from {len( content )} chars of content" )

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug( f"[FILENAME] Content truncated from {len( content )} to {max_content_chars} chars" )

	prompt = FILENAME_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=FILENAME_SYSTEM_PROMPT )

	if result :
		cleaned = (
			result
			.strip( "\"'`" )
			.replace( " " , "_" )
			.replace( "-" , "_" )
			.lower( )
			.split( "." )[ 0 ]
		)
		logger.info( f"[FILENAME] Generated filename: '{cleaned}'" )
		return cleaned

	logger.warning( "[FILENAME] LLM returned no result, filename generation failed" )
	return None


def _pdf_pages_to_png_bytes(
		logger: logging.Logger ,
		pdf_path: Path ,
		max_pages: int = 3 ,
		dpi: int = 200 ,
) -> List[ bytes ] :
	"""
	Render the first N pages of a PDF as PNG byte arrays.

	Returns a list of PNG byte buffers (one per page), or an empty list on failure.
	Uses pymupdf (fitz) which has no external dependencies like poppler.
	"""
	logger.debug( f"[PDF→PNG] Opening '{pdf_path.name}' for rendering (max_pages={max_pages}, dpi={dpi})" )
	png_buffers: List[ bytes ] = [ ]

	try :
		doc = fitz.open( str( pdf_path ) )
		total = len( doc )
		render_count = min( total , max_pages )
		logger.debug( f"[PDF→PNG] '{pdf_path.name}' has {total} pages, rendering {render_count}" )

		for i in range( render_count ) :
			try :
				page = doc[ i ]
				pix = page.get_pixmap( dpi=dpi )
				png_bytes = pix.tobytes( "png" )
				png_buffers.append( png_bytes )
				logger.debug(
						f"[PDF→PNG] Page {i + 1}/{render_count} rendered: "
						f"{pix.width}x{pix.height}px, {len( png_bytes ) / 1024:.1f} KB" ,
				)
			except Exception as e :
				logger.warning(
						f"[PDF→PNG] Failed to render page {i + 1} of '{pdf_path.name}': "
						f"{type( e ).__name__}: {e}" ,
				)

		doc.close( )
		logger.info(
				f"[PDF→PNG] Rendered {len( png_buffers )}/{render_count} pages from '{pdf_path.name}'" ,
		)

	except Exception as e :
		logger.error(
				f"[PDF→PNG] Failed to open '{pdf_path.name}': {type( e ).__name__}: {e}" ,
				exc_info=True ,
		)

	return png_buffers


# ──────────────────────────────────────────────────────────────────────────────
# Fixed: _vision_generate — now accepts optional raw bytes
# ──────────────────────────────────────────────────────────────────────────────

def _vision_generate(
		logger: logging.Logger ,
		user_prompt: str ,
		system: str ,
		file_path: Optional[ Path ] = None ,
		image_bytes: Optional[ bytes ] = None ,
) -> Optional[ str ] :
	"""
	Send an image + prompt to the vision model and return the response string.

	Accepts EITHER:
		- file_path: reads raw bytes from disk (for actual image files)
		- image_bytes: pre-rendered PNG/JPEG bytes (for PDF pages converted to images)

	At least one must be provided. If both are given, image_bytes takes priority.
	"""
	source_label = file_path.name if file_path else "<raw bytes>"
	logger.debug( f"[VISION] Processing '{source_label}' with {VISION_MODEL}" )
	start = time.perf_counter( )

	# ── Resolve image bytes ───────────────────────────────────────────────
	if image_bytes is not None :
		raw_bytes = image_bytes
		logger.debug( f"[VISION] Using provided image bytes ({len( raw_bytes ) / 1024:.1f} KB)" )
	elif file_path is not None :
		try :
			raw_bytes = file_path.read_bytes( )
			logger.debug( f"[VISION] Read {len( raw_bytes ) / 1024:.1f} KB from '{file_path.name}'" )
		except Exception as e :
			elapsed = time.perf_counter( ) - start
			logger.error(
					f"[VISION] Failed to read '{file_path.name}' after {elapsed:.2f}s: "
					f"{type( e ).__name__}: {e}" ,
					exc_info=True ,
			)
			return None
	else :
		logger.error( "[VISION] Neither file_path nor image_bytes provided — nothing to process" )
		return None

	if len( raw_bytes ) == 0 :
		logger.error( f"[VISION] Image data is empty (0 bytes) for '{source_label}' — aborting" )
		return None

	# ── Encode and send ───────────────────────────────────────────────────
	b64_image = base64.b64encode( raw_bytes ).decode( "utf-8" )
	file_size_mb = len( raw_bytes ) / (1024 * 1024)
	logger.debug( f"[VISION] Base64 encoded '{source_label}' ({file_size_mb:.2f} MB, {len( b64_image )} b64 chars)" )

	try :
		client = ollama.Client( )
		logger.debug( f"[VISION] Sending request to Ollama ({VISION_MODEL})..." )
		response = client.generate(
				model=VISION_MODEL ,
				system=system ,
				prompt=user_prompt ,
				images=[ b64_image ] ,
				stream=False ,
		)
		elapsed = time.perf_counter( ) - start
		result = response[ "response" ].strip( )
		logger.debug(
				f"[VISION] Response received for '{source_label}' in {elapsed:.2f}s "
				f"({len( result )} chars)" ,
		)
		logger.debug( f"[VISION] Response preview: {result[ :200 ]}" )
		return result

	except Exception as e :
		elapsed = time.perf_counter( ) - start
		logger.error(
				f"[VISION] Generation failed for '{source_label}' after {elapsed:.2f}s: "
				f"{type( e ).__name__}: {e}" ,
				exc_info=True ,
		)
		return None


IMAGE_EXTENSIONS = { "png" , "jpg" , "jpeg" , "bmp" , "tiff" , "tif" , "webp" , "heic" , "heif" }


def extract_document_text(
		logger: logging.Logger ,
		file_path: Path ,
		timeout: int = 120 ,
) -> Optional[ str ] :
	"""
	Extract text content from a document file using Apache Tika.

	Falls back to vision-based OCR if Tika fails or returns empty content.
	For PDFs, renders pages to images before sending to the vision model.
	"""
	logger.info( f"[SCAN-TEXT] Extracting text from '{file_path.name}'" )
	logger.debug( f"[SCAN-TEXT] Full path: {file_path}" )
	logger.debug( f"[SCAN-TEXT] File exists: {file_path.exists( )}" )
	if file_path.exists( ) :
		file_size = file_path.stat( ).st_size
		logger.debug( f"[SCAN-TEXT] File size: {file_size} bytes ({file_size / 1024:.1f} KB)" )

	start = time.perf_counter( )
	file_ext = file_path.suffix.lower( ).strip( ).strip( "." )

	# ── Attempt 1: Apache Tika CLI ────────────────────────────────────────
	java_str = str( JAVA_PATH )
	tika_jar_str = str( TIKA_APP_JAR_PATH )
	file_str = str( file_path )

	logger.debug( f"[SCAN-TEXT] JAVA_PATH type={type( JAVA_PATH ).__name__}, value='{JAVA_PATH}'" )
	logger.debug(
			f"[SCAN-TEXT] TIKA_APP_JAR_PATH type={type( TIKA_APP_JAR_PATH ).__name__}, value='{TIKA_APP_JAR_PATH}'" )
	logger.debug( f"[SCAN-TEXT] Java exists: {Path( java_str ).exists( )}" )
	logger.debug( f"[SCAN-TEXT] Tika JAR exists: {Path( tika_jar_str ).exists( )}" )

	cmd = [ java_str , "-jar" , tika_jar_str , "--text" , file_str ]
	logger.debug( f"[SCAN-TEXT] Running Tika: {' '.join( cmd )}" )

	try :
		proc = subprocess.run( cmd , capture_output=True , text=True , timeout=timeout )
		logger.debug( f"[SCAN-TEXT] Tika exit code: {proc.returncode}" )

		if proc.returncode == 0 and proc.stdout.strip( ) :
			content = proc.stdout.strip( )
			elapsed = time.perf_counter( ) - start
			logger.info(
					f"[SCAN-TEXT] Tika extracted {len( content )} chars from "
					f"'{file_path.name}' in {elapsed:.2f}s" ,
			)
			return content

		logger.warning(
				f"[SCAN-TEXT] Tika returned empty or failed for '{file_path.name}' "
				f"(exit code {proc.returncode}, stdout length: {len( proc.stdout )})" ,
		)
		if proc.stderr :
			logger.debug( f"[SCAN-TEXT] Tika stderr: {proc.stderr[ :500 ]}" )

	except subprocess.TimeoutExpired :
		logger.warning( f"[SCAN-TEXT] Tika timed out after {timeout}s for '{file_path.name}'" )
	except FileNotFoundError as e :
		logger.error(
				f"[SCAN-TEXT] Java or Tika JAR not found — check JAVA_PATH ({java_str}) "
				f"and TIKA_APP_JAR_PATH ({tika_jar_str}): {e}" ,
		)
	except Exception as e :
		logger.error(
				f"[SCAN-TEXT] Tika error for '{file_path.name}': {type( e ).__name__}: {e}" ,
				exc_info=True ,
		)

	# ── Attempt 2: Vision-based OCR fallback ──────────────────────────────
	logger.info( f"[SCAN-TEXT] Falling back to vision OCR for '{file_path.name}'" )

	if file_ext == "pdf" :
		# PDFs must be rendered to images first — vision models can't read PDF bytes
		logger.info( f"[SCAN-TEXT] '{file_path.name}' is a PDF — rendering pages to PNG for vision model" )
		png_pages = _pdf_pages_to_png_bytes( logger , file_path , max_pages=3 )

		if not png_pages :
			elapsed = time.perf_counter( ) - start
			logger.error(
					f"[SCAN-TEXT] PDF rendering produced no images for '{file_path.name}' "
					f"— cannot perform vision OCR. Elapsed: {elapsed:.2f}s" ,
			)
			return None

		# OCR each rendered page and concatenate
		all_text_parts: List[ str ] = [ ]
		for idx , png_bytes in enumerate( png_pages ) :
			logger.debug(
					f"[SCAN-TEXT] Running vision OCR on page {idx + 1}/{len( png_pages )} "
					f"({len( png_bytes ) / 1024:.1f} KB)" ,
			)
			page_text = _vision_generate(
					logger ,
					user_prompt=ARTIFACT_SCANNING_PROMPT ,
					system=SCAN_SYSTEM_PROMPT ,
					image_bytes=png_bytes ,
			)
			if page_text :
				all_text_parts.append( page_text )
				logger.debug(
						f"[SCAN-TEXT] Page {idx + 1} OCR returned {len( page_text )} chars" ,
				)
			else :
				logger.warning( f"[SCAN-TEXT] Page {idx + 1} OCR returned nothing" )

		if all_text_parts :
			combined = "\n\n".join( all_text_parts )
			elapsed = time.perf_counter( ) - start
			logger.info(
					f"[SCAN-TEXT] Vision OCR extracted {len( combined )} chars from "
					f"{len( all_text_parts )} page(s) of '{file_path.name}' in {elapsed:.2f}s" ,
			)
			return combined

	elif file_ext in IMAGE_EXTENSIONS :
		# Actual image file — can send directly
		logger.debug( f"[SCAN-TEXT] '{file_path.name}' is an image — sending directly to vision model" )
		result = _vision_generate(
				logger ,
				user_prompt=ARTIFACT_SCANNING_PROMPT ,
				system=SCAN_SYSTEM_PROMPT ,
				file_path=file_path ,
		)
		if result :
			elapsed = time.perf_counter( ) - start
			logger.info(
					f"[SCAN-TEXT] Vision OCR extracted {len( result )} chars from "
					f"'{file_path.name}' in {elapsed:.2f}s" ,
			)
			return result

	else :
		logger.warning(
				f"[SCAN-TEXT] '{file_path.name}' (ext='{file_ext}') is not a PDF or image "
				f"— vision OCR fallback not applicable" ,
		)

	elapsed = time.perf_counter( ) - start
	logger.error( f"[SCAN-TEXT] All extraction methods failed for '{file_path.name}' after {elapsed:.2f}s" )
	return None


def extract_visual_description(
		logger: logging.Logger ,
		file_path: Path ,
) -> Optional[ str ] :
	"""
	Generate a visual description of a document or image using the vision model.

	Returns a concise 2-4 sentence description of visible content, or None on failure.
	"""
	logger.info( f"[VISUAL-DESC] Generating visual description for '{file_path.name}'" )

	result = _vision_generate(
			logger=logger ,
			file_path=file_path ,
			user_prompt=VISUAL_DESC_PROMPT ,
			system=VISUAL_DESC_SYSTEM_PROMPT ,
	)

	if result :
		logger.info( f"[VISUAL-DESC] Description generated for '{file_path.name}' ({len( result )} chars)" )
		logger.debug( f"[VISUAL-DESC] Preview for '{file_path.name}': {result[ :150 ]}..." )
		return result

	logger.warning( f"[VISUAL-DESC] Failed to generate description for '{file_path.name}'" )
	return None


def generate_tags(
		logger: logging.Logger ,
		content: str ,
		max_content_chars: int = 9000 ,
) -> Optional[ List[ str ] ] :
	"""
	Generate comma-separated classification tags from document content.

	Returns a string of 5-10 lowercase tags, or None on failure.
	"""
	logger.info( f"[TAGS] Generating classification tags from {len( content )} chars of content" )

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug( f"[TAGS] Content truncated from {len( content )} to {max_content_chars} chars" )

	prompt = ORGANISATIONAL_TAGS_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=TAGS_SYSTEM_PROMPT )

	if result :
		tags = (
			(tag.strip( ).lower( )
			 for tag in result.split( "," )
			 if tag.strip( )))

		logger.info( f"[TAGS] Generated tags: {tags}" )
		return tags

	logger.warning( "[TAGS] LLM returned no result, tag generation failed" )
	return None


def compile_doc_subset( input_pdf: Path , logger: logging.Logger ) -> Path :
	"""
	Create a subset of a large PDF by sampling pages.

	Keeps the first page, last page, and a random sample of middle pages
	up to PDF_SUBSET_SIZE total. Returns the original path if the PDF is
	already small enough.
	"""
	reader = pypdf.PdfReader( str( input_pdf ) )
	total_pages = len( reader.pages )
	logger.debug( f"[SUBSET] '{input_pdf.name}' has {total_pages} pages (subset threshold: {PDF_SUBSET_SIZE})" )

	if total_pages <= PDF_SUBSET_SIZE :
		logger.debug( f"[SUBSET] '{input_pdf.name}' is within threshold, no subset needed" )
		return input_pdf

	first , last = 0 , total_pages - 1
	middle_indices = list( range( 1 , last ) )
	sampled = random.sample( middle_indices , min( PDF_SUBSET_SIZE - 2 , len( middle_indices ) ) )
	selected = sorted( [ first ] + sampled + [ last ] )
	logger.debug( f"[SUBSET] Selected {len( selected )} pages from '{input_pdf.name}': {selected}" )

	writer = pypdf.PdfWriter( )
	for i in selected :
		writer.add_page( reader.pages[ i ] )

	output_path = Path( TEMP_DIR ) / f"SUBSET-{input_pdf.stem}.pdf"
	with open( output_path , "wb" ) as f :
		writer.write( f )

	logger.info(
			f"[SUBSET] Created subset of '{input_pdf.name}' -> '{output_path.name}' ({len( selected )}/{total_pages} pages)" )
	return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API — Thematic detection functions
# ──────────────────────────────────────────────────────────────────────────────

def detect_video_course_theme( artifact_location: Path , logger: logging.Logger , ) -> bool :
	"""
	Detect whether a video file is part of an educational course or tutorial.

	Uses file metadata (via Tika server) to determine if the video has
	indicators of being course/lecture material. Returns False if metadata
	cannot be retrieved.
	"""
	logger.info( f"[DETECT-VIDEO-COURSE] Analyzing '{artifact_location.name}'" )

	metadata = get_metadata(
			logger=logger ,
			artifact=artifact_location ,
			tika_server_process=None ,
	)

	if not metadata :
		logger.warning( f"[DETECT-VIDEO-COURSE] No metadata available for '{artifact_location.name}', cannot classify" )
		return False

	meta_lines = [ ]
	for key , value in metadata.items( ) :
		if isinstance( value , list ) :
			value = "; ".join( str( v ) for v in value )
		meta_lines.append( f"{key}: {value}" )
	meta_text = "\n".join( meta_lines )
	logger.debug(
			f"[DETECT-VIDEO-COURSE] Extracted {len( meta_text )} chars of metadata from '{artifact_location.name}'" )

	prompt = _VIDEO_COURSE_PROMPT.format( content=meta_text[ :9000 ] )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_course = _parse_bool_response( result )
	logger.info( f"[DETECT-VIDEO-COURSE] '{artifact_location.name}' -> {is_course} (raw response: '{result}')" )
	return is_course


def detect_book_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an officially published book (not a textbook).

	Analyzes extracted text content for book indicators like ISBN, publisher
	info, chapter structure, and literary narrative patterns.
	"""
	logger.info( f"[DETECT-BOOK] Running book theme detection ({len( content )} chars of content)" )

	prompt = _BOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_book = _parse_bool_response( result )
	logger.info( f"[DETECT-BOOK] Result: {is_book} (raw response: '{result}')" )
	return is_book


def detect_textbook_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an educational textbook.

	Looks for textbook-specific indicators: edition numbers, exercises,
	academic publishers, learning objectives, glossaries, and
	subject-oriented structure.
	"""
	logger.info( f"[DETECT-TEXTBOOK] Running textbook theme detection ({len( content )} chars of content)" )

	prompt = _TEXTBOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_textbook = _parse_bool_response( result )
	logger.info( f"[DETECT-TEXTBOOK] Result: {is_textbook} (raw response: '{result}')" )
	return is_textbook


def detect_professional_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a professional/employment artifact.

	Identifies resumes, contracts, certificates, offer letters, and other
	career-related documents that belong to a specific individual.
	"""
	logger.info( f"[DETECT-PROFESSIONAL] Running professional theme detection ({len( content )} chars of content)" )

	prompt = _PROFESSIONAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_professional = _parse_bool_response( result )
	logger.info( f"[DETECT-PROFESSIONAL] Result: {is_professional} (raw response: '{result}')" )
	return is_professional


def detect_financial_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is related to finances or monetary transactions.

	Catches invoices, receipts, tax forms, bank statements, credit card
	offers, tracking info, and any document involving money or financial
	accounts.
	"""
	logger.info( f"[DETECT-FINANCIAL] Running financial theme detection ({len( content )} chars of content)" )

	prompt = _FINANCIAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_financial = _parse_bool_response( result )
	logger.info( f"[DETECT-FINANCIAL] Result: {is_financial} (raw response: '{result}')" )
	return is_financial


def detect_immigration_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is related to immigration or travel authorization.

	Identifies documents from immigration bodies (IRCC, CBSA, ICE, USCIS,
	etc.), visa applications, permits, travel bookings, and border service
	records.
	"""
	logger.info( f"[DETECT-IMMIGRATION] Running immigration theme detection ({len( content )} chars of content)" )

	prompt = _IMMIGRATION_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_immigration = _parse_bool_response( result )
	logger.info( f"[DETECT-IMMIGRATION] Result: {is_immigration} (raw response: '{result}')" )
	return is_immigration


def detect_legal_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is a legal or legally binding document.

	Identifies contracts, terms and conditions, legal notices, wills,
	court filings, lawyer correspondence, and any law-related or legally
	binding material.
	"""
	logger.info( f"[DETECT-LEGAL] Running legal theme detection ({len( content )} chars of content)" )

	prompt = _LEGAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_legal = _parse_bool_response( result )
	logger.info( f"[DETECT-LEGAL] Result: {is_legal} (raw response: '{result}')" )
	return is_legal


def detect_academic_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is related to academia or school work.

	Identifies research papers, grades, assignments, lecture notes, syllabi,
	exams, and any document produced for or by educational institutions at
	any level.
	"""
	logger.info( f"[DETECT-ACADEMIC] Running academic theme detection ({len( content )} chars of content)" )

	prompt = _ACADEMIC_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_academic = _parse_bool_response( result )
	logger.info( f"[DETECT-ACADEMIC] Result: {is_academic} (raw response: '{result}')" )
	return is_academic


def detect_instruction_manual_theme( logger: logging.Logger , content: str ) -> bool :
	"""
	Detect whether a document is an instruction manual or product guide.

	Identifies user manuals, setup guides, assembly instructions, owner's
	manuals, troubleshooting guides, and any document designed to instruct
	on product usage or maintenance.
	"""
	logger.info( f"[DETECT-MANUAL] Running instruction manual theme detection ({len( content )} chars of content)" )

	prompt = _INSTRUCTION_MANUAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_manual = _parse_bool_response( result )
	logger.info( f"[DETECT-MANUAL] Result: {is_manual} (raw response: '{result}')" )
	return is_manual


def detect_document_scan( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether an image file is a scan/photo of a text document.

	Uses the vision model to analyze the image for indicators of a
	document scan: large blocks of text, paragraph structure, page-like
	layout, etc. Only processes image files — returns False for non-images.
	"""
	logger.info( f"[DETECT-SCAN] Analyzing '{artifact_location.name}' for document scan indicators" )

	image_extensions = { "png" , "jpg" , "jpeg" , "bmp" , "tiff" , "tif" , "webp" , "heic" , "heif" }
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	if file_ext not in image_extensions :
		logger.debug( f"[DETECT-SCAN] '{artifact_location.name}' is not an image (ext='{file_ext}'), skipping" )
		return False

	result = _vision_generate(
			logger=logger ,
			file_path=artifact_location ,
			user_prompt=_DOCUMENT_SCAN_PROMPT ,
			system=_DOCUMENT_SCAN_SYSTEM_PROMPT ,
	)

	is_scan = _parse_bool_response( result )
	logger.info( f"[DETECT-SCAN] '{artifact_location.name}' -> {is_scan} (raw response: '{result}')" )
	return is_scan
