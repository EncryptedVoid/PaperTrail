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
from typing import Optional

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
	TEMP_DIR , TEXT_MODEL ,
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
		logger.debug( f"[LLM] Response received in {elapsed:.2f}s ({len( result )} chars)" )
		return result

	except Exception as e :
		elapsed = time.perf_counter( ) - start
		logger.error( f"[LLM] Generation failed after {elapsed:.2f}s — {type( e ).__name__}: {e}" )
		return None


def _vision_generate(
		logger: logging.Logger ,
		file_path: Path ,
		user_prompt: str ,
		system: str ,
) -> Optional[ str ] :
	"""Send an image + prompt to the vision model and return the response string."""
	logger.debug( f"[VISION] Processing {file_path.name} with {VISION_MODEL}" )
	start = time.perf_counter( )

	try :
		raw_bytes = file_path.read_bytes( )
		b64_image = base64.b64encode( raw_bytes ).decode( "utf-8" )
		file_size_mb = len( raw_bytes ) / (1024 * 1024)
		logger.debug( f"[VISION] Image loaded: {file_size_mb:.2f} MB" )

		client = ollama.Client( )
		response = client.generate(
				model=VISION_MODEL ,
				system=system ,
				prompt=user_prompt ,
				images=[ b64_image ] ,
				stream=False ,
		)
		elapsed = time.perf_counter( ) - start
		result = response[ "response" ].strip( )
		logger.debug( f"[VISION] Response received in {elapsed:.2f}s ({len( result )} chars)" )
		return result

	except Exception as e :
		elapsed = time.perf_counter( ) - start
		logger.error( f"[VISION] Generation failed after {elapsed:.2f}s — {type( e ).__name__}: {e}" )
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


def _extract_text_for_detection(
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
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	source_path = artifact_location

	# Subset large PDFs before extraction
	if file_ext == "pdf" :
		try :
			source_path = compile_doc_subset( artifact_location )
			if source_path != artifact_location :
				logger.debug(
						f"[DETECT] Subsetted {artifact_location.name} → {source_path.name}" ,
				)
		except Exception as e :
			logger.warning(
					f"[DETECT] PDF subset failed for {artifact_location.name}, "
					f"using original: {type( e ).__name__}: {e}" ,
			)
			source_path = artifact_location

	content = extract_document_text( logger , source_path )

	# Clean up temporary subset file
	if source_path != artifact_location and source_path.exists( ) :
		try :
			source_path.unlink( )
		except OSError :
			pass

	if not content :
		return None

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug(
				f"[DETECT] Content truncated from {len( content )} to {max_content_chars} chars" ,
		)
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
	logger.info( "[FILENAME] Generating descriptive filename" )

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
		logger.info( f"[FILENAME] Generated: {cleaned}" )
		return cleaned

	logger.warning( "[FILENAME] Failed to generate filename" )
	return None


def extract_document_text(
		logger: logging.Logger ,
		file_path: Path ,
		timeout: int = 120 ,
) -> Optional[ str ] :
	"""
	Extract text content from a document file using Apache Tika.

	Falls back to vision-based OCR if Tika fails or returns empty content.
	Returns extracted text string or None on complete failure.
	"""
	logger.info( f"[SCAN-TEXT] Extracting text from: {file_path.name}" )
	start = time.perf_counter( )

	# ── Attempt 1: Apache Tika ────────────────────────────────────────────
	try :
		cmd = [ JAVA_PATH , "-jar" , str( TIKA_APP_JAR_PATH ) , "--text" , str( file_path ) ]
		logger.debug( f"[SCAN-TEXT] Running Tika: {' '.join( cmd )}" )

		proc = subprocess.run( cmd , capture_output=True , text=True , timeout=timeout )

		if proc.returncode == 0 and proc.stdout.strip( ) :
			content = proc.stdout.strip( )
			elapsed = time.perf_counter( ) - start
			logger.info( f"[SCAN-TEXT] Tika extracted {len( content )} chars in {elapsed:.2f}s" )
			return content

		logger.warning( f"[SCAN-TEXT] Tika returned empty or failed (code {proc.returncode})" )
		if proc.stderr :
			logger.debug( f"[SCAN-TEXT] Tika stderr: {proc.stderr[ :500 ]}" )

	except subprocess.TimeoutExpired :
		logger.warning( f"[SCAN-TEXT] Tika timed out after {timeout}s" )
	except FileNotFoundError :
		logger.error( "[SCAN-TEXT] Java or Tika JAR not found — check paths" )
	except Exception as e :
		logger.error( f"[SCAN-TEXT] Tika error — {type( e ).__name__}: {e}" )

	# ── Attempt 2: Vision-based OCR fallback ──────────────────────────────
	logger.info( f"[SCAN-TEXT] Falling back to vision OCR for: {file_path.name}" )
	result = _vision_generate( logger , file_path , user_prompt=ARTIFACT_SCANNING_PROMPT , system=SCAN_SYSTEM_PROMPT )

	if result :
		elapsed = time.perf_counter( ) - start
		logger.info( f"[SCAN-TEXT] Vision OCR extracted {len( result )} chars in {elapsed:.2f}s" )
		return result

	elapsed = time.perf_counter( ) - start
	logger.error( f"[SCAN-TEXT] All extraction methods failed for {file_path.name} after {elapsed:.2f}s" )
	return None


def extract_visual_description(
		logger: logging.Logger ,
		file_path: Path ,
) -> Optional[ str ] :
	"""
	Generate a visual description of a document or image using the vision model.

	Returns a concise 2-4 sentence description of visible content, or None on failure.
	"""
	logger.info( f"[VISUAL-DESC] Describing: {file_path.name}" )

	result = _vision_generate(
			logger=logger ,
			file_path=file_path ,
			user_prompt=VISUAL_DESC_PROMPT ,
			system=VISUAL_DESC_SYSTEM_PROMPT ,
	)

	if result :
		logger.info( f"[VISUAL-DESC] Description generated ({len( result )} chars)" )
		logger.debug( f"[VISUAL-DESC] Preview: {result[ :150 ]}..." )
		return result

	logger.warning( f"[VISUAL-DESC] Failed to describe {file_path.name}" )
	return None


def generate_tags(
		logger: logging.Logger ,
		content: str ,
		max_content_chars: int = 9000 ,
) -> Optional[ str ] :
	"""
	Generate comma-separated classification tags from document content.

	Returns a string of 5-10 lowercase tags, or None on failure.
	"""
	logger.info( "[TAGS] Generating classification tags" )

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug( f"[TAGS] Content truncated from {len( content )} to {max_content_chars} chars" )

	prompt = ORGANISATIONAL_TAGS_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=TAGS_SYSTEM_PROMPT )

	if result :
		tags = ", ".join(
				(tag.strip( ).lower( )
				 for tag in result.split( "," )
				 if tag.strip( )) ,
		)
		logger.info( f"[TAGS] Generated: {tags}" )
		return tags

	logger.warning( "[TAGS] Failed to generate tags" )
	return None


def compile_doc_subset( input_pdf: Path ) -> Path :
	reader = pypdf.PdfReader( str( input_pdf ) )
	total_pages = len( reader.pages )

	if total_pages <= PDF_SUBSET_SIZE :
		return input_pdf

	first , last = 0 , total_pages - 1
	middle_indices = list( range( 1 , last ) )
	sampled = random.sample( middle_indices , min( PDF_SUBSET_SIZE - 2 , len( middle_indices ) ) )
	selected = sorted( [ first ] + sampled + [ last ] )

	writer = pypdf.PdfWriter( )
	for i in selected :
		writer.add_page( reader.pages[ i ] )

	output_path = Path( TEMP_DIR ) / f"SUBSET-{input_pdf.stem}.pdf"
	with open( output_path , "wb" ) as f :
		writer.write( f )

	return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API — Thematic detection functions
# ──────────────────────────────────────────────────────────────────────────────

def detect_video_course_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a video file is part of an educational course or tutorial.

	Uses file metadata (via Tika server) to determine if the video has
	indicators of being course/lecture material. Returns False if metadata
	cannot be retrieved.
	"""

	logger.info( f"[DETECT-VIDEO-COURSE] Analyzing: {artifact_location.name}" )

	# We need a running Tika server — attempt without a process handle
	# by passing None; get_metadata will log the error and return None
	# if the server isn't running.
	metadata = get_metadata(
			logger=logger ,
			artifact=artifact_location ,
			tika_server_process=None ,
	)

	if not metadata :
		logger.warning(
				f"[DETECT-VIDEO-COURSE] No metadata available for {artifact_location.name} "
				f"— cannot determine video course status" ,
		)
		return False

	# Flatten the metadata dict into a readable text block for the LLM
	meta_lines = [ ]
	for key , value in metadata.items( ) :
		if isinstance( value , list ) :
			value = "; ".join( str( v ) for v in value )
		meta_lines.append( f"{key}: {value}" )
	meta_text = "\n".join( meta_lines )

	prompt = _VIDEO_COURSE_PROMPT.format( content=meta_text[ :9000 ] )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_course = _parse_bool_response( result )
	logger.info( f"[DETECT-VIDEO-COURSE] {artifact_location.name} → {is_course}" )
	return is_course


def detect_book_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is an officially published book (not a textbook).

	Subsets large PDFs before extraction to stay within token limits.
	Analyzes extracted text content for book indicators like ISBN, publisher
	info, chapter structure, and literary narrative patterns.
	"""
	logger.info( f"[DETECT-BOOK] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-BOOK] No content extracted from {artifact_location.name}" )
		return False

	prompt = _BOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_book = _parse_bool_response( result )
	logger.info( f"[DETECT-BOOK] {artifact_location.name} → {is_book}" )
	return is_book


def detect_textbook_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is an educational textbook.

	Subsets large PDFs before extraction. Looks for textbook-specific
	indicators: edition numbers, exercises, academic publishers, learning
	objectives, glossaries, and subject-oriented structure.
	"""
	logger.info( f"[DETECT-TEXTBOOK] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-TEXTBOOK] No content extracted from {artifact_location.name}" )
		return False

	prompt = _TEXTBOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_textbook = _parse_bool_response( result )
	logger.info( f"[DETECT-TEXTBOOK] {artifact_location.name} → {is_textbook}" )
	return is_textbook


def detect_professional_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is a professional/employment artifact.

	Subsets large PDFs before extraction. Identifies resumes, contracts,
	certificates, offer letters, and other career-related documents that
	belong to a specific individual.
	"""
	logger.info( f"[DETECT-PROFESSIONAL] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-PROFESSIONAL] No content extracted from {artifact_location.name}" )
		return False

	prompt = _PROFESSIONAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_professional = _parse_bool_response( result )
	logger.info( f"[DETECT-PROFESSIONAL] {artifact_location.name} → {is_professional}" )
	return is_professional


def detect_financial_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is related to finances or monetary transactions.

	Subsets large PDFs before extraction. Catches invoices, receipts, tax
	forms, bank statements, credit card offers, tracking info, and any
	document involving money or financial accounts.
	"""
	logger.info( f"[DETECT-FINANCIAL] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-FINANCIAL] No content extracted from {artifact_location.name}" )
		return False

	prompt = _FINANCIAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_financial = _parse_bool_response( result )
	logger.info( f"[DETECT-FINANCIAL] {artifact_location.name} → {is_financial}" )
	return is_financial


def detect_immigration_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is related to immigration or travel authorization.

	Subsets large PDFs before extraction. Identifies documents from
	immigration bodies (IRCC, CBSA, ICE, USCIS, etc.), visa applications,
	permits, travel bookings, and border service records.
	"""
	logger.info( f"[DETECT-IMMIGRATION] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-IMMIGRATION] No content extracted from {artifact_location.name}" )
		return False

	prompt = _IMMIGRATION_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_immigration = _parse_bool_response( result )
	logger.info( f"[DETECT-IMMIGRATION] {artifact_location.name} → {is_immigration}" )
	return is_immigration


def detect_legal_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is a legal or legally binding document.

	Subsets large PDFs before extraction. Identifies contracts, terms and
	conditions, legal notices, wills, court filings, lawyer correspondence,
	and any law-related or legally binding material.
	"""
	logger.info( f"[DETECT-LEGAL] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-LEGAL] No content extracted from {artifact_location.name}" )
		return False

	prompt = _LEGAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_legal = _parse_bool_response( result )
	logger.info( f"[DETECT-LEGAL] {artifact_location.name} → {is_legal}" )
	return is_legal


def detect_academic_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is related to academia or school work.

	Subsets large PDFs before extraction. Identifies research papers, grades,
	assignments, lecture notes, syllabi, exams, and any document produced
	for or by educational institutions at any level.
	"""
	logger.info( f"[DETECT-ACADEMIC] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-ACADEMIC] No content extracted from {artifact_location.name}" )
		return False

	prompt = _ACADEMIC_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_academic = _parse_bool_response( result )
	logger.info( f"[DETECT-ACADEMIC] {artifact_location.name} → {is_academic}" )
	return is_academic


def detect_instruction_manual_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a document is an instruction manual or product guide.

	Subsets large PDFs before extraction. Identifies user manuals, setup
	guides, assembly instructions, owner's manuals, troubleshooting guides,
	and any document designed to instruct on product usage or maintenance.
	"""
	logger.info( f"[DETECT-MANUAL] Analyzing: {artifact_location.name}" )

	content = _extract_text_for_detection( logger , artifact_location )
	if not content :
		logger.warning( f"[DETECT-MANUAL] No content extracted from {artifact_location.name}" )
		return False

	prompt = _INSTRUCTION_MANUAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_manual = _parse_bool_response( result )
	logger.info( f"[DETECT-MANUAL] {artifact_location.name} → {is_manual}" )
	return is_manual


def detect_document_scan( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether an image file is a scan/photo of a text document.

	Uses the vision model to analyze the image for indicators of a
	document scan: large blocks of text, paragraph structure, page-like
	layout, etc. Only processes image files — returns False for non-images.
	"""
	logger.info( f"[DETECT-SCAN] Analyzing: {artifact_location.name}" )

	# Only process image files
	image_extensions = { "png" , "jpg" , "jpeg" , "bmp" , "tiff" , "tif" , "webp" , "heic" , "heif" }
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	if file_ext not in image_extensions :
		logger.debug( f"[DETECT-SCAN] {artifact_location.name} is not an image — skipping" )
		return False

	result = _vision_generate(
			logger=logger ,
			file_path=artifact_location ,
			user_prompt=_DOCUMENT_SCAN_PROMPT ,
			system=_DOCUMENT_SCAN_SYSTEM_PROMPT ,
	)

	is_scan = _parse_bool_response( result )
	logger.info( f"[DETECT-SCAN] {artifact_location.name} → {is_scan}" )
	return is_scan
