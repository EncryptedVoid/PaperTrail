"""
LLM-Based Semantic Extraction Services

Provides focused functions for document analysis using local OLLAMA models:
- generate_filename: Produces a clean, descriptive filename from content
- extract_text_for_detection: Unified text/visual extraction for PDFs and images
- extract_visual_description: Describes images/documents visually via vision model
- generate_tags: Creates comma-separated categorical tags from content
- detect_*_theme: Content-based thematic classifiers for file sorting

All PDFs are converted to PNGs and processed through the vision model,
eliminating Tika dependency for PDFs and handling image-based PDFs natively.

Non-PDF, non-image document types (e.g. .docx, .txt) still use Apache Tika.

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

from config import (ARTIFACT_SCANNING_PROMPT , IMAGE_TYPES ,
										JAVA_PATH , OLLAMA_PORT , ORGANISATIONAL_TAGS_PROMPT_TEMPLATE , PDF_SUBSET_PAGE_LIMIT ,
										SCAN_SYSTEM_PROMPT , TAGS_SYSTEM_PROMPT , TEMP_DIR , TEXT_MODEL , TIKA_APP_JAR_PATH , VISION_MODEL ,
										VISUAL_DESC_PROMPT , VISUAL_DESC_SYSTEM_PROMPT)
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
	"This does NOT include:\n"
	"- Books about careers, job-seeking guides, general business documents, or "
	"educational materials about professions.\n"
	"- ANY financial transaction records, even if employment-related. Pay stubs, "
	"pay statements, salary deposit confirmations, T4 slips, W-2 forms, tax "
	"withholding summaries, direct deposit records, commission statements, or "
	"any document whose primary purpose is recording a monetary amount paid — "
	"these are FINANCIAL documents, not professional ones.\n\n"
	"The document must be an actual professional artifact (about a role, credential, "
	"or career action), NOT a record of money changing hands.\n\n"
	"Content:\n{content}\n\n"
	"Is this a professional/employment document? Respond ONLY with True or False."
)

_FINANCIAL_PROMPT = (
	"Below is text extracted from a document. Determine whether this document is "
	"related to finances, monetary transactions, or financial products.\n\n"
	"This includes: invoices, receipts, purchase confirmations, order tracking "
	"details, bank statements, tax forms (T4, T4A, T5, T1, W-2, 1099, etc.), "
	"pay stubs, pay statements, salary deposit confirmations, commission "
	"statements, credit card statements, credit card offers, loan documents, "
	"insurance policies, investment statements, budget spreadsheets, expense "
	"reports, financial advisories, billing notices, subscription confirmations, "
	"refund notices, or any other document involving money, payments, or "
	"financial accounts.\n\n"
	"IMPORTANT — Government tax and revenue body documents ARE financial:\n"
	"Documents from tax and revenue authorities — such as Canada Revenue Agency "
	"(CRA), IRS, HMRC, state/provincial tax agencies, or any equivalent body — "
	"are FINANCIAL documents. This includes: notices of assessment, tax returns, "
	"benefit statements (GST/HST credit, Canada Child Benefit, stimulus payments), "
	"tax account summaries, instalment reminders, reassessment notices, refund "
	"confirmations, contribution room statements (RRSP, TFSA), and any "
	"correspondence from these bodies about taxes, credits, deductions, or "
	"amounts owed/refunded. These are financial even if they reference "
	"legislation or use formal/legal-sounding language.\n\n"
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
	"IMPORTANT — The following are NOT legal documents:\n"
	"Documents from government tax/revenue authorities (CRA, IRS, HMRC, or "
	"equivalents) whose primary purpose is reporting tax assessments, credits, "
	"deductions, refunds, benefit amounts, contribution room, or account balances "
	"are FINANCIAL, not legal — even if they cite legislation, use formal language, "
	"or carry government letterhead. Only classify a document from these bodies as "
	"legal if it is genuinely about a legal dispute, audit appeal hearing, court "
	"proceeding, or penalty enforcement action — not routine tax correspondence.\n\n"
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

_FILENAME_SYSTEM_PROMPT = (
	"You are a precise document-naming assistant. You generate filenames that are "
	"specific, descriptive, and immediately tell someone what the document contains "
	"without opening it. Your filenames should read like a short, informative label."
)

_FILENAME_PROMPT_TEMPLATE = (
	"Below is content extracted from a document. Generate a single descriptive "
	"filename (NO file extension) that captures the specific identity of this "
	"document.\n\n"
	"Rules:\n"
	"- Use lowercase words separated by underscores.\n"
	"- Be SPECIFIC: include key identifying details like dates, names, subjects, "
	"companies, account types, or document types visible in the content.\n"
	"- Aim for 4-8 words. Shorter is worse — vague names like 'tax_document' or "
	"'bank_statement' are too generic.\n"
	"- Good examples:\n"
	"    cra_notice_of_assessment_2023_tax_year\n"
	"    john_smith_resume_software_engineer\n"
	"    td_visa_statement_march_2024\n"
	"    lease_agreement_45_queen_st_2024\n"
	"    udemy_python_bootcamp_section_12_decorators\n"
	"- Bad examples (too vague):\n"
	"    tax_form\n"
	"    resume\n"
	"    bank_document\n"
	"    lease\n"
	"    video_lecture\n\n"
	"Content:\n{content}\n\n"
	"Filename:"
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
		client = ollama.Client( host=f"http://localhost:{OLLAMA_PORT}" )
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
		logger.error(
				f"[LLM] Generation failed on {model} after {elapsed:.2f}s: {type( e ).__name__}: {e}" ,
				exc_info=True ,
		)
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


# ──────────────────────────────────────────────────────────────────────────────
# PDF → PNG pipeline
# ──────────────────────────────────────────────────────────────────────────────

def compile_doc_subset( input_pdf: Path , logger: logging.Logger ) -> Path :
	"""
	Create a subset of a large PDF by sampling pages.

	If the PDF has more than PDF_SUBSET_PAGE_LIMIT (6) pages, keeps the
	first page, last page, and 4 randomly sampled middle pages.
	Returns the original path if the PDF is already small enough.
	"""
	reader = pypdf.PdfReader( str( input_pdf ) )
	total_pages = len( reader.pages )
	logger.debug( f"[SUBSET] '{input_pdf.name}' has {total_pages} pages (limit: {PDF_SUBSET_PAGE_LIMIT})" )

	if total_pages <= PDF_SUBSET_PAGE_LIMIT :
		logger.debug( f"[SUBSET] '{input_pdf.name}' is within limit, no subset needed" )
		return input_pdf

	first , last = 0 , total_pages - 1
	middle_indices = list( range( 1 , last ) )
	sample_count = min( PDF_SUBSET_PAGE_LIMIT - 2 , len( middle_indices ) )  # 4 middle pages
	sampled = random.sample( middle_indices , sample_count )
	selected = sorted( [ first ] + sampled + [ last ] )
	logger.debug( f"[SUBSET] Selected {len( selected )} pages from '{input_pdf.name}': {selected}" )

	writer = pypdf.PdfWriter( )
	for i in selected :
		writer.add_page( reader.pages[ i ] )

	output_path = Path( TEMP_DIR ) / f"SUBSET-{input_pdf.stem}.pdf"
	with open( output_path , "wb" ) as f :
		writer.write( f )

	logger.info(
			f"[SUBSET] Created subset of '{input_pdf.name}' -> '{output_path.name}' "
			f"({len( selected )}/{total_pages} pages)" ,
	)
	return output_path


def pdf_to_png_files(
		pdf_path: Path ,
		logger: logging.Logger ,
		dpi: int = 200 ,
) -> List[ Path ] :
	"""
	Convert every page of a PDF into individual PNG files saved in TEMP_DIR.

	The PDF should already be subsetted via compile_doc_subset before calling
	this function. Returns a list of PNG file paths, or an empty list on failure.
	"""
	logger.info( f"[PDF→PNG] Converting '{pdf_path.name}' pages to PNG files in {TEMP_DIR}" )
	png_paths: List[ Path ] = [ ]

	try :
		doc = fitz.open( str( pdf_path ) )
		total = len( doc )
		logger.debug( f"[PDF→PNG] '{pdf_path.name}' has {total} pages to render" )

		for i in range( total ) :
			try :
				page = doc[ i ]
				pix = page.get_pixmap( dpi=dpi )
				png_filename = f"{pdf_path.stem}_page_{i + 1:03d}.png"
				png_path = Path( TEMP_DIR ) / png_filename

				pix.save( str( png_path ) )
				png_paths.append( png_path )

				logger.debug(
						f"[PDF→PNG] Page {i + 1}/{total} saved: '{png_filename}' "
						f"({pix.width}x{pix.height}px, {png_path.stat( ).st_size / 1024:.1f} KB)" ,
				)
			except Exception as e :
				logger.warning(
						f"[PDF→PNG] Failed to render page {i + 1} of '{pdf_path.name}': "
						f"{type( e ).__name__}: {e}" ,
				)

		doc.close( )
		logger.info( f"[PDF→PNG] Saved {len( png_paths )}/{total} PNGs from '{pdf_path.name}'" )

	except Exception as e :
		logger.error(
				f"[PDF→PNG] Failed to open '{pdf_path.name}': {type( e ).__name__}: {e}" ,
				exc_info=True ,
		)

	return png_paths


def _cleanup_temp_files( paths: List[ Path ] , logger: logging.Logger ) -> None :
	"""Remove a list of temporary files, logging any failures."""
	for p in paths :
		try :
			if p.exists( ) :
				p.unlink( )
				logger.debug( f"[CLEANUP] Deleted temp file '{p.name}'" )
		except OSError as e :
			logger.warning( f"[CLEANUP] Failed to delete '{p.name}': {e}" )


# ──────────────────────────────────────────────────────────────────────────────
# Unified text / visual extraction for images and PDFs
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pages(
		png_paths: List[ Path ] ,
		logger: logging.Logger ,
) -> Optional[ str ] :
	"""
	Run text-extraction OCR on each PNG via the vision model.

	Returns the concatenated text if any page produced content, or None
	if every page came back empty.
	"""
	text_parts: List[ str ] = [ ]

	for idx , png_path in enumerate( png_paths ) :
		logger.debug(
				f"[OCR] Running text OCR on page {idx + 1}/{len( png_paths )}: '{png_path.name}'" ,
		)
		page_text = _vision_generate(
				logger ,
				user_prompt=ARTIFACT_SCANNING_PROMPT ,
				system=SCAN_SYSTEM_PROMPT ,
				file_path=png_path ,
		)
		if page_text :
			text_parts.append( page_text )
			logger.debug( f"[OCR] Page {idx + 1} returned {len( page_text )} chars" )
		else :
			logger.debug( f"[OCR] Page {idx + 1} returned no text" )

	if text_parts :
		combined = "\n\n".join( text_parts )
		logger.info(
				f"[OCR] Text extracted from {len( text_parts )}/{len( png_paths )} pages ({len( combined )} chars total)" )
		return combined

	logger.info( f"[OCR] No text extracted from any of the {len( png_paths )} pages" )
	return None


def _extract_visual_from_pages(
		png_paths: List[ Path ] ,
		logger: logging.Logger ,
) -> Optional[ str ] :
	"""
	Run visual-description generation on each PNG via the vision model.

	Returns the concatenated descriptions if any page produced content,
	or None if every page came back empty.
	"""
	desc_parts: List[ str ] = [ ]

	for idx , png_path in enumerate( png_paths ) :
		logger.debug(
				f"[VISUAL] Running visual description on page {idx + 1}/{len( png_paths )}: '{png_path.name}'" ,
		)
		desc = _vision_generate(
				logger=logger ,
				file_path=png_path ,
				user_prompt=VISUAL_DESC_PROMPT ,
				system=VISUAL_DESC_SYSTEM_PROMPT ,
		)
		if desc :
			desc_parts.append( desc )
			logger.debug( f"[VISUAL] Page {idx + 1} description: {len( desc )} chars" )
		else :
			logger.debug( f"[VISUAL] Page {idx + 1} returned no description" )

	if desc_parts :
		combined = "\n\n".join( desc_parts )
		logger.info(
				f"[VISUAL] Descriptions generated for {len( desc_parts )}/{len( png_paths )} pages "
				f"({len( combined )} chars total)" ,
		)
		return combined

	logger.info( f"[VISUAL] No descriptions generated from any of the {len( png_paths )} pages" )
	return None


def extract_text_for_detection(
		logger: logging.Logger ,
		artifact_location: Path ,
		max_content_chars: int = 18000 ,
) -> Optional[ str ] :
	"""
	Unified content extraction for PDFs, images, and other document types.

	Pipeline:
		1. PDFs  → subset (6-page limit) → convert to PNGs in TEMP_DIR
		   Images → treat the image file itself as a single-page PNG list
		   Other  → extract via Apache Tika (existing path)

		2. Attempt text OCR on each PNG via the vision model.
		   If text is found → return it (stop here).

		3. If no text was extracted → generate a visual description instead.

	Returns truncated content string or None on failure.
	Temporary PNG and subset files are cleaned up before returning.
	"""
	logger.info( f"[DETECT] Extracting content for detection from '{artifact_location.name}'" )
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	temp_files_to_cleanup: List[ Path ] = [ ]
	subset_path: Optional[ Path ] = None

	try :
		# ── Route 1: PDF → subset → PNGs ──────────────────────────────────
		if file_ext == "pdf" :
			logger.info( f"[DETECT] '{artifact_location.name}' is a PDF, entering vision pipeline" )

			# Subset large PDFs
			try :
				subset_path = compile_doc_subset( artifact_location , logger )
			except Exception as e :
				logger.warning(
						f"[DETECT] PDF subset failed for '{artifact_location.name}', "
						f"using original: {type( e ).__name__}: {e}" ,
				)
				subset_path = artifact_location

			if subset_path != artifact_location :
				temp_files_to_cleanup.append( subset_path )

			# Convert to PNGs
			png_paths = pdf_to_png_files( subset_path , logger )
			temp_files_to_cleanup.extend( png_paths )

			if not png_paths :
				logger.error(
						f"[DETECT] PDF→PNG conversion produced no images for "
						f"'{artifact_location.name}' — cannot process" ,
				)
				return None

		# ── Route 2: Image file → treat as single-page list ───────────────
		elif file_ext in IMAGE_TYPES :
			logger.info( f"[DETECT] '{artifact_location.name}' is an image, entering vision pipeline" )
			png_paths = [ artifact_location ]
		# Don't add to cleanup — this is the original file

		# ── Route 3: Other document types → Tika extraction ───────────────
		else :
			logger.info(
					f"[DETECT] '{artifact_location.name}' is a non-PDF document (ext='{file_ext}'), "
					f"using Tika extraction" ,
			)
			content = _extract_via_tika( logger , artifact_location )
			if content :
				truncated = content[ :max_content_chars ]
				if len( content ) > max_content_chars :
					logger.debug(
							f"[DETECT] Content truncated from {len( content )} to "
							f"{max_content_chars} chars for '{artifact_location.name}'" ,
					)
				logger.info( f"[DETECT] Extracted {len( truncated )} chars from '{artifact_location.name}'" )
				return truncated
			logger.warning( f"[DETECT] Tika returned no content for '{artifact_location.name}'" )
			return None

		# ── Step 2: Try text OCR on the PNGs ──────────────────────────────
		text_content = _extract_text_from_pages( png_paths , logger )

		if text_content :
			logger.info(
					f"[DETECT] Text OCR succeeded for '{artifact_location.name}', "
					f"skipping visual description" ,
			)
			truncated = text_content[ :max_content_chars ]
			if len( text_content ) > max_content_chars :
				logger.debug(
						f"[DETECT] Content truncated from {len( text_content )} to "
						f"{max_content_chars} chars" ,
				)
			return truncated

		# ── Step 3: No text found — fall back to visual description ───────
		logger.info(
				f"[DETECT] No text extracted from '{artifact_location.name}', "
				f"falling back to visual description" ,
		)
		visual_content = _extract_visual_from_pages( png_paths , logger )

		if visual_content :
			truncated = visual_content[ :max_content_chars ]
			if len( visual_content ) > max_content_chars :
				logger.debug(
						f"[DETECT] Visual content truncated from {len( visual_content )} to "
						f"{max_content_chars} chars" ,
				)
			return truncated

		logger.error(
				f"[DETECT] Both text OCR and visual description failed for "
				f"'{artifact_location.name}'" ,
		)
		return None

	finally :
		# ── Always clean up temp files ────────────────────────────────────
		if temp_files_to_cleanup :
			logger.debug(
					f"[DETECT] Cleaning up {len( temp_files_to_cleanup )} temp files "
					f"for '{artifact_location.name}'" ,
			)
			_cleanup_temp_files( temp_files_to_cleanup , logger )


# ──────────────────────────────────────────────────────────────────────────────
# Tika extraction (kept for non-PDF, non-image documents like .docx, .txt)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_via_tika(
		logger: logging.Logger ,
		file_path: Path ,
		timeout: int = 120 ,
) -> Optional[ str ] :
	"""
	Extract text content from a non-PDF document file using Apache Tika.

	Used only for document types that are not PDFs or images (e.g. .docx,
	.odt, .rtf, .txt). PDFs and images go through the vision pipeline.
	"""
	logger.info( f"[TIKA] Extracting text from '{file_path.name}'" )

	java_str = str( JAVA_PATH )
	tika_jar_str = str( TIKA_APP_JAR_PATH )
	file_str = str( file_path )

	cmd = [ java_str , "-jar" , tika_jar_str , "--text" , file_str ]
	logger.debug( f"[TIKA] Running: {' '.join( cmd )}" )

	try :
		proc = subprocess.run(
				cmd ,
				capture_output=True ,
				text=True ,
				timeout=timeout ,
				encoding="utf-8" ,
				errors="replace" ,
		)

		if proc.returncode == 0 and proc.stdout.strip( ) :
			content = proc.stdout.strip( )
			logger.info( f"[TIKA] Extracted {len( content )} chars from '{file_path.name}'" )
			return content

		logger.warning(
				f"[TIKA] Empty or failed for '{file_path.name}' "
				f"(exit code {proc.returncode})" ,
		)
		if proc.stderr :
			logger.debug( f"[TIKA] stderr: {proc.stderr[ :500 ]}" )

	except subprocess.TimeoutExpired :
		logger.warning( f"[TIKA] Timed out after {timeout}s for '{file_path.name}'" )
	except FileNotFoundError as e :
		logger.error( f"[TIKA] Java or Tika JAR not found: {e}" )
	except Exception as e :
		logger.error(
				f"[TIKA] Error for '{file_path.name}': {type( e ).__name__}: {e}" ,
				exc_info=True ,
		)

	return None


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

	prompt = _FILENAME_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=_FILENAME_SYSTEM_PROMPT )

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

	Returns a list of 5-10 lowercase tags, or None on failure.
	"""
	logger.info( f"[TAGS] Generating classification tags from {len( content )} chars of content" )

	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug( f"[TAGS] Content truncated from {len( content )} to {max_content_chars} chars" )

	prompt = ORGANISATIONAL_TAGS_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=TAGS_SYSTEM_PROMPT )

	if result :
		tags = [
			tag.strip( ).lower( )
			for tag in result.split( "," )
			if tag.strip( )
		]
		logger.info( f"[TAGS] Generated tags: {tags}" )
		return tags

	logger.warning( "[TAGS] LLM returned no result, tag generation failed" )
	return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API — Thematic detection functions
# ──────────────────────────────────────────────────────────────────────────────

def detect_video_course_theme( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether a video file is part of an educational course or tutorial.

	Uses file metadata (via Tika server) to determine if the video has
	indicators of being course/lecture material.
	"""
	logger.info( f"[DETECT-VIDEO-COURSE] Analyzing '{artifact_location.name}'" )

	metadata = get_metadata(
			logger=logger ,
			artifact=artifact_location ,
	)

	if not metadata :
		logger.warning( f"[DETECT-VIDEO-COURSE] No metadata for '{artifact_location.name}'" )
		return False

	meta_lines = [ ]
	for key , value in metadata.items( ) :
		if isinstance( value , list ) :
			value = "; ".join( str( v ) for v in value )
		meta_lines.append( f"{key}: {value}" )
	meta_text = "\n".join( meta_lines )

	prompt = _VIDEO_COURSE_PROMPT.format( content=meta_text[ :9000 ] )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )

	is_course = _parse_bool_response( result )
	logger.info( f"[DETECT-VIDEO-COURSE] '{artifact_location.name}' -> {is_course}" )
	return is_course


def detect_book_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is an officially published book (not a textbook)."""
	logger.info( f"[DETECT-BOOK] Running book theme detection ({len( content )} chars)" )
	prompt = _BOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_book = _parse_bool_response( result )
	logger.info( f"[DETECT-BOOK] Result: {is_book} (raw: '{result}')" )
	return is_book


def detect_textbook_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is an educational textbook."""
	logger.info( f"[DETECT-TEXTBOOK] Running textbook theme detection ({len( content )} chars)" )
	prompt = _TEXTBOOK_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_textbook = _parse_bool_response( result )
	logger.info( f"[DETECT-TEXTBOOK] Result: {is_textbook} (raw: '{result}')" )
	return is_textbook


def detect_professional_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is a professional/employment artifact."""
	logger.info( f"[DETECT-PROFESSIONAL] Running professional theme detection ({len( content )} chars)" )
	prompt = _PROFESSIONAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_professional = _parse_bool_response( result )
	logger.info( f"[DETECT-PROFESSIONAL] Result: {is_professional} (raw: '{result}')" )
	return is_professional


def detect_financial_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is related to finances or monetary transactions."""
	logger.info( f"[DETECT-FINANCIAL] Running financial theme detection ({len( content )} chars)" )
	prompt = _FINANCIAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_financial = _parse_bool_response( result )
	logger.info( f"[DETECT-FINANCIAL] Result: {is_financial} (raw: '{result}')" )
	return is_financial


def detect_immigration_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is related to immigration or travel authorization."""
	logger.info( f"[DETECT-IMMIGRATION] Running immigration theme detection ({len( content )} chars)" )
	prompt = _IMMIGRATION_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_immigration = _parse_bool_response( result )
	logger.info( f"[DETECT-IMMIGRATION] Result: {is_immigration} (raw: '{result}')" )
	return is_immigration


def detect_legal_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is a legal or legally binding document."""
	logger.info( f"[DETECT-LEGAL] Running legal theme detection ({len( content )} chars)" )
	prompt = _LEGAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_legal = _parse_bool_response( result )
	logger.info( f"[DETECT-LEGAL] Result: {is_legal} (raw: '{result}')" )
	return is_legal


def detect_academic_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is related to academia or school work."""
	logger.info( f"[DETECT-ACADEMIC] Running academic theme detection ({len( content )} chars)" )
	prompt = _ACADEMIC_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_academic = _parse_bool_response( result )
	logger.info( f"[DETECT-ACADEMIC] Result: {is_academic} (raw: '{result}')" )
	return is_academic


def detect_instruction_manual_theme( logger: logging.Logger , content: str ) -> bool :
	"""Detect whether a document is an instruction manual or product guide."""
	logger.info( f"[DETECT-MANUAL] Running instruction manual theme detection ({len( content )} chars)" )
	prompt = _INSTRUCTION_MANUAL_PROMPT.format( content=content )
	result = _llm_generate( logger , prompt=prompt , system=_DETECTION_SYSTEM_PROMPT )
	is_manual = _parse_bool_response( result )
	logger.info( f"[DETECT-MANUAL] Result: {is_manual} (raw: '{result}')" )
	return is_manual


def detect_document_scan( artifact_location: Path , logger: logging.Logger ) -> bool :
	"""
	Detect whether an image file is a scan/photo of a text document.

	Uses the vision model to analyze the image for document scan indicators.
	Only processes image files — returns False for non-images.
	"""
	logger.info( f"[DETECT-SCAN] Analyzing '{artifact_location.name}' for document scan indicators" )

	image_extensions = { "png" , "jpg" , "jpeg" , "bmp" , "tiff" , "tif" , "webp" , "heic" , "heif" }
	file_ext = artifact_location.suffix.lower( ).strip( ).strip( "." )
	if file_ext not in image_extensions :
		logger.debug( f"[DETECT-SCAN] '{artifact_location.name}' is not an image, skipping" )
		return False

	result = _vision_generate(
			logger=logger ,
			file_path=artifact_location ,
			user_prompt=_DOCUMENT_SCAN_PROMPT ,
			system=_DOCUMENT_SCAN_SYSTEM_PROMPT ,
	)

	is_scan = _parse_bool_response( result )
	logger.info( f"[DETECT-SCAN] '{artifact_location.name}' -> {is_scan} (raw: '{result}')" )
	return is_scan
