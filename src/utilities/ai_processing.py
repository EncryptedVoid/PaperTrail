"""
LLM-Based Semantic Extraction Services

Provides focused functions for document analysis using local OLLAMA models:
- generate_filename: Produces a clean, descriptive filename from content
- extract_document_text: Pulls text from PDFs/documents via Apache Tika
- extract_visual_description: Describes images/documents visually via vision model
- generate_tags: Creates comma-separated categorical tags from content

Optimized for 12GB VRAM (RTX 3060) using:
- qwen2.5:7b for text tasks (fast, strong instruction-following)
- minicpm-v:8b for vision tasks (multimodal, VRAM-efficient)

Author: Ashiq Gazi
"""

import base64
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import ollama

from config import (
	ARTIFACT_SCANNING_PROMPT ,
	FILENAME_PROMPT_TEMPLATE ,
	FILENAME_SYSTEM_PROMPT ,
	JAVA_PATH ,
	ORGANISATIONAL_TAGS_PROMPT_TEMPLATE ,
	SCAN_SYSTEM_PROMPT ,
	TAGS_SYSTEM_PROMPT ,
	TEXT_MODEL ,
	TIKA_APP_JAR_PATH ,
	VISION_MODEL ,
	VISUAL_DESC_PROMPT ,
	VISUAL_DESC_SYSTEM_PROMPT
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
		# Read and encode the image
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


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_filename(
		logger: logging.Logger ,
		content: str ,
		max_content_chars: int = 3000 ,
) -> Optional[ str ] :
	"""
	Generate a clean, descriptive filename from document content.

	Works for any file whose content has already been extracted (text or visual description).
	Returns a lowercase underscore-separated name without extension, or None on failure.
	"""
	logger.info( "[FILENAME] Generating descriptive filename" )

	# Truncate content to keep prompt size manageable
	truncated = content[ :max_content_chars ]
	if len( content ) > max_content_chars :
		logger.debug( f"[FILENAME] Content truncated from {len( content )} to {max_content_chars} chars" )

	prompt = FILENAME_PROMPT_TEMPLATE.format( content=truncated )
	result = _llm_generate( logger , prompt=prompt , system=FILENAME_SYSTEM_PROMPT )

	if result :
		# Clean up: remove quotes, extension, enforce underscores
		cleaned = (
			result
			.strip( "\"'`" )
			.replace( " " , "_" )
			.replace( "-" , "_" )
			.lower( )
			.split( "." )[ 0 ]  # strip any extension the model might add
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
		max_content_chars: int = 3000 ,
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
		# Normalize: strip whitespace around each tag, lowercase, remove empty
		tags = ", ".join(
				(tag.strip( ).lower( )
				 for tag in result.split( "," )
				 if tag.strip( )) ,
		)
		logger.info( f"[TAGS] Generated: {tags}" )
		return tags

	logger.warning( "[TAGS] Failed to generate tags" )
	return None
