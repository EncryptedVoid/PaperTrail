"""
Apache Tika Metadata Extractor Module

This module provides functionality to extract metadata from files using Apache Tika.
It processes a file, retrieves all available metadata fields, and saves them to a JSON file.

Usage:
    from pathlib import Path
    import logging
    from tika_metadata_extractor import extract_metadata

    logger = logging.getLogger(__name__)
    file_path = Path("document.pdf")
    output_json = Path("metadata.json")

    extract_metadata(file_path, output_json, logger)
"""
import datetime
import logging
from typing import Any , Dict , Optional , TypedDict

import ollama

from config import FIELD_PROMPTS , SYSTEM_PROMPT


class LanguageExtractionReport( TypedDict ) :
	"""
	Structured type definition for field extraction results.

	This TypedDict ensures consistent return format from the extract_fields method
	and provides clear documentation of expected fields in the response.
	"""

	success: bool
	processing_timestamp: str
	model_used: str
	processing_time_seconds: float
	extracted_fields: Optional[ Dict[ str , str ] ]  # Only present on success
	error: Optional[ str ]  # Only present on failure
	error_type: Optional[ str ]  # Only present on failure


def extract_fields( logger: logging.Logger , content: str ) -> LanguageExtractionReport :
	"""
	Extract all document fields using LLM with comprehensive error handling.

	This method processes document content through the configured LLM model to extract
	structured field data. It iterates through all configured field prompts, handles
	individual field extraction failures gracefully, and provides detailed performance
	metrics and error reporting.

	Returns:
									ExtractionReport: Structured report containing either:
																	- On success: extracted_fields dict with field_name -> extracted_value mappings
																	- On failure: error details and diagnostic information
																	Both cases include processing metadata and performance metrics

	Note:
									Individual field extraction failures are logged but don't stop the overall
									process. Failed fields are marked as "UNKNOWN" in the results.
	"""

	# Initialize ALL instance variables FIRST to ensure consistent object state
	logger: logging.Logger = logger

	# Start timing for performance monitoring
	start_time = datetime.now( )

	logger.debug( "Extracting fields from summaries..." )

	try :
		# Attempt to establish connection to OLLAMA service
		client = ollama.Client( )

		# Log successful initialization with model details
		logger.info( "Language processor initialization completed successfully" )
		logger.info( f"Active model: {PREFERRED_LANGUAGE_MODEL}" )

		# Main extraction loop - process each configured field independently
		extracted_fields: Dict[ str , str ] = { }

		# Iterate through all field prompts defined in configuration
		for field_name , field_instruction in FIELD_PROMPTS.items( ) :
			try :
				logger.debug( f"Extracting field: {field_name}" )

				# Construct complete prompt combining system instructions with document data
				# This ensures the model has full context for accurate extraction
				complete_prompt: str = f"""
{SYSTEM_PROMPT}

DOCUMENT CONTENTS:
{content}

TASK:
=====
{field_instruction}"""

				# Check if client is properly initialized
				if client is None :
					raise RuntimeError( "OLLAMA client is not initialized" )

				# Send prompt to LLM and wait for complete response
				# stream=False ensures we get the full response before proceeding
				response: Dict[ str , Any ] = client.generate(
						model=PREFERRED_LANGUAGE_MODEL ,
						prompt=complete_prompt ,
						stream=False ,
				)

				# Extract the text response from the LLM output
				response_text: str = response[ "response" ]
				extracted_fields[ field_name ] = response_text

				# Log successful extraction with response preview
				logger.info(
						f"Extracted field '{field_name}': {response_text[ :100 ]}..."
						if len( response_text ) > 100
						else f"Extracted field '{field_name}': {response_text}" ,
				)

			except Exception as e :
				# Handle individual field extraction failures gracefully
				# Log the error but continue processing other fields
				logger.warning( f"Failed to extract {field_name}: {e}" )
				extracted_fields[ field_name ] = "UNKNOWN"

		# Calculate total processing time for performance monitoring
		processing_time: float = (datetime.now( ) - start_time).total_seconds( )

		logger.info( f"Processing time: {processing_time:.2f}s" )

		# Return successful extraction report with all metadata
		return LanguageExtractionReport(
				success=True ,
				extracted_fields=extracted_fields ,
				processing_timestamp=datetime.now( ).isoformat( ) ,
				model_used=PREFERRED_LANGUAGE_MODEL ,
				processing_time_seconds=processing_time ,
				error=None ,
				error_type=None ,
		)

	except Exception as e :
		# Comprehensive error handling with context preservation
		processing_time = (datetime.now( ) - start_time).total_seconds( )

		# Log detailed error information for debugging
		logger.error( f"Error type: {type( e ).__name__}" )
		logger.error( f"Error message: {str( e )}" )
		logger.error( f"Processing time before failure: {processing_time:.2f}s" )

		# Log full traceback for debugging if available
		if hasattr( e , "__traceback__" ) :
			import traceback

			logger.debug( "Full traceback:" , exc_info=True )

		# Return failure report with diagnostic information
		return LanguageExtractionReport(
				success=False ,
				error=str( e ) ,
				error_type=type( e ).__name__ ,
				processing_timestamp=datetime.now( ).isoformat( ) ,
				processing_time_seconds=processing_time ,
				model_used=PREFERRED_LANGUAGE_MODEL ,
				extracted_fields=None ,
		)
