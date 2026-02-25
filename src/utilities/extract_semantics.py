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
import json
import logging
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any , Dict , Optional , TypedDict

import ollama

from config import FIELD_PROMPTS , SYSTEM_PROMPT , TIKA_APP_JAR_PATH


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


def extract_metadata( file_path: Path , output_json_path: Path , logger ) -> Dict[ str , Any ] :
	"""
	Extract metadata from a file using Apache Tika and save to JSON.

	This function calls Apache Tika via subprocess to extract all available metadata
	from the specified file, then saves the results as formatted JSON.

	Args:
									file_path: Path object pointing to the file to extract metadata from
									output_json_path: Path object specifying where to save the JSON output
									logger: Logger instance for recording operation details and progress

	Returns:
									Dictionary containing all extracted metadata fields

	Raises:
									FileNotFoundError: If the input file or Tika JAR doesn't exist
									RuntimeError: If Tika extraction fails or output cannot be parsed
	"""
	# Record the start time to calculate total processing duration
	start_time = time.time( )

	logger.info( f"Starting metadata extraction for file: {file_path}" )

	# Validate that the input file exists on the filesystem
	if not file_path.exists( ) :
		logger.error( f"Input file does not exist: {file_path}" )
		raise FileNotFoundError( f"Input file not found: {file_path}" )

	# Log basic file information
	file_size = file_path.stat( ).st_size
	logger.info( f"Input file size: {file_size:,} bytes" )

	# Validate that the Tika JAR file exists at the specified location
	if not TIKA_APP_JAR_PATH.exists( ) :
		logger.error( f"Tika JAR file does not exist: {TIKA_APP_JAR_PATH}" )
		raise FileNotFoundError( f"Tika JAR not found: {TIKA_APP_JAR_PATH}" )

	logger.info( f"Using Tika JAR: {TIKA_APP_JAR_PATH}" )

	# Construct the command to run Tika
	# java: Invokes the Java runtime
	# -jar: Specifies we're running a JAR file
	# --json: Tells Tika to output metadata in JSON format
	cmd = [ "java" , "-jar" , str( TIKA_APP_JAR_PATH ) , "--json" , str( file_path ) ]

	logger.info( "Executing Tika extraction process" )

	# Record timing for the Tika subprocess execution
	tika_start = time.time( )

	try :
		# Execute the Tika command as a subprocess
		# capture_output=True: Captures stdout and stderr for processing
		# text=True: Returns output as string rather than bytes
		# check=True: Raises CalledProcessError if command returns non-zero exit code
		result = subprocess.run( cmd , capture_output=True , text=True , check=True )

		# Calculate how long the Tika extraction took
		tika_duration = time.time( ) - tika_start
		logger.info( f"Tika extraction completed in {tika_duration:.2f} seconds" )

		# Parse the JSON string returned by Tika into a Python dictionary
		# json.loads() converts the JSON string to a dict object
		metadata = json.loads( result.stdout )

		# Count total number of metadata fields extracted
		total_fields = len( metadata )
		logger.info( f"Successfully extracted {total_fields} metadata fields" )

		# Analyze the types of values in the metadata for statistics
		# Counter creates a frequency count of value types
		type_counts = Counter( )
		for key , value in metadata.items( ) :
			# Get the type name (e.g., 'str', 'int', 'list', 'dict')
			type_name = type( value ).__name__
			type_counts[ type_name ] += 1

		# Log breakdown of metadata field types
		logger.info( f"Metadata type breakdown: {dict( type_counts )}" )

		# Log sample of metadata keys for verification (first 5 keys)
		sample_keys = list( metadata.keys( ) )[ :5 ]
		logger.info( f"Sample metadata fields: {', '.join( sample_keys )}" )

		logger.info( f"Writing metadata to output file: {output_json_path}" )

		# Write the metadata dictionary to a JSON file
		# 'w': Open file in write mode
		# encoding='utf-8': Use UTF-8 encoding to support international characters
		# indent=2: Pretty-print JSON with 2-space indentation
		# ensure_ascii=False: Allow non-ASCII characters in output
		with open( output_json_path , "w" , encoding="utf-8" ) as f :
			json.dump( metadata , f , indent=2 , ensure_ascii=False )

		# Get the size of the output JSON file for logging
		output_size = output_json_path.stat( ).st_size
		logger.info( f"Output JSON file size: {output_size:,} bytes" )

		# Calculate total operation duration
		total_duration = time.time( ) - start_time
		logger.info(
				f"Metadata extraction completed successfully in {total_duration:.2f} seconds" ,
				)

		# Return the metadata dictionary for potential further processing
		return metadata

	except subprocess.CalledProcessError as e :
		# Tika process failed (non-zero exit code)
		logger.error( f"Tika extraction process failed with exit code {e.returncode}" )
		logger.error( f"Tika error output: {e.stderr}" )
		raise RuntimeError( f"Tika extraction failed: {e.stderr}" )

	except json.JSONDecodeError as e :
		# Tika output was not valid JSON
		logger.error( f"Failed to parse Tika output as JSON: {e}" )
		logger.error( f"Tika raw output: {result.stdout[ :500 ]}" )  # Log first 500 chars
		raise RuntimeError( f"Failed to parse Tika output as JSON: {e}" )

	except Exception as e :
		# Catch any other unexpected errors
		logger.error( f"Unexpected error during metadata extraction: {e}" )
		raise


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
