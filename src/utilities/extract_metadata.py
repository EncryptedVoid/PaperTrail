import json
import logging
import subprocess
from pathlib import Path

from config import (
	AUDIO_TYPES ,
	TIKA_APP_JAR_PATH ,
	VIDEO_TYPES
)


# Good news — the path fix worked! The .pptx extracted successfully. This is a different error.
# The key line is: Cannot run program "env": CreateProcess error=2
# Tika is trying to use an ExternalParser for the .mp4 file, which invokes the Unix env command to shell out to an
# external tool (likely ffmpeg). That command doesn't exist on Windows, so it crashes.
# This is a known Tika limitation on Windows for media files. You have two options:
# Option A: Install ffmpeg and handle gracefully. Install ffmpeg and add it to PATH — but Tika may still fail because
# it uses the Unix env command internally. So you'd also want to catch these failures gracefully rather than treating
# them as hard errors.


def extract_metadata( artifact_location: Path , logger: logging.Logger ) -> subprocess.CompletedProcess[ str ] :
	logger.info( f"Processing artifact: {artifact_location.name}	" )

	artifact_ext = artifact_location.suffix.strip( ).strip( "." ).lower( )

	logger.info( f"Starting metadata extraction for {artifact_location} using Tika JAR: {TIKA_APP_JAR_PATH}" )

	if artifact_ext in AUDIO_TYPES or artifact_ext in VIDEO_TYPES :
		artifact_metadata_extraction_cmd = [
			"ffprobe" ,
			"-v" ,
			"quiet" ,
			"-print_format" ,
			"json" ,
			"-show_format" ,
			"-show_streams" ,
			str( artifact_location ) ,
		]

	else :
		artifact_metadata_extraction_cmd = [
			"java" ,
			"-jar" ,
			str( TIKA_APP_JAR_PATH ) ,
			"--json" ,
			str( artifact_location ) ,
		]

	try :
		# Execute the Tika command as a subprocess
		# capture_output=True: Captures stdout and stderr for processing
		# text=True: Returns output as string rather than bytes
		# check=True: Raises CalledProcessError if command returns non-zero exit code
		result = subprocess.run(
				artifact_metadata_extraction_cmd ,
				capture_output=True ,
				text=True ,
				check=True ,
		)

		# Parse the JSON string returned by Tika into a Python dictionary
		# json.loads() converts the JSON string to a dict object
		metadata = json.loads( result.stdout )

		# Count total number of metadata fields extracted
		total_fields = len( metadata )
		logger.info( f"Successfully extracted {total_fields} metadata fields" )
	except subprocess.CalledProcessError as e :
		# Tika process failed (non-zero exit code)
		logger.error(
				f"Tika extraction process failed with exit code {e.returncode}" ,
		)
		logger.error( f"Tika error output: {e.stderr}" )
		raise RuntimeError( f"Tika extraction failed: {e.stderr}" )

	except json.JSONDecodeError as e :
		# Tika output was not valid JSON
		logger.error( f"Failed to parse Tika output as JSON: {e}" )
		raise RuntimeError( f"Failed to parse Tika output as JSON: {e}" )

	except Exception as e :
		# Catch any other unexpected errors
		logger.error( f"Unexpected error during metadata extraction: {e}" )
		raise

	return metadata
