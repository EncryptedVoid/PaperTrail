import shutil
import subprocess
from pathlib import Path
from config import NOTESHRINK_PATH

def process_handwritten_notes( input_files , output_pdf ) :
	# Build command
	if isinstance( input_files , str ) :
		if "*" in input_files :
			# Glob pattern - let shell expand it
			cmd = (
				f"cd noteshrink && ./noteshrink.py ../{input_files} -o ../{output_pdf}"
			)
			subprocess.run( cmd , shell=True , check=True )
		else :
			# Single file
			cmd = [ "./noteshrink.py" , f"../{input_files}" , "-o" , f"../{output_pdf}" ]
			subprocess.run( cmd , cwd="noteshrink" , check=True )
	else :
		# List of files
		files = [ f"../{f}" for f in input_files ]
		cmd = [ "./noteshrink.py" ] + files + [ "-o" , f"../{output_pdf}" ]
		subprocess.run( cmd , cwd="noteshrink" , check=True )

	print( f"\n✓ Done! Check: {output_pdf}" )
	return output_pdf


def process_printed_documents( input_path , output_dir ) :
	input_path = Path( input_path )
	output_dir = Path( output_dir )
	output_dir.mkdir( exist_ok=True )

	# Check if it's a PDF
	if input_path.suffix.lower( ) == ".pdf" :
		from pdf2image import convert_from_path

		print( "Converting PDF to images..." )
		images = convert_from_path( str( input_path ) , dpi=300 )

		temp_dir = output_dir / "temp"
		temp_dir.mkdir( exist_ok=True )

		for i , img in enumerate( images ) :
			img.save( temp_dir / f"page_{i + 1}.png" )

		input_path = temp_dir

	# Process with document-scanner
	if input_path.is_dir( ) :
		cmd = [ "document-scanner" , "--images" , str( input_path ) , str( output_dir ) ]
	else :
		cmd = [ "document-scanner" , "--image" , str( input_path ) , str( output_dir ) ]

	print( "Processing documents..." )
	subprocess.run( cmd , check=True )

	# Clean up temp files
	if "temp" in str( input_path ) :
		shutil.rmtree( input_path )

	print( f"\n✓ Done! Check: {output_dir}" )
	return output_dir
