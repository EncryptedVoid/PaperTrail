import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import ollama
import requests

from config import JAVA_PATH , LOG_DIR , TIKA_SERVER_JAR_PATH


def check_command_exists( command: str ) -> bool :
	"""
    Check if a command exists, refreshing PATH from registry first on Windows
    to catch recently installed programs.
    """
	# Refresh PATH on Windows before checking
	if platform.system( ) == "Windows" :
		_refresh_windows_path( )

	return shutil.which( command ) is not None


def run_cmd(
		command: str ,
		description: str = "" ,
		capture: bool = True ,
		timeout: int = 300 ,
) -> Tuple[ bool , str ] :
	"""
    Run a shell command with optional output capture and a description label.

    Args:
        command:     Shell command string to execute
        description: Human-readable label printed before running (e.g. "Installing ffmpeg")
        capture:     If True, capture and return stdout+stderr as a string.
                     If False, output streams directly to the terminal (useful for
                     long-running installs where you want live progress).
        timeout:     Max seconds to wait before giving up (default: 300)

    Returns:
        Tuple of (success: bool, output: str).
        output is the combined stdout+stderr when capture=True, empty string otherwise.
    """
	if description :
		print( f"  → {description}..." )

	try :
		if capture :
			result = subprocess.run(
					command ,
					shell=True ,
					capture_output=True ,
					text=True ,
					timeout=timeout ,
					check=False ,
			)
			return result.returncode == 0 , result.stdout + result.stderr
		else :
			# Stream output directly to terminal — useful for interactive installs
			result = subprocess.run(
					command ,
					shell=True ,
					timeout=timeout ,
					check=False ,
			)
			return result.returncode == 0 , ""

	except subprocess.TimeoutExpired :
		print( f"  ⚠ Command timed out after {timeout}s: {command}" )
		return False , f"Timeout after {timeout}s"
	except Exception as e :
		print( f"  ⚠ Command failed: {e}" )
		return False , str( e )


def _run_command( cmd: list , timeout: int = 30 , shell: bool = False ) -> Tuple[ bool , str ] :
	"""Helper function to run shell commands safely."""
	try :
		result = subprocess.run(
				cmd ,
				capture_output=True ,
				text=True ,
				timeout=timeout ,
				check=False ,
				shell=shell ,
		)
		return result.returncode == 0 , result.stdout + result.stderr
	except Exception as e :
		return False , str( e )


def _get_os( ) -> str :
	"""Get the current operating system."""
	return platform.system( )


def _ensure_wsl( ) -> bool :
	"""Ensure WSL is available on Windows. Returns True if available."""
	if _get_os( ) != "Windows" :
		return False

	# Check if wsl command exists
	if shutil.which( "wsl" ) :
		# Check if a distro is installed
		success , output = _run_command( [ "wsl" , "--list" , "--quiet" ] )
		if success and output.strip( ) :
			print( "✓ WSL is available" )
			return True

	print( "⚠ WSL not found. Install with: wsl --install" )
	return False


def _run_in_wsl( command: str , timeout: int = 300 ) -> Tuple[ bool , str ] :
	"""Run a command in WSL."""
	wsl_cmd = [ "wsl" , "bash" , "-c" , command ]
	return _run_command( wsl_cmd , timeout=timeout )


def _clean_choco_lock_files( ) :
	"""Clean up Chocolatey lock files that cause issues."""
	try :
		lock_patterns = [
			"C:\\ProgramData\\chocolatey\\lib\\*lock*" ,
			"C:\\ProgramData\\chocolatey\\lib-bad" ,
		]
		for pattern in lock_patterns :
			subprocess.run(
					[
						"powershell" ,
						"-Command" ,
						f"Remove-Item -Force -Recurse '{pattern}' -ErrorAction SilentlyContinue" ,
					] ,
					capture_output=True ,
					timeout=10 ,
			)
	except :
		pass


def _install_via_choco( package: str , timeout: int = 300 ) -> bool :
	"""Install package via Chocolatey with automatic cleanup and fallback."""
	if not shutil.which( "choco" ) :
		return False

	# Clean lock files first
	_clean_choco_lock_files( )

	try :
		# Try normal install with force and yes to all
		result = subprocess.run(
				[ "choco" , "install" , "-y" , "--force" , "--ignore-checksums" , package ] ,
				capture_output=True ,
				text=True ,
				timeout=timeout ,
				check=False ,
				input="Y\n" ,  # Auto-answer yes
		)

		if result.returncode == 0 :
			return True

		# If failed, clean again and try elevated
		_clean_choco_lock_files( )

		# Try with PowerShell elevation (will prompt for admin)
		ps_cmd = f'Start-Process choco -ArgumentList "install -y --force --ignore-checksums {package}" -Verb RunAs -Wait -WindowStyle Hidden'
		result = subprocess.run(
				[ "powershell" , "-Command" , ps_cmd ] ,
				capture_output=True ,
				timeout=timeout ,
				check=False ,
		)

		if result.returncode == 0 :
			return True

		return False
	except Exception as e :
		print( f"⚠ Chocolatey error: {e}" )
		return False


def _install_via_winget( package_id: str , timeout: int = 300 ) -> bool :
	"""Install package via winget."""
	if not shutil.which( "winget" ) :
		return False

	try :
		result = subprocess.run(
				[
					"winget" ,
					"install" ,
					"--id" ,
					package_id ,
					"-e" ,
					"--accept-package-agreements" ,
					"--accept-source-agreements" ,
					"--silent" ,
				] ,
				capture_output=True ,
				timeout=timeout ,
				check=False ,
		)
		return result.returncode == 0
	except Exception :
		return False


def ensure_ffmpeg( min_version: str = "4.0" ) -> str :
	"""Ensure FFmpeg is installed with WSL fallback."""
	# Check if already installed
	if shutil.which( "ffmpeg" ) :
		success , output = _run_command( [ "ffmpeg" , "-version" ] )
		if success :
			version_line = output.split( "\n" )[ 0 ]
			print( f"✓ FFmpeg already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing FFmpeg on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "ffmpeg" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try winget first
			if _install_via_winget( "Gyan.FFmpeg" ) :
				print( "✓ Installed via winget" )
			# Try chocolatey
			elif _install_via_choco( "ffmpeg" ) :
				print( "✓ Installed via chocolatey" )
			# Fallback to WSL
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y ffmpeg" )
					success , output = _run_in_wsl( "ffmpeg -version" )
					if success :
						print( "✓ FFmpeg installed in WSL" )
						return "ffmpeg (WSL)"
					else :
						print( "⚠ WSL install failed, ffmpeg may not work" )
						return "ffmpeg (unavailable)"
				else :
					print( "⚠ WSL unavailable, ffmpeg not installed" )
					return "ffmpeg (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "ffmpeg" ] , check=False , timeout=300 )

		# Verify installation
		if shutil.which( "ffmpeg" ) :
			success , output = _run_command( [ "ffmpeg" , "-version" ] )
			if success :
				version_line = output.split( "\n" )[ 0 ]
				print( f"✓ FFmpeg installed: {version_line}" )
				return version_line

		# Check WSL as last resort
		if os_type == "Windows" :
			success , output = _run_in_wsl( "ffmpeg -version" )
			if success :
				print( "✓ FFmpeg available in WSL" )
				return "ffmpeg (WSL)"

		print( "⚠ FFmpeg installation incomplete, may not work" )
		return "ffmpeg (unavailable)"

	except Exception as e :
		print( f"⚠ FFmpeg installation error: {e}" )
		return "ffmpeg (unavailable)"


def ensure_imagemagick( min_version: str = "6.9" ) -> str :
	"""Ensure ImageMagick is installed with WSL fallback."""
	# Check if already installed
	if shutil.which( "convert" ) or shutil.which( "magick" ) :
		cmd = "magick" if shutil.which( "magick" ) else "convert"
		success , output = _run_command( [ cmd , "-version" ] )
		if success and "ImageMagick" in output :
			version_line = output.split( "\n" )[ 0 ]
			print( f"✓ ImageMagick already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing ImageMagick on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "imagemagick" ] ,
					check=False ,
					timeout=300 ,
			)

		elif os_type == "Windows" :
			if _install_via_winget( "ImageMagick.ImageMagick" ) :
				print( "✓ Installed via winget" )
			elif _install_via_choco( "imagemagick" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					_run_in_wsl( "sudo apt update -y && sudo apt install -y imagemagick" )
					success , output = _run_in_wsl( "convert -version" )
					if success :
						print( "✓ ImageMagick installed in WSL" )
						return "imagemagick (WSL)"
				print( "⚠ ImageMagick not fully installed" )
				return "imagemagick (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , "imagemagick" ] , check=False , timeout=300 ,
				)

		# Verify
		if shutil.which( "convert" ) or shutil.which( "magick" ) :
			cmd = "magick" if shutil.which( "magick" ) else "convert"
			success , output = _run_command( [ cmd , "-version" ] )
			if success :
				version_line = output.split( "\n" )[ 0 ]
				print( f"✓ ImageMagick installed: {version_line}" )
				return version_line

		if os_type == "Windows" :
			success , output = _run_in_wsl( "convert -version" )
			if success :
				print( "✓ ImageMagick available in WSL" )
				return "imagemagick (WSL)"

		return "imagemagick (unavailable)"

	except Exception as e :
		print( f"⚠ ImageMagick error: {e}" )
		return "imagemagick (unavailable)"


def ensure_pandoc( min_version: str = "2.0" ) -> str :
	"""Ensure Pandoc is installed with WSL fallback."""
	if shutil.which( "pandoc" ) :
		success , output = _run_command( [ "pandoc" , "--version" ] )
		if success :
			version_line = output.split( "\n" )[ 0 ]
			print( f"✓ Pandoc already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing Pandoc on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "pandoc" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			if _install_via_winget( "JohnMacFarlane.Pandoc" ) :
				print( "✓ Installed via winget" )
			elif _install_via_choco( "pandoc" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					_run_in_wsl( "sudo apt update -y && sudo apt install -y pandoc" )
					success , output = _run_in_wsl( "pandoc --version" )
					if success :
						print( "✓ Pandoc installed in WSL" )
						return "pandoc (WSL)"
				return "pandoc (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "pandoc" ] , check=False , timeout=300 )

		if shutil.which( "pandoc" ) :
			success , output = _run_command( [ "pandoc" , "--version" ] )
			if success :
				version_line = output.split( "\n" )[ 0 ]
				print( f"✓ Pandoc installed: {version_line}" )
				return version_line

		if os_type == "Windows" :
			success , output = _run_in_wsl( "pandoc --version" )
			if success :
				print( "✓ Pandoc available in WSL" )
				return "pandoc (WSL)"

		return "pandoc (unavailable)"

	except Exception as e :
		print( f"⚠ Pandoc error: {e}" )
		return "pandoc (unavailable)"


def ensure_unpaper( min_version: str = "6.1" ) -> str :
	"""Ensure unpaper is installed with WSL fallback. NEVER crashes."""
	# Check if already installed
	if shutil.which( "unpaper" ) :
		success , output = _run_command( [ "unpaper" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ unpaper already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing unpaper on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "unpaper" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try chocolatey (will likely fail due to your permissions)
			if _install_via_choco( "unpaper" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Chocolatey failed (expected), falling back to WSL..." )
				if _ensure_wsl( ) :
					print( "Installing unpaper in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y unpaper" )
					success , output = _run_in_wsl( "unpaper --version" )
					if success :
						print( "✓ unpaper installed in WSL" )
						return "unpaper (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. unpaper will not be available." )
					print( "   Document cleaning features will be limited." )
				return "unpaper (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "unpaper" ] , check=False , timeout=300 )

		# Verify
		if shutil.which( "unpaper" ) :
			success , output = _run_command( [ "unpaper" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ unpaper installed: {version_line}" )
				return version_line

		# Check WSL
		if os_type == "Windows" :
			success , output = _run_in_wsl( "unpaper --version" )
			if success :
				print( "✓ unpaper available in WSL" )
				return "unpaper (WSL)"

		print( "⚠ unpaper not installed, features will be limited" )
		return "unpaper (unavailable)"

	except Exception as e :
		print( f"⚠ unpaper installation issue: {e}" )
		print( "   Document cleaning will be limited" )
		return "unpaper (unavailable)"


def ensure_par2( min_version: str = "0.8" ) -> str :
	"""Ensure par2 is installed with WSL fallback."""
	if shutil.which( "par2" ) :
		success , output = _run_command( [ "par2" , "-V" ] )
		if success :
			version_line = output.split( "\n" )[ 0 ] if output else "par2 (version unknown)"
			print( f"✓ par2 already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing par2 on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "par2" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			if _install_via_choco( "par2cmdline" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Chocolatey failed, using WSL..." )
				if _ensure_wsl( ) :
					_run_in_wsl( "sudo apt update -y && sudo apt install -y par2" )
					success , output = _run_in_wsl( "par2 -V" )
					if success :
						print( "✓ par2 installed in WSL" )
						return "par2 (WSL)"
				return "par2 (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "par2" ] , check=False , timeout=300 )

		if shutil.which( "par2" ) :
			success , output = _run_command( [ "par2" , "-V" ] )
			if success :
				version_line = (
					output.split( "\n" )[ 0 ] if output else "par2 (version unknown)"
				)
				print( f"✓ par2 installed: {version_line}" )
				return version_line

		if os_type == "Windows" :
			success , output = _run_in_wsl( "par2 -V" )
			if success :
				print( "✓ par2 available in WSL" )
				return "par2 (WSL)"

		return "par2 (unavailable)"

	except Exception as e :
		print( f"⚠ par2 error: {e}" )
		return "par2 (unavailable)"


def ensure_java( min_version: int = 11 ) -> str :
	"""Ensure Java is installed with WSL fallback."""
	if shutil.which( "java" ) :
		success , output = _run_command( [ "java" , "-version" ] )
		if success :
			import re

			version_output = output.lower( )
			match = re.search( r'version "?(\d+)\.?(\d+)?' , version_output )
			if match :
				major = int( match.group( 1 ) )
				if major == 1 and match.group( 2 ) :
					major = int( match.group( 2 ) )

				version_line = output.split( "\n" )[ 0 ]
				if major >= min_version :
					print( f"✓ Java already installed: {version_line}" )
					return version_line

	os_type = _get_os( )
	print( f"Installing Java {min_version}+ on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , f"openjdk-{min_version}-jdk" ] ,
					check=False ,
					timeout=300 ,
			)

		elif os_type == "Windows" :
			if _install_via_winget( "EclipseAdoptium.Temurin.11.JDK" ) :
				print( "✓ Installed via winget" )
			elif _install_via_choco( f"openjdk{min_version}" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					_run_in_wsl(
							f"sudo apt update -y && sudo apt install -y openjdk-{min_version}-jdk" ,
					)
					success , output = _run_in_wsl( "java -version" )
					if success :
						print( "✓ Java installed in WSL" )
						return f"java-{min_version} (WSL)"
				return "java (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , f"openjdk@{min_version}" ] ,
						check=False ,
						timeout=300 ,
				)

		if shutil.which( "java" ) :
			success , output = _run_command( [ "java" , "-version" ] )
			if success :
				version_line = output.split( "\n" )[ 0 ]
				print( f"✓ Java installed: {version_line}" )
				return version_line

		if os_type == "Windows" :
			success , output = _run_in_wsl( "java -version" )
			if success :
				print( "✓ Java available in WSL" )
				return "java (WSL)"

		return "java (unavailable)"

	except Exception as e :
		print( f"⚠ Java error: {e}" )
		return "java (unavailable)"


def ensure_libpff_python( min_version: str = "20180714" ) -> str :
	"""
    Ensure libpff-python is installed for reading Outlook PST/OST files.

    Args:
                    min_version: Minimum required version (default: "20180714")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If libpff-python cannot be installed or verified

    Example verification commands:
                    python -c "import pypff; print(pypff.__version__)"
    """
	# Check if already installed
	try :
		import pypff

		version = getattr( pypff , "__version__" , "unknown" )
		print( f"✓ libpff-python already installed: version {version}" )
		return f"libpff-python {version}"
	except ImportError :
		pass

	# Install based on OS
	os_type = _get_os( )
	print( f"Installing libpff-python on {os_type}..." )

	try :
		if os_type == "Linux" :
			# Install system dependencies first
			subprocess.run( [ "sudo" , "apt" , "update" ] , check=True , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "libpff-dev" ] , check=True , timeout=300 ,
			)
			# Install Python package
			subprocess.run(
					[ sys.executable , "-m" , "pip" , "install" , "libpff-python" ] ,
					check=True ,
					timeout=300 ,
			)
		elif os_type == "Windows" :
			# Windows typically uses pre-built wheels
			subprocess.run(
					[ sys.executable , "-m" , "pip" , "install" , "libpff-python" ] ,
					check=True ,
					timeout=300 ,
			)
		elif os_type == "Darwin" :
			# Install dependencies via Homebrew
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "libpff" ] , check=True , timeout=300 )
			subprocess.run(
					[ sys.executable , "-m" , "pip" , "install" , "libpff-python" ] ,
					check=True ,
					timeout=300 ,
			)
		else :
			raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

		# Verify installation
		try :
			import pypff

			version = getattr( pypff , "__version__" , "unknown" )
			print( f"✓ libpff-python installed successfully: version {version}" )
			print(
					f"  Verify with: {sys.executable} -c 'import pypff; print(pypff.__version__)'" ,
			)
			return f"libpff-python {version}"
		except ImportError :
			raise ModuleNotFoundError(
					"libpff-python pip install succeeded but module cannot be imported" ,
			)

	except subprocess.CalledProcessError as e :
		raise ModuleNotFoundError( f"libpff-python installation failed: {e}" )


def ensure_java( min_version: int = 11 ) -> str :
	"""
    Ensure Java (JDK/JRE) is installed and meets minimum version requirements.

    Args:
                    min_version: Minimum required major version (default: 11)

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If Java cannot be installed or version is insufficient

    Example verification commands:
                    java -version
                    javac -version
    """
	# Check if already installed
	if shutil.which( "java" ) :
		success , output = _run_command( [ "java" , "-version" ] )
		if success :
			# Java version output goes to stderr typically
			version_output = output.lower( )
			# Parse version (handles both old and new format)
			import re

			match = re.search( r'version "?(\d+)\.?(\d+)?' , version_output )
			if match :
				major = int( match.group( 1 ) )
				# Java 9+ uses single digit versioning (11, 17, etc)
				# Java 8 and below use 1.x format
				if major == 1 and match.group( 2 ) :
					major = int( match.group( 2 ) )

				version_line = output.split( "\n" )[ 0 ]
				if major >= min_version :
					print( f"✓ Java already installed: {version_line}" )
					return version_line
				else :
					print( f"⚠ Java {major} found, but need version {min_version}+" )

	# Install based on OS
	os_type = _get_os( )
	print( f"Installing Java {min_version}+ on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" ] , check=True , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , f"openjdk-{min_version}-jdk" ] ,
					check=True ,
					timeout=300 ,
			)
		elif os_type == "Windows" :
			if shutil.which( "winget" ) :
				subprocess.run(
						[
							"winget" ,
							"install" ,
							"--id" ,
							"EclipseAdoptium.Temurin.11.JDK" ,
							"-e" ,
						] ,
						check=True ,
						timeout=300 ,
				)
			elif shutil.which( "choco" ) :
				subprocess.run(
						[ "choco" , "install" , "-y" , f"openjdk{min_version}" ] ,
						check=True ,
						timeout=300 ,
				)
			else :
				raise ModuleNotFoundError(
						f"Java installation failed: No package manager found.\n"
						f"Download from: https://adoptium.net/" ,
				)
		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , f"openjdk@{min_version}" ] ,
						check=True ,
						timeout=300 ,
				)
			else :
				raise ModuleNotFoundError(
						"Java installation failed: Homebrew not found.\n"
						f"Install with: brew install openjdk@{min_version}" ,
				)
		else :
			raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

		# Verify installation
		if shutil.which( "java" ) :
			success , output = _run_command( [ "java" , "-version" ] )
			if success :
				version_line = output.split( "\n" )[ 0 ]
				print( f"✓ Java installed successfully: {version_line}" )
				print( "  Verify with: java -version" )
				return version_line

		raise ModuleNotFoundError(
				"Java installation completed but command not found in PATH" ,
		)

	except subprocess.CalledProcessError as e :
		raise ModuleNotFoundError( f"Java installation failed: {e}" )


def _find_java_executable( min_version: int = 11 ) -> str | None :
	"""
    Search common installation paths for a Java executable meeting the minimum version.
    Updates os.environ["PATH"] at runtime if a suitable Java is found outside current PATH.

    Returns the path to the java executable, or None if not found.
    """
	import os
	import re

	def _get_java_version( java_path: str ) -> int | None :
		"""Returns the major version of the given java executable, or None on failure."""
		try :
			result = subprocess.run(
					[ java_path , "-version" ] ,
					capture_output=True ,
					text=True ,
					timeout=10 ,
			)
			output = result.stdout + result.stderr  # java -version writes to stderr
			match = re.search( r'version "?(\d+)\.?(\d+)?' , output )
			if match :
				major = int( match.group( 1 ) )
				if major == 1 and match.group( 2 ) :
					major = int( match.group( 2 ) )
				return major
		except Exception :
			pass
		return None

	# Step 1: Check current PATH first
	current_java = shutil.which( "java" )
	if current_java :
		version = _get_java_version( current_java )
		if version and version >= min_version :
			return current_java  # Already good, nothing to do

	# Step 2: Search common install locations for a suitable Java
	search_roots = [ ]

	if platform.system( ) == "Windows" :
		search_roots = [
			r"C:\Program Files\Eclipse Adoptium" ,
			r"C:\Program Files\Java" ,
			r"C:\Program Files\Microsoft" ,
			r"C:\Program Files\OpenJDK" ,
			r"C:\Program Files\BellSoft" ,
			r"C:\Program Files\Amazon Corretto" ,
		]
		# Also check JAVA_HOME env var
		java_home = os.environ.get( "JAVA_HOME" , "" )
		if java_home :
			search_roots.insert( 0 , java_home )
	else :
		search_roots = [
			"/usr/lib/jvm" ,
			"/usr/local/lib/jvm" ,
			"/opt/java" ,
			"/opt/jdk" ,
		]
		java_home = os.environ.get( "JAVA_HOME" , "" )
		if java_home :
			search_roots.insert( 0 , java_home )

	best_java = None
	best_version = 0

	for root in search_roots :
		if not os.path.isdir( root ) :
			continue
		# Walk up to 3 levels deep looking for java/java.exe
		for dirpath , dirnames , filenames in os.walk( root ) :
			# Limit search depth
			depth = dirpath.replace( root , "" ).count( os.sep )
			if depth > 3 :
				dirnames.clear( )
				continue

			java_exe = "java.exe" if platform.system( ) == "Windows" else "java"
			candidate = os.path.join( dirpath , java_exe )

			if os.path.isfile( candidate ) :
				version = _get_java_version( candidate )
				if version and version >= min_version and version > best_version :
					best_java = candidate
					best_version = version

	if best_java :
		# Inject the found Java's bin directory at the front of PATH for this process
		java_bin_dir = os.path.dirname( best_java )
		os.environ[ "PATH" ] = java_bin_dir + os.pathsep + os.environ.get( "PATH" , "" )
		print( f"✓ Found Java {best_version} at: {best_java}" )
		print( f"  Updated runtime PATH to use this Java for this session" )
		return best_java

	return None


def ensure_apache_tika( tika_version: str = "2.9.2" , install_dir: str = None ) -> str :
	"""
    Ensure Apache Tika JAR is downloaded and available.

    Automatically searches for a suitable Java 11+ installation and patches
    the runtime PATH if the system PATH points to an older version.

    Args:
        tika_version: Version to download (default: "2.9.2")
        install_dir: Directory to install Tika JAR (default: ~/.tika/)

    Returns:
        Path to the Tika JAR file

    Raises:
        ModuleNotFoundError: If Tika cannot be downloaded or Java 11+ is not available
    """
	import os
	import urllib.request

	# Find a suitable Java, patching PATH at runtime if needed
	java_exe = _find_java_executable( min_version=11 )

	if not java_exe :
		# Last resort: try to install Java then search again
		try :
			ensure_java( min_version=11 )
			java_exe = _find_java_executable( min_version=11 )
		except Exception :
			pass

	if not java_exe :
		raise ModuleNotFoundError(
				"Apache Tika requires Java 11+. Could not find or install it.\n"
				"Download from: https://adoptium.net/" ,
		)

	# Set install directory
	if install_dir is None :
		install_dir = os.path.expanduser( "~/.tika" )

	os.makedirs( install_dir , exist_ok=True )

	jar_name = f"tika-app-{tika_version}.jar"
	jar_path = os.path.join( install_dir , jar_name )

	# Check if already downloaded and working
	if os.path.exists( jar_path ) :
		success , output = _run_command( [ java_exe , "-jar" , jar_path , "--version" ] )
		if success and tika_version in output :
			print( f"✓ Apache Tika already installed: {jar_path}" )
			return jar_path
		else :
			# JAR exists but doesn't work — delete and re-download
			print( f"⚠ Tika JAR exists but failed verification, re-downloading..." )
			os.remove( jar_path )

	# Download Tika
	print( f"Downloading Apache Tika {tika_version}..." )
	tika_url = f"https://archive.apache.org/dist/tika/{tika_version}/{jar_name}"

	try :
		print( f"  From: {tika_url}" )
		print( f"  To:   {jar_path}" )

		urllib.request.urlretrieve( tika_url , jar_path )

		if os.path.exists( jar_path ) and os.path.getsize( jar_path ) > 1_000_000 :
			success , output = _run_command( [ java_exe , "-jar" , jar_path , "--version" ] )
			if success :
				print( f"✓ Apache Tika downloaded and verified: {jar_path}" )
				return jar_path

		raise ModuleNotFoundError( "Tika JAR downloaded but verification failed" )

	except Exception as e :
		if os.path.exists( jar_path ) :
			os.remove( jar_path )
		raise ModuleNotFoundError( f"Apache Tika download failed: {e}" )


def ensure_pdfarranger( min_version: str = "1.8" ) -> str :
	"""
    Ensure pdfarranger is installed for PDF manipulation.

    Args:
                    min_version: Minimum required version (default: "1.8")

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If pdfarranger cannot be installed or verified

    Example verification commands:
                    pdfarranger --version
                    python -c "import pdfarranger"
    """
	# Check if already installed
	if shutil.which( "pdfarranger" ) :
		success , output = _run_command( [ "pdfarranger" , "--version" ] )
		if success :
			version_line = output.strip( )
			print( f"✓ pdfarranger already installed: {version_line}" )
			return version_line

	# Install based on OS
	os_type = _get_os( )
	print( f"Installing pdfarranger on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" ] , check=True , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "pdfarranger" ] , check=True , timeout=300 ,
			)
		elif os_type == "Windows" :
			# Windows typically uses pip installation
			print( "Installing pdfarranger via pip..." )
			subprocess.run(
					[ sys.executable , "-m" , "pip" , "install" , "pdfarranger" ] ,
					check=True ,
					timeout=300 ,
			)
		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , "pdfarranger" ] , check=True , timeout=300 ,
				)
			else :
				# Fallback to pip
				print( "Homebrew not found, installing via pip..." )
				subprocess.run(
						[ sys.executable , "-m" , "pip" , "install" , "pdfarranger" ] ,
						check=True ,
						timeout=300 ,
				)
		else :
			raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

		# Verify installation
		if shutil.which( "pdfarranger" ) :
			success , output = _run_command( [ "pdfarranger" , "--version" ] )
			if success :
				version_line = output.strip( )
				print( f"✓ pdfarranger installed successfully: {version_line}" )
				print( "  Verify with: pdfarranger --version" )
				return version_line

		raise ModuleNotFoundError(
				"pdfarranger installation completed but command not found in PATH" ,
		)

	except subprocess.CalledProcessError as e :
		raise ModuleNotFoundError( f"pdfarranger installation failed: {e}" )


def ensure_ollama(
		min_version: str = "0.1" , start_server: bool = True , port: int = 11434 ,
) -> str :
	"""
    Ensure Ollama is installed and optionally start the server.
    Kills any existing Ollama servers to ensure clean state.

    Args:
                    min_version: Minimum required version (default: "0.1")
                    start_server: Whether to start ollama serve (default: True)
                    port: Port for Ollama server (default: 11434)

    Returns:
                    Installed version string

    Raises:
                    ModuleNotFoundError: If Ollama cannot be installed or verified

    Example verification commands:
                    ollama --version
                    ollama list
                    ollama serve (to start the server)
    """
	import psutil
	import time
	import requests

	# Check if already installed
	if shutil.which( "ollama" ) :
		success , output = _run_command( [ "ollama" , "--version" ] )
		if success :
			version_line = output.strip( )
			print( f"✓ Ollama already installed: {version_line}" )
		else :
			version_line = "unknown"
	else :
		# Install based on OS
		os_type = _get_os( )
		print( f"Installing Ollama on {os_type}..." )

		try :
			if os_type == "Linux" :
				# Use official install script
				print( "Running official Ollama install script..." )
				install_script = "curl -fsSL https://ollama.com/install.sh | sh"
				subprocess.run( install_script , shell=True , check=True , timeout=300 )
			elif os_type == "Windows" :
				if shutil.which( "winget" ) :
					subprocess.run(
							[ "winget" , "install" , "--id" , "Ollama.Ollama" , "-e" ] ,
							check=True ,
							timeout=300 ,
					)
				else :
					raise ModuleNotFoundError(
							"Ollama installation failed: winget not found.\n"
							"Download installer from: https://ollama.com/download/windows" ,
					)
			elif os_type == "Darwin" :
				# Check if Homebrew available
				if shutil.which( "brew" ) :
					subprocess.run(
							[ "brew" , "install" , "ollama" ] , check=True , timeout=300 ,
					)
				else :
					# Suggest manual download
					raise ModuleNotFoundError(
							"Ollama installation failed: Homebrew not found.\n"
							"Download from: https://ollama.com/download/mac\n"
							"Or install with: brew install ollama" ,
					)
			else :
				raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

			# Verify installation
			if shutil.which( "ollama" ) :
				success , output = _run_command( [ "ollama" , "--version" ] )
				if success :
					version_line = output.strip( )
					print( f"✓ Ollama installed successfully: {version_line}" )
				else :
					version_line = "unknown"
			else :
				raise ModuleNotFoundError(
						"Ollama installation completed but command not found in PATH" ,
				)

		except subprocess.CalledProcessError as e :
			raise ModuleNotFoundError( f"Ollama installation failed: {e}" )

	# =========================================================================
	# SERVER MANAGEMENT
	# =========================================================================

	if start_server :
		print( "\n" + "=" * 60 )
		print( "Ollama Server Management" )
		print( "=" * 60 )

		# Step 1: Kill any existing Ollama processes
		print( "Checking for existing Ollama servers..." )
		killed_count = 0

		try :
			for proc in psutil.process_iter( [ "pid" , "name" , "cmdline" ] ) :
				try :
					# Check if process is ollama serve
					if proc.info[ "name" ] and "ollama" in proc.info[ "name" ].lower( ) :
						cmdline = proc.info.get( "cmdline" , [ ] )
						if cmdline and any( "serve" in str( arg ).lower( ) for arg in cmdline ) :
							print(
									f"  Killing existing Ollama server (PID: {proc.info[ 'pid' ]})" ,
							)
							proc.kill( )
							killed_count += 1
				except (psutil.NoSuchProcess , psutil.AccessDenied) :
					pass

			if killed_count > 0 :
				print( f"✓ Killed {killed_count} existing Ollama server(s)" )
				time.sleep( 2 )  # Give processes time to clean up
			else :
				print( "✓ No existing Ollama servers found" )

		except Exception as e :
			print( f"⚠️  Warning: Could not check for existing processes: {e}" )

		# Step 2: Start new Ollama server
		print( f"\nStarting Ollama server on port {port}..." )

		os_type = _get_os( )

		try :
			if os_type == "Windows" :
				# Windows: Start as detached process
				startupinfo = subprocess.STARTUPINFO( )
				startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
				startupinfo.wShowWindow = subprocess.SW_HIDE

				server_process = subprocess.Popen(
						[ "ollama" , "serve" ] ,
						stdout=subprocess.DEVNULL ,
						stderr=subprocess.DEVNULL ,
						startupinfo=startupinfo ,
						creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
													| subprocess.DETACHED_PROCESS ,
				)
			else :
				# Linux/Mac: Start as background process
				server_process = subprocess.Popen(
						[ "ollama" , "serve" ] ,
						stdout=subprocess.DEVNULL ,
						stderr=subprocess.DEVNULL ,
						start_new_session=True ,  # Detach from parent
				)

			print( f"  Started Ollama server (PID: {server_process.pid})" )

			# Step 3: Wait for server to be ready
			print( "  Waiting for server to be ready..." , end="" , flush=True )

			max_wait = 30  # seconds
			start_time = time.time( )
			server_ready = False

			while time.time( ) - start_time < max_wait :
				try :
					response = requests.get(
							f"http://localhost:{port}/api/tags" , timeout=2 ,
					)
					if response.status_code == 200 :
						server_ready = True
						break
				except requests.exceptions.RequestException :
					pass

				print( "." , end="" , flush=True )
				time.sleep( 1 )

			print( )  # Newline

			if server_ready :
				print( f"✓ Ollama server is ready on http://localhost:{port}" )
				print( f"  Server PID: {server_process.pid}" )
				print( f"  Verify with: curl http://localhost:{port}/api/tags" )
			else :
				print(
						f"⚠️  Warning: Server started but not responding after {max_wait}s" ,
				)
				print( f"  Server may still be initializing..." )
				print( f"  Check manually: curl http://localhost:{port}/api/tags" )

		except FileNotFoundError :
			raise ModuleNotFoundError(
					"Ollama command not found. Installation may have failed." ,
			)
		except Exception as e :
			raise ModuleNotFoundError( f"Failed to start Ollama server: {e}" )

		print( "=" * 60 )

	return version_line


def ensure_unpaper( min_version: str = "6.1" ) -> str :
	"""
    Ensure unpaper is installed for cleaning up scanned images.

    Args:
        min_version: Minimum required version (default: "6.1")

    Returns:
        Installed version string

    Raises:
        ModuleNotFoundError: If unpaper cannot be installed or verified

    Example verification commands:
        unpaper --version
        unpaper --help
    """
	# Check if already installed
	if shutil.which( "unpaper" ) :
		success , output = _run_command( [ "unpaper" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ unpaper already installed: {version_line}" )
			return version_line

	# Install based on OS
	os_type = _get_os( )
	print( f"Installing unpaper on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" ] , check=True , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "unpaper" ] , check=True , timeout=300 ,
			)
		elif os_type == "Windows" :
			if shutil.which( "choco" ) :
				subprocess.run(
						[ "choco" , "install" , "-y" , "unpaper" ] , check=True , timeout=300 ,
				)
			else :
				raise ModuleNotFoundError(
						"unpaper installation failed: Chocolatey not found.\n"
						"Install Chocolatey from: https://chocolatey.org/install\n"
						"Then run: choco install unpaper\n"
						"Or download binaries from: https://github.com/unpaper/unpaper/releases" ,
				)
		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "unpaper" ] , check=True , timeout=300 )
			else :
				raise ModuleNotFoundError(
						"unpaper installation failed: Homebrew not found.\n"
						"Install Homebrew from: https://brew.sh\n"
						"Then run: brew install unpaper" ,
				)
		else :
			raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

		# Verify installation
		if shutil.which( "unpaper" ) :
			success , output = _run_command( [ "unpaper" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ unpaper installed successfully: {version_line}" )
				print( "  Verify with: unpaper --version" )
				return version_line

		raise ModuleNotFoundError(
				"unpaper installation completed but command not found in PATH" ,
		)

	except subprocess.CalledProcessError as e :
		raise ModuleNotFoundError( f"unpaper installation failed: {e}" )


def ensure_pngquant( min_version: str = "2.0" ) -> str :
	"""Ensure pngquant is installed with WSL fallback."""
	# Check if already installed
	if shutil.which( "pngquant" ) :
		success , output = _run_command( [ "pngquant" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ pngquant already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing pngquant on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "pngquant" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try chocolatey first
			if _install_via_choco( "pngquant" ) :
				print( "✓ Installed via chocolatey" )
			else :
				print( "⚠ Chocolatey failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing pngquant in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y pngquant" )
					success , output = _run_in_wsl( "pngquant --version" )
					if success :
						print( "✓ pngquant installed in WSL" )
						return "pngquant (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. pngquant will not be available." )
					print( "   PDF optimization will be disabled." )
				return "pngquant (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , "pngquant" ] , check=False , timeout=300 ,
				)

		# Verify installation
		if shutil.which( "pngquant" ) :
			success , output = _run_command( [ "pngquant" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ pngquant installed: {version_line}" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "pngquant --version" )
			if success :
				print( "✓ pngquant available in WSL" )
				return "pngquant (WSL)"

		print( "⚠ pngquant not installed, optimization will be disabled" )
		return "pngquant (unavailable)"

	except Exception as e :
		print( f"⚠ pngquant installation issue: {e}" )
		print( "   PDF optimization will be limited" )
		return "pngquant (unavailable)"


def _refresh_windows_path( ) :
	"""Refresh PATH on Windows after installation."""
	if _get_os( ) != "Windows" :
		return

	try :
		import winreg
		import os

		# Read system PATH
		key = winreg.OpenKey(
				winreg.HKEY_LOCAL_MACHINE ,
				r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment" ,
		)
		system_path = winreg.QueryValueEx( key , "PATH" )[ 0 ]
		winreg.CloseKey( key )

		# Read user PATH
		key = winreg.OpenKey( winreg.HKEY_CURRENT_USER , r"Environment" )
		user_path = winreg.QueryValueEx( key , "PATH" )[ 0 ]
		winreg.CloseKey( key )

		# Update current process PATH
		os.environ[ "PATH" ] = system_path + ";" + user_path + ";" + os.environ[ "PATH" ]
	except :
		pass


def ensure_tesseract( min_version: str = "4.0" ) -> str :
	"""Ensure Tesseract OCR is installed."""
	if shutil.which( "tesseract" ) :
		success , output = _run_command( [ "tesseract" , "--version" ] )
		if success :
			print( f"✓ Tesseract already installed: {output.strip( ).split( chr( 10 ) )[ 0 ]}" )
			return output.strip( ).split( "\n" )[ 0 ]

	os_type = _get_os( )
	print( f"Installing Tesseract OCR on {os_type}..." )

	if os_type == "Windows" :
		# Try chocolatey
		if shutil.which( "choco" ) :
			result = subprocess.run(
					[ "choco" , "install" , "-y" , "tesseract" ] ,
					capture_output=True ,
					timeout=300 ,
					check=False ,
			)
			if result.returncode == 0 :
				print( "✓ Chocolatey install completed, refreshing PATH..." )
				_refresh_windows_path( )  # Refresh PATH

				# Check again
				if shutil.which( "tesseract" ) :
					print( "✓ Tesseract verified in PATH" )
					return "tesseract (installed)"
				else :
					print( "⚠ Tesseract installed but not in PATH yet" )
					print( "   Close and reopen your terminal, then run again" )
					return "tesseract (restart required)"

	return "tesseract (unavailable)"


def ensure_ghostscript( min_version: str = "9.50" ) -> str :
	"""Ensure Ghostscript is installed."""
	gs_cmd = "gswin64c" if _get_os( ) == "Windows" else "gs"

	if shutil.which( gs_cmd ) or shutil.which( "gs" ) :
		cmd = gs_cmd if shutil.which( gs_cmd ) else "gs"
		success , output = _run_command( [ cmd , "--version" ] )
		if success :
			print( f"✓ Ghostscript already installed: {output.strip( )}" )
			return output.strip( )

	os_type = _get_os( )
	print( f"Installing Ghostscript on {os_type}..." )

	if os_type == "Windows" :
		if shutil.which( "choco" ) :
			result = subprocess.run(
					[ "choco" , "install" , "-y" , "ghostscript" ] ,
					capture_output=True ,
					timeout=300 ,
					check=False ,
			)
			if result.returncode == 0 :
				print( "✓ Chocolatey install completed, refreshing PATH..." )
				_refresh_windows_path( )

				if shutil.which( "gswin64c" ) or shutil.which( "gs" ) :
					print( "✓ Ghostscript verified in PATH" )
					return "ghostscript (installed)"
				else :
					print( "⚠ Ghostscript installed but not in PATH yet" )
					print( "   Close and reopen your terminal, then run again" )
					return "ghostscript (restart required)"

	return "ghostscript (unavailable)"


def ensure_jbig2enc( ) -> str :
	"""Ensure jbig2enc is installed for better PDF compression."""
	if shutil.which( "jbig2" ) :
		print( "✓ jbig2enc already installed" )
		return "jbig2enc (installed)"

	os_type = _get_os( )
	print( f"Installing jbig2enc on {os_type}..." )

	if os_type == "Windows" :
		if shutil.which( "choco" ) :
			subprocess.run(
					[ "choco" , "install" , "-y" , "jbig2enc" ] , check=False , timeout=300 ,
			)
			_refresh_windows_path( )
			if shutil.which( "jbig2" ) :
				print( "✓ jbig2enc installed" )
				return "jbig2enc (installed)"

	print( "⚠ jbig2enc unavailable (optional, PDFs will be larger)" )
	return "jbig2enc (unavailable)"


def ensure_cuda_gpu( min_cuda_version: str = "11.0" ) -> dict :
	"""
    Ensure CUDA-capable GPU with proper drivers is available.

    Args:
                    min_cuda_version: Minimum required CUDA version (default: "11.0")

    Returns:
                    Dict containing:
                                    - status: str ("available", "partial", "unavailable")
                                    - cuda_version: str or None
                                    - driver_version: str or None
                                    - gpu_name: str or None
                                    - gpu_memory_gb: float or None
                                    - compute_capability: str or None
                                    - message: str with detailed information

    Example verification commands:
                    nvidia-smi
                    nvcc --version
                    python -c "import torch; print(torch.cuda.is_available())"
    """
	import re

	result = {
		"status"             : "unavailable" ,
		"cuda_version"       : None ,
		"driver_version"     : None ,
		"gpu_name"           : None ,
		"gpu_memory_gb"      : None ,
		"compute_capability" : None ,
		"message"            : "" ,
	}

	print( "=" * 60 )
	print( "CUDA & GPU Detection" )
	print( "=" * 60 )

	# Check 1: nvidia-smi (driver and GPU detection)
	print( "\n[1/4] Checking NVIDIA drivers and GPU..." )
	if shutil.which( "nvidia-smi" ) :
		success , output = _run_command( [ "nvidia-smi" ] , timeout=10 )
		if success :
			print( "✓ nvidia-smi found and working" )

			# Parse driver version
			driver_match = re.search( r"Driver Version: ([\d.]+)" , output )
			if driver_match :
				result[ "driver_version" ] = driver_match.group( 1 )
				print( f"  Driver Version: {result[ 'driver_version' ]}" )

			# Parse CUDA version from driver
			cuda_match = re.search( r"CUDA Version: ([\d.]+)" , output )
			if cuda_match :
				cuda_from_driver = cuda_match.group( 1 )
				print( f"  CUDA Version (from driver): {cuda_from_driver}" )

			# Parse GPU name
			gpu_match = re.search(
					r"(?:NVIDIA|GeForce|Tesla|Quadro|RTX)\s+([^\n|]+)" , output ,
			)
			if gpu_match :
				result[ "gpu_name" ] = gpu_match.group( 0 ).strip( )
				print( f"  GPU: {result[ 'gpu_name' ]}" )

			# Parse GPU memory
			mem_match = re.search( r"(\d+)MiB\s+/\s+(\d+)MiB" , output )
			if mem_match :
				total_mem_mb = int( mem_match.group( 2 ) )
				result[ "gpu_memory_gb" ] = total_mem_mb / 1024
				print( f"  GPU Memory: {result[ 'gpu_memory_gb' ]:.1f} GB" )

			result[ "status" ] = "partial"
		else :
			print( "⚠ nvidia-smi found but failed to execute" )
			print( "  This usually means driver issues" )
	else :
		print( "✗ nvidia-smi not found" )
		print( "  NVIDIA drivers are not installed or not in PATH" )

	# Check 2: NVCC (CUDA toolkit)
	print( "\n[2/4] Checking CUDA toolkit (nvcc)..." )
	if shutil.which( "nvcc" ) :
		success , output = _run_command( [ "nvcc" , "--version" ] , timeout=10 )
		if success :
			version_match = re.search( r"release ([\d.]+)" , output )
			if version_match :
				result[ "cuda_version" ] = version_match.group( 1 )
				print( f"✓ CUDA toolkit installed: {result[ 'cuda_version' ]}" )

				# Check if meets minimum version
				try :
					cuda_major = float( result[ "cuda_version" ].split( "." )[ 0 ] )
					min_major = float( min_cuda_version.split( "." )[ 0 ] )

					if cuda_major >= min_major :
						print( f"  Meets minimum requirement ({min_cuda_version})" )
					else :
						print( f"⚠ Below minimum requirement ({min_cuda_version})" )
						print( f"  Consider upgrading CUDA toolkit" )
				except :
					pass
			else :
				print( "✓ nvcc found but couldn't parse version" )
		else :
			print( "⚠ nvcc found but failed to execute" )
	else :
		print( "✗ nvcc not found" )
		print( "  CUDA toolkit is not installed or not in PATH" )

	# Check 3: PyTorch CUDA support
	print( "\n[3/4] Checking PyTorch CUDA support..." )
	try :
		import torch

		if torch.cuda.is_available( ) :
			print( f"✓ PyTorch CUDA available: {torch.version.cuda}" )
			print( f"  PyTorch version: {torch.__version__}" )
			print( f"  CUDA devices found: {torch.cuda.device_count( )}" )

			if torch.cuda.device_count( ) > 0 :
				device_name = torch.cuda.get_device_name( 0 )
				print( f"  Primary GPU: {device_name}" )

				# Get compute capability
				props = torch.cuda.get_device_properties( 0 )
				compute_cap = f"{props.major}.{props.minor}"
				result[ "compute_capability" ] = compute_cap
				print( f"  Compute Capability: {compute_cap}" )

				# Update GPU info if not found earlier
				if not result[ "gpu_name" ] :
					result[ "gpu_name" ] = device_name
				if not result[ "gpu_memory_gb" ] :
					result[ "gpu_memory_gb" ] = props.total_memory / (1024 ** 3)
					print( f"  GPU Memory: {result[ 'gpu_memory_gb' ]:.1f} GB" )

				result[ "status" ] = "available"
		else :
			print( "✗ PyTorch found but CUDA not available" )
			print( "  You may have CPU-only PyTorch installed" )
			print(
					"  Reinstall with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
			)
	except ImportError :
		print( "⚠ PyTorch not installed" )
		print(
				"  Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
		)
	except Exception as e :
		print( f"⚠ Error checking PyTorch: {e}" )

	# Check 4: cuDNN (optional but recommended)
	print( "\n[4/4] Checking cuDNN..." )
	try :
		import torch

		if torch.cuda.is_available( ) :
			if torch.backends.cudnn.is_available( ) :
				cudnn_version = torch.backends.cudnn.version( )
				print( f"✓ cuDNN available: {cudnn_version}" )
			else :
				print( "⚠ cuDNN not available" )
				print( "  Some operations may be slower" )
	except :
		print( "⚠ Could not check cuDNN status" )

	# Generate summary message
	print( "\n" + "=" * 60 )
	print( "Summary" )
	print( "=" * 60 )

	if result[ "status" ] == "available" :
		result[ "message" ] = (
			f"✓ CUDA GPU fully available: {result[ 'gpu_name' ]} ({result[ 'gpu_memory_gb' ]:.1f} GB)"
		)
		print( f"✓ Status: FULLY AVAILABLE" )
		print( f"  GPU: {result[ 'gpu_name' ]}" )
		print( f"  Memory: {result[ 'gpu_memory_gb' ]:.1f} GB" )
		print( f"  Driver: {result[ 'driver_version' ]}" )
		print( f"  CUDA: {result[ 'cuda_version' ] or 'N/A (using driver version)'}" )
		print( f"  Compute Capability: {result[ 'compute_capability' ]}" )

	elif result[ "status" ] == "partial" :
		result[ "message" ] = "⚠ GPU detected but CUDA support incomplete"
		print( f"⚠ Status: PARTIAL" )
		print( f"  GPU detected: {result[ 'gpu_name' ] or 'Unknown'}" )
		print( f"  Driver: {result[ 'driver_version' ] or 'Unknown'}" )

		# Provide guidance
		print( "\n  To enable full CUDA support:" )
		if not result[ "cuda_version" ] :
			print(
					"  1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads" ,
			)

		try :
			import torch

			if not torch.cuda.is_available( ) :
				print( "  2. Reinstall PyTorch with CUDA:" )
				print(
						"     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
				)
		except ImportError :
			print( "  2. Install PyTorch with CUDA:" )
			print(
					"     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
			)

	else :
		result[ "message" ] = "✗ CUDA GPU not available"
		print( f"✗ Status: NOT AVAILABLE" )

		os_type = _get_os( )
		print( f"\n  System: {os_type}" )
		print( "  To enable CUDA GPU support:" )

		if os_type == "Windows" :
			print( "\n  1. Install NVIDIA drivers:" )
			print( "     - Visit: https://www.nvidia.com/download/index.aspx" )
			print( "     - Or use GeForce Experience for automatic updates" )
			print( "\n  2. Install CUDA Toolkit:" )
			print( "     - Download from: https://developer.nvidia.com/cuda-downloads" )
			print( "     - Recommended version: 11.8 or 12.1" )
			print( "\n  3. Install PyTorch with CUDA:" )
			print(
					"     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
			)

		elif os_type == "Linux" :
			print( "\n  1. Install NVIDIA drivers:" )
			print( "     sudo apt update" )
			print( "     sudo apt install nvidia-driver-535" )  # Recent stable version
			print( "\n  2. Install CUDA Toolkit:" )
			print(
					"     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb" ,
			)
			print( "     sudo dpkg -i cuda-keyring_1.0-1_all.deb" )
			print( "     sudo apt-get update" )
			print( "     sudo apt-get install cuda" )
			print( "\n  3. Install PyTorch with CUDA:" )
			print(
					"     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" ,
			)

		elif os_type == "Darwin" :
			print( "\n  ⚠ CUDA is not supported on macOS" )
			print(
					"  Apple Silicon Macs can use Metal Performance Shaders (MPS) with PyTorch" ,
			)
			print( "  For MPS support:" )
			print( "     pip install torch torchvision" )
			print( "  Then use: torch.device('mps') instead of torch.device('cuda')" )

	print( "=" * 60 )

	return result


# Helper function to get detailed CUDA info
def get_cuda_info( ) -> dict :
	"""
    Get detailed CUDA and GPU information.

    Returns:
                    Dict with comprehensive CUDA/GPU details
    """
	info = ensure_cuda_gpu( )

	# Add additional PyTorch-specific info if available
	try :
		import torch

		if torch.cuda.is_available( ) :
			info[ "pytorch_cuda_version" ] = torch.version.cuda
			info[ "pytorch_version" ] = torch.__version__
			info[ "cudnn_version" ] = (
				torch.backends.cudnn.version( )
				if torch.backends.cudnn.is_available( )
				else None
			)
			info[ "device_count" ] = torch.cuda.device_count( )

			# Per-device info
			devices = [ ]
			for i in range( torch.cuda.device_count( ) ) :
				props = torch.cuda.get_device_properties( i )
				devices.append(
						{
							"index"                 : i ,
							"name"                  : torch.cuda.get_device_name( i ) ,
							"compute_capability"    : f"{props.major}.{props.minor}" ,
							"total_memory_gb"       : props.total_memory / (1024 ** 3) ,
							"multi_processor_count" : props.multi_processor_count ,
						} ,
				)
			info[ "devices" ] = devices
	except :
		pass

	return info


def ensure_exiftool( min_version: str = "12.0" ) -> str :
	"""
    Ensure ExifTool is installed for reading/writing file metadata.

    Args:
        min_version: Minimum required version (default: "12.0")

    Returns:
        Installed version string

    Raises:
        ModuleNotFoundError: If ExifTool cannot be installed or verified

    Example verification commands:
        exiftool -ver
        exiftool -h
    """
	# Check if already installed
	if shutil.which( "exiftool" ) :
		success , output = _run_command( [ "exiftool" , "-ver" ] )
		if success :
			version = output.strip( )
			print( f"✓ ExifTool already installed: version {version}" )
			return f"exiftool {version}"

	os_type = _get_os( )
	print( f"Installing ExifTool on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "libimage-exiftool-perl" ] ,
					check=False ,
					timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try chocolatey first
			if _install_via_choco( "exiftool" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing ExifTool in WSL..." )
					_run_in_wsl(
							"sudo apt update -y && sudo apt install -y libimage-exiftool-perl" ,
					)
					success , output = _run_in_wsl( "exiftool -ver" )
					if success :
						print( "✓ ExifTool installed in WSL" )
						return "exiftool (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. ExifTool not installed." )
					print( "  Manual install: https://exiftool.org/" )
				return "exiftool (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , "exiftool" ] , check=False , timeout=300 ,
				)
			else :
				raise ModuleNotFoundError(
						"ExifTool installation failed: Homebrew not found.\n"
						"Install Homebrew from: https://brew.sh\n"
						"Then run: brew install exiftool\n"
						"Or download from: https://exiftool.org/" ,
				)

		# Verify installation
		if shutil.which( "exiftool" ) :
			success , output = _run_command( [ "exiftool" , "-ver" ] )
			if success :
				version = output.strip( )
				print( f"✓ ExifTool installed successfully: version {version}" )
				print( "  Verify with: exiftool -ver" )
				return f"exiftool {version}"

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "exiftool -ver" )
			if success :
				print( "✓ ExifTool available in WSL" )
				return "exiftool (WSL)"

		print( "⚠ ExifTool not installed, metadata features unavailable" )
		return "exiftool (unavailable)"

	except Exception as e :
		print( f"⚠ ExifTool installation error: {e}" )
		print( "   Download manually from: https://exiftool.org/" )
		return "exiftool (unavailable)"


def ensure_7zip( min_version: str = "16.0" ) -> str :
	"""
    Ensure 7-Zip is installed for file compression/encryption.

    Args:
                    min_version: Minimum required version (default: "16.0")

    Returns:
                    Installed version string

    Example verification commands:
                    7z --version
                    7z i (shows detailed info)
    """
	# Check if already installed
	cmd = "7z" if shutil.which( "7z" ) else "7za" if shutil.which( "7za" ) else None

	if cmd :
		success , output = _run_command( [ cmd ] , timeout=5 )
		if success and "7-Zip" in output :
			# Parse version from output
			import re

			version_match = re.search( r"7-Zip.*?(\d+\.\d+)" , output )
			if version_match :
				version = version_match.group( 1 )
				print( f"✓ 7-Zip already installed: version {version}" )
				return f"7-Zip {version}"
			print( f"✓ 7-Zip already installed" )
			return "7-Zip (installed)"

	os_type = _get_os( )
	print( f"Installing 7-Zip on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "p7zip-full" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try winget first
			if _install_via_winget( "7zip.7zip" ) :
				print( "✓ Installed via winget" )
				_refresh_windows_path( )
			# Try chocolatey
			elif _install_via_choco( "7zip" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			# Fallback to WSL
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing 7-Zip in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y p7zip-full" )
					success , output = _run_in_wsl( "7z" )
					if success :
						print( "✓ 7-Zip installed in WSL" )
						return "7-Zip (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. 7-Zip not installed." )
					print( "  Download from: https://www.7-zip.org/download.html" )
				return "7-Zip (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "p7zip" ] , check=False , timeout=300 )
			else :
				print( "⚠ Homebrew not found" )
				print( "  Install with: brew install p7zip" )
				return "7-Zip (unavailable)"

		# Verify installation
		cmd = "7z" if shutil.which( "7z" ) else "7za" if shutil.which( "7za" ) else None
		if cmd :
			success , output = _run_command( [ cmd ] , timeout=5 )
			if success :
				import re

				version_match = re.search( r"7-Zip.*?(\d+\.\d+)" , output )
				version_str = version_match.group( 1 ) if version_match else "unknown"
				print( f"✓ 7-Zip installed successfully: version {version_str}" )
				print( "  Verify with: 7z --version" )
				return f"7-Zip {version_str}"

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "7z" )
			if success :
				print( "✓ 7-Zip available in WSL" )
				return "7-Zip (WSL)"

		print( "⚠ 7-Zip not installed, encryption features unavailable" )
		return "7-Zip (unavailable)"

	except Exception as e :
		print( f"⚠ 7-Zip installation error: {e}" )
		print( "   Download manually from: https://www.7-zip.org/" )
		return "7-Zip (unavailable)"


def ensure_keepassxc( min_version: str = "2.6" ) -> str :
	"""
    Ensure KeePassXC is installed for password database management.

    Args:
                    min_version: Minimum required version (default: "2.6")

    Returns:
                    Installed version string

    Example verification commands:
                    keepassxc-cli --version
                    keepassxc-cli db-info database.kdbx
    """
	# Check if already installed
	if shutil.which( "keepassxc-cli" ) :
		success , output = _run_command( [ "keepassxc-cli" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ KeePassXC already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing KeePassXC on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "keepassxc" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try winget first
			if _install_via_winget( "KeePassXCTeam.KeePassXC" ) :
				print( "✓ Installed via winget" )
				_refresh_windows_path( )
			# Try chocolatey
			elif _install_via_choco( "keepassxc" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			# Fallback to WSL
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing KeePassXC in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y keepassxc" )
					success , output = _run_in_wsl( "keepassxc-cli --version" )
					if success :
						print( "✓ KeePassXC installed in WSL" )
						return "KeePassXC (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. KeePassXC not installed." )
					print( "  Download from: https://keepassxc.org/download/" )
				return "KeePassXC (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run(
						[ "brew" , "install" , "--cask" , "keepassxc" ] , check=False , timeout=300 ,
				)
			else :
				print( "⚠ Homebrew not found" )
				print( "  Install with: brew install --cask keepassxc" )
				return "KeePassXC (unavailable)"

		# Verify installation
		if shutil.which( "keepassxc-cli" ) :
			success , output = _run_command( [ "keepassxc-cli" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ KeePassXC installed successfully: {version_line}" )
				print( "  Verify with: keepassxc-cli --version" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "keepassxc-cli --version" )
			if success :
				print( "✓ KeePassXC available in WSL" )
				return "KeePassXC (WSL)"

		print( "⚠ KeePassXC not installed, password management unavailable" )
		return "KeePassXC (unavailable)"

	except Exception as e :
		print( f"⚠ KeePassXC installation error: {e}" )
		print( "   Download manually from: https://keepassxc.org/download/" )
		return "KeePassXC (unavailable)"


def ensure_isync( min_version: str = "1.3" ) -> str :
	"""
    Ensure isync (mbsync) is installed for IMAP mailbox synchronization.

    Args:
                    min_version: Minimum required version (default: "1.3")

    Returns:
                    Installed version string

    Example verification commands:
                    mbsync --version
                    mbsync -h
    """
	# Check if already installed
	if shutil.which( "mbsync" ) :
		success , output = _run_command( [ "mbsync" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ isync (mbsync) already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing isync (mbsync) on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "isync" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try winget first
			if _install_via_winget( "isync.isync" ) :
				print( "✓ Installed via winget" )
				_refresh_windows_path( )
			# Try chocolatey
			elif _install_via_choco( "isync" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			# Fallback to WSL
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing isync in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y isync" )
					success , output = _run_in_wsl( "mbsync --version" )
					if success :
						print( "✓ isync (mbsync) installed in WSL" )
						return "isync (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. isync not installed." )
					print( "  Build from source: https://isync.sourceforge.io/" )
				return "isync (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "isync" ] , check=False , timeout=300 )
			else :
				print( "⚠ Homebrew not found" )
				print( "  Install with: brew install isync" )
				return "isync (unavailable)"

		# Verify installation
		if shutil.which( "mbsync" ) :
			success , output = _run_command( [ "mbsync" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ isync (mbsync) installed successfully: {version_line}" )
				print( "  Verify with: mbsync --version" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "mbsync --version" )
			if success :
				print( "✓ isync (mbsync) available in WSL" )
				return "isync (WSL)"

		print( "⚠ isync (mbsync) not installed, email sync unavailable" )
		return "isync (unavailable)"

	except Exception as e :
		print( f"⚠ isync installation error: {e}" )
		print( "   Visit: https://isync.sourceforge.io/" )
		return "isync (unavailable)"


def ensure_rspamd( min_version: str = "2.0" ) -> str :
	"""
    Ensure rspamd is installed for spam/phishing detection (optional).

    Args:
                    min_version: Minimum required version (default: "2.0")

    Returns:
                    Installed version string

    Example verification commands:
                    rspamd --version
                    rspamadm --help
    """
	# Check if already installed
	if shutil.which( "rspamd" ) :
		success , output = _run_command( [ "rspamd" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ rspamd already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing rspamd on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "rspamd" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			print( "⚠ rspamd has limited Windows support, using WSL..." )
			if _ensure_wsl( ) :
				print( "Installing rspamd in WSL..." )
				_run_in_wsl( "sudo apt update -y && sudo apt install -y rspamd" )
				success , output = _run_in_wsl( "rspamd --version" )
				if success :
					print( "✓ rspamd installed in WSL" )
					return "rspamd (WSL)"
				else :
					print( "⚠ WSL install incomplete" )
			else :
				print( "⚠ WSL unavailable. rspamd not installed." )
				print( "  This is optional - email will work without it" )
			return "rspamd (unavailable - optional)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "rspamd" ] , check=False , timeout=300 )
			else :
				print( "⚠ Homebrew not found" )
				print( "  Install with: brew install rspamd" )
				return "rspamd (unavailable - optional)"

		# Verify installation
		if shutil.which( "rspamd" ) :
			success , output = _run_command( [ "rspamd" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ rspamd installed successfully: {version_line}" )
				print( "  Verify with: rspamd --version" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "rspamd --version" )
			if success :
				print( "✓ rspamd available in WSL" )
				return "rspamd (WSL)"

		print( "⚠ rspamd not installed (optional - spam detection unavailable)" )
		return "rspamd (unavailable - optional)"

	except Exception as e :
		print( f"⚠ rspamd installation error: {e}" )
		print( "   This is optional - email will work without it" )
		return "rspamd (unavailable - optional)"


def ensure_wget( min_version: str = "1.19" ) -> str :
	"""
    Ensure wget is installed for downloading files.

    Args:
                    min_version: Minimum required version (default: "1.19")

    Returns:
                    Installed version string

    Example verification commands:
                    wget --version
    """
	# Check if already installed
	if shutil.which( "wget" ) :
		success , output = _run_command( [ "wget" , "--version" ] )
		if success :
			version_line = output.strip( ).split( "\n" )[ 0 ]
			print( f"✓ wget already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing wget on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "wget" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# wget is built into Windows 10+ PowerShell, but the GNU version is better
			if _install_via_winget( "GNU.Wget2" ) :
				print( "✓ Installed via winget" )
				_refresh_windows_path( )
			elif _install_via_choco( "wget" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing wget in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y wget" )
					success , output = _run_in_wsl( "wget --version" )
					if success :
						print( "✓ wget installed in WSL" )
						return "wget (WSL)"
				return "wget (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "wget" ] , check=False , timeout=300 )
			else :
				print( "⚠ Homebrew not found" )
				print( "  Install with: brew install wget" )
				return "wget (unavailable)"

		# Verify installation
		if shutil.which( "wget" ) :
			success , output = _run_command( [ "wget" , "--version" ] )
			if success :
				version_line = output.strip( ).split( "\n" )[ 0 ]
				print( f"✓ wget installed successfully: {version_line}" )
				print( "  Verify with: wget --version" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "wget --version" )
			if success :
				print( "✓ wget available in WSL" )
				return "wget (WSL)"

		print( "⚠ wget not installed" )
		return "wget (unavailable)"

	except Exception as e :
		print( f"⚠ wget installation error: {e}" )
		return "wget (unavailable)"


def ensure_git( min_version: str = "2.0" ) -> str :
	"""
    Ensure git is installed for version control.

    Args:
                    min_version: Minimum required version (default: "2.0")

    Returns:
                    Installed version string

    Example verification commands:
                    git --version
    """
	# Check if already installed
	if shutil.which( "git" ) :
		success , output = _run_command( [ "git" , "--version" ] )
		if success :
			version_line = output.strip( )
			print( f"✓ git already installed: {version_line}" )
			return version_line

	os_type = _get_os( )
	print( f"Installing git on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "git" ] , check=False , timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try winget first
			if _install_via_winget( "Git.Git" ) :
				print( "✓ Installed via winget" )
				_refresh_windows_path( )
			# Try chocolatey
			elif _install_via_choco( "git" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			# Fallback to WSL
			else :
				print( "⚠ Native install failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing git in WSL..." )
					_run_in_wsl( "sudo apt update -y && sudo apt install -y git" )
					success , output = _run_in_wsl( "git --version" )
					if success :
						print( "✓ git installed in WSL" )
						return "git (WSL)"
				return "git (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "git" ] , check=False , timeout=300 )
			else :
				# Git is often pre-installed on macOS via Xcode tools
				print( "⚠ Homebrew not found, but git may be available via Xcode" )
				print( "  Install with: xcode-select --install" )
				print( "  Or: brew install git" )
				return "git (unavailable)"

		# Verify installation
		if shutil.which( "git" ) :
			success , output = _run_command( [ "git" , "--version" ] )
			if success :
				version_line = output.strip( )
				print( f"✓ git installed successfully: {version_line}" )
				print( "  Verify with: git --version" )
				return version_line

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "git --version" )
			if success :
				print( "✓ git available in WSL" )
				return "git (WSL)"

		print( "⚠ git not installed" )
		return "git (unavailable)"

	except Exception as e :
		print( f"⚠ git installation error: {e}" )
		return "git (unavailable)"


def ensure_oauth2ms( min_version: str = "0.1" ) -> str :
	"""
    Ensure oauth2ms Python package is installed for OAuth2 authentication.

    Args:
                    min_version: Minimum required version (default: "0.1")

    Returns:
                    Installed version string

    Example verification commands:
                    oauth2ms --version
                    python -c "import oauth2ms; print(oauth2ms.__version__)"
    """
	# Check if already installed
	if shutil.which( "oauth2ms" ) :
		success , output = _run_command( [ "oauth2ms" , "--version" ] )
		if success :
			version_line = output.strip( )
			print( f"✓ oauth2ms already installed: {version_line}" )
			return version_line

	# Try importing as Python module
	try :
		import oauth2ms

		version = getattr( oauth2ms , "__version__" , "unknown" )
		print( f"✓ oauth2ms already installed (Python module): version {version}" )
		return f"oauth2ms {version}"
	except ImportError :
		pass

	os_type = _get_os( )
	print( f"Installing oauth2ms on {os_type}..." )

	try :
		# Install via pip
		print( "Installing oauth2ms via pip..." )
		subprocess.run(
				[ sys.executable , "-m" , "pip" , "install" , "oauth2ms" ] ,
				check=False ,
				timeout=300 ,
		)

		# Verify installation
		if shutil.which( "oauth2ms" ) :
			success , output = _run_command( [ "oauth2ms" , "--version" ] )
			if success :
				version_line = output.strip( )
				print( f"✓ oauth2ms installed successfully: {version_line}" )
				print( "  Verify with: oauth2ms --version" )
				return version_line

		# Try importing as module
		try :
			import oauth2ms

			version = getattr( oauth2ms , "__version__" , "unknown" )
			print( f"✓ oauth2ms installed successfully: version {version}" )
			print( f"  Verify with: {sys.executable} -m oauth2ms --version" )
			return f"oauth2ms {version}"
		except ImportError :
			pass

		# Windows WSL fallback
		if os_type == "Windows" :
			if _ensure_wsl( ) :
				print( "Installing oauth2ms in WSL..." )
				_run_in_wsl( "pip3 install oauth2ms" )
				success , output = _run_in_wsl( "oauth2ms --version" )
				if success :
					print( "✓ oauth2ms installed in WSL" )
					return "oauth2ms (WSL)"

		print( "⚠ oauth2ms not installed" )
		return "oauth2ms (unavailable)"

	except Exception as e :
		print( f"⚠ oauth2ms installation error: {e}" )
		print( "   Try manually: pip install oauth2ms" )
		return "oauth2ms (unavailable)"


def ensure_mutt_oauth2( install_dir: str = None ) -> str :
	"""
    Ensure mutt_oauth2.py script is downloaded and available.

    Args:
                    install_dir: Directory to install the script (default: ~/bin)

    Returns:
                    Path to the mutt_oauth2.py script

    Example verification commands:
                    ls -la ~/bin/mutt_oauth2.py
                    python ~/bin/mutt_oauth2.py --help
    """
	# Set install directory
	if install_dir is None :
		install_dir = os.path.expanduser( "~/bin" )

	script_path = os.path.join( install_dir , "mutt_oauth2.py" )

	# Check if already installed
	if os.path.exists( script_path ) :
		# Verify it's executable and valid Python
		if os.access( script_path , os.X_OK ) :
			print( f"✓ mutt_oauth2.py already installed: {script_path}" )
			return script_path
		else :
			print( f"⚠ mutt_oauth2.py exists but not executable, fixing..." )
			try :
				os.chmod( script_path , 0o755 )
				print( f"✓ Made executable: {script_path}" )
				return script_path
			except :
				pass

	os_type = _get_os( )
	print( f"Installing mutt_oauth2.py on {os_type}..." )

	try :
		# Create install directory
		os.makedirs( install_dir , exist_ok=True )

		# URL to download from
		script_url = "https://raw.githubusercontent.com/alejandrogallo/mutt-oauth2/master/mutt_oauth2.py"

		print( f"Downloading mutt_oauth2.py..." )
		print( f"  From: {script_url}" )
		print( f"  To: {script_path}" )

		# Try using wget first
		if shutil.which( "wget" ) :
			success , _ = _run_command(
					[ "wget" , "-O" , script_path , script_url ] , timeout=60 ,
			)
			if not success :
				print( "⚠ wget download failed, trying curl..." )

		# Try curl if wget failed or unavailable
		if not os.path.exists( script_path ) and shutil.which( "curl" ) :
			success , _ = _run_command(
					[ "curl" , "-L" , "-o" , script_path , script_url ] , timeout=60 ,
			)
			if not success :
				print( "⚠ curl download failed, trying Python urllib..." )

		# Fallback to Python urllib
		if not os.path.exists( script_path ) :
			import urllib.request

			print( "Downloading with Python urllib..." )
			urllib.request.urlretrieve( script_url , script_path )

		# Verify download
		if os.path.exists( script_path ) and os.path.getsize( script_path ) > 1000 :
			# Make executable
			os.chmod( script_path , 0o755 )

			print( f"✓ mutt_oauth2.py downloaded successfully: {script_path}" )
			print( f"  Verify with: ls -la {script_path}" )
			print( f"  Test with: python {script_path} --help" )

			return script_path

		raise Exception( "Download completed but file is invalid or too small" )

	except Exception as e :
		# Clean up partial download
		if os.path.exists( script_path ) :
			try :
				os.remove( script_path )
			except :
				pass

		print( f"⚠ mutt_oauth2.py download failed: {e}" )
		print( f"   Manual download:" )
		print( f"   mkdir -p {install_dir}" )
		print( f"   wget -O {script_path} {script_url}" )
		print( f"   chmod +x {script_path}" )
		return "mutt_oauth2.py (unavailable)"


def ensure_poppler( min_version: str = "0.86" ) -> str :
	"""
    Ensure Poppler is installed for PDF rendering and conversion.
    Required by pdf2image library.

    Args:
        min_version: Minimum required version (default: "0.86")

    Returns:
        Installed version string

    Raises:
        ModuleNotFoundError: If Poppler cannot be installed or verified

    Example verification commands:
        pdftoppm -v
        pdfinfo -v
        pdftotext -v
    """
	# Check if already installed (check for pdftoppm which is part of poppler-utils)
	if shutil.which( "pdftoppm" ) :
		success , output = _run_command( [ "pdftoppm" , "-v" ] )
		if success :
			# Parse version from output
			import re

			version_match = re.search( r"(\d+\.\d+)" , output )
			if version_match :
				version = version_match.group( 1 )
				print( f"✓ Poppler already installed: version {version}" )
				return f"poppler {version}"
			print( f"✓ Poppler already installed" )
			return "poppler (installed)"

	os_type = _get_os( )
	print( f"Installing Poppler on {os_type}..." )

	try :
		if os_type == "Linux" :
			subprocess.run( [ "sudo" , "apt" , "update" , "-y" ] , check=False , timeout=60 )
			subprocess.run(
					[ "sudo" , "apt" , "install" , "-y" , "poppler-utils" ] ,
					check=False ,
					timeout=300 ,
			)

		elif os_type == "Windows" :
			# Try chocolatey first
			if _install_via_choco( "poppler" ) :
				print( "✓ Installed via chocolatey" )
				_refresh_windows_path( )
			else :
				print( "⚠ Chocolatey failed, using WSL..." )
				if _ensure_wsl( ) :
					print( "Installing Poppler in WSL..." )
					_run_in_wsl(
							"sudo apt update -y && sudo apt install -y poppler-utils" ,
					)
					success , output = _run_in_wsl( "pdftoppm -v" )
					if success :
						print( "✓ Poppler installed in WSL" )
						print(
								"\n⚠️ IMPORTANT: pdf2image will use WSL for PDF conversion" ,
						)
						print( "   This may be slower than native installation" )
						return "poppler (WSL)"
					else :
						print( "⚠ WSL install incomplete" )
				else :
					print( "⚠ WSL unavailable. Poppler not installed." )
					print( "\n📦 Manual Installation:" )
					print(
							"   1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/" ,
					)
					print( "   2. Extract to C:\\Program Files\\poppler" )
					print( "   3. Add C:\\Program Files\\poppler\\Library\\bin to PATH" )
					raise ModuleNotFoundError(
							"Poppler installation failed.\n"
							"Download from: https://github.com/oschwartz10612/poppler-windows/releases/" ,
					)
				return "poppler (unavailable)"

		elif os_type == "Darwin" :
			if shutil.which( "brew" ) :
				subprocess.run( [ "brew" , "install" , "poppler" ] , check=False , timeout=300 )
			else :
				raise ModuleNotFoundError(
						"Poppler installation failed: Homebrew not found.\n"
						"Install Homebrew from: https://brew.sh\n"
						"Then run: brew install poppler" ,
				)

		else :
			raise ModuleNotFoundError( f"Unsupported OS: {os_type}" )

		# Verify installation
		if shutil.which( "pdftoppm" ) :
			success , output = _run_command( [ "pdftoppm" , "-v" ] )
			if success :
				import re

				version_match = re.search( r"(\d+\.\d+)" , output )
				version_str = version_match.group( 1 ) if version_match else "unknown"
				print( f"✓ Poppler installed successfully: version {version_str}" )
				print( "  Verify with: pdftoppm -v" )
				print(
						'  Test pdf2image with: python -c "from pdf2image import convert_from_path"' ,
				)
				return f"poppler {version_str}"

		# Check WSL as fallback
		if os_type == "Windows" :
			success , output = _run_in_wsl( "pdftoppm -v" )
			if success :
				print( "✓ Poppler available in WSL" )
				return "poppler (WSL)"

		raise ModuleNotFoundError(
				"Poppler installation completed but pdftoppm not found in PATH" ,
		)

	except subprocess.CalledProcessError as e :
		raise ModuleNotFoundError( f"Poppler installation failed: {e}" )


def ensure_gmvault( ) :
	"""Install Gmvault if not present"""
	print( "\n📦 Checking Gmvault..." )

	if check_command_exists( "gmvault" ) :
		print( "   ✅ Gmvault already installed" )
		return True

	print( "   📥 Installing Gmvault via pip..." )
	success , output = run_cmd(
			f"{sys.executable} -m pip install gmvault" , "Installing gmvault" , capture=False ,
	)

	if success :
		print( "   ✅ Gmvault installed successfully" )
		return True
	else :
		print( "   ❌ Failed to install Gmvault" )
		return False


def ensure_gyb( ) :
	"""Check/Install GYB"""
	print( "\n📦 Checking GYB (Got Your Back)..." )

	if check_command_exists( "gyb" ) :
		print( "   ✅ GYB already installed" )
		return True

	system = platform.system( )

	if system == "Windows" :
		print( "   ⚠️  GYB not found. Please install manually:" )
		print(
				"   👉 Download: https://github.com/GAM-team/got-your-back/releases/Setup.msi" ,
		)
		print( "   👉 Run the installer, then run this script again" )
		return False

	elif system in [ "Linux" , "Darwin" ] :  # Darwin = macOS
		print( "   📥 Installing GYB..." )
		success , _ = run_cmd(
				"bash <(curl -s -S -L https://git.io/gyb-install)" ,
				"Installing GYB" ,
				capture=False ,
		)

		if success :
			print( "   ✅ GYB installed successfully" )
			return True
		else :
			print( "   ❌ Failed to install GYB automatically" )
			print(
					"   👉 Try manual install: bash <(curl -s -S -L https://git.io/gyb-install)" ,
			)
			return False

	return False


def ensure_oauth( email_address , tool_name , test_cmd ) :
	"""Guide user through OAuth setup if needed"""
	print( f"\n🔐 Checking OAuth for {email_address}..." )

	# Try to run a test command
	success , output = run_cmd( test_cmd , f"Testing {tool_name}" , capture=True )

	if "oauth" in output.lower( ) or "authori" in output.lower( ) or not success :
		print( f"   ⚠️  OAuth not set up yet for {email_address}" )
		print( f"\n   {'=' * 50}" )
		print( f"   🌐 OAuth Setup Required" )
		print( f"   {'=' * 50}" )
		print( f"   This will open your browser to authorize {tool_name}." )
		print( f"   You only need to do this ONCE.\n" )

		input( "   Press ENTER to start OAuth setup..." )

		# Run the auth command interactively
		result = subprocess.run( test_cmd , shell=True )

		if result.returncode == 0 :
			print( f"   ✅ OAuth set up successfully!" )
			return True
		else :
			print( f"   ⚠️  OAuth setup incomplete. You can try again by running:" )
			print( f"   {test_cmd}" )
			return False
	else :
		print( f"   ✅ OAuth already configured" )
		return True


def install_noteshrink( ) :
	"""Clone and setup noteshrink."""
	if Path( "noteshrink" ).exists( ) :
		print( "✓ noteshrink already installed" )
		return

	print( "Installing noteshrink..." )
	subprocess.run(
			[ "git" , "clone" , "https://github.com/mzucker/noteshrink.git" ] , check=True ,
	)
	print( "✓ noteshrink installed" )


def ensure_bitsandbytes( min_version: str = "0.41.0" ) -> str :
	"""
    Ensure bitsandbytes is installed for model quantization (int8 / nf4).

    Required by VisualProcessor for quantized model tiers on GPUs with
    limited VRAM (< 18 GB). Uses --prefer-binary on all platforms to pull
    pre-compiled wheels with CUDA support rather than building from source,
    which fails on Windows without a full MSVC toolchain.

    Args:
        min_version: Minimum required version (default: "0.41.0" — first
                     stable Windows CUDA release).

    Returns:
        Installed version string, or "(unavailable)" if install fails.

    Example verification commands:
        python -c "import bitsandbytes; print(bitsandbytes.__version__)"
        python -m bitsandbytes  (prints CUDA / build diagnostics)
    """
	import importlib.metadata
	import re

	def _get_installed_version( ) -> str | None :
		"""Return installed bitsandbytes version string, or None if not found."""
		try :
			return importlib.metadata.version( "bitsandbytes" )
		except importlib.metadata.PackageNotFoundError :
			return None

	def _version_meets_minimum( version: str , minimum: str ) -> bool :
		"""Return True if version >= minimum using simple numeric comparison."""
		try :
			v = tuple( int( x ) for x in re.findall( r"\d+" , version ) )
			m = tuple( int( x ) for x in re.findall( r"\d+" , minimum ) )
			return v >= m
		except Exception :
			return False

	# ── Already installed? ────────────────────────────────────────────────
	installed = _get_installed_version( )
	if installed :
		if _version_meets_minimum( installed , min_version ) :
			print( f"✓ bitsandbytes already installed: version {installed}" )
			return f"bitsandbytes {installed}"
		else :
			print(
					f"⚠ bitsandbytes {installed} found but below minimum "
					f"({min_version}), upgrading..." ,
			)

	os_type = _get_os( )
	print( f"Installing bitsandbytes on {os_type}..." )

	# --prefer-binary skips source builds and grabs the pre-compiled wheel.
	# This is critical on Windows where building from source requires MSVC.
	base_cmd = [
		sys.executable , "-m" , "pip" , "install" ,
		"--prefer-binary" ,
		f"bitsandbytes>={min_version}" ,
	]

	try :
		if os_type == "Windows" :
			# The standard PyPI wheel has Windows CUDA support since ~0.41.
			# If that fails, fall back to the community-maintained Windows build.
			print( "  Trying official PyPI wheel (--prefer-binary)..." )
			result = subprocess.run(
					base_cmd ,
					capture_output=True ,
					text=True ,
					timeout=300 ,
					check=False ,
			)

			if result.returncode != 0 :
				print( "  ⚠ Official wheel failed, trying HuggingFace Windows index..." )
				result = subprocess.run(
						[
							sys.executable , "-m" , "pip" , "install" ,
							"--prefer-binary" ,
							f"bitsandbytes>={min_version}" ,
							"--index-url" ,
							"https://huggingface.github.io/bitsandbytes-windows-webui" ,
						] ,
						capture_output=True ,
						text=True ,
						timeout=300 ,
						check=False ,
				)

				if result.returncode != 0 :
					print( f"  ✗ Both install attempts failed." )
					print( f"    Output: {result.stdout + result.stderr}" )
					print(
							"    Manual fix:\n"
							"      pip install bitsandbytes --prefer-binary\n"
							"    Or from HF index:\n"
							"      pip install bitsandbytes "
							"--index-url https://huggingface.github.io/bitsandbytes-windows-webui" ,
					)
					return "bitsandbytes (unavailable)"

		else :
			# Linux / macOS — standard install is reliable
			result = subprocess.run(
					base_cmd ,
					capture_output=True ,
					text=True ,
					timeout=300 ,
					check=False ,
			)

			if result.returncode != 0 :
				print( f"  ✗ Install failed: {result.stdout + result.stderr}" )
				return "bitsandbytes (unavailable)"

		# ── Verify ────────────────────────────────────────────────────────
		installed = _get_installed_version( )
		if installed :
			print( f"✓ bitsandbytes installed successfully: version {installed}" )
			print(
					f"  Verify with: "
					f"{sys.executable} -c \"import bitsandbytes; print(bitsandbytes.__version__)\"" ,
			)

			# Quick CUDA sanity check — non-fatal, informational only
			try :
				import torch
				if torch.cuda.is_available( ) :
					import bitsandbytes as bnb  # noqa: F401
					print( "  ✓ bitsandbytes imported successfully with CUDA available" )
				else :
					print(
							"  ⚠ bitsandbytes installed but no CUDA device found — "
							"quantization will not be available at runtime" ,
					)
			except Exception as e :
				print( f"  ⚠ Post-install CUDA check failed: {e}" )
				print( "    This is not always fatal — try running your code to confirm." )

			return f"bitsandbytes {installed}"

		print( "⚠ bitsandbytes install reported success but package not found in metadata" )
		return "bitsandbytes (unavailable)"

	except subprocess.TimeoutExpired :
		print( "⚠ bitsandbytes install timed out (300s)" )
		return "bitsandbytes (unavailable)"
	except Exception as e :
		print( f"⚠ bitsandbytes installation error: {e}" )
		return "bitsandbytes (unavailable)"


def find_libreoffice( ) -> str :
	"""
	Locate the LibreOffice 'soffice' executable on the current system.

	On Windows, checks the two standard install paths first, then falls back to PATH.
	On Linux/macOS, checks PATH for 'soffice' and 'libreoffice'.

	Returns:
		The full path to the soffice executable.

	Raises:
		RuntimeError: If LibreOffice cannot be found anywhere.
	"""
	import os

	if platform.system( ) == "Windows" :
		# Windows: LibreOffice is rarely on PATH — check default install locations
		candidates = [
			r"C:\Program Files\LibreOffice\program\soffice.exe" ,
			r"C:\Program Files (x86)\LibreOffice\program\soffice.exe" ,
		]
		for path in candidates :
			if os.path.isfile( path ) :
				return path

		# Last resort: maybe the user added it to PATH manually
		found = shutil.which( "soffice" )
		if found :
			return found

		raise RuntimeError(
				"LibreOffice not found. Expected at "
				r"'C:\Program Files\LibreOffice\program\soffice.exe'. "
				"Download from https://www.libreoffice.org/download/" ,
		)

	# Linux / macOS: check PATH for common executable names
	for cmd in ("soffice" , "libreoffice") :
		found = shutil.which( cmd )
		if found :
			return found

	raise RuntimeError(
			"LibreOffice not found on PATH. "
			"Install with: sudo apt install libreoffice  OR  brew install libreoffice" ,
	)


def is_apache_tika_server_alive( ) -> bool :
	try :
		requests.get( f"http://localhost:9998/tika" , timeout=2 )
		return True
	except requests.ConnectionError :
		return False


def ensure_apache_tika_server(
		logger: logging.Logger ,
		tika_server_process: subprocess.Popen | None ,
) -> subprocess.Popen :
	"""
	Ensure the Apache Tika server is running.

	Args:
			:param logger: Logger instance.
			:param tika_server_process:
	"""

	# Already running?
	if tika_server_process is not None and tika_server_process.poll( ) is None :
		if is_apache_tika_server_alive( ) :
			logger.debug( "Tika server already running" )
			return tika_server_process

	log_path = LOG_DIR / f"APACHE-TIKA-SERVER-{datetime.now( ).strftime( '%Y-%m-%d_%H-%M-%S' )}.log"
	log_file = open( log_path , "w" )

	cmd = [
		str( JAVA_PATH ) ,
		"-jar" ,
		str( TIKA_SERVER_JAR_PATH ) ,
		"-p" ,
		"9998" ,
	]

	logger.info( f"Java path: {JAVA_PATH}" )
	logger.info( f"Java path exists: {Path( JAVA_PATH ).exists( )}" )
	logger.info( f"Tika JAR path: {TIKA_SERVER_JAR_PATH}" )
	logger.info( f"Tika JAR exists: {Path( TIKA_SERVER_JAR_PATH ).exists( )}" )
	logger.info( f"Tika server command: {cmd}" )
	logger.info( f"Tika server log: {log_path}" )

	kwargs: dict = dict( stdout=log_file , stderr=subprocess.STDOUT )
	if sys.platform == "win32" :
		kwargs[ "creationflags" ] = subprocess.CREATE_NO_WINDOW

	proc = subprocess.Popen( cmd , **kwargs )

	for i in range( 90 ) :
		if proc.poll( ) is not None :
			log_file.close( )
			snippet = log_path.read_text( )[ -2000 : ]
			raise RuntimeError(
					f"Tika server exited with code {proc.returncode}:\n{snippet}" ,
			)
		if is_apache_tika_server_alive( ) :
			logger.info( f"Apache Tika server ready on port 9998 (took {i + 1}s)" )
			return proc
		time.sleep( 1 )

	log_file.close( )
	proc.kill( )
	raise RuntimeError( f"Tika server did not respond within 90s. Check log: {log_path}" )


def ensure_ollama_model(
		model: str ,
		logger: logging.Logger | None = None ,
) -> None :
	"""
	Ensure an Ollama model is available locally, pulling it if necessary.

	Raises RuntimeError if the model cannot be pulled.
	"""
	client = ollama.Client( )

	# Check if the model already exists locally
	try :
		existing = client.list( )
		installed = { m.model for m in existing.models }
		if model in installed :
			if logger :
				logger.info( f"[MODEL] '{model}' already available" )
			return
	except Exception as e :
		if logger :
			logger.warning( f"[MODEL] Could not list models — {type( e ).__name__}: {e}" )

	# Pull the model
	if logger :
		logger.info( f"[MODEL] '{model}' not found locally — pulling..." )

	try :
		client.pull( model )
		if logger :
			logger.info( f"[MODEL] '{model}' pulled successfully" )
	except Exception as e :
		msg = f"Failed to pull model '{model}' — {type( e ).__name__}: {e}"
		if logger :
			logger.error( f"[MODEL] {msg}" )
		raise RuntimeError( msg ) from e
