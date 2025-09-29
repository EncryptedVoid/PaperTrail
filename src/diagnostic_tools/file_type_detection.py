#!/usr/bin/env python3
"""
Find LibreOffice Installation on Windows
"""

import os
from pathlib import Path
import subprocess


def find_libreoffice():
    """Find LibreOffice installation and test it"""

    print("Searching for LibreOffice installation...")

    # Common Windows installation paths
    possible_paths = [
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        Path(
            os.path.expanduser(
                "~/AppData/Local/Programs/LibreOffice/program/soffice.exe"
            )
        ),
        # Check Program Files for any LibreOffice folder
    ]

    # Also search Program Files directories
    program_files_dirs = [
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]

    for pf_dir in program_files_dirs:
        if pf_dir.exists():
            for item in pf_dir.iterdir():
                if item.is_dir() and "libreoffice" in item.name.lower():
                    soffice_path = item / "program" / "soffice.exe"
                    if not soffice_path in possible_paths:
                        possible_paths.append(soffice_path)

    found_paths = []
    for path in possible_paths:
        if path.exists():
            found_paths.append(path)
            print(f"Found LibreOffice at: {path}")

    if not found_paths:
        print("LibreOffice not found in common locations")
        print("Check if it actually installed successfully")
        return None

    # Test the first found installation
    libreoffice_exe = found_paths[0]
    print(f"\nTesting LibreOffice at: {libreoffice_exe}")

    try:
        # Test version command
        result = subprocess.run(
            [str(libreoffice_exe), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"LibreOffice version: {result.stdout.strip()}")
            return libreoffice_exe
        else:
            print(f"LibreOffice test failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error testing LibreOffice: {e}")
        return None


def add_to_path_instructions(libreoffice_exe):
    """Provide instructions to add LibreOffice to PATH"""
    program_dir = libreoffice_exe.parent

    print(f"\nTo add LibreOffice to your PATH:")
    print("1. Press Win+R, type 'sysdm.cpl', press Enter")
    print("2. Click 'Environment Variables'")
    print("3. Under 'System Variables', find 'Path' and click 'Edit'")
    print("4. Click 'New' and add this path:")
    print(f"   {program_dir}")
    print("5. Click OK on all dialogs")
    print("6. Restart your PowerShell/Command Prompt")

    print(f"\nOR use this PowerShell command (run as Administrator):")
    print(
        f'[Environment]::SetEnvironmentVariable("Path", $env:Path + ";{program_dir}", "Machine")'
    )


def create_wrapper_script(libreoffice_exe):
    """Create a wrapper batch file as alternative to PATH"""
    wrapper_path = Path("libreoffice.bat")

    with open(wrapper_path, "w") as f:
        f.write(f'@echo off\n"{libreoffice_exe}" %*\n')

    print(f"\nCreated wrapper script: {wrapper_path.absolute()}")
    print("You can now use 'libreoffice.bat' instead of 'libreoffice'")

    return wrapper_path


def update_config_file(libreoffice_exe):
    """Show how to update the conversion code with explicit path"""
    print(f"\nAlternative: Update your type_conversion.py to use explicit path:")
    print("Replace this line:")
    print('    cmd = ["libreoffice", "--headless", ...]')
    print("With:")
    print(f'    cmd = [r"{libreoffice_exe}", "--headless", ...]')


def main():
    print("LibreOffice Path Fixer")
    print("=" * 50)

    libreoffice_exe = find_libreoffice()

    if libreoffice_exe:
        print(f"\nLibreOffice is installed and working!")
        add_to_path_instructions(libreoffice_exe)

        print(f"\nQUICK FIXES:")
        create_wrapper_script(libreoffice_exe)
        update_config_file(libreoffice_exe)

    else:
        print("\nLibreOffice installation not found or not working")
        print("Try reinstalling LibreOffice with these steps:")
        print("1. Download from https://www.libreoffice.org/download/")
        print("2. Right-click installer, 'Run as Administrator'")
        print("3. Choose 'Complete' installation")
        print("4. Restart your computer after installation")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Detection Diagnostic - Prove what's causing the mismatch
# """

# import magic
# from pathlib import Path

# # Your extension mapping from config
# EXTENSION_MAPPING = {
#     ".jpeg": "image",
#     ".jpg": "image",
#     ".png": "image",
#     ".heic": "image",
#     ".cr2": "image",
#     ".arw": "image",
#     ".nef": "image",
#     ".webp": "image",
#     ".mov": "video",
#     ".mp4": "video",
#     ".webm": "video",
#     ".amv": "video",
#     ".3gp": "video_audio",
#     ".wav": "audio",
#     ".mp3": "audio",
#     ".m4a": "audio",
#     ".ogg": "audio",
#     ".pptx": "document",
#     ".doc": "document",
#     ".docx": "document",
#     ".rtf": "document",
#     ".epub": "document",
#     ".pub": "document",
#     ".djvu": "document",
#     ".pdf": "document",
# }


# def detect_by_extension(file_path: Path):
#     """Your exact extension detection logic"""
#     ext = file_path.suffix.lower()
#     return EXTENSION_MAPPING.get(ext)


# def detect_by_content(file_path: Path):
#     """Your exact content detection logic"""
#     try:
#         mime_type = magic.from_file(str(file_path), mime=True)

#         if mime_type.startswith("image/"):
#             return "image"
#         elif mime_type.startswith("video/"):
#             return "video"
#         elif mime_type.startswith("audio/"):
#             return "audio"
#         elif mime_type in [
#             "application/pdf",
#             "application/msword",
#             "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#             "application/vnd.ms-powerpoint",
#             "application/vnd.openxmlformats-officedocument.presentationml.presentation",
#         ]:
#             return "document"
#         else:
#             return None
#     except Exception:
#         return None


# def test_file(file_path: Path):
#     """Test a single file and show exactly what each detection method returns"""
#     print(f"\n{'='*60}")
#     print(f"TESTING: {file_path.name}")
#     print(f"{'='*60}")

#     if not file_path.exists():
#         print("❌ FILE NOT FOUND")
#         return

#     # Extension detection
#     ext_result = detect_by_extension(file_path)
#     print(f"📁 Extension (.{file_path.suffix}): {ext_result}")

#     # Content detection
#     try:
#         mime_type = magic.from_file(str(file_path), mime=True)
#         print(f"🔍 MIME Type: {mime_type}")
#     except Exception as e:
#         print(f"❌ MIME Detection Failed: {e}")
#         mime_type = "ERROR"

#     content_result = detect_by_content(file_path)
#     print(f"📄 Content Detection: {content_result}")

#     # Agreement check
#     agree = ext_result == content_result and ext_result is not None
#     print(f"✅ AGREEMENT: {agree}")

#     if not agree:
#         print(f"💥 MISMATCH DETECTED!")
#         print(f"   Extension says: {ext_result}")
#         print(f"   Content says: {content_result}")
#         print(f"   This will trigger ConversionStatus.DETECTION_MISMATCH")

#     return agree


# def main():
#     # Test files in your review directory (files that failed)
#     failure_dir = Path(r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run")

#     if not failure_dir.exists():
#         print(f"❌ Review directory not found: {failure_dir}")
#         print("Update the path to match your directory structure")
#         return

#     files = list(failure_dir.glob("*"))
#     if not files:
#         print("No files found in review directory")
#         return

#     print(f"🔍 Testing {len(files)} files from review directory")
#     print("These are files that failed with DETECTION_MISMATCH")

#     mismatches = 0
#     for file_path in files:
#         if file_path.is_file():
#             if not test_file(file_path):
#                 mismatches += 1

#     print(f"\n{'='*60}")
#     print(f"SUMMARY: {mismatches}/{len(files)} files have detection mismatches")
#     print(f"{'='*60}")

#     if mismatches > 0:
#         print("\n💡 TO FIX:")
#         print("1. Set REQUIRE_DETECTION_AGREEMENT = False in config.py")
#         print("2. OR fix the actual file corruption issues")
#         print("3. OR set USE_CONTENT_DETECTION = False to use extension only")


# if __name__ == "__main__":
#     main()
