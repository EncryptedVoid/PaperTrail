#!/usr/bin/env python
"""
Bulletproof self-contained Tika extractor.
"""
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


def kill_zombie_java_processes():
    """Kill any hanging Java/Tika processes."""
    try:
        if platform.system() == "Windows":
            # Kill any java.exe processes running tika
            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/FI",
                    "IMAGENAME eq java.exe",
                    "/FI",
                    "WINDOWTITLE eq *tika*",
                ],
                capture_output=True,
                timeout=10,
            )
        else:
            # Unix-like systems
            subprocess.run(
                ["pkill", "-f", "tika-server"], capture_output=True, timeout=10
            )
        print("✓ Cleaned up any zombie processes")
    except Exception:
        pass  # Not critical if this fails


def clear_tika_cache():
    """Clear corrupted Tika cache files."""
    try:
        # Find temp directory
        temp_dir = Path(tempfile.gettempdir())

        # Delete Tika files
        for pattern in [
            "tika-server.jar",
            "tika-server.jar.md5",
            "tika.log",
            "tika-server.log",
        ]:
            for file in temp_dir.glob(pattern):
                try:
                    file.unlink()
                    print(f"✓ Deleted cached {file.name}")
                except Exception:
                    pass

        # Also check user home .tika directory
        tika_home = Path.home() / ".tika"
        if tika_home.exists():
            shutil.rmtree(tika_home, ignore_errors=True)
            print("✓ Cleared .tika directory")

    except Exception as e:
        print(f"⚠ Could not clear cache: {e}")


def extract_with_tika(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata and content using Apache Tika.
    Bulletproof version that handles common issues.
    """
    try:
        # Step 1: Clean up any issues
        print("Preparing Tika environment...")
        kill_zombie_java_processes()
        clear_tika_cache()

        # Step 2: Set environment variables BEFORE importing tika
        os.environ["TIKA_LOG_FILE"] = ""  # Disable log file to avoid permission issues
        os.environ["TIKA_SERVER_ENDPOINT"] = "http://localhost:9998"
        os.environ["TIKA_STARTUP_SLEEP"] = "1"  # Give it more time
        os.environ["TIKA_STARTUP_MAX_RETRY"] = "5"  # More retries

        # Step 3: NOW import tika
        print("Importing Tika...")
        from tika import parser

        # Step 4: Validate file
        filepath = Path(file_path)
        if not filepath.exists():
            return {"success": False, "error": "File not found"}

        file_size = filepath.stat().st_size
        print(f"\nProcessing: {filepath.name} ({file_size / 1024:.1f} KB)")
        print("Starting extraction...")
        print("-" * 80)

        # Step 5: Parse with timeout
        print("Calling Tika parser (this may take a moment on first run)...")
        parsed = parser.from_file(str(filepath))

        if not parsed:
            return {"success": False, "error": "Tika returned empty response"}

        # Step 6: Build result
        result = {
            "success": True,
            "file_info": {
                "filename": filepath.name,
                "full_path": str(filepath.absolute()),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "extension": filepath.suffix.lower(),
            },
            "tika_metadata": parsed.get("metadata", {}),
            "tika_content": parsed.get("content", ""),
            "extraction_status": parsed.get("status"),
        }

        content_length = len(result["tika_content"]) if result["tika_content"] else 0
        result["content_stats"] = {
            "content_length_chars": content_length,
            "content_length_kb": round(content_length / 1024, 2),
            "has_content": content_length > 0,
            "word_count_estimate": (
                len(result["tika_content"].split()) if result["tika_content"] else 0
            ),
        }

        metadata = result["tika_metadata"]
        result["common_fields"] = {
            "mime_type": metadata.get("Content-Type"),
            "author": metadata.get("Author")
            or metadata.get("creator")
            or metadata.get("meta:author"),
            "title": metadata.get("title")
            or metadata.get("dc:title")
            or metadata.get("Title"),
            "created": metadata.get("Creation-Date")
            or metadata.get("created")
            or metadata.get("meta:creation-date"),
            "modified": metadata.get("Last-Modified")
            or metadata.get("modified")
            or metadata.get("Last-Save-Date"),
            "page_count": metadata.get("xmpTPg:NPages")
            or metadata.get("Page-Count")
            or metadata.get("meta:page-count"),
            "word_count": metadata.get("Word-Count") or metadata.get("meta:word-count"),
            "language": metadata.get("language") or metadata.get("dc:language"),
        }

        print(f"✓ Extraction successful!")
        print(f"  - Metadata fields: {len(result['tika_metadata'])}")
        print(f"  - Content length: {content_length:,} characters")
        print(f"  - MIME type: {result['common_fields']['mime_type']}")

        return result

    except ImportError:
        return {
            "success": False,
            "error": "tika-python not installed. Run: pip install tika",
        }
    except Exception as e:
        import traceback

        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def main():
    file_path = r"C:\Users\UserX\Desktop\PaperTrail-Load\(Ashiq Gazi)  [Scenario 3] BTT - Computor purchasi.docx"

    print("=" * 80)
    print("Bulletproof Apache Tika Extractor")
    print("=" * 80)

    # Check Java
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=5
        )
        print(
            f"✓ Java found: {result.stderr.split()[2] if result.stderr else 'version unknown'}"
        )
    except Exception:
        print("✗ Java not found - please install Java 8 or higher")
        print("  Download: https://adoptium.net/temurin/releases/")
        return

    result = extract_with_tika(file_path)

    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_output)

    if result.get("success"):
        output_file = Path(file_path).with_suffix(".metadata.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"\n✓ Saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠ Could not save: {e}")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """
# Self-contained Tika extractor that auto-downloads Java if needed.
# """
# import json
# import os
# import platform
# import subprocess
# import urllib.request
# import zipfile
# from pathlib import Path
# from typing import Any, Dict
#
#
# def ensure_java() -> Path:
#     """
#     Ensure Java is available. Downloads portable JRE if not found.
#     Returns path to Java executable.
#     """
#     # Check if Java already exists
#     try:
#         result = subprocess.run(
#             ["java", "-version"], capture_output=True, text=True, timeout=5
#         )
#         if result.returncode == 0:
#             print("✓ Java already installed")
#             return Path("java")  # Use system Java
#     except (FileNotFoundError, subprocess.TimeoutExpired):
#         pass
#
#     print("Java not found. Downloading portable JRE...")
#
#     # Create a local jre directory in the project
#     jre_dir = Path(__file__).parent / "jre"
#     jre_dir.mkdir(exist_ok=True)
#
#     # Detect platform
#     system = platform.system().lower()
#     machine = platform.machine().lower()
#
#     # Amazon Corretto 17 portable (no admin needed)
#     if system == "windows":
#         if "64" in machine or "amd64" in machine:
#             java_url = "https://corretto.aws/downloads/latest/amazon-corretto-17-x64-windows-jdk.zip"
#             java_exe = jre_dir / "amazon-corretto-17" / "bin" / "java.exe"
#         else:
#             raise RuntimeError("32-bit Windows not supported")
#     elif system == "linux":
#         java_url = "https://corretto.aws/downloads/latest/amazon-corretto-17-x64-linux-jdk.tar.gz"
#         java_exe = jre_dir / "amazon-corretto-17" / "bin" / "java"
#     elif system == "darwin":  # macOS
#         java_url = "https://corretto.aws/downloads/latest/amazon-corretto-17-aarch64-macos-jdk.tar.gz"
#         java_exe = (
#             jre_dir / "amazon-corretto-17.jdk" / "Contents" / "Home" / "bin" / "java"
#         )
#     else:
#         raise RuntimeError(f"Unsupported platform: {system}")
#
#     # Check if already downloaded
#     if java_exe.exists():
#         print(f"✓ Using cached Java at {java_exe}")
#         return java_exe
#
#     # Download
#     zip_path = jre_dir / "java_download.zip"
#     print(f"Downloading from {java_url}...")
#
#     try:
#         urllib.request.urlretrieve(java_url, zip_path)
#         print(f"✓ Downloaded ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
#
#         # Extract
#         print("Extracting...")
#         if system == "windows":
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(jre_dir)
#         else:  # Linux/Mac - use tar
#             import tarfile
#
#             with tarfile.open(zip_path, "r:gz") as tar_ref:
#                 tar_ref.extractall(jre_dir)
#
#         zip_path.unlink()  # Delete archive
#         print(f"✓ Java installed at {java_exe}")
#
#         # Make executable on Unix
#         if system != "windows":
#             java_exe.chmod(0o755)
#
#         return java_exe
#
#     except Exception as e:
#         print(f"✗ Failed to download/extract Java: {e}")
#         raise
#
#
# def extract_with_tika(file_path: str) -> Dict[str, Any]:
#     """
#     Extract metadata and content using Apache Tika.
#     Auto-downloads Java if needed.
#     """
#     try:
#         # Ensure Java is available
#         java_path = ensure_java()
#
#         # Set JAVA_HOME if using bundled Java
#         if not java_path.name == "java":
#             java_home = java_path.parent.parent
#             os.environ["JAVA_HOME"] = str(java_home)
#             os.environ["PATH"] = f"{java_path.parent}{os.pathsep}{os.environ['PATH']}"
#             print(f"Set JAVA_HOME to {java_home}")
#
#         # NOW import tika (after Java is ready)
#         from tika import parser
#
#         filepath = Path(file_path)
#         if not filepath.exists():
#             return {"success": False, "error": "File not found"}
#
#         file_size = filepath.stat().st_size
#         print(f"\nProcessing: {filepath.name} ({file_size / 1024:.1f} KB)")
#         print("Extracting with Apache Tika...")
#         print("-" * 80)
#
#         # Parse file (Tika will auto-download tika-server.jar)
#         parsed = parser.from_file(str(filepath))
#
#         result = {
#             "success": True,
#             "file_info": {
#                 "filename": filepath.name,
#                 "full_path": str(filepath.absolute()),
#                 "file_size_bytes": file_size,
#                 "file_size_mb": round(file_size / 1024 / 1024, 2),
#                 "extension": filepath.suffix.lower(),
#             },
#             "tika_metadata": parsed.get("metadata", {}),
#             "tika_content": parsed.get("content", ""),
#             "extraction_status": parsed.get("status"),
#         }
#
#         content_length = len(result["tika_content"]) if result["tika_content"] else 0
#         result["content_stats"] = {
#             "content_length_chars": content_length,
#             "content_length_kb": round(content_length / 1024, 2),
#             "has_content": content_length > 0,
#             "word_count_estimate": (
#                 len(result["tika_content"].split()) if result["tika_content"] else 0
#             ),
#         }
#
#         metadata = result["tika_metadata"]
#         result["common_fields"] = {
#             "mime_type": metadata.get("Content-Type"),
#             "author": metadata.get("Author")
#             or metadata.get("creator")
#             or metadata.get("meta:author"),
#             "title": metadata.get("title")
#             or metadata.get("dc:title")
#             or metadata.get("Title"),
#             "created": metadata.get("Creation-Date")
#             or metadata.get("created")
#             or metadata.get("meta:creation-date"),
#             "modified": metadata.get("Last-Modified")
#             or metadata.get("modified")
#             or metadata.get("Last-Save-Date"),
#             "page_count": metadata.get("xmpTPg:NPages")
#             or metadata.get("Page-Count")
#             or metadata.get("meta:page-count"),
#             "word_count": metadata.get("Word-Count") or metadata.get("meta:word-count"),
#             "language": metadata.get("language") or metadata.get("dc:language"),
#         }
#
#         print(f"✓ Extraction successful!")
#         print(f"  - Metadata fields: {len(result['tika_metadata'])}")
#         print(f"  - Content length: {content_length:,} characters")
#         print(f"  - MIME type: {result['common_fields']['mime_type']}")
#
#         return result
#
#     except ImportError:
#         return {
#             "success": False,
#             "error": "tika-python not installed. Run: pip install tika",
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "error_type": type(e).__name__,
#         }
#
#
# def main():
#     file_path = r"C:\Users\UserX\Desktop\PaperTrail-Load\(Ashiq Gazi)  [Scenario 3] BTT - Computor purchasi.docx"
#
#     print("=" * 80)
#     print("Self-Contained Apache Tika Extractor")
#     print("=" * 80)
#
#     result = extract_with_tika(file_path)
#
#     print("\n" + "=" * 80)
#     print("EXTRACTION RESULTS")
#     print("=" * 80)
#
#     json_output = json.dumps(result, indent=2, ensure_ascii=False)
#     print(json_output)
#
#     if result.get("success"):
#         output_file = Path(file_path).with_suffix(".metadata.json")
#         try:
#             with open(output_file, "w", encoding="utf-8") as f:
#                 f.write(json_output)
#             print(f"\n✓ Saved to: {output_file}")
#         except Exception as e:
#             print(f"\n⚠ Could not save: {e}")
#
#
# if __name__ == "__main__":
#     main()
