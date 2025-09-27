#!/usr/bin/env python3
"""
Inspect DOCX files that are failing detection
"""

import magic
from pathlib import Path
import zipfile


def inspect_docx_file(file_path: Path):
    """Inspect a DOCX file to understand why detection is failing"""
    print(f"\nInspecting: {file_path.name}")
    print("-" * 50)

    if not file_path.exists():
        print("File not found!")
        return

    # Check MIME type
    try:
        mime_type = magic.from_file(str(file_path), mime=True)
        print(f"MIME Type: {mime_type}")

        expected_mime = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        print(f"Expected:  {expected_mime}")
        print(f"Match: {'YES' if mime_type == expected_mime else 'NO'}")

    except Exception as e:
        print(f"MIME detection failed: {e}")

    # Try to open as ZIP (DOCX should be a ZIP file)
    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            files = zip_file.namelist()
            print(f"\nZIP Contents ({len(files)} files):")

            # Look for key DOCX structure files
            key_files = [
                "[Content_Types].xml",
                "word/document.xml",
                "_rels/.rels",
                "word/_rels/document.xml.rels",
            ]

            for key_file in key_files:
                if key_file in files:
                    print(f"  ✓ {key_file}")
                else:
                    print(f"  ✗ {key_file} (MISSING)")

            # Show first few files
            print(f"\nFirst 5 files in archive:")
            for f in files[:5]:
                print(f"  - {f}")

    except zipfile.BadZipFile:
        print("\n❌ NOT A VALID ZIP FILE")
        print("This file is not a proper DOCX format")

        # Check if it might be RTF or old DOC format
        try:
            with open(file_path, "rb") as f:
                header = f.read(10)
                print(f"File header: {header}")

                if header.startswith(b"{\\rtf"):
                    print("→ This appears to be RTF format with .docx extension")
                elif header.startswith(b"\xd0\xcf\x11\xe0"):
                    print("→ This appears to be old .DOC format with .docx extension")
                else:
                    print("→ Unknown file format")

        except Exception as e:
            print(f"Header inspection failed: {e}")

    except Exception as e:
        print(f"ZIP inspection failed: {e}")


def main():
    """Inspect all DOCX files in review directory"""
    review_dir = Path(
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run\01_unprocessed"
    )

    if not review_dir.exists():
        print("Review directory not found")
        return

    docx_files = list(review_dir.glob("*.docx"))

    if not docx_files:
        print("No .docx files found in review directory")
        return

    print(f"Found {len(docx_files)} DOCX files to inspect")

    for docx_file in docx_files:
        inspect_docx_file(docx_file)

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(
        "- If files show 'NOT A VALID ZIP FILE' → Files are corrupted or wrong format"
    )
    print("- If MIME type doesn't match → Set REQUIRE_DETECTION_AGREEMENT = False")
    print("- If RTF detected → Rename to .rtf extension")
    print("- If old DOC detected → Rename to .doc extension")


if __name__ == "__main__":
    main()
