#!/usr/bin/env python3
"""
Tesseract Debug Script - Find out why Tesseract isn't working
"""

import os
import subprocess
import sys
from pathlib import Path


def check_tesseract_locations():
    """Check common Tesseract installation locations"""

    print("🔍 SEARCHING FOR TESSERACT BINARY...")
    print("=" * 60)

    # Common installation paths
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\%USERNAME%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\tesseract\tesseract.exe",
        r"C:\tools\tesseract\tesseract.exe",
    ]

    found_paths = []

    for path in common_paths:
        # Expand environment variables
        expanded_path = os.path.expandvars(path)

        print(f"Checking: {expanded_path}")

        if os.path.exists(expanded_path):
            print(f"  ✅ FOUND!")
            found_paths.append(expanded_path)

            # Check if it's executable
            try:
                result = subprocess.run(
                    [expanded_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"  ✅ WORKING! Version: {result.stdout.strip()}")
                else:
                    print(f"  ❌ Not working: {result.stderr}")
            except Exception as e:
                print(f"  ❌ Error running: {e}")
        else:
            print(f"  ❌ Not found")

    return found_paths


def check_path_environment():
    """Check if Tesseract is in PATH"""

    print("\n🔍 CHECKING PATH ENVIRONMENT...")
    print("=" * 60)

    path_env = os.environ.get("PATH", "")
    path_dirs = path_env.split(os.pathsep)

    tesseract_in_path = []

    for path_dir in path_dirs:
        if "tesseract" in path_dir.lower():
            print(f"Found Tesseract in PATH: {path_dir}")
            tesseract_in_path.append(path_dir)

            # Check if tesseract.exe is actually there
            exe_path = os.path.join(path_dir, "tesseract.exe")
            if os.path.exists(exe_path):
                print(f"  ✅ tesseract.exe exists in this PATH directory")
            else:
                print(f"  ❌ tesseract.exe NOT found in this PATH directory")

    # Try to run tesseract from command line
    try:
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✅ tesseract command works from PATH: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ tesseract command failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ tesseract command not found in PATH")
    except Exception as e:
        print(f"❌ Error running tesseract from PATH: {e}")

    return False


def check_pytesseract_config():
    """Check pytesseract configuration"""

    print("\n🔍 CHECKING PYTESSERACT CONFIGURATION...")
    print("=" * 60)

    try:
        import pytesseract

        print(f"PyTesseract version: {pytesseract.__version__}")

        # Check current tesseract command setting
        current_cmd = pytesseract.pytesseract.tesseract_cmd
        print(f"Current tesseract_cmd: {current_cmd}")

        # Try to get version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ PyTesseract can get version: {version}")
            return True
        except Exception as e:
            print(f"❌ PyTesseract version check failed: {e}")
            return False

    except ImportError:
        print("❌ PyTesseract not installed")
        return False


def try_manual_paths(found_paths):
    """Try to manually set tesseract paths"""

    if not found_paths:
        print("\n❌ No Tesseract installations found!")
        return

    print(f"\n🔧 TESTING MANUAL PATH CONFIGURATION...")
    print("=" * 60)

    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        for tesseract_path in found_paths:
            print(f"\nTrying path: {tesseract_path}")

            # Set the tesseract command
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

            try:
                # Create a simple test image
                img_array = np.ones((100, 300, 3), dtype=np.uint8) * 255

                # Try to add text if OpenCV available
                try:
                    import cv2

                    cv2.putText(
                        img_array,
                        "TEST123",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                    )
                except:
                    pass

                test_img = Image.fromarray(img_array)

                # Try OCR
                text = pytesseract.image_to_string(test_img)
                print(f"  ✅ OCR SUCCESS! Detected text: '{text.strip()}'")

                # Try getting languages
                langs = pytesseract.get_languages()
                print(f"  ✅ Available languages: {langs[:5]}...")  # Show first 5

                return tesseract_path

            except Exception as e:
                print(f"  ❌ OCR failed with this path: {e}")

    except ImportError as e:
        print(f"❌ Missing Python libraries: {e}")

    return None


def generate_fix_instructions(working_path=None):
    """Generate specific fix instructions"""

    print(f"\n🔧 FIX INSTRUCTIONS")
    print("=" * 60)

    if working_path:
        print(f"✅ GOOD NEWS: Found working Tesseract at: {working_path}")
        print(f"\n📝 ADD THIS TO YOUR OCR SCRIPT:")
        print(f"import pytesseract")
        print(f"pytesseract.pytesseract.tesseract_cmd = r'{working_path}'")

        # Create a config file
        config_content = f"""# Tesseract configuration
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'{working_path}'
"""

        with open("tesseract_config.py", "w") as f:
            f.write(config_content)

        print(f"\n💾 CREATED: tesseract_config.py (import this in your scripts)")

    else:
        print("❌ NO WORKING TESSERACT FOUND")
        print("\n🔧 SOLUTIONS:")
        print("1. REINSTALL TESSERACT:")
        print("   - Go to: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Download: tesseract-ocr-w64-setup-5.3.3.20231005.exe")
        print("   - IMPORTANT: Run installer AS ADMINISTRATOR")
        print("   - Choose 'Add to PATH' option during install")

        print("\n2. MANUAL INSTALLATION CHECK:")
        print("   - Open File Explorer")
        print("   - Go to C:\\Program Files\\Tesseract-OCR\\")
        print("   - Confirm tesseract.exe is there")
        print("   - Right-click tesseract.exe → Properties → Unblock (if present)")

        print("\n3. ADD TO PATH MANUALLY:")
        print("   - Windows key + R → sysdm.cpl")
        print("   - Advanced → Environment Variables")
        print("   - System Variables → PATH → Edit → New")
        print("   - Add: C:\\Program Files\\Tesseract-OCR")
        print("   - Restart command prompt")


def main():
    """Run complete Tesseract debugging"""

    print("🔍 TESSERACT DEBUGGING TOOL")
    print("=" * 80)
    print("Finding out why Tesseract isn't working...\n")

    # Step 1: Find Tesseract installations
    found_paths = check_tesseract_locations()

    # Step 2: Check PATH
    in_path = check_path_environment()

    # Step 3: Check PyTesseract
    pytess_ok = check_pytesseract_config()

    # Step 4: Try manual configuration
    working_path = try_manual_paths(found_paths)

    # Step 5: Generate fix instructions
    generate_fix_instructions(working_path)

    print(f"\n📊 SUMMARY:")
    print(f"Tesseract installations found: {len(found_paths)}")
    print(f"Tesseract in PATH: {'Yes' if in_path else 'No'}")
    print(f"PyTesseract working: {'Yes' if pytess_ok else 'No'}")
    print(f"Working configuration: {'Yes' if working_path else 'No'}")


if __name__ == "__main__":
    main()
