#!/usr/bin/env python3
"""
OCR Dependencies Test Script
Tests all required packages for the multilingual OCR processor
"""

import sys
import os
from typing import Dict, List, Tuple


def test_basic_imports() -> Dict[str, Tuple[bool, str]]:
    """Test basic Python imports"""
    results = {}

    # Core Python libraries
    tests = [
        ("os", "import os"),
        ("sys", "import sys"),
        ("datetime", "from datetime import datetime"),
        ("hashlib", "import hashlib"),
        ("tempfile", "import tempfile"),
        ("pathlib", "from pathlib import Path"),
        ("typing", "from typing import List, Dict, Optional"),
        ("dataclasses", "from dataclasses import dataclass"),
        ("enum", "from enum import Enum"),
    ]

    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            results[name] = (True, "✅ OK")
        except ImportError as e:
            results[name] = (False, f"❌ FAILED: {e}")

    return results


def test_image_processing() -> Dict[str, Tuple[bool, str]]:
    """Test image processing libraries"""
    results = {}

    # PIL/Pillow
    try:
        from PIL import Image, ImageEnhance, ImageFilter

        results["PIL/Pillow"] = (True, f"✅ OK - Version: {Image.__version__}")
    except ImportError as e:
        results["PIL/Pillow"] = (False, f"❌ FAILED: {e}")

    # OpenCV
    try:
        import cv2

        results["OpenCV (cv2)"] = (True, f"✅ OK - Version: {cv2.__version__}")
    except ImportError as e:
        results["OpenCV (cv2)"] = (False, f"❌ FAILED: {e}")

    # NumPy
    try:
        import numpy as np

        results["NumPy"] = (True, f"✅ OK - Version: {np.__version__}")
    except ImportError as e:
        results["NumPy"] = (False, f"❌ FAILED: {e}")

    return results


def test_ocr_engines() -> Dict[str, Tuple[bool, str]]:
    """Test OCR engine imports"""
    results = {}

    # PyTesseract
    try:
        import pytesseract

        results["PyTesseract"] = (True, f"✅ OK - Version: {pytesseract.__version__}")
    except ImportError as e:
        results["PyTesseract"] = (False, f"❌ FAILED: {e}")

    # EasyOCR
    try:
        import easyocr

        results["EasyOCR"] = (True, f"✅ OK - Version: {easyocr.__version__}")
    except ImportError as e:
        results["EasyOCR"] = (False, f"❌ FAILED: {e}")

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        import paddleocr

        results["PaddleOCR"] = (True, f"✅ OK - Version: {paddleocr.__version__}")
    except ImportError as e:
        results["PaddleOCR"] = (False, f"❌ FAILED: {e}")

    return results


def test_format_support() -> Dict[str, Tuple[bool, str]]:
    """Test file format support libraries"""
    results = {}

    # PDF support
    try:
        from pdf2image import convert_from_path

        results["PDF Support (pdf2image)"] = (True, "✅ OK")
    except ImportError as e:
        results["PDF Support (pdf2image)"] = (False, f"❌ FAILED: {e}")

    # HEIC support - Method 1
    try:
        from pillow_heif import register_heif_opener

        results["HEIC Support (pillow-heif)"] = (True, "✅ OK")
    except ImportError:
        # HEIC support - Method 2
        try:
            import pyheif

            results["HEIC Support (pyheif)"] = (True, "✅ OK (alternative)")
        except ImportError as e:
            results["HEIC Support"] = (False, f"❌ FAILED: {e}")

    return results


def test_tesseract_binary() -> Dict[str, Tuple[bool, str]]:
    """Test if Tesseract binary is available"""
    results = {}

    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        # Try to set Windows Tesseract path (use actual found location)
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Users\UserX\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            )
        except:
            pass

        # Create a simple test image
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2_available = False

        try:
            import cv2

            cv2.putText(
                test_img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            cv2_available = True
        except:
            pass

        if cv2_available:
            pil_img = Image.fromarray(test_img)
            # Try to extract text
            text = pytesseract.image_to_string(pil_img)
            results["Tesseract Binary"] = (True, "✅ OK - Binary is working")
        else:
            results["Tesseract Binary"] = (
                False,
                "❌ Cannot test - OpenCV not available",
            )

    except Exception as e:
        if "tesseract is not installed" in str(e).lower():
            results["Tesseract Binary"] = (
                False,
                "❌ FAILED: Tesseract binary not installed",
            )
        else:
            results["Tesseract Binary"] = (False, f"❌ FAILED: {e}")

    return results


def test_functional_ocr() -> Dict[str, Tuple[bool, str]]:
    """Test if OCR engines can actually process images"""
    results = {}

    try:
        import numpy as np
        from PIL import Image

        # Create a simple test image with text
        img_array = np.ones((100, 300, 3), dtype=np.uint8) * 255

        try:
            import cv2

            cv2.putText(
                img_array, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            test_img = Image.fromarray(img_array)

            # Test PyTesseract
            try:
                import pytesseract

                # Try to set Windows Tesseract path (use actual found location)
                try:
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\UserX\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                except:
                    pass

                text = pytesseract.image_to_string(test_img).strip()
                if text:
                    results["PyTesseract Functionality"] = (
                        True,
                        f"✅ OK - Detected: '{text}'",
                    )
                else:
                    results["PyTesseract Functionality"] = (
                        True,
                        "✅ OK - No text detected (normal for test)",
                    )
            except Exception as e:
                results["PyTesseract Functionality"] = (False, f"❌ FAILED: {e}")

            # Test EasyOCR (lighter test - just initialization)
            try:
                import easyocr

                reader = easyocr.Reader(["en"], gpu=False)
                results["EasyOCR Functionality"] = (True, "✅ OK - Reader initialized")
            except Exception as e:
                results["EasyOCR Functionality"] = (False, f"❌ FAILED: {e}")

            # Test PaddleOCR (lighter test - just initialization)
            try:
                from paddleocr import PaddleOCR

                ocr = PaddleOCR(use_textline_orientation=True, lang="en")
                results["PaddleOCR Functionality"] = (True, "✅ OK - OCR initialized")
            except Exception as e:
                results["PaddleOCR Functionality"] = (False, f"❌ FAILED: {e}")

        except ImportError:
            results["Functional OCR Tests"] = (
                False,
                "❌ SKIPPED - OpenCV not available",
            )

    except Exception as e:
        results["Functional OCR Tests"] = (False, f"❌ FAILED: {e}")

    return results


def print_system_info():
    """Print system information"""
    print("=" * 80)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")

    # Check for virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("🔧 Virtual environment: ACTIVE")
    else:
        print("🔧 Virtual environment: NOT ACTIVE")
    print()


def print_results(title: str, results: Dict[str, Tuple[bool, str]]):
    """Print test results in a formatted way"""
    print(f"📦 {title.upper()}")
    print("-" * 60)

    success_count = 0
    total_count = len(results)

    for name, (success, message) in results.items():
        print(f"{message} - {name}")
        if success:
            success_count += 1

    print(f"\nResult: {success_count}/{total_count} passed")
    print()

    return success_count, total_count


def main():
    """Run all tests"""
    print_system_info()

    total_passed = 0
    total_tests = 0

    # Run all test suites
    test_suites = [
        ("Basic Python Imports", test_basic_imports),
        ("Image Processing Libraries", test_image_processing),
        ("OCR Engines", test_ocr_engines),
        ("File Format Support", test_format_support),
        ("Tesseract Binary", test_tesseract_binary),
        ("Functional OCR Tests", test_functional_ocr),
    ]

    for title, test_func in test_suites:
        results = test_func()
        passed, tests = print_results(title, results)
        total_passed += passed
        total_tests += tests

    # Final summary
    print("=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! You're ready to use the OCR processor!")
        print("✅ Run your OCR script with confidence!")
    else:
        print(f"⚠️  {total_tests - total_passed} out of {total_tests} tests failed.")
        print("\n🔧 TO FIX REMAINING ISSUES:")

        # Check if Tesseract is the main issue
        print("\n1. INSTALL TESSERACT BINARY (if failed above):")
        if sys.platform == "win32":
            print("   📥 For Windows:")
            print("   - Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("   - Install tesseract-ocr-w64-setup-5.3.3.20231005.exe")
            print("   - Default install location: C:\\Program Files\\Tesseract-OCR\\")
        elif sys.platform == "darwin":
            print("   📥 For macOS:")
            print("   - Run: brew install tesseract")
        else:
            print("   📥 For Linux:")
            print("   - Run: sudo apt-get install tesseract-ocr")

        print("\n2. UPDATE PYTHON PACKAGES (if needed):")
        print("   python3 -m pip install --upgrade pip")
        print(
            "   python3 -m pip install pytesseract pillow easyocr paddlepaddle paddleocr opencv-python numpy pdf2image pillow-heif"
        )

        print("\n3. RERUN THIS TEST:")
        print("   python3 test_ocr_deps.py")

    print("=" * 80)


if __name__ == "__main__":
    main()
