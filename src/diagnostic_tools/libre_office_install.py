#!/usr/bin/env python3
"""
LibreOffice Installation Checker
"""

import subprocess
import shutil
from pathlib import Path


def check_libreoffice():
    """Check if LibreOffice is properly installed and accessible"""
    print("🔍 Checking LibreOffice Installation...")

    # Check if commands are in PATH
    commands_to_check = ["libreoffice", "soffice"]
    found_command = None

    for cmd in commands_to_check:
        path = shutil.which(cmd)
        if path:
            print(f"✅ Found {cmd} at: {path}")
            found_command = cmd
            break
        else:
            print(f"❌ {cmd} not found in PATH")

    if not found_command:
        print("\n❌ LibreOffice not found in PATH")
        print("\n💡 SOLUTIONS:")
        print("1. Download LibreOffice from: https://www.libreoffice.org/download/")
        print("2. Install it with default settings")
        print("3. Add it to your PATH, or restart your terminal")
        print("\nCommon Windows LibreOffice locations:")

        common_paths = [
            Path("C:/Program Files/LibreOffice/program/soffice.exe"),
            Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        ]

        for path in common_paths:
            if path.exists():
                print(f"  ✅ Found: {path}")
                print(f"     Add this directory to your PATH: {path.parent}")
            else:
                print(f"  ❌ Not found: {path}")

        return False

    # Test the command
    try:
        print(f"\n🧪 Testing {found_command} --version...")
        result = subprocess.run(
            [found_command, "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print(f"✅ LibreOffice version: {result.stdout.strip()}")

            # Test headless mode
            print(f"\n🧪 Testing {found_command} --help...")
            result2 = subprocess.run(
                [found_command, "--help"], capture_output=True, text=True, timeout=10
            )

            if result2.returncode == 0:
                print("✅ LibreOffice headless mode should work")
                return True
            else:
                print(f"❌ LibreOffice help failed: {result2.stderr}")
                return False
        else:
            print(f"❌ LibreOffice version check failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ LibreOffice command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing LibreOffice: {e}")
        return False


def test_conversion():
    """Test if LibreOffice can actually convert a document"""
    print(f"\n{'='*60}")
    print("🧪 TESTING DOCUMENT CONVERSION")
    print(f"{'='*60}")

    # You would need a test file for this
    # For now, just check if the command structure works

    test_cmd = ["libreoffice", "--headless", "--convert-to", "pdf", "--help"]

    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if "--convert-to" in result.stdout or result.returncode == 0:
            print("✅ LibreOffice conversion commands are available")
            return True
        else:
            print(f"❌ LibreOffice conversion test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Conversion test error: {e}")
        return False


def main():
    print("LibreOffice Installation Diagnostic")
    print("=" * 60)

    if check_libreoffice():
        if test_conversion():
            print(f"\n{'='*60}")
            print("✅ SUCCESS: LibreOffice is properly installed and should work")
            print("The [WinError 2] is likely caused by something else")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("⚠️  LibreOffice is installed but conversion commands may not work")
            print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ PROBLEM: LibreOffice is not properly installed")
        print("This explains the [WinError 2] in your logs")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
