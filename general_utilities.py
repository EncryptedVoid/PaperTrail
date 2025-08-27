from pathlib import Path
from datetime import datetime
import hashlib
import uuid

# Basic UUID generation
unique_id: uuid.UUID = uuid.uuid4()
print(f"Generated UUID: {unique_id}")

# Convert to string with proper typing
unique_string: str = str(unique_id)
print(f"UUID as string: {unique_string}")


def calculate_checksum(file_path, algorithm="sha512"):
    """Calculate checksum of a file using specified algorithm."""
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


# def demonstrate_deterministic_checksums(file_paths):
#     """Demonstrate that checksums are deterministic by calculating multiple times."""

#     for file_path in file_paths:
#         print(f"\n=== Testing file: {file_path} ===")

#         checksums = []

#         # Calculate checksum 5 times with small delays
#         for i in range(5):
#             checksum = calculate_checksum(file_path)
#             checksums.append(checksum)
#             timestamp = time.strftime("%H:%M:%S")
#             print(f"Run {i+1} at {timestamp}: {checksum}")
#             time.sleep(0.1)  # Small delay to show different timestamps

#         # Verify all checksums are identical
#         all_same = all(cs == checksums[0] for cs in checksums)
#         print(f"All checksums identical: {all_same}")

#         if all_same:
#             print("✅ DETERMINISTIC - Same file always produces same checksum")
#         else:
#             print("❌ ERROR - Checksums should be identical!")


# Example usage with your file variables
# file1: str = (
#     r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (1).pdf"
# )
# file2: str = (
#     r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (3).pdf"
# )
# file3: str = (
#     r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (5).pdf"
# )

# files_to_test = [file1, file2, file3]

# Run the demonstration
# demonstrate_deterministic_checksums(files_to_test)

# Additional demonstration: Show that different files have different checksums
# print("\n=== Comparing different files ===")
# for i, file_path in enumerate(files_to_test):
#     try:
#         checksum = calculate_checksum(file_path)
#         print(f"File {i+1} ({file_path}): {checksum[:16]}...")  # Show first 16 chars
#     except FileNotFoundError:
#         print(f"File {i+1} ({file_path}): [File not found - just for demo]")

# print("\n🔍 Key Points:")
# print("1. Same file = Same checksum EVERY time")
# print("2. Different files = Different checksums")
# print("3. Even tiny changes = Completely different checksum")


def read_file_attributes_pathlib(filepath):
    """Read file attributes using pathlib (Python 3.4+)"""
    try:
        path = Path(filepath)

        if not path.exists():
            print(f"File {filepath} not found")
            return

        file_stats = path.stat()

        print(f"File: {filepath}")
        print("-" * 40)
        print(f"Name: {path.name}")
        print(f"Suffix: {path.suffix}")
        print(f"Parent directory: {path.parent}")
        print(f"Absolute path: {path.absolute()}")
        print(f"Size: {file_stats.st_size} bytes")
        print(f"Modified: {datetime.fromtimestamp(file_stats.st_mtime)}")
        print(f"Is file: {path.is_file()}")
        print(f"Is directory: {path.is_dir()}")
        print(f"Is symlink: {path.is_symlink()}")

        # Human-readable size
        size_mb = file_stats.st_size / (1024 * 1024)
        if size_mb > 1:
            print(f"Size (MB): {size_mb:.2f} MB")
        else:
            size_kb = file_stats.st_size / 1024
            print(f"Size (KB): {size_kb:.2f} KB")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except OSError as e:
        print(f"OS error: {e}")
    except Exception as e:
        print(f"Error reading file attributes: {e}")


# Example usage
if __name__ == "__main__":
    print("\n=== Using pathlib ===")
    read_file_attributes_pathlib(
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\samples\Adobe Scan Aug 23, 2025 (1).pdf"
    )
