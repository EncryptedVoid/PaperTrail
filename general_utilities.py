from pathlib import Path
from datetime import datetime
import hashlib
import uuid


# Basic UUID generation
def generate_item_id() -> str:
    return str(uuid.uuid4())


def calculate_checksum(file_path, algorithm="sha512"):
    """Calculate checksum of a file using specified algorithm."""
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


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
