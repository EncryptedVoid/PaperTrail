from pathlib import Path


def move_file_safely(source: Path, destination: Path) -> bool:
    """
    Safely move a file from source to destination, handling naming conflicts gracefully.

    If the destination file already exists, this method automatically appends a counter
    to the filename (e.g., file.txt -> file_1.txt -> file_2.txt) to avoid overwrites.

    Args:
        source: Source file path to move from
        destination: Target destination path to move to

    Returns:
        True if the file was successfully moved, False if an error occurred

    Note:
        This method preserves file extensions and handles edge cases like
        files without extensions or complex naming scenarios.
    """
    try:
        # Handle naming conflicts by appending a counter to avoid overwrites
        counter = 1
        original_destination = destination

        # Keep incrementing counter until we find an available filename
        while destination.exists():
            name_part = original_destination.stem  # Filename without extension
            ext_part = original_destination.suffix  # File extension including the dot
            destination = (
                original_destination.parent / f"{name_part}_{counter}{ext_part}"
            )
            counter += 1

        # Perform the actual file move operation
        source.rename(destination)
        return True

    except Exception as e:
        return False
