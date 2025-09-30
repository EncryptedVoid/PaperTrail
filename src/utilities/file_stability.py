import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def create_par2(
    logger: logging.Logger,
    file_path: str,
    redundancy_percent: int = 5,
    block_size: Optional[int] = None,
) -> bool:
    """
    Create PAR2 recovery files for a given file.

    This MUST be done before you can verify or repair files.
    Creates .par2 files that contain parity data for error detection/correction.

    Args:
                    logger: Logger instance for logging messages
                    file_path: Path to the file to protect
                    redundancy_percent: Percentage of recovery data (1-100). Default 5%.
                                                                                            Higher = more corruption can be repaired, but more storage used
                    block_size: Optional block size in bytes. None = auto-calculate

    Returns:
                    True if PAR2 files created successfully, False otherwise

    Example:
                    create_par2(logger, "important.zip", redundancy_percent=10)
                    Creates: important.zip.par2, important.zip.vol0+1.par2, etc.
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return False

    if not 1 <= redundancy_percent <= 100:
        logger.error(
            f"Redundancy percent must be between 1-100, got: {redundancy_percent}"
        )
        return False

    logger.info(f"Creating PAR2 files for: {file_path}")
    logger.info(f"Redundancy: {redundancy_percent}%")

    try:
        # Build par2 create command
        # Format: par2 create -r<redundancy> [-s<blocksize>] <par2file> <sourcefile>
        file_path_obj = Path(file_path)
        par2_output = str(file_path_obj.with_suffix(file_path_obj.suffix + ".par2"))

        cmd = ["par2", "create", f"-r{redundancy_percent}"]

        if block_size:
            cmd.append(f"-s{block_size}")

        cmd.extend([par2_output, file_path])

        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run par2 create
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for large files
        )

        if result.returncode == 0:
            logger.info(f"PAR2 files created successfully for: {file_path}")
            logger.debug(f"PAR2 output: {result.stdout}")
            return True
        else:
            logger.error(f"Failed to create PAR2 files for: {file_path}")
            logger.debug(f"PAR2 output: {result.stdout}")
            logger.debug(f"PAR2 errors: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"PAR2 creation timeout for: {file_path}")
        return False
    except FileNotFoundError:
        logger.error("PAR2 command not found. Please install par2 utility.")
        logger.error("Linux: sudo apt install par2")
        logger.error(
            "Windows: Download from https://github.com/animetosho/par2cmdline/releases"
        )
        return False
    except Exception as e:
        logger.error(f"Error creating PAR2 files: {e}")
        return False


def find_par2_file(file_path: str) -> Optional[str]:
    """
    Find the corresponding PAR2 file for the given file.
    PAR2 files are typically named: filename.par2 or filename.vol00+01.par2

    Returns the main .par2 file path if found, None otherwise.
    """
    file_path_obj = Path(file_path)
    directory = file_path_obj.parent
    filename = file_path_obj.name

    # Look for the main PAR2 file
    main_par2 = directory / f"{filename}.par2"
    if main_par2.exists():
        return str(main_par2)

    # Look for volume-based PAR2 files (e.g., filename.vol00+01.par2)
    for par2_file in directory.glob(f"{filename}.vol*.par2"):
        return str(par2_file)

    return None


def is_stable(logger: logging.Logger, file_path: str) -> bool:
    """
    Check if a file is stable/uncorrupted using PAR2 verification.

    Args:
                    logger: Logger instance for logging messages
                    file_path: Path to the file to verify

    Returns:
                    True if file is stable, False if corrupted or PAR2 files not found
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    # Find the PAR2 file
    par2_file = find_par2_file(file_path)
    if not par2_file:
        logger.warning(f"No PAR2 file found for: {file_path}")
        return False

    logger.info(f"Verifying file stability: {file_path}")
    logger.debug(f"Using PAR2 file: {par2_file}")

    try:
        # Run par2 verify command
        result = subprocess.run(
            ["par2", "verify", par2_file],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # PAR2 returns 0 for success, non-zero for errors/corruption
        if result.returncode == 0:
            logger.info(f"File is stable: {file_path}")
            return True
        else:
            logger.warning(f"File verification failed: {file_path}")
            logger.debug(f"PAR2 output: {result.stdout}")
            logger.debug(f"PAR2 errors: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Verification timeout for: {file_path}")
        return False
    except FileNotFoundError:
        logger.error("PAR2 command not found. Please install par2 utility.")
        return False
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False


def repair_instability(
    logger: logging.Logger, file_path: str, temp_directory: str, archive_directory: str
) -> bool:
    """
    Attempt to repair a corrupted file using PAR2.

    Args:
                    logger: Logger instance for logging messages
                    file_path: Path to the corrupted file
                    temp_directory: Directory to store temporary backup
                    archive_directory: Directory to move failed files

    Returns:
                    True if repair successful, False if repair failed

    Workflow:
                    1. Copy file to temp directory (backup)
                    2. Attempt PAR2 repair on original file
                    3. If successful: delete temp backup, keep repaired file
                    4. If failed: delete corrupted file, move backup to archive
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    # Find PAR2 file
    par2_file = find_par2_file(file_path)
    if not par2_file:
        logger.error(f"No PAR2 file found for repair: {file_path}")
        return False

    # Create directories if they don't exist
    os.makedirs(temp_directory, exist_ok=True)
    os.makedirs(archive_directory, exist_ok=True)

    file_path_obj = Path(file_path)
    filename = file_path_obj.name
    temp_backup_path = os.path.join(temp_directory, filename)
    archive_path = os.path.join(archive_directory, filename)

    # Handle duplicate names in archive
    if os.path.exists(archive_path):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(archive_path):
            archive_path = os.path.join(archive_directory, f"{base}_{counter}{ext}")
            counter += 1

    logger.info(f"Starting repair process for: {file_path}")

    try:
        # Step 1: Create backup in temp directory
        logger.debug(f"Creating backup: {temp_backup_path}")
        shutil.copy2(file_path, temp_backup_path)

        # Step 2: Attempt repair with PAR2
        logger.info("Attempting PAR2 repair...")
        result = subprocess.run(
            ["par2", "repair", par2_file],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for repair
        )

        # Step 3: Check repair result
        if result.returncode == 0:
            # Repair successful
            logger.info(f"Repair successful: {file_path}")
            logger.debug(f"PAR2 output: {result.stdout}")

            # Delete temp backup
            os.remove(temp_backup_path)
            logger.debug(f"Deleted temp backup: {temp_backup_path}")

            return True
        else:
            # Repair failed
            logger.warning(f"Repair failed: {file_path}")
            logger.debug(f"PAR2 output: {result.stdout}")
            logger.debug(f"PAR2 errors: {result.stderr}")

            # Delete corrupted file
            os.remove(file_path)
            logger.debug(f"Deleted corrupted file: {file_path}")

            # Move backup to archive
            shutil.move(temp_backup_path, archive_path)
            logger.info(f"Moved backup to archive: {archive_path}")

            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Repair timeout for: {file_path}")
        # Clean up: remove temp backup, keep original
        if os.path.exists(temp_backup_path):
            os.remove(temp_backup_path)
        return False

    except FileNotFoundError:
        logger.error("PAR2 command not found. Please install par2 utility.")
        # Clean up temp backup
        if os.path.exists(temp_backup_path):
            os.remove(temp_backup_path)
        return False

    except Exception as e:
        logger.error(f"Error during repair: {e}")
        # Try to clean up temp backup
        try:
            if os.path.exists(temp_backup_path):
                os.remove(temp_backup_path)
        except:
            pass
        return False
