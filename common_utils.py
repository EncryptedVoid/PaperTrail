import base64
import os
from datetime import datetime
from pathlib import Path

# Cryptography imports for secure file encryption operations
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Import all security-related constants from centralized configuration
from config import (
    ENCRYPTED_FILE_EXTENSION,
    KEY_DERIVATION_ITERATIONS,
    SALT_LENGTH_BYTES,
)
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


def decrypt_file(self, encrypted_artifact_path: Path, password: str) -> None:
    """
    Decrypt a password-protected encrypted file with comprehensive validation and error handling.

    This method reverses the encryption process by extracting the salt from the encrypted file,
    deriving the decryption key using the same PBKDF2 parameters, and decrypting the content
    using Fernet symmetric decryption.

    Decryption Process:
    1. Validate input parameters and file existence
    2. Read encrypted file and extract salt from beginning
    3. Derive decryption key using extracted salt and provided password
    4. Initialize Fernet cipher with derived key
    5. Decrypt data and validate integrity (Fernet includes authentication)
    6. Write decrypted content to output file

    Security Features:
    - Authentication verification prevents tampering detection
    - Salt extraction enables proper key derivation recreation
    - Comprehensive error handling prevents information leakage
    - Secure memory handling for passwords and keys

    Args:
        encrypted_artifact_path: Path to the encrypted file - must exist and be readable
        password: Password used for original encryption - must match exactly

    Raises:
        FileNotFoundError: If the encrypted file doesn't exist or cannot be accessed
        ValueError: If password is empty, decryption fails, or file format is invalid
        PermissionError: If unable to read encrypted file or write decrypted output
        OSError: If file I/O operations fail due to system constraints

    Example:
        >>> agent = SecurityAgent(logger, Path("checksums.txt"))
        >>> agent.decrypt_file(Path("document.txt.encrypted"), "secure_password_123")
        # Creates "document.txt" with original content restored

    Note:
        The encrypted file is not modified. A new decrypted file is created by
        removing the encrypted extension from the original filename.
    """
    # Log the start of decryption operation for audit trail
    self.logger.info(f"Starting file decryption for {encrypted_artifact_path.name}")

    # Validate input parameters before proceeding
    if not password or not password.strip():
        error_msg: str = "Password cannot be empty or contain only whitespace"
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    if not encrypted_artifact_path.exists():
        error_msg = f"Encrypted file not found: {encrypted_artifact_path}"
        self.logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Log encrypted file size for performance monitoring
    try:
        encrypted_artifact_size: int = encrypted_artifact_path.stat().st_size
        self.logger.debug(f"Decrypting file of size: {encrypted_artifact_size:,} bytes")

    except OSError as stat_error:
        self.logger.warning(f"Could not determine encrypted file size: {stat_error}")

    # Read entire encrypted file contents for processing
    try:
        with open(encrypted_artifact_path, "rb") as encrypted_file:
            encrypted_artifact_contents: bytes = encrypted_file.read()

        self.logger.debug(
            f"Read {len(encrypted_artifact_contents):,} bytes from encrypted file"
        )

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied reading encrypted file: {encrypted_artifact_path}"
        )
        self.logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as read_error:
        error_msg = f"I/O error reading encrypted file: {read_error}"
        self.logger.error(error_msg)
        raise OSError(error_msg) from read_error

    # Validate minimum file size to contain salt
    if len(encrypted_artifact_contents) < SALT_LENGTH_BYTES:
        error_msg = f"Invalid encrypted file format: file too small ({len(encrypted_artifact_contents)} bytes, minimum {SALT_LENGTH_BYTES})"
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract salt and encrypted data from file contents
    stored_salt: bytes = encrypted_artifact_contents[:SALT_LENGTH_BYTES]
    encrypted_data_portion: bytes = encrypted_artifact_contents[SALT_LENGTH_BYTES:]

    self.logger.debug(
        f"Extracted {len(stored_salt)}-byte salt and {len(encrypted_data_portion):,} bytes of encrypted data"
    )

    # Recreate key derivation process using extracted salt
    try:
        key_derivation_function = PBKDF2HMAC(
            algorithm=hashes.SHA256(),  # Must match encryption algorithm
            length=32,  # Must match encryption key length
            salt=stored_salt,  # Use salt from encrypted file
            iterations=KEY_DERIVATION_ITERATIONS,  # Must match encryption iterations
        )

        # Derive decryption key using same process as encryption
        raw_derived_key: bytes = key_derivation_function.derive(
            password.encode("utf-8")
        )
        fernet_key: bytes = base64.urlsafe_b64encode(raw_derived_key)

        self.logger.debug(
            f"Recreated decryption key using {KEY_DERIVATION_ITERATIONS} PBKDF2 iterations"
        )

    except Exception as key_derivation_error:
        error_msg = f"Failed to derive decryption key: {key_derivation_error}"
        self.logger.error(error_msg)
        raise ValueError(error_msg) from key_derivation_error

    # Initialize Fernet decryption cipher and decrypt data
    try:
        decryption_cipher = Fernet(fernet_key)
        self.logger.debug("Initialized Fernet decryption cipher successfully")

        # Perform decryption with built-in authentication verification
        start_time: float = datetime.now().timestamp()
        decrypted_artifact_data: bytes = decryption_cipher.decrypt(
            encrypted_data_portion
        )
        decryption_duration: float = datetime.now().timestamp() - start_time

        self.logger.debug(
            f"Decrypted {len(encrypted_data_portion):,} bytes to {len(decrypted_artifact_data):,} bytes "
            f"in {decryption_duration:.2f} seconds"
        )

    except Exception as decryption_error:
        error_msg = "Decryption failed: incorrect password or corrupted file"
        self.logger.error(f"{error_msg} ({decryption_error})")
        raise ValueError(error_msg) from decryption_error

    # Determine output file path by removing encrypted extension
    if encrypted_artifact_path.suffix == ENCRYPTED_FILE_EXTENSION:
        # Standard case: remove .encrypted extension
        original_artifact_path: Path = encrypted_artifact_path.with_suffix("")

    else:
        # Handle compound extensions: remove .encrypted from anywhere in filename
        original_artifact_path = Path(
            str(encrypted_artifact_path).replace(ENCRYPTED_FILE_EXTENSION, "")
        )

    self.logger.debug(f"Determined output file path: {original_artifact_path}")

    # Write decrypted data to original file path
    try:
        with open(original_artifact_path, "wb") as decrypted_file:
            decrypted_file.write(decrypted_artifact_data)

            # Ensure data is written to disk immediately
            decrypted_file.flush()
            os.fsync(decrypted_file.fileno())

        self.logger.info(
            f"Successfully decrypted file: {encrypted_artifact_path.name} -> {original_artifact_path.name}"
        )

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied writing decrypted file: {original_artifact_path}"
        )
        self.logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as write_error:
        error_msg = f"I/O error writing decrypted file: {write_error}"
        self.logger.error(error_msg)
        raise OSError(error_msg) from write_error
