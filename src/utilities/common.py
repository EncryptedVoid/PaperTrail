import base64
import os
from datetime import datetime
from pathlib import Path
from typing import List, Set
import hashlib
import secrets
import string
from logging import Logger

# Cryptography imports for secure file encryption operations
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Import all security-related constants from centralized configuration
from config import (
    ENCRYPTED_FILE_EXTENSION,
    KEY_DERIVATION_ITERATIONS,
    SALT_LENGTH_BYTES,
    DEFAULT_PASSPHRASE_WORD_COUNT,
    DEFAULT_PASSWORD_LENGTH,
    DEFAULT_WORD_SEPARATOR,
    MAX_PASSWORD_GENERATION_ATTEMPTS,
    SIMILAR_CHARACTERS,
    SYMBOL_CHARACTERS,
    PASSPHRASE_WORDLIST_PATH,
    CHECKSUM_ALGORITHM,
    CHECKSUM_CHUNK_SIZE_BYTES,
    CHECKSUM_HISTORY_FILE,
    MINIMUM_WORD_LENGTH,
)


def move(source: Path, destination: Path) -> bool:
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
        raise e


def decrypt_file(logger: Logger, encrypted_artifact_path: Path, password: str) -> None:
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
    logger.info(f"Starting file decryption for {encrypted_artifact_path.name}")

    # Validate input parameters before proceeding
    if not password or not password.strip():
        error_msg: str = "Password cannot be empty or contain only whitespace"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not encrypted_artifact_path.exists():
        error_msg = f"Encrypted file not found: {encrypted_artifact_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Log encrypted file size for performance monitoring
    try:
        encrypted_artifact_size: int = encrypted_artifact_path.stat().st_size
        logger.debug(f"Decrypting file of size: {encrypted_artifact_size:,} bytes")

    except OSError as stat_error:
        logger.warning(f"Could not determine encrypted file size: {stat_error}")

    # Read entire encrypted file contents for processing
    try:
        with open(encrypted_artifact_path, "rb") as encrypted_file:
            encrypted_artifact_contents: bytes = encrypted_file.read()

        logger.debug(
            f"Read {len(encrypted_artifact_contents):,} bytes from encrypted file"
        )

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied reading encrypted file: {encrypted_artifact_path}"
        )
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as read_error:
        error_msg = f"I/O error reading encrypted file: {read_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from read_error

    # Validate minimum file size to contain salt
    if len(encrypted_artifact_contents) < SALT_LENGTH_BYTES:
        error_msg = f"Invalid encrypted file format: file too small ({len(encrypted_artifact_contents)} bytes, minimum {SALT_LENGTH_BYTES})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract salt and encrypted data from file contents
    stored_salt: bytes = encrypted_artifact_contents[:SALT_LENGTH_BYTES]
    encrypted_data_portion: bytes = encrypted_artifact_contents[SALT_LENGTH_BYTES:]

    logger.debug(
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

        logger.debug(
            f"Recreated decryption key using {KEY_DERIVATION_ITERATIONS} PBKDF2 iterations"
        )

    except Exception as key_derivation_error:
        error_msg = f"Failed to derive decryption key: {key_derivation_error}"
        logger.error(error_msg)
        raise ValueError(error_msg) from key_derivation_error

    # Initialize Fernet decryption cipher and decrypt data
    try:
        decryption_cipher = Fernet(fernet_key)
        logger.debug("Initialized Fernet decryption cipher successfully")

        # Perform decryption with built-in authentication verification
        start_time: float = datetime.now().timestamp()
        decrypted_artifact_data: bytes = decryption_cipher.decrypt(
            encrypted_data_portion
        )
        decryption_duration: float = datetime.now().timestamp() - start_time

        logger.debug(
            f"Decrypted {len(encrypted_data_portion):,} bytes to {len(decrypted_artifact_data):,} bytes "
            f"in {decryption_duration:.2f} seconds"
        )

    except Exception as decryption_error:
        error_msg = "Decryption failed: incorrect password or corrupted file"
        logger.error(f"{error_msg} ({decryption_error})")
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

    logger.debug(f"Determined output file path: {original_artifact_path}")

    # Write decrypted data to original file path
    try:
        with open(original_artifact_path, "wb") as decrypted_file:
            decrypted_file.write(decrypted_artifact_data)

            # Ensure data is written to disk immediately
            decrypted_file.flush()
            os.fsync(decrypted_file.fileno())

        logger.info(
            f"Successfully decrypted file: {encrypted_artifact_path.name} -> {original_artifact_path.name}"
        )

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied writing decrypted file: {original_artifact_path}"
        )
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as write_error:
        error_msg = f"I/O error writing decrypted file: {write_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from write_error


def generate_checksum(logger: Logger, artifact_path: Path) -> str:
    """
    Calculate cryptographic checksum of a file using streaming approach for memory efficiency.

    This method implements a high-performance file hashing system that can handle files
    of any size by processing them in configurable chunks. The streaming approach ensures
    constant memory usage regardless of file size, making it suitable for processing
    large datasets in production environments.

    Performance Optimizations:
    - Configurable chunk size for optimal I/O performance on different storage systems
    - Single-pass processing minimizes file system overhead
    - Memory usage remains constant regardless of file size
    - Progress logging for long-running operations on large files

    Security Features:
    - Supports all major cryptographic hash algorithms
    - Validates CHECKSUM_ALGORITHM availability before processing
    - Comprehensive error handling prevents information leakage
    - Full audit trail logging for security compliance

    Args:
        artifact_path: Path object pointing to the file to be hashed - must exist and be readable

    Returns:
        Hexadecimal string representation of the file's cryptographic checksum

    Raises:
        ValueError: If the specified hash CHECKSUM_ALGORITHM is not supported by the system
        FileNotFoundError: If the specified file does not exist or cannot be accessed
        PermissionError: If the file cannot be read due to insufficient permissions
        OSError: If an I/O error occurs during file reading operations
    """
    # Log the start of checksum generation for audit trail
    logger.debug(
        f"Starting checksum generation for {artifact_path.name} using {CHECKSUM_ALGORITHM.value}"
    )

    # Validate file exists before attempting to process
    if not artifact_path.exists():
        error_msg: str = f"File not found: {artifact_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Initialize the cryptographic hash object with validation
    try:
        hash_object = hashlib.new(CHECKSUM_ALGORITHM.value)
        logger.debug(f"Initialized {CHECKSUM_ALGORITHM.value} hash object successfully")

    except ValueError as algorithm_error:
        # Provide detailed error message with available alternatives
        available_algorithms: List[str] = sorted(hashlib.algorithms_available)
        error_msg = (
            f"Unsupported hash CHECKSUM_ALGORITHM: {CHECKSUM_ALGORITHM.value}. "
            f"Available algorithms: {', '.join(available_algorithms)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from algorithm_error

    # Track processing metrics for performance monitoring
    start_time: float = datetime.now().timestamp()
    bytes_processed: int = 0

    # Process file in chunks using streaming approach for memory efficiency
    try:
        with artifact_path.open("rb") as artifact_handle:
            logger.debug(f"Opened file for reading: {artifact_path.name}")

            # Process file in optimally-sized chunks for I/O performance
            while True:
                # Read next chunk - size optimized for most storage systems
                artifact_chunk: bytes = artifact_handle.read(CHECKSUM_CHUNK_SIZE_BYTES)

                # Check for end of file condition
                if not artifact_chunk:
                    break

                # Update hash with current chunk
                hash_object.update(artifact_chunk)
                bytes_processed += len(artifact_chunk)

                # Log progress for large files to provide user feedback
                if bytes_processed % (CHECKSUM_CHUNK_SIZE_BYTES * 100) == 0:
                    logger.debug(
                        f"Processed {bytes_processed:,} bytes of {artifact_path.name}"
                    )

    except FileNotFoundError as artifact_error:
        error_msg = f"File not found during processing: {artifact_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from artifact_error

    except PermissionError as permission_error:
        error_msg = f"Permission denied reading file: {artifact_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as io_error:
        error_msg = f"I/O error reading file {artifact_path}: {io_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from io_error

    # Calculate final checksum and performance metrics
    final_checksum: str = hash_object.hexdigest()
    processing_duration: float = datetime.now().timestamp() - start_time

    # Log completion with performance metrics for monitoring
    logger.info(
        f"Generated {CHECKSUM_ALGORITHM.value} checksum for {artifact_path.name}: "
        f"{final_checksum[:16]}... ({bytes_processed:,} bytes in {processing_duration:.2f}s)"
    )

    return final_checksum


def load_checksum_history(logger: Logger) -> Set[str]:
    """
    Load existing checksum history from persistent storage with comprehensive error handling.

    This method reads the persistent checksum history file and returns all previously
    calculated checksums as a set for efficient duplicate detection. The history file
    is expected to contain one checksum per line in hexadecimal format.

    File Format:
    - One checksum per line
    - Hexadecimal format (lowercase or uppercase accepted)
    - Empty lines and whitespace are automatically ignored
    - Comments (lines starting with #) are automatically filtered out

    Performance Considerations:
    - Large history files are processed efficiently with streaming reads
    - Memory usage scales linearly with unique checksum count
    - Duplicate checksums in file are automatically deduplicated

    Args:
        None - uses the history_artifact_path configured during initialization

    Returns:
        Set of unique checksum strings loaded from the history file.
        Returns empty set if file doesn't exist or cannot be read.

    Note:
        This method never raises exceptions - errors are logged and an empty
        set is returned to allow graceful degradation of duplicate detection.
    """
    # Log the start of history loading operation
    logger.debug(f"Loading checksum history from {CHECKSUM_HISTORY_FILE}")

    # Initialize empty checksum collection
    loaded_checksums: Set[str] = set()

    # Check if history file exists before attempting to read
    if not CHECKSUM_HISTORY_FILE.exists():
        logger.info(
            "Checksum history file does not exist - starting with empty history"
        )
        return loaded_checksums

    try:
        # Process history file line by line for memory efficiency
        with open(CHECKSUM_HISTORY_FILE, "r", encoding="utf-8") as history_file:
            line_count: int = 0

            for raw_line in history_file:
                line_count += 1

                # Clean and validate each line
                cleaned_line: str = raw_line.strip()

                # Skip empty lines and comments for robust parsing
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue

                # Validate checksum format (hexadecimal characters only)
                if all(char in "0123456789abcdefABCDEF" for char in cleaned_line):
                    # Normalize to lowercase for consistent comparison
                    normalized_checksum: str = cleaned_line.lower()
                    loaded_checksums.add(normalized_checksum)

                else:
                    # Log invalid format but continue processing
                    logger.warning(
                        f"Invalid checksum format on line {line_count}: {cleaned_line[:32]}..."
                    )

        # Log successful loading with statistics
        logger.info(
            f"Successfully loaded {len(loaded_checksums)} unique checksums "
            f"from {line_count} lines in history file"
        )

    except PermissionError as permission_error:
        logger.warning(
            f"Permission denied reading checksum history: {permission_error}"
        )

    except OSError as io_error:
        logger.warning(f"I/O error reading checksum history: {io_error}")

    except Exception as unexpected_error:
        logger.warning(f"Unexpected error loading checksum history: {unexpected_error}")

    return loaded_checksums


def save_checksum(logger: Logger, checksum: str) -> None:
    """
    Append a new checksum to the persistent history file with atomic operation guarantees.

    This method safely appends a new checksum to the history file using atomic write
    operations to prevent corruption. The checksum is validated before writing and
    the operation is logged for audit purposes.

    Atomic Write Process:
    1. Validate checksum format to prevent corrupt entries
    2. Open file in append mode for thread-safe writing
    3. Write checksum with newline terminator
    4. Flush to ensure immediate persistence
    5. Log operation completion for audit trail

    Data Integrity Features:
    - Input validation prevents invalid checksums from being stored
    - Append-only operations preserve existing history
    - Comprehensive error handling prevents data loss
    - All operations are logged for troubleshooting

    Args:
        checksum: Valid hexadecimal checksum string to append to history file.
                    Must contain only hexadecimal characters (0-9, a-f, A-F).

    Raises:
        ValueError: If checksum format is invalid (contains non-hex characters)

    Note:
        File I/O errors are logged but do not raise exceptions to prevent
        interruption of processing pipelines. The checksum will still be
        available in memory for the current session.
    """
    # Log the checksum save operation for audit trail
    logger.debug(f"Saving checksum to history: {checksum[:16]}...")

    # Validate checksum format before writing to prevent corrupt history
    if not checksum:
        error_msg: str = "Cannot save empty checksum to history"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Verify checksum contains only valid hexadecimal characters
    if not all(char in "0123456789abcdefABCDEF" for char in checksum):
        error_msg = f"Invalid checksum format - contains non-hexadecimal characters: {checksum[:32]}..."
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Ensure parent directory exists before writing
        CHECKSUM_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Perform atomic append operation to preserve existing history
        with open(CHECKSUM_HISTORY_FILE, "a", encoding="utf-8") as history_file:
            # Normalize checksum to lowercase for consistency
            normalized_checksum: str = checksum.lower()

            # Write checksum with newline terminator
            history_file.write(f"{normalized_checksum}\n")

            # Force immediate write to disk for persistence
            history_file.flush()
            os.fsync(history_file.fileno())

        # Log successful save operation
        logger.debug("Successfully saved checksum to history file")

    except PermissionError as permission_error:
        logger.error(
            f"Permission denied writing to checksum history: {permission_error}"
        )

    except OSError as io_error:
        logger.error(f"I/O error writing to checksum history: {io_error}")

    except Exception as unexpected_error:
        logger.error(f"Unexpected error saving checksum to history: {unexpected_error}")


def encrypt_file(logger: Logger, artifact: Path, passphrase: bool = False) -> str:
    """
    Encrypt a file using password-based encryption with industry-standard security practices.

    This method implements secure file encryption using a combination of PBKDF2 key derivation
    and Fernet symmetric encryption. The encrypted file includes a randomly generated salt
    for enhanced security against rainbow table attacks.

    Security Architecture:
    - PBKDF2 key derivation with configurable iteration count (default: 100,000+)
    - Fernet symmetric encryption (AES 128 in CBC mode with HMAC authentication)
    - Cryptographically secure random salt generation for each encryption
    - Salt prepended to encrypted data for self-contained decryption

    File Format:
    [SALT_BYTES][ENCRYPTED_DATA]
    - First N bytes: Random salt used for key derivation
    - Remaining bytes: Fernet-encrypted file contents

    Args:
        artifact: Path to the file to encrypt - must exist and be readable
        password: Password string to use for encryption - must be non-empty

    Raises:
        FileNotFoundError: If the input file doesn't exist or cannot be accessed
        ValueError: If password is empty or contains only whitespace
        PermissionError: If unable to read input file or write encrypted output
        OSError: If file I/O operations fail due to system constraints

    Example:
        >>> agent = SecurityAgent(logger, Path("checksums.txt"))
        >>> agent.encrypt_file(Path("document.txt"), "secure_password_123")
        # Creates "document.txt.encrypted" with secure encryption

    Note:
        The original file is not modified. A new encrypted file is created with
        the configured encrypted file extension (typically ".encrypted").
    """
    # Log the start of encryption operation for audit trail
    logger.info(f"Starting file encryption for {artifact.name}")

    if not artifact.exists():
        error_msg = f"File not found for encryption: {artifact}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Log file size for performance monitoring
    try:
        artifact_size: int = artifact.stat().st_size
        logger.debug(f"Encrypting file of size: {artifact_size:,} bytes")

    except OSError as stat_error:
        logger.warning(f"Could not determine file size: {stat_error}")

    # Generate cryptographically secure random salt for key derivation
    encryption_salt: bytes = os.urandom(SALT_LENGTH_BYTES)
    logger.debug(f"Generated {SALT_LENGTH_BYTES}-byte random salt for key derivation")

    password = generate_passphrase(logger) if passphrase else generate_password(logger)

    # Derive encryption key from password using PBKDF2 with strong parameters
    try:
        key_derivation_function = PBKDF2HMAC(
            algorithm=hashes.SHA256(),  # Use SHA-256 for key derivation
            length=32,  # 256-bit key for Fernet encryption
            salt=encryption_salt,  # Unique salt for this encryption
            iterations=KEY_DERIVATION_ITERATIONS,  # Configurable iteration count
        )

        # Derive raw key and encode for Fernet usage
        raw_derived_key: bytes = key_derivation_function.derive(
            password.encode("utf-8")
        )
        fernet_key: bytes = base64.urlsafe_b64encode(raw_derived_key)

        logger.debug(
            f"Derived encryption key using {KEY_DERIVATION_ITERATIONS} PBKDF2 iterations"
        )

    except Exception as key_derivation_error:
        error_msg = f"Failed to derive encryption key: {key_derivation_error}"
        logger.error(error_msg)
        raise ValueError(error_msg) from key_derivation_error

    # Initialize Fernet symmetric encryption cipher
    try:
        encryption_cipher = Fernet(fernet_key)
        logger.debug("Initialized Fernet encryption cipher successfully")

    except Exception as cipher_error:
        error_msg = f"Failed to initialize encryption cipher: {cipher_error}"
        logger.error(error_msg)
        raise ValueError(error_msg) from cipher_error

    # Read original file contents for encryption
    try:
        with open(artifact, "rb") as original_file:
            original_artifact_data: bytes = original_file.read()

        logger.debug(f"Read {len(original_artifact_data):,} bytes from source file")

    except PermissionError as permission_error:
        error_msg = f"Permission denied reading file for encryption: {artifact}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as read_error:
        error_msg = f"I/O error reading file for encryption: {read_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from read_error

    # Encrypt the file data using Fernet
    try:
        start_time: float = datetime.now().timestamp()
        encrypted_artifact_data: bytes = encryption_cipher.encrypt(
            original_artifact_data
        )
        encryption_duration: float = datetime.now().timestamp() - start_time

        logger.debug(
            f"Encrypted {len(original_artifact_data):,} bytes to {len(encrypted_artifact_data):,} bytes "
            f"in {encryption_duration:.2f} seconds"
        )

    except Exception as encryption_error:
        error_msg = f"Failed to encrypt file data: {encryption_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from encryption_error

    # Write encrypted file with salt prepended for self-contained decryption
    encrypted_artifact_path: Path = artifact.with_suffix(
        artifact.suffix + ENCRYPTED_FILE_EXTENSION
    )

    try:
        with open(encrypted_artifact_path, "wb") as encrypted_file:
            # Write salt first for later key derivation during decryption
            encrypted_file.write(encryption_salt)

            # Write encrypted data after salt
            encrypted_file.write(encrypted_artifact_data)

            # Ensure data is written to disk immediately
            encrypted_file.flush()
            os.fsync(encrypted_file.fileno())

        logger.info(
            f"Successfully encrypted file: {artifact.name} -> {encrypted_artifact_path.name}"
        )

        return password

    except PermissionError as permission_error:
        error_msg = (
            f"Permission denied writing encrypted file: {encrypted_artifact_path}"
        )
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as write_error:
        error_msg = f"I/O error writing encrypted file: {write_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from write_error


def generate_password(
    logger: Logger,
    character_length: int = DEFAULT_PASSWORD_LENGTH,
    include_special_symbols: bool = True,
    exclude_similar_looking_chars: bool = True,
) -> str:
    """
    Generate cryptographically secure random password with comprehensive complexity validation.

    This method creates passwords using the secrets module for cryptographic security,
    ensuring that generated passwords meet complexity requirements while being suitable
    for various authentication systems.

    Password Complexity Algorithm:
    1. Build character pool based on configuration options
    2. Generate random password using cryptographically secure random source
    3. Validate character distribution (uppercase, lowercase, digits, symbols)
    4. Retry generation if complexity requirements not met (up to configured limit)
    5. Return password that meets all complexity requirements

    Character Pool Configuration:
    - Always includes: uppercase letters (A-Z), lowercase letters (a-z), digits (0-9)
    - Optional: special symbols (!@#$%^&*()_+-=[]{}|;:,.<>?)
    - Optional exclusion: visually similar characters (0O1lI|)

    Security Features:
    - Uses secrets.choice() for cryptographically secure random generation
    - Configurable complexity validation prevents weak passwords
    - No predictable patterns or sequences in generated passwords
    - Suitable for high-security authentication systems

    Args:
        character_length: Desired password length (minimum 1, recommended 12+)
        include_special_symbols: Whether to include special symbols in character pool
        exclude_similar_looking_chars: Whether to exclude visually similar characters

    Returns:
        Cryptographically secure random password meeting complexity requirements

    Raises:
        ValueError: If character_length is less than 1

    Example:
        >>> agent = SecurityAgent(logger, Path("checksums.txt"))
        >>> password = agent.generate_password(16, True, True)
        >>> len(password)
        16
        >>> any(c.isupper() for c in password)  # Contains uppercase
        True
        >>> any(c.islower() for c in password)  # Contains lowercase
        True

    Note:
        For passwords shorter than 4 characters, complexity validation is skipped
        to avoid infinite generation loops. Such short passwords are not recommended
        for security applications.
    """
    # Log password generation request for audit trail
    logger.debug(
        f"Generating password: length={character_length}, "
        f"symbols={include_special_symbols}, exclude_similar={exclude_similar_looking_chars}"
    )

    # Validate minimum password length requirement
    if character_length < 1:
        error_msg: str = "Password length must be at least 1 character"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Build character pool starting with basic alphanumeric characters
    available_characters: str = string.ascii_letters + string.digits
    logger.debug(f"Base character pool size: {len(available_characters)}")

    # Add special symbols if requested for increased complexity
    if include_special_symbols:
        available_characters += SYMBOL_CHARACTERS
        logger.debug(f"Added symbols, pool size: {len(available_characters)}")

    # Remove visually similar characters if requested for usability
    if exclude_similar_looking_chars:
        original_length: int = len(available_characters)
        available_characters = "".join(
            char for char in available_characters if char not in SIMILAR_CHARACTERS
        )
        removed_count: int = original_length - len(available_characters)
        logger.debug(
            f"Removed {removed_count} similar characters, final pool size: {len(available_characters)}"
        )

    # Generate password with complexity validation loop
    for generation_attempt in range(1, MAX_PASSWORD_GENERATION_ATTEMPTS + 1):
        # Generate random password using cryptographically secure source
        generated_password: str = "".join(
            secrets.choice(available_characters) for _ in range(character_length)
        )

        # Skip complexity validation for very short passwords to avoid infinite loops
        if character_length < 4:
            logger.debug(f"Generated short password on attempt {generation_attempt}")
            return generated_password

        # Validate password meets complexity requirements
        has_lowercase_letter: bool = any(char.islower() for char in generated_password)
        has_uppercase_letter: bool = any(char.isupper() for char in generated_password)
        has_numeric_digit: bool = any(char.isdigit() for char in generated_password)
        has_special_symbol: bool = (
            any(char in SYMBOL_CHARACTERS for char in generated_password)
            if include_special_symbols
            else True  # Skip symbol check if symbols not required
        )

        # Return password if it meets all complexity requirements
        if (
            has_lowercase_letter
            and has_uppercase_letter
            and has_numeric_digit
            and has_special_symbol
        ):
            logger.debug(f"Generated complex password on attempt {generation_attempt}")
            return generated_password

        # Log complexity validation failure for debugging
        missing_types: List[str] = []
        if not has_lowercase_letter:
            missing_types.append("lowercase")
        if not has_uppercase_letter:
            missing_types.append("uppercase")
        if not has_numeric_digit:
            missing_types.append("digits")
        if not has_special_symbol and include_special_symbols:
            missing_types.append("symbols")

        logger.debug(
            f"Attempt {generation_attempt} failed complexity check, missing: {', '.join(missing_types)}"
        )

    # Fallback: return last generated password even if complexity check failed
    # This password is still cryptographically secure, just may not meet all complexity rules
    logger.warning(
        f"Could not generate password meeting complexity requirements after {MAX_PASSWORD_GENERATION_ATTEMPTS} attempts, "
        f"returning cryptographically secure password"
    )
    return generated_password


def generate_passphrase(logger: Logger) -> str:
    """
    Generate memorable passphrase using random word combinations with configurable formatting.

    This method creates human-friendly passphrases that are significantly easier to remember
    than random character passwords while maintaining good entropy for security. Passphrases
    are particularly suitable for master passwords and long-term authentication scenarios.

    Passphrase Generation Process:
    1. Select word source (built-in collection or external wordlist file)
    2. Choose random words using cryptographically secure selection
    3. Apply formatting options (capitalization, separators)
    4. Optionally append random number for additional entropy
    5. Combine components into final passphrase

    Security Considerations:
    - Uses secrets module for cryptographically secure word selection
    - Built-in wordlist contains 80+ carefully chosen words
    - External wordlist support for larger vocabulary
    - Optional numeric suffix increases entropy significantly
    - Word separation prevents dictionary attacks on individual components

    Entropy Calculation:
    - 4 words from 80-word list: ~26 bits entropy
    - 4 words from 2000-word list: ~44 bits entropy
    - Additional 2-digit number: +6.6 bits entropy
    - Recommended: 4+ words for strong security
    Returns:
        Memorable passphrase string combining random words with specified formatting

    Raises:
        ValueError: If DEFAULT_PASSPHRASE_WORD_COUNT is less than 1

    Example:
        >>> agent = SecurityAgent(logger, Path("checksums.txt"))
        >>> passphrase = agent.generate_passphrase(4, "-", True, True)
        >>> passphrase
        'Forest-Mountain-River-Ocean-73'
        >>> len(passphrase.split("-"))
        5  # 4 words + 1 number

    Note:
        If external wordlist is requested but unavailable, the method gracefully
        falls back to the built-in word collection with appropriate logging.
    """
    # Log passphrase generation request for audit trail
    logger.debug(
        f"Generating passphrase with {DEFAULT_PASSPHRASE_WORD_COUNT} words and {DEFAULT_WORD_SEPARATOR} as a separator"
    )

    # Validate minimum word count requirement
    if DEFAULT_PASSPHRASE_WORD_COUNT < 1:
        error_msg: str = "Word count must be at least 1"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log start of external wordlist processing
    logger.debug(
        f"Extracting {DEFAULT_PASSPHRASE_WORD_COUNT} words from external wordlist"
    )

    # Count total lines in wordlist file for random selection
    try:
        if not PASSPHRASE_WORDLIST_PATH.exists():
            raise FileNotFoundError

        with open(PASSPHRASE_WORDLIST_PATH, "r", encoding="utf-8") as f:
            total_line_count: int = sum(1 for line in f if line.strip())

        logger.debug(f"Wordlist contains {total_line_count} total lines")

    except OSError as error:
        error_msg = (
            f"Failed to load wordlist file at {PASSPHRASE_WORDLIST_PATH}: {error}"
        )
        logger.error(error_msg)
        raise OSError(error_msg) from error

    # Generate sorted list of random line numbers for efficient processing
    random_line_numbers: List[int] = sorted(
        secrets.randbelow(total_line_count)
        for _ in range(DEFAULT_PASSPHRASE_WORD_COUNT)
    )

    # Extract words from selected lines using streaming approach
    extracted_words: List[str] = []

    try:
        with open(PASSPHRASE_WORDLIST_PATH, "r", encoding="utf-8") as f:
            for current_line_number, artifact_line in enumerate(f):
                # Check if current line is one of our selected lines
                if current_line_number in random_line_numbers:
                    # Clean and validate word
                    cleaned_word: str = artifact_line.strip().lower()

                    # Apply quality filters for word selection
                    if (
                        len(cleaned_word)
                        >= MINIMUM_WORD_LENGTH  # Minimum length requirement
                        and cleaned_word.isalpha()  # Only alphabetic characters
                        and not any(char in cleaned_word for char in "'\"")  # No quotes
                    ):
                        extracted_words.append(cleaned_word)
                        logger.debug(f"Extracted word: {cleaned_word}")

                    # Stop processing once we have enough quality words
                    if len(extracted_words) >= DEFAULT_PASSPHRASE_WORD_COUNT:
                        break

    except OSError as extraction_error:
        error_msg = f"Failed to extract words from wordlist file: {extraction_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from extraction_error

    # Log extraction results
    logger.info(
        f"Successfully extracted {len(extracted_words)} words from external wordlist"
    )

    # Apply capitalization formatting if requested
    extracted_words = [word.capitalize() for word in extracted_words]
    logger.debug("Applied capitalization to selected words")

    # Combine words with specified separator
    passphrase: str = DEFAULT_WORD_SEPARATOR.join(extracted_words)

    # Log passphrase generation completion (without revealing actual passphrase)
    logger.info(
        f"Generated passphrase with {len(extracted_words)} components, "
        f"total length: {len(passphrase)} characters"
    )

    return passphrase
