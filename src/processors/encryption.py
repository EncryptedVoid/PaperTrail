import base64
import json
import logging
import os
import secrets
import shutil
import string
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any , Dict , List

# Cryptography imports for secure file encryption operations
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from tqdm import tqdm

# Import all security-related constants from centralized configuration
from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	DEFAULT_PASSPHRASE_WORD_COUNT ,
	DEFAULT_PASSWORD_LENGTH ,
	DEFAULT_WORD_SEPARATOR ,
	ENCRYPTED_FILE_EXTENSION ,
	EXCLUDE_VISUALLY_SIMILAR_CHARS_IN_PASSWORD ,
	INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD ,
	KEY_DERIVATION_ITERATIONS ,
	MAX_PASSWORD_GENERATION_ATTEMPTS ,
	MINIMUM_WORD_LENGTH ,
	PASSPHRASE_WORDLIST_PATH ,
	PROFILE_PREFIX ,
	SALT_LENGTH_BYTES ,
	SIMILAR_CHARACTERS ,
	SYMBOL_CHARACTERS ,
)


def password_protect(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:
    """
    Extract metadata and text from all files in a directory and update their profiles.

    This method performs comprehensive metadata extraction by validating dependencies,
    discovering artifact files, loading existing profiles, extracting metadata and text
    using Apache Tika, updating profiles with extraction results, and moving processed
    files to appropriate directories based on success or failure.

    The extraction process includes:
    - Java runtime validation (version 11+)
    - Apache Tika JAR validation
    - Filesystem metadata extraction
    - Technical metadata extraction (EXIF, IPTC, document properties, etc.)
    - Full text content extraction
    - Profile updates with extraction timestamps and results
    - Graceful error handling with detailed logging

    Args:
            logger: Logger instance for tracking operations and errors
            source_dir: Directory containing artifact files to process. Files must follow
                    the ARTIFACT-{uuid}.ext naming convention
            failure_dir: Directory to move files that fail processing. Files are moved here
                    when profile loading, metadata extraction, or text extraction fails
            success_dir: Directory to move files after successful metadata extraction and
                    profile updates

    Returns:
            Tuple containing (metadata, text) from the last successfully processed file,
            or None if no files were processed successfully. Both elements may be None
            if no extraction occurred.

    Raises:
            FileNotFoundError: If Tika JAR is not found at the configured path or if
                    source directory does not exist
            RuntimeError: If Java is not installed, not in PATH, or version is below
                    the minimum required version (Java 11+)
            EnvironmentError: If Java version cannot be determined or other environment
                    issues prevent execution

    Note:
            - Files are processed in order of size (smallest first) for faster feedback
            - Requires corresponding PROFILE-{uuid}.json files in ARTIFACT_PROFILES_DIR
            - Tika commands timeout after 120 seconds to prevent hanging on large files
            - All profile updates include timestamps for audit trail
            - Extraction failures are logged with full exception details (exc_info=True)
    """

    # Log metadata extraction stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("ENCRYPTION STAGE")
    logger.info("=" * 80)

    # Discover all artifact files in the source directory
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]

    # Handle empty directory case
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

    # Sort files by size for consistent processing order (smaller files first)
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Process each artifact file
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Encrypting with passwords",
        unit="artifacts",
    ):
        try:
            logger.info(f"Processing artifact: {artifact.name}")

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            password = _encrypt_file(logger=logger, artifact=artifact)

            # Store extracted text if available
            if password and len(password) > 0:
                artifact_profile_data["password_protection_encryption"] = {
                    "success": True,
                    "password": password,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                artifact_profile_data["password_protection_encryption"] = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            # Update stage tracking
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            artifact_profile_data["stage_progression_data"][
                "password_protection_encryption"
            ] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile back to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory
            success_location: Path = success_dir / artifact.name

            # Handle naming conflicts
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                while success_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

        except Exception as e:
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                while failure_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue

    logger.info("Encryption stage completed")


def _decrypt_file(logger: Logger, file_path: Path, password: str) -> None:
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
        file_path: Path to the encrypted file - must exist and be readable
        password: Password used for original encryption - must match exactly

    Raises:
        FileNotFoundError: If the encrypted file doesn't exist or cannot be accessed
        ValueError: If password is empty, decryption fails, or file format is invalid
        PermissionError: If unable to read encrypted file or write decrypted output
        OSError: If file I/O operations fail due to system constraints

    Note:
        The encrypted file is not modified. A new decrypted file is created by
        removing the encrypted extension from the original filename.
    """
    # Log the start of decryption operation for audit trail
    logger.info(f"Starting file decryption for {file_path.name}")

    # Validate input parameters before proceeding
    if not password or not password.strip():
        error_msg: str = "Password cannot be empty or contain only whitespace"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not file_path.exists():
        error_msg = f"Encrypted file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Log encrypted file size for performance monitoring
    try:
        encrypted_artifact_size: int = file_path.stat().st_size
        logger.debug(f"Decrypting file of size: {encrypted_artifact_size:,} bytes")

    except OSError as stat_error:
        logger.warning(f"Could not determine encrypted file size: {stat_error}")

    # Read entire encrypted file contents for processing
    try:
        with open(file_path, "rb") as encrypted_file:
            encrypted_artifact_contents: bytes = encrypted_file.read()

        logger.debug(
            f"Read {len(encrypted_artifact_contents):,} bytes from encrypted file"
        )

    except PermissionError as permission_error:
        error_msg = f"Permission denied reading encrypted file: {file_path}"
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
    if file_path.suffix == ENCRYPTED_FILE_EXTENSION:
        # Standard case: remove .encrypted extension
        original_artifact_path: Path = file_path.with_suffix("")

    else:
        # Handle compound extensions: remove .encrypted from anywhere in filename
        original_artifact_path = Path(
            str(file_path).replace(ENCRYPTED_FILE_EXTENSION, "")
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
            f"Successfully decrypted file: {file_path.name} -> {original_artifact_path.name}"
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


def _encrypt_file(logger: Logger, artifact: Path, passphrase: bool = False) -> str:
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

    password = (
        _generate_passphrase(logger) if passphrase else _generate_password(logger)
    )

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
    file_path: Path = artifact.with_suffix(artifact.suffix + ENCRYPTED_FILE_EXTENSION)

    try:
        with open(file_path, "wb") as encrypted_file:
            # Write salt first for later key derivation during decryption
            encrypted_file.write(encryption_salt)

            # Write encrypted data after salt
            encrypted_file.write(encrypted_artifact_data)

            # Ensure data is written to disk immediately
            encrypted_file.flush()
            os.fsync(encrypted_file.fileno())

        logger.info(f"Successfully encrypted file: {artifact.name} -> {file_path.name}")

        return password

    except PermissionError as permission_error:
        error_msg = f"Permission denied writing encrypted file: {file_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg) from permission_error

    except OSError as write_error:
        error_msg = f"I/O error writing encrypted file: {write_error}"
        logger.error(error_msg)
        raise OSError(error_msg) from write_error


def _generate_password(logger: Logger) -> str:
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



    Returns:
        Cryptographically secure random password meeting complexity requirements

    Raises:
        ValueError: If DEFAULT_PASSWORD_LENGTH is less than 1

    Note:
        For passwords shorter than 4 characters, complexity validation is skipped
        to avoid infinite generation loops. Such short passwords are not recommended
        for security applications.
    """
    # Log password generation request for audit trail
    logger.debug(
        f"Generating password: length={DEFAULT_PASSWORD_LENGTH}, "
        f"symbols={INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD}, exclude_similar={EXCLUDE_VISUALLY_SIMILAR_CHARS_IN_PASSWORD}"
    )

    # Validate minimum password length requirement
    if DEFAULT_PASSWORD_LENGTH < 1:
        error_msg: str = "Password length must be at least 1 character"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Build character pool starting with basic alphanumeric characters
    available_characters: str = string.ascii_letters + string.digits
    logger.debug(f"Base character pool size: {len(available_characters)}")

    # Add special symbols if requested for increased complexity
    if INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD:
        available_characters += SYMBOL_CHARACTERS
        logger.debug(f"Added symbols, pool size: {len(available_characters)}")

    # Remove visually similar characters if requested for usability
    if EXCLUDE_VISUALLY_SIMILAR_CHARS_IN_PASSWORD:
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
            secrets.choice(available_characters) for _ in range(DEFAULT_PASSWORD_LENGTH)
        )

        # Skip complexity validation for very short passwords to avoid infinite loops
        if DEFAULT_PASSWORD_LENGTH < 4:
            logger.debug(f"Generated short password on attempt {generation_attempt}")
            return generated_password

        # Validate password meets complexity requirements
        has_lowercase_letter: bool = any(char.islower() for char in generated_password)
        has_uppercase_letter: bool = any(char.isupper() for char in generated_password)
        has_numeric_digit: bool = any(char.isdigit() for char in generated_password)
        has_special_symbol: bool = (
            any(char in SYMBOL_CHARACTERS for char in generated_password)
            if INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD
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
        if not has_special_symbol and INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD:
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


def _generate_passphrase(logger: Logger) -> str:
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
