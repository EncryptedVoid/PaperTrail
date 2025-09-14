"""
Password generator with file encryption capabilities.

This module provides secure password generation, memorable passphrase creation,
and cross-platform file encryption using industry-standard cryptographic methods.

Features:
    - Cryptographically secure random password generation
    - Memorable passphrase creation with customizable options
    - File encryption/decryption with password protection
    - Cross-platform compatibility
    - Type-safe implementation with comprehensive error handling

Dependencies:
    - cryptography: For secure file encryption/decryption
    - secrets: For cryptographically secure random number generation
"""

import secrets
import string
import os
from typing import List, Optional, Union
from pathlib import Path

# Cryptography imports for file encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


# File encryption constants
SALT_LENGTH_BYTES: int = 16
KEY_DERIVATION_ITERATIONS: int = 100000
ENCRYPTED_FILE_EXTENSION: str = ".encrypted"

# Password generation constants
DEFAULT_PASSWORD_LENGTH: int = 16
DEFAULT_PASSPHRASE_WORD_COUNT: int = 6
DEFAULT_WORD_SEPARATOR: str = "-"
SIMILAR_CHARACTERS: str = "0O1lI|"  # Characters that look similar and cause confusion
SYMBOL_CHARACTERS: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
MINIMUM_WORD_LENGTH: int = 3
RANDOM_NUMBER_MIN: int = 10
RANDOM_NUMBER_MAX: int = 99
MAX_PASSWORD_GENERATION_ATTEMPTS: int = 50


def encrypt_file(file_path: Union[str, Path], password: str) -> None:
    """
    Encrypt a file using password-based encryption with PBKDF2 key derivation.

    Creates an encrypted version of the file with a randomly generated salt
    for enhanced security. The encrypted file will have '.encrypted' extension.

    Args:
        file_path: Path to the file to encrypt (string or Path object)
        password: Password to use for encryption (must be non-empty string)

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If password is empty or invalid
        PermissionError: If unable to read input file or write encrypted file
        OSError: If file I/O operations fail

    Example:
        >>> encrypt_file("document.txt", "secure_password_123")
        # Creates "document.txt.encrypted"
    """
    if not password:
        raise ValueError("Password cannot be empty")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Generate cryptographically secure random salt
    encryption_salt: bytes = os.urandom(SALT_LENGTH_BYTES)

    # Derive encryption key from password using PBKDF2
    key_derivation_function = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=encryption_salt,
        iterations=KEY_DERIVATION_ITERATIONS,
    )
    derived_key: bytes = base64.urlsafe_b64encode(
        key_derivation_function.derive(password.encode("utf-8"))
    )

    # Initialize Fernet symmetric encryption cipher
    encryption_cipher = Fernet(derived_key)

    # Read original file contents
    try:
        with open(file_path, "rb") as original_file:
            original_file_data: bytes = original_file.read()
    except PermissionError:
        raise PermissionError(f"Permission denied reading file: {file_path}")

    # Encrypt the file data
    encrypted_file_data: bytes = encryption_cipher.encrypt(original_file_data)

    # Write encrypted file with salt prepended for later key derivation
    encrypted_file_path = file_path.with_suffix(
        file_path.suffix + ENCRYPTED_FILE_EXTENSION
    )
    try:
        with open(encrypted_file_path, "wb") as encrypted_file:
            # Salt must be stored with encrypted data for decryption
            encrypted_file.write(encryption_salt + encrypted_file_data)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing encrypted file: {encrypted_file_path}"
        )


def decrypt_file(encrypted_file_path: Union[str, Path], password: str) -> None:
    """
    Decrypt a password-protected encrypted file.

    Reads the encrypted file, extracts the salt, derives the decryption key,
    and creates the original file. The decrypted file will have the same name
    as the encrypted file but without the '.encrypted' extension.

    Args:
        encrypted_file_path: Path to the encrypted file (string or Path object)
        password: Password used for original encryption

    Raises:
        FileNotFoundError: If the encrypted file doesn't exist
        ValueError: If password is empty or decryption fails (wrong password)
        PermissionError: If unable to read encrypted file or write decrypted file
        OSError: If file I/O operations fail

    Example:
        >>> decrypt_file("document.txt.encrypted", "secure_password_123")
        # Creates "document.txt"
    """
    if not password:
        raise ValueError("Password cannot be empty")

    encrypted_file_path = Path(encrypted_file_path)
    if not encrypted_file_path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")

    # Read encrypted file contents
    try:
        with open(encrypted_file_path, "rb") as encrypted_file:
            encrypted_file_contents: bytes = encrypted_file.read()
    except PermissionError:
        raise PermissionError(
            f"Permission denied reading encrypted file: {encrypted_file_path}"
        )

    if len(encrypted_file_contents) < SALT_LENGTH_BYTES:
        raise ValueError("Invalid encrypted file format: file too small")

    # Extract salt and encrypted data
    stored_salt: bytes = encrypted_file_contents[:SALT_LENGTH_BYTES]
    encrypted_data_portion: bytes = encrypted_file_contents[SALT_LENGTH_BYTES:]

    # Recreate the same key derivation process used during encryption
    key_derivation_function = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=stored_salt,
        iterations=KEY_DERIVATION_ITERATIONS,
    )

    try:
        derived_key: bytes = base64.urlsafe_b64encode(
            key_derivation_function.derive(password.encode("utf-8"))
        )
    except Exception:
        raise ValueError("Failed to derive decryption key")

    # Initialize decryption cipher and decrypt data
    decryption_cipher = Fernet(derived_key)
    try:
        decrypted_file_data: bytes = decryption_cipher.decrypt(encrypted_data_portion)
    except Exception:
        raise ValueError("Decryption failed: incorrect password or corrupted file")

    # Determine output file path by removing encrypted extension
    if encrypted_file_path.suffix == ENCRYPTED_FILE_EXTENSION:
        original_file_path = encrypted_file_path.with_suffix("")
    else:
        # Handle cases where .encrypted might be part of a compound extension
        original_file_path = Path(
            str(encrypted_file_path).replace(ENCRYPTED_FILE_EXTENSION, "")
        )

    # Write decrypted data to original file
    try:
        with open(original_file_path, "wb") as decrypted_file:
            decrypted_file.write(decrypted_file_data)
    except PermissionError:
        raise PermissionError(
            f"Permission denied writing decrypted file: {original_file_path}"
        )


def generate_password(
    character_length: int = DEFAULT_PASSWORD_LENGTH,
    include_special_symbols: bool = True,
    exclude_similar_looking_chars: bool = True,
) -> str:
    """
    Generate a cryptographically secure random password with customizable options.

    Uses the secrets module for cryptographically strong random generation.
    Ensures password complexity by verifying character type distribution.

    Args:
        character_length: Desired password length (minimum 4 for complexity checks)
        include_special_symbols: Whether to include special symbols (!@#$%^&* etc.)
        exclude_similar_looking_chars: Whether to exclude visually similar chars (0O1lI|)

    Returns:
        A securely generated random password string

    Raises:
        ValueError: If character_length is less than 1

    Example:
        >>> password = generate_password(20, True, True)
        >>> len(password)
        20
        >>> any(c.isupper() for c in password)  # Contains uppercase
        True
    """
    if character_length < 1:
        raise ValueError("Password length must be at least 1 character")

    # Build character pool starting with letters and digits
    available_characters: str = string.ascii_letters + string.digits

    # Add special symbols if requested
    if include_special_symbols:
        available_characters += SYMBOL_CHARACTERS

    # Remove visually similar characters if requested
    if exclude_similar_looking_chars:
        available_characters = "".join(
            char for char in available_characters if char not in SIMILAR_CHARACTERS
        )

    # Generate password with complexity validation
    for generation_attempt in range(MAX_PASSWORD_GENERATION_ATTEMPTS):
        generated_password: str = "".join(
            secrets.choice(available_characters) for _ in range(character_length)
        )

        # For shorter passwords, skip complexity validation
        if character_length < 4:
            return generated_password

        # Validate password has good character distribution
        has_lowercase_letter: bool = any(char.islower() for char in generated_password)
        has_uppercase_letter: bool = any(char.isupper() for char in generated_password)
        has_numeric_digit: bool = any(char.isdigit() for char in generated_password)
        has_special_symbol: bool = (
            any(char in SYMBOL_CHARACTERS for char in generated_password)
            if include_special_symbols
            else True
        )

        # Return password if it meets complexity requirements
        if (
            has_lowercase_letter
            and has_uppercase_letter
            and has_numeric_digit
            and has_special_symbol
        ):
            return generated_password

    # Fallback: return random password even if complexity check fails
    # (still cryptographically secure)
    return "".join(
        secrets.choice(available_characters) for _ in range(character_length)
    )


def generate_passphrase(
    word_count: int = DEFAULT_PASSPHRASE_WORD_COUNT,
    word_separator: str = DEFAULT_WORD_SEPARATOR,
    append_random_number: bool = False,
    capitalize_each_word: bool = False,
    use_external_wordlist: bool = False,
) -> str:
    """
    Generate a memorable passphrase using random word combinations.

    Creates human-friendly passphrases that are easier to remember than
    random character passwords while maintaining good entropy.

    Args:
        word_count: Number of words to include in passphrase
        word_separator: Character(s) to place between words
        append_random_number: Whether to add a 2-digit number at the end
        capitalize_each_word: Whether to capitalize the first letter of each word
        use_external_wordlist: Whether to attempt using external MIT wordlist file

    Returns:
        A memorable passphrase string

    Raises:
        ValueError: If word_count is less than 1
        FileNotFoundError: If use_external_wordlist is True but wordlist file not found

    Example:
        >>> passphrase = generate_passphrase(4, "-", True, True)
        >>> len(passphrase.split("-"))
        5  # 4 words + 1 number
        >>> passphrase
        'Forest-Mountain-River-Ocean-73'
    """
    if word_count < 1:
        raise ValueError("Word count must be at least 1")

    # Default word pool for passphrase generation
    default_word_collection: List[str] = [
        "apple",
        "beach",
        "cloud",
        "dance",
        "eagle",
        "flame",
        "grape",
        "house",
        "island",
        "jungle",
        "kite",
        "lemon",
        "moon",
        "night",
        "ocean",
        "piano",
        "quiet",
        "river",
        "stone",
        "tower",
        "voice",
        "water",
        "yellow",
        "zebra",
        "magic",
        "forest",
        "mountain",
        "thunder",
        "crystal",
        "dragon",
        "castle",
        "wizard",
        "tiger",
        "phoenix",
        "storm",
        "garden",
        "bridge",
        "rocket",
        "compass",
        "diamond",
        "emerald",
        "falcon",
        "harbor",
        "knight",
        "library",
        "marble",
        "anchor",
        "bronze",
        "copper",
        "desert",
        "engine",
        "feather",
        "golden",
        "helmet",
        "ivory",
        "jasper",
        "keystone",
        "lantern",
        "mirror",
        "nectar",
        "opal",
        "palace",
        "quartz",
        "ruby",
        "silver",
        "topaz",
        "umbrella",
        "valley",
        "willow",
        "cosmic",
        "shadow",
        "bright",
        "winter",
        "summer",
        "spring",
        "autumn",
        "plasma",
        "neutron",
        "galaxy",
        "stellar",
    ]

    # Choose words from external file or default collection
    if use_external_wordlist:
        try:
            selected_words: List[str] = _extract_random_words_from_wordlist_file(
                word_count
            )
        except FileNotFoundError:
            # Fallback to default wordlist if external file not available
            selected_words = [
                secrets.choice(default_word_collection) for _ in range(word_count)
            ]
    else:
        selected_words = [
            secrets.choice(default_word_collection) for _ in range(word_count)
        ]

    # Apply capitalization if requested
    if capitalize_each_word:
        selected_words = [word.capitalize() for word in selected_words]

    # Append random number if requested
    if append_random_number:
        random_two_digit_number: str = str(
            secrets.randbelow(RANDOM_NUMBER_MAX - RANDOM_NUMBER_MIN + 1)
            + RANDOM_NUMBER_MIN
        )
        selected_words.append(random_two_digit_number)

    return word_separator.join(selected_words)


def _extract_random_words_from_wordlist_file(requested_word_count: int) -> List[str]:
    """
    Memory-efficient extraction of random words from MIT wordlist file.

    Reads only the specific random lines needed rather than loading the
    entire wordlist into memory, making it suitable for large wordlist files.

    Args:
        requested_word_count: Number of words to extract from wordlist

    Returns:
        List of random words from the wordlist file

    Raises:
        FileNotFoundError: If MIT wordlist file cannot be found in expected locations
        OSError: If file reading operations fail

    Note:
        Searches for wordlist in: assets/mit_wordlist.txt, mit_wordlist.txt,
        or relative to current module location.
    """
    # Potential locations for MIT wordlist file
    wordlist_search_paths: List[Path] = [
        Path("assets/mit_wordlist.txt"),
        Path("mit_wordlist.txt"),
        Path(__file__).parent / "assets" / "mit_wordlist.txt",
    ]

    # Find the first existing wordlist file
    wordlist_file_path: Optional[Path] = None
    for potential_path in wordlist_search_paths:
        if potential_path.exists():
            wordlist_file_path = potential_path
            break

    if wordlist_file_path is None:
        raise FileNotFoundError(
            f"MIT wordlist not found in any of these locations: {wordlist_search_paths}"
        )

    # Count total lines in wordlist file for random selection
    try:
        with open(wordlist_file_path, "r", encoding="utf-8") as wordlist_file:
            total_line_count: int = sum(1 for line in wordlist_file if line.strip())
    except OSError as e:
        raise OSError(f"Failed to read wordlist file {wordlist_file_path}: {e}")

    # Generate sorted list of random line numbers to read
    random_line_numbers: List[int] = sorted(
        secrets.randbelow(total_line_count) for _ in range(requested_word_count)
    )

    # Extract words from specific random lines
    extracted_words: List[str] = []
    try:
        with open(wordlist_file_path, "r", encoding="utf-8") as wordlist_file:
            for current_line_number, file_line in enumerate(wordlist_file):
                if current_line_number in random_line_numbers:
                    cleaned_word: str = file_line.strip().lower()
                    # Only include words that meet minimum length requirement
                    if len(cleaned_word) >= MINIMUM_WORD_LENGTH:
                        extracted_words.append(cleaned_word)
                    # Stop once we have enough words
                    if len(extracted_words) >= requested_word_count:
                        break
    except OSError as e:
        raise OSError(f"Failed to extract words from wordlist file: {e}")

    # Return exactly the requested number of words
    return extracted_words[:requested_word_count]


# Example usage and demonstration functions
# def demonstrate_password_generation() -> None:
#     """
#     Demonstrate various password and passphrase generation options.

#     Prints examples of different password generation configurations
#     to showcase the available functionality and options.
#     """
#     print("Enhanced Password Generator Demonstration")
#     print("=" * 50)

#     print("\nSecure Random Passwords:")
#     print(f"Standard (16 chars):     {generate_password()}")
#     print(f"Long (24 chars):         {generate_password(24)}")
#     print(f"Ultra-long (32 chars):   {generate_password(32)}")
#     print(f"No symbols:              {generate_password(16, include_special_symbols=False)}")
#     print(f"Allow similar chars:     {generate_password(16, exclude_similar_looking_chars=False)}")

#     print("\nMemorable Passphrases:")
#     print(f"Standard:                {generate_passphrase()}")
#     print(f"5 words:                 {generate_passphrase(5)}")
#     print(f"Underscore separator:    {generate_passphrase(4, '_')}")
#     print(f"With random number:      {generate_passphrase(4, append_random_number=True)}")
#     print(f"Capitalized words:       {generate_passphrase(4, capitalize_each_word=True)}")

#     print("\nFile Encryption Example:")
#     print("# To encrypt a file:")
#     print("encrypt_file('document.txt', 'your_secure_password')")
#     print("# To decrypt a file:")
#     print("decrypt_file('document.txt.encrypted', 'your_secure_password')")
