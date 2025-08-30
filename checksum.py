from enum import Enum
import hashlib


class HashAlgorithm(Enum):
    """Enumeration of supported hash algorithms for checksum calculation."""

    # SHA-2 family (recommended)
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA224 = "sha224"

    # SHA-3 family (modern, secure)
    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"
    SHA3_224 = "sha3_224"

    # BLAKE2 family (fast and secure)
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

    # SHAKE (extendable output functions)
    SHAKE_128 = "shake_128"
    SHAKE_256 = "shake_256"

    # Legacy algorithms (not recommended for security-critical applications)
    MD5 = "md5"  # Deprecated - only use for non-security purposes
    SHA1 = "sha1"  # Deprecated - only use for non-security purposes


def calculate_checksum(file_path, algorithm: HashAlgorithm = HashAlgorithm.SHA512):
    """Calculate checksum of a file using specified algorithm.

    Args:
        file_path (str): Path to the file
        algorithm (HashAlgorithm): Hash algorithm to use (default: SHA512)

    Returns:
        str: Hexadecimal digest of the file checksum

    Raises:
        ValueError: If algorithm is not supported
        FileNotFoundError: If file doesn't exist
    """

    try:
        hash_obj = hashlib.new(algorithm.value)
    except ValueError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm.value}") from e

    try:
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_obj.update(chunk)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e

    return hash_obj.hexdigest()
