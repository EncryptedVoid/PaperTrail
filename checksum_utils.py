from enum import Enum
import hashlib
from pathlib import Path
from typing import Union


class HashAlgorithm(Enum):
    """Enumeration of supported hash algorithms for checksum calculation.

    This enum provides a comprehensive list of cryptographic hash algorithms
    available in Python's hashlib module, categorized by family and security level.

    Security recommendations:
        - Use SHA-2 or SHA-3 family for general security purposes
        - Use BLAKE2 for high-performance applications
        - Avoid MD5 and SHA1 for security-critical applications
    """

    # SHA-2 family (recommended for general use)
    SHA256 = "sha256"  # Most commonly used, good balance of security and performance
    SHA384 = "sha384"  # Truncated SHA-512, faster on 32-bit systems
    SHA512 = "sha512"  # Highest security in SHA-2 family
    SHA224 = "sha224"  # Truncated SHA-256

    # SHA-3 family (modern, secure alternative to SHA-2)
    SHA3_256 = "sha3_256"  # SHA-3 with 256-bit output
    SHA3_384 = "sha3_384"  # SHA-3 with 384-bit output
    SHA3_512 = "sha3_512"  # SHA-3 with 512-bit output
    SHA3_224 = "sha3_224"  # SHA-3 with 224-bit output

    # BLAKE2 family (fast and secure, good for high-performance needs)
    BLAKE2B = "blake2b"  # BLAKE2b - optimized for 64-bit platforms
    BLAKE2S = "blake2s"  # BLAKE2s - optimized for 32-bit platforms

    # SHAKE (extendable output functions from SHA-3 family)
    SHAKE_128 = "shake_128"  # Variable-length output based on SHAKE128
    SHAKE_256 = "shake_256"  # Variable-length output based on SHAKE256

    # Legacy algorithms (deprecated for security-critical applications)
    MD5 = "md5"  # Fast but cryptographically broken - only for non-security uses
    SHA1 = "sha1"  # Deprecated due to collision vulnerabilities


def generate_checksum(
    file_path: Union[str, Path], algorithm: HashAlgorithm = HashAlgorithm.SHA256
) -> str:
    """Calculate the cryptographic checksum of a file using the specified algorithm.

    This function efficiently processes files of any size by reading them in chunks
    to avoid loading the entire file into memory at once.

    Args:
        file_path: Path to the file to be hashed. Can be a string or Path object.
        algorithm: Hash algorithm to use for checksum calculation.
                  Defaults to SHA256 for optimal security/performance balance.

    Returns:
        The hexadecimal string representation of the file's checksum.

    Raises:
        ValueError: If the specified hash algorithm is not supported by the system.
        FileNotFoundError: If the specified file does not exist or cannot be accessed.
        PermissionError: If the file cannot be read due to insufficient permissions.
        OSError: If an I/O error occurs while reading the file.

    Example:
        >>> checksum = calculate_checksum("document.pdf", HashAlgorithm.SHA256)
        >>> print(f"SHA256: {checksum}")
        SHA256: a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3

        >>> # Using default algorithm
        >>> checksum = calculate_checksum("data.txt")

        >>> # With Path object
        >>> from pathlib import Path
        >>> checksum = calculate_checksum(Path("./files/data.csv"))
    """
    # Convert to Path object for consistent handling
    file_path = Path(file_path)

    # Initialize the hash object - this may raise ValueError for unsupported algorithms
    try:
        hash_obj = hashlib.new(algorithm.value)
    except ValueError as e:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm.value}. "
            f"Available algorithms: {', '.join(sorted(hashlib.algorithms_available))}"
        ) from e

    # Read and hash the file in chunks to handle large files efficiently
    try:
        with file_path.open("rb") as file:
            # Process file in 64KB chunks for optimal memory usage and I/O performance
            chunk_size = 65536  # 64KB chunks - good balance for most systems

            # Read file in chunks until EOF
            while chunk := file.read(chunk_size):
                hash_obj.update(chunk)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading file: {file_path}") from e
    except OSError as e:
        raise OSError(f"I/O error reading file {file_path}: {e}") from e

    # Return the hexadecimal digest
    return hash_obj.hexdigest()
