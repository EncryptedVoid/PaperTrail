"""
Configuration Module

Central configuration management for the PaperTrail processing pipeline.
Contains all settings for file processing, security, directory structures,
and system parameters.

Author: Ashiq Gazi
"""

from pathlib import Path
from typing import Set
from enum import Enum


# Import HashAlgorithm enum (assuming it's defined in security_agent module)
class HashAlgorithm(Enum):
    """
    Comprehensive enumeration of supported cryptographic hash algorithms.

    This enum provides access to all major cryptographic hash algorithms available
    in Python's hashlib module, organized by family and security characteristics.
    Each CHECKSUM_ALGORITHM is suitable for different use cases based on security requirements
    and performance constraints.

    Security Classification:
        RECOMMENDED: SHA-2, SHA-3, BLAKE2 families - suitable for all security applications
        LEGACY: MD5, SHA1 - deprecated for security-critical applications
        SPECIALIZED: SHAKE functions - for applications requiring variable-length output

    Performance Characteristics:
        FASTEST: MD5, SHA1 (but cryptographically broken)
        BALANCED: SHA256, BLAKE2S/BLAKE2B (recommended for most use cases)
        HIGHEST_SECURITY: SHA512, SHA3_512 (for maximum security requirements)

    Algorithm Selection Guidelines:
        - Use SHA256 for general-purpose security applications (default choice)
        - Use BLAKE2B for high-performance applications on 64-bit systems
        - Use SHA3 family for applications requiring resistance to length extension attacks
        - Use SHA512 for applications requiring maximum security margin
        - Avoid MD5/SHA1 except for non-security purposes like simple file integrity
    """

    # SHA-2 family - Industry standard, FIPS approved, widely supported
    SHA256 = "sha256"  # Most commonly used, optimal security/performance balance
    SHA384 = "sha384"  # Truncated SHA-512, faster on 32-bit systems than SHA-512
    SHA512 = "sha512"  # Highest security in SHA-2 family, best for sensitive data
    SHA224 = (
        "sha224"  # Truncated SHA-256, compatible with systems requiring shorter hashes
    )

    # SHA-3 family - Modern Keccak-based algorithms, resistant to length extension
    SHA3_256 = "sha3_256"  # SHA-3 with 256-bit output, equivalent security to SHA256
    SHA3_384 = "sha3_384"  # SHA-3 with 384-bit output, equivalent security to SHA384
    SHA3_512 = "sha3_512"  # SHA-3 with 512-bit output, maximum security available
    SHA3_224 = "sha3_224"  # SHA-3 with 224-bit output, compact hash for space-constrained systems

    # BLAKE2 family - Modern, fast, and secure alternative to SHA-2
    BLAKE2B = "blake2b"  # Optimized for 64-bit platforms, excellent performance
    BLAKE2S = "blake2s"  # Optimized for 32-bit platforms, mobile-friendly

    # SHAKE family - Extendable output functions based on SHA-3 internals
    SHAKE_128 = "shake_128"  # Variable-length output, 128-bit security level
    SHAKE_256 = "shake_256"  # Variable-length output, 256-bit security level

    # Legacy algorithms - Deprecated for security-critical applications
    MD5 = "md5"  # Fast but cryptographically broken, only for non-security checksums
    SHA1 = "sha1"  # Deprecated due to collision vulnerabilities, avoid for new systems


# Model settings
GPU_MEMORY_LIMIT_PERCENT: float = 75.0
MODEL_REFRESH_INTERVAL: int = 20
MAX_MEMORY_THRESHOLD_PERCENT: float = 75.0

# Checksum algorithm for duplicate detection and integrity verification
CHECKSUM_ALGORITHM: HashAlgorithm = HashAlgorithm.SHA3_512

# Enhanced supported extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    # Text & Documents
    ".txt",
    ".md",
    ".rtf",
    ".doc",
    ".docx",
    ".pdf",
    ".odt",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".ods",
    ".odp",
    ".odg",
    # Data formats
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
    ".heif",
    ".raw",
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".tga",
    ".psd",
    # Email & Communication
    ".eml",
    ".msg",
    ".mbox",
    ".ics",
    ".vcs",
}

# Enhanced unsupported extensions
UNSUPPORTED_EXTENSIONS: Set[str] = {
    # Audio
    ".mp3",
    ".aac",
    ".ogg",
    ".wma",
    ".m4a",
    ".wav",
    ".flac",
    ".aiff",
    # Video
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".ogv",
    ".m4v",
    # 3D & CAD
    ".obj",
    ".fbx",
    ".dae",
    ".3ds",
    ".blend",
    ".dwg",
    ".dxf",
    ".step",
    ".stl",
    ".gcode",
    # Executables & System
    ".exe",
    ".msi",
    ".dmg",
    ".pkg",
    ".deb",
    ".rpm",
    ".dll",
    ".so",
    ".dylib",
    # Databases
    ".db",
    ".sqlite",
    ".mdb",
    ".accdb",
    # Proprietary
    ".indd",
    ".fla",
    ".swf",
    ".sav",
    ".dta",
    ".sas7bdat",
    ".mat",
    ".hdf5",
    # Archives (if extraction not implemented)
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

# File names for permanent storage
CHECKSUM_HISTORY_FILE: Path = Path("checksum_history.txt")

# Base directory for all processing
BASE_DIR: Path = Path(r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run")

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

# Wordlist path for passphrase generation
PASSPHRASE_WORDLIST_PATH: Path = (
    BASE_DIR / "resources" / "wordlists" / "common_words.txt"
)

# Profile and artifact tracking directories
ARTIFACT_PROFILES_DIR: Path = BASE_DIR / "profiles"

# Logging configuration
LOG_DIR: Path = BASE_DIR / "logs"

# System directories to exclude from processing
SYSTEM_DIRECTORIES: Set[Path] = {
    UNPROCESSED_ARTIFACTS_DIR,
    FOR_REVIEW_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    RENAMED_ARTIFACTS_DIR,
    METADATA_ARTIFACTS_DIR,
    ANALYZED_ARTIFACTS_DIR,
    COMPLETED_ARTIFACTS_DIR,
}

# Processing pipeline directories
UNPROCESSED_ARTIFACTS_DIR: Path = BASE_DIR / "01_unprocessed"
FOR_REVIEW_ARTIFACTS_DIR: Path = BASE_DIR / "02_for_review"
SANITIZED_ARTIFACTS_DIR: Path = BASE_DIR / "03_sanitized"
RENAMED_ARTIFACTS_DIR: Path = BASE_DIR / "04_renamed"
METADATA_ARTIFACTS_DIR: Path = BASE_DIR / "05_metadata"
ANALYZED_ARTIFACTS_DIR: Path = BASE_DIR / "06_analyzed"
COMPLETED_ARTIFACTS_DIR: Path = BASE_DIR / "07_completed"

# Resource directories
WORDLISTS_DIR: Path = "assets/mit_wordlist.txt"

# Session and tracking files
SESSION_TRACKING_FILE: Path = BASE_DIR / "session_tracking.json"
PROCESSING_HISTORY_FILE: Path = BASE_DIR / "processing_history.json"

# Performance and processing limits
MAX_FILE_SIZE_MB: int = 500
MAX_FILES_PER_BATCH: int = 1000
CHECKSUM_CHUNK_SIZE: int = 8192
METADATA_EXTRACTION_TIMEOUT: int = 300  # seconds

# Logging configuration
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT: int = 5

# Processing flags
ENABLE_ADVANCED_METADATA: bool = True
ENABLE_INTEGRITY_VERIFICATION: bool = True
ENABLE_ROLLBACK_RECORDS: bool = True
ENABLE_PROGRESS_TRACKING: bool = True
ENABLE_DETAILED_LOGGING: bool = True

METADATA_EXTRACTED_DIR
