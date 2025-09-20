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

max_gpu_vram = 100.0
max_ram = 75.0
max_cpu_cores = 0.0
PREFERRED_VISUAL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


FIELD_PROMPTS = {
    "title": """Create a descriptive title that captures the document's purpose and content.

    If the document has an official title, use it. If not, synthesize a descriptive title based on:
    - Document type and purpose
    - Issuing organization
    - Main subject matter

    Examples:
    - "Immigration and Refugee Board Virtual Hearing Notice"
    - "Refugee Protection Division Hearing Preparation Guide"
    - "IRB Document Submission Requirements and Procedures"

    Return ONLY the title. No quotes, no explanations.""",
    "document_type": """Identify the specific type of document this is.

Examples of document types:
- birth_certificate, passport_us, driver_license_ca
- invoice, receipt, bank_statement, credit_report
- contract, lease_agreement, employment_contract
- medical_record, prescription, insurance_policy
- w2_tax_form, 1099, tax_return
- court_order, legal_notice, business_license
- university_transcript, diploma, certificate
- property_deed, mortgage_document, warranty

Return ONLY the specific document type in lowercase with underscores. If uncertain, return "UNKNOWN".""",
    "language": """Analyze the document text and identify ALL languages present.

Document contains sections in multiple languages. List ALL languages found.
Examples:
- If English only: "English"
- If French only: "French"
- If both: "English, French"
- If trilingual: "English, French, Spanish"

Return ONLY language names, comma-separated.""",
    "confidentiality_level": """Determine the confidentiality or security classification of this document.

Look for markings or indicators such as:
- CONFIDENTIAL, CLASSIFIED, SECRET, TOP SECRET
- INTERNAL USE ONLY, PROPRIETARY, RESTRICTED
- PUBLIC, FOR PUBLIC RELEASE
- Security stamps, watermarks, or headers

Classification levels:
- Public: No restrictions, can be freely shared
- Internal: Internal use within organization
- Confidential: Sensitive information, restricted access
- Restricted: Highly sensitive, top-secret information

Return ONLY one of these words: Public, Internal, Confidential, Restricted""",
    "translator_name": """If this document has been translated, identify the translator's name.

Look for:
- Translator certifications or signatures
- Translation agency information
- "Translated by" notices
- Official translation stamps or seals
- Translator contact information

Return ONLY the translator's full name. If this is not a translated document or no translator is identified, return "UNKNOWN".""",
    "issuer_name": """Identify who issued, created, or published this document.

Look for:
- Organization names, agencies, departments
- Company names or letterheads
- Government agencies or institutions
- Individual names (for personally issued documents)
- Official stamps or seals with issuer information

Return ONLY the full official name of the issuer. If unclear, return "UNKNOWN".""",
    "officiater_name": """Identify any official authority that validated, certified, witnessed, or authorized this document.

Look for:
- Notary public names and seals
- Certifying agency names
- Official witnesses or authorizing bodies
- Government officials who signed or stamped
- Licensing boards or regulatory authorities

Return ONLY the name of the official authority. If no official validation exists, return "UNKNOWN".""",
    "date_created": """Find when this document was originally created, written, or authored.

Look for:
- Creation dates, authored dates, written dates
- "Created on", "Date created", "Authored"
- Document composition or drafting dates

Return the date in YYYY-MM-DD format. If no creation date is found, return "UNKNOWN".""",
    "date_of_reception": """Find when this document was received by the current holder.

Look for:
- "Received", "Date received", "Arrival date"
- Postal stamps or delivery confirmations
- Filing dates or intake dates
- "Delivered on" stamps

Return the date in YYYY-MM-DD format. If no reception date is found, return "UNKNOWN".""",
    "date_of_issue": """Find the official issue, publication, or release date of this document.

Look for:
- "Issued", "Date of issue", "Publication date"
- "Released", "Effective date"
- Official dating stamps or seals
- Government or agency issue dates

Return the date in YYYY-MM-DD format. If no issue date is found, return "UNKNOWN".""",
    "date_of_expiry": """Find when this document expires, becomes invalid, or requires renewal.

Look for:
- "Expires", "Expiration date", "Valid until"
- "Renewal required", "Valid through"
- License or certification expiry dates
- "Not valid after" dates

Return the date in YYYY-MM-DD format. If no expiration date exists, return "UNKNOWN".""",
    "tags": """Create comprehensive keywords that describe this document for search and categorization purposes.

Include keywords for:
- Document category (legal, medical, financial, educational, personal, business, government, technical)
- Subject matter (taxes, healthcare, employment, education, property, travel, identification, insurance)
- Content type (contract, certificate, statement, report, application, notice, invoice, receipt)
- Industry/field (healthcare, legal, finance, education, technology, government, military)
- Geographic relevance (federal, state, local, international, specific regions)
- Time relevance (annual, quarterly, monthly, historical, current)
- Action items (renewal_required, payment_due, action_needed, informational_only)
- Format type (official, certified, notarized, electronic, handwritten, typed)

Return 15-25 comma-separated keywords. If document content is unclear, return "UNKNOWN".""",
    "version_notes": """Analyze document versioning, revision history, and administrative metadata.

Look for:
- Version numbers, revision dates, edition information
- Document control numbers, form numbers
- "Supersedes" notices, amendment references
- Administrative tracking information

Provide a professional assessment of document currency and version status.
If no versioning found, state: "No explicit version control information identified."

Use formal, administrative language.""",
    "utility_notes": """Provide a professional analysis of this document's administrative function and legal purpose.

Analyze:
- Regulatory or statutory requirements this document fulfills
- Administrative processes it initiates or supports
- Legal obligations or rights it establishes
- Institutional workflows it facilitates
- Compliance requirements it addresses

Write in formal, bureaucratic language appropriate for government documentation.""",
    "additional_notes": """Document significant administrative, security, or procedural characteristics not covered elsewhere.

Note:
- Security classifications, handling restrictions
- Authentication elements, official markings
- Distribution methods, transmission records
- Document quality, preservation concerns
- Cross-references to related administrative instruments

Present observations in formal, official terminology suitable for administrative records.""",
}

SYSTEM_PROMPT = """You are a document extraction tool. Extract ONLY the requested information.

CRITICAL RULES:
- Return ONLY the answer, nothing else
- NO explanations, NO reasoning, NO "based on", NO "therefore"
- NO sentences, just the raw information
- If not found, return exactly: UNKNOWN

Examples:
GOOD: "Immigration and Refugee Board of Canada"
BAD: "The document appears to be issued by the Immigration and Refugee Board of Canada"

GOOD: "UNKNOWN"
BAD: "document does not appear to have any official authority validating, certifying, witnessing, or authorizing it. Therefore, the answer is: UNKNOWN"

Extract the information. Nothing else.

Return ONLY the requested information. Any additional text, explanation, or reasoning will be considered an error and rejected causing immediate shutdown
"""


ARTIFACT_PREFIX = "ARTIFACT"
PROFILE_PREFIX = "PROFILE"

# Image Processing Settings
IMAGE_TARGET_SIZE = 1920  # HD minimum for upscaling
IMAGE_SHARPENING_FACTOR = 1.1  # Slight sharpening enhancement
IMAGE_CONTRAST_FACTOR = 1.05  # Slight contrast boost
IMAGE_UPSCALE_THRESHOLD = 1920  # Only upscale if smaller than this
PNG_COMPRESS_LEVEL = 0  # 0 = no compression, 9 = max compression

# Video Processing Settings
VIDEO_CRF = 18  # Lower = better quality (0-51 scale)
VIDEO_PRESET = "slow"  # FFmpeg preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_UPSCALE_THRESHOLD = 1920  # Upscale videos smaller than this
VIDEO_4K_THRESHOLD = 3840  # Don't upscale beyond 4K if source is smaller
VIDEO_TARGET_1080P = 1080
VIDEO_TARGET_4K = 2160

# Audio Processing Settings
AUDIO_BITRATE = "320k"  # Highest quality MP3
AUDIO_SAMPLE_RATE = 48000  # High sample rate
AUDIO_CODEC = "libmp3lame"

# Document Processing Settings
LIBREOFFICE_TIMEOUT = 60  # Seconds before timing out LibreOffice conversion

# File Detection Settings
USE_CONTENT_DETECTION = True  # Enable magic number detection
REQUIRE_DETECTION_AGREEMENT = True  # Both extension and content must agree

# Quality Enhancement Settings
ENABLE_IMAGE_ENHANCEMENT = True
ENABLE_VIDEO_ENHANCEMENT = True
ENABLE_AUDIO_ENHANCEMENT = True
ENABLE_UPSCALING = True

# File type mappings
EXTENSION_MAPPING = {
    # Images
    ".jpeg": "image",
    ".jpg": "image",
    ".png": "image",
    ".heic": "image",
    ".cr2": "image",
    ".arw": "image",
    ".nef": "image",
    ".webp": "image",
    # Videos
    ".mov": "video",
    ".mp4": "video",
    ".webm": "video",
    ".amv": "video",
    ".3gp": "video_audio",  # Special case - need to probe
    # Audio
    ".wav": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    # Documents
    ".pptx": "document",
    ".doc": "document",
    ".docx": "document",
    ".rtf": "document",
    ".epub": "document",
    ".pub": "document",
    ".djvu": "document",
    ".pdf": "document",
}

# Target formats for each type
TARGET_FORMATS = {
    "image": ".png",
    "video": ".mp4",
    "audio": ".mp3",
    "document": ".pdf",
}

# Default processing settings
DEFAULT_REFRESH_INTERVAL = 5
DEFAULT_MEMORY_THRESHOLD = 80.0
DEFAULT_AUTO_MODEL_SELECTION = True
DEFAULT_PROCESSING_MODE = "balanced"

# Hardware resource allocation ratios
RAM_USAGE_RATIO = 0.7  # Use 70% of total RAM
VRAM_USAGE_RATIO = 0.8  # Use 80% of total VRAM

# PDF processing zoom factors
ZOOM_FACTOR_FAST = 1.5
ZOOM_FACTOR_BALANCED = 2.0
ZOOM_FACTOR_HIGH_QUALITY = 3.0

# Performance thresholds
MAX_PROCESSING_TIME_PER_DOC = 60.0  # seconds
MIN_TEXT_SUCCESS_RATE = 0.8
MIN_DESCRIPTION_SUCCESS_RATE = 0.8
MAX_AVERAGE_PROCESSING_TIME = 30  # seconds
MAX_GPU_MEMORY_PERCENT = 90

# Model specifications
QWEN2VL_2B_MIN_VRAM = 4.0
QWEN2VL_2B_MIN_RAM = 8.0
QWEN2VL_2B_QUALITY = 7
QWEN2VL_2B_MAX_TOKENS = 512

QWEN2VL_7B_MIN_VRAM = 14.0
QWEN2VL_7B_MIN_RAM = 16.0
QWEN2VL_7B_QUALITY = 9
QWEN2VL_7B_MAX_TOKENS = 512

QWEN2VL_72B_MIN_VRAM = 144.0
QWEN2VL_72B_MIN_RAM = 200.0
QWEN2VL_72B_QUALITY = 10
QWEN2VL_72B_MAX_TOKENS = 1024

QWEN2VL_7B_CPU_MIN_VRAM = 0.0
QWEN2VL_7B_CPU_MIN_RAM = 32.0
QWEN2VL_7B_CPU_QUALITY = 8
QWEN2VL_7B_CPU_MAX_TOKENS = 512

SESSION_LOG_FILE_PREFIX = "PAPERTRAIL-SESSION"

UUID_PREFIX: str = ""
INCLUDE_TIMESTAMP_ON_UUID: bool = False
UUID_ENTROPY: int = 16

CHECKSUM_ALGORITHM,
UNSUPPORTED_EXTENSIONS,
ARTIFACT_PROFILES_DIR,
CHECKSUM_CHUNK_SIZE_BYTES,
CHECKSUM_HISTORY_FILE,
PROFILE_PREFIX,
ARTIFACT_PREFIX,
UUID_PREFIX,
INCLUDE_TIMESTAMP_ON_UUID,
UUID_ENTROPY,

PREFERRED_LANGUAGE_MODEL = "mistral:7b"
