"""
Configuration Module

Central configuration management for the PaperTrail processing pipeline.
Contains all settings for file processing, security, directory structures,
and system parameters.

Author: Ashiq Gazi
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Set


class PipelineStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    INCOMPLETE = "incomplete"


class HashAlgorithm(Enum):
    """
    Comprehensive enumeration of supported cryptographic hash algorithms.

    Security Classification:
        RECOMMENDED: SHA-2, SHA-3, BLAKE2 families - suitable for all security applications
        LEGACY: MD5, SHA1 - deprecated for security-critical applications
        SPECIALIZED: SHAKE functions - for applications requiring variable-length output
    """

    # SHA-2 family - Industry standard, FIPS approved, widely supported
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA224 = "sha224"

    # SHA-3 family - Modern Keccak-based algorithms
    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"
    SHA3_224 = "sha3_224"

    # BLAKE2 family - Modern, fast, and secure
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

    # SHAKE family - Extendable output functions
    SHAKE_128 = "shake_128"
    SHAKE_256 = "shake_256"

    # Legacy algorithms - Deprecated for security-critical applications
    MD5 = "md5"
    SHA1 = "sha1"


# ============================================================================
# BASE DIRECTORIES AND PATHS
# ============================================================================

BASE_DIR: Path = Path(r"C:\Users\UserX\Desktop\PaperTrail")

# Resource directories
PASSPHRASE_WORDLIST_PATH: Path = Path("assets/mit_wordlist.txt")
CHECKSUM_HISTORY_FILE: Path = BASE_DIR / "checksum_history.txt"

# Main processing pipeline stages
ARTIFACT_PROFILES_DIR: Path = BASE_DIR / "artifact_profiles"
LOG_DIR: Path = BASE_DIR / "logs"
UNPROCESSED_ARTIFACTS_DIR: Path = Path(r"C:\Users\UserX\Desktop\PaperTrail-Load")
ARCHIVE_DIR: Path = BASE_DIR / "archive"
FOR_REVIEW_ARTIFACTS_DIR: Path = BASE_DIR / "for_review"
SANITIZED_ARTIFACTS_DIR: Path = BASE_DIR / "01_sanitized"
METADATA_EXTRACTED_DIR: Path = BASE_DIR / "02_metadata"
SEMANTICS_EXTRACTED_DIR: Path = BASE_DIR / "03_semantics"
CONVERTED_ARTIFACT_DIR: Path = BASE_DIR / "04_converted"
COMPLETED_ARTIFACTS_DIR: Path = BASE_DIR / "05_completed"

# System directories collection
SYSTEM_DIRECTORIES: Set[Path] = {
    ARTIFACT_PROFILES_DIR,
    LOG_DIR,
    UNPROCESSED_ARTIFACTS_DIR,
    ARCHIVE_DIR,
    FOR_REVIEW_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    METADATA_EXTRACTED_DIR,
    SEMANTICS_EXTRACTED_DIR,
    CONVERTED_ARTIFACT_DIR,
    COMPLETED_ARTIFACTS_DIR,
}


# ============================================================================
# FILE TYPE DEFINITIONS
# ============================================================================

# Supported file extensions
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

# Unsupported file extensions
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
    # Archives
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

# File type mappings for conversion
EXTENSION_MAPPING: Dict[str, str] = {
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
TARGET_FORMATS: Dict[str, str] = {
    "image": ".png",
    "video": ".mp4",
    "audio": ".mp3",
    "document": ".pdf",
}


# ============================================================================
# SECURITY AND ENCRYPTION SETTINGS
# ============================================================================

# Checksum algorithm for duplicate detection and integrity verification
CHECKSUM_ALGORITHM: HashAlgorithm = HashAlgorithm.SHA3_512
CHECKSUM_CHUNK_SIZE_BYTES: int = 8192

# File encryption constants
SALT_LENGTH_BYTES: int = 16
KEY_DERIVATION_ITERATIONS: int = 100000
ENCRYPTED_FILE_EXTENSION: str = ".encrypted"

# Password generation constants
DEFAULT_PASSWORD_LENGTH: int = 16
DEFAULT_PASSPHRASE_WORD_COUNT: int = 6
DEFAULT_WORD_SEPARATOR: str = "-"
SIMILAR_CHARACTERS: str = "0O1lI|"  # Characters that look similar
SYMBOL_CHARACTERS: str = r"!@#$%^&*()_+-=[]{}|;:,.<>?"
MINIMUM_WORD_LENGTH: int = 3
RANDOM_NUMBER_MIN: int = 10
RANDOM_NUMBER_MAX: int = 99
MAX_PASSWORD_GENERATION_ATTEMPTS: int = 50
MINIMUM_WORD_LENGTH = 3  # Minimum word length for passphrase generation

# ============================================================================
# ARTIFACT AND PROFILE NAMING
# ============================================================================

ARTIFACT_PREFIX: str = "ARTIFACT"
PROFILE_PREFIX: str = "PROFILE"

# UUID generation settings
UUID_PREFIX: str = ""
INCLUDE_TIMESTAMP_ON_UUID: bool = False
UUID_ENTROPY: int = 16


# ============================================================================
# PROCESSING LIMITS AND THRESHOLDS
# ============================================================================

# Performance and processing limits
MAX_FILE_SIZE_MB: int = 500
MAX_FILES_PER_BATCH: int = 1000
METADATA_EXTRACTION_TIMEOUT: int = 300  # seconds

# GPU and memory settings
GPU_MEMORY_LIMIT_PERCENT: float = 75.0
MODEL_REFRESH_INTERVAL: int = 20
MAX_MEMORY_THRESHOLD_PERCENT: float = 75.0

# Processing thresholds
MAX_PROCESSING_TIME_PER_DOC: float = 60.0  # seconds
MIN_TEXT_SUCCESS_RATE: float = 0.8
MIN_DESCRIPTION_SUCCESS_RATE: float = 0.8
MAX_AVERAGE_PROCESSING_TIME: float = 30.0  # seconds
MAX_GPU_MEMORY_PERCENT: float = 90.0


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT: int = 5
SESSION_LOG_FILE_PREFIX: str = "PAPERTRAIL-SESSION"


# ============================================================================
# PROCESSING FLAGS
# ============================================================================

ENABLE_ADVANCED_METADATA: bool = True
ENABLE_INTEGRITY_VERIFICATION: bool = True
ENABLE_ROLLBACK_RECORDS: bool = True
ENABLE_PROGRESS_TRACKING: bool = True
ENABLE_DETAILED_LOGGING: bool = True


# ============================================================================
# CONVERSION AND ENHANCEMENT SETTINGS
# ============================================================================

# File detection settings
USE_CONTENT_DETECTION: bool = True
REQUIRE_DETECTION_AGREEMENT: bool = False

# Quality enhancement settings
ENABLE_IMAGE_ENHANCEMENT: bool = True
ENABLE_VIDEO_ENHANCEMENT: bool = True
ENABLE_AUDIO_ENHANCEMENT: bool = True
ENABLE_UPSCALING: bool = True

# Image processing settings
IMAGE_TARGET_SIZE: int = 1920  # HD minimum for upscaling
IMAGE_SHARPENING_FACTOR: float = 1.1
IMAGE_CONTRAST_FACTOR: float = 1.05
IMAGE_UPSCALE_THRESHOLD: int = 1920
PNG_COMPRESS_LEVEL: int = 0  # 0 = no compression, 9 = max compression

# Video processing settings
VIDEO_CRF: int = 18  # Lower = better quality (0-51 scale)
VIDEO_PRESET: str = "slow"
VIDEO_CODEC: str = "libx264"
VIDEO_PIXEL_FORMAT: str = "yuv420p"
VIDEO_UPSCALE_THRESHOLD: int = 1920
VIDEO_4K_THRESHOLD: int = 3840
VIDEO_TARGET_1080P: int = 1080
VIDEO_TARGET_4K: int = 2160

# Audio processing settings
AUDIO_BITRATE: str = "320k"
AUDIO_SAMPLE_RATE: int = 48000
AUDIO_CODEC: str = "libmp3lame"

# Document processing settings
LIBREOFFICE_TIMEOUT: int = 60  # seconds


# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

# Preferred models
PREFERRED_VISUAL_MODEL: str = "Qwen/Qwen2-VL-2B-Instruct"
PREFERRED_LANGUAGE_MODEL: str = "mistral:7b"

# Visual processing settings
DEFAULT_REFRESH_INTERVAL: int = 5
DEFAULT_MEMORY_THRESHOLD: float = 80.0
DEFAULT_AUTO_MODEL_SELECTION: bool = True
DEFAULT_PROCESSING_MODE: str = "balanced"

# Hardware resource allocation ratios
RAM_USAGE_RATIO: float = 0.7  # Use 70% of total RAM
VRAM_USAGE_RATIO: float = 0.8  # Use 80% of total VRAM

# PDF processing zoom factors
ZOOM_FACTOR_FAST: float = 1.5
ZOOM_FACTOR_BALANCED: float = 2.0
ZOOM_FACTOR_HIGH_QUALITY: float = 3.0

# Model specifications
QWEN2VL_2B_MIN_VRAM: float = 4.0
QWEN2VL_2B_MIN_RAM: float = 8.0
QWEN2VL_2B_QUALITY: int = 7
QWEN2VL_2B_MAX_TOKENS: int = 512

QWEN2VL_7B_MIN_VRAM: float = 14.0
QWEN2VL_7B_MIN_RAM: float = 16.0
QWEN2VL_7B_QUALITY: int = 9
QWEN2VL_7B_MAX_TOKENS: int = 512

QWEN2VL_72B_MIN_VRAM: float = 144.0
QWEN2VL_72B_MIN_RAM: float = 200.0
QWEN2VL_72B_QUALITY: int = 10
QWEN2VL_72B_MAX_TOKENS: int = 1024

QWEN2VL_7B_CPU_MIN_VRAM: float = 0.0
QWEN2VL_7B_CPU_MIN_RAM: float = 32.0
QWEN2VL_7B_CPU_QUALITY: int = 8
QWEN2VL_7B_CPU_MAX_TOKENS: int = 512


# ============================================================================
# LLM FIELD EXTRACTION PROMPTS
# ============================================================================

SYSTEM_PROMPT: str = """You are a document extraction tool. Extract ONLY the requested information.

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

FIELD_PROMPTS: Dict[str, str] = {
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
    #     "document_type": """Identify the specific type of document this is.
    # Examples of document types:
    # - birth_certificate, passport_us, driver_license_ca
    # - invoice, receipt, bank_statement, credit_report
    # - contract, lease_agreement, employment_contract
    # - medical_record, prescription, insurance_policy
    # - w2_tax_form, 1099, tax_return
    # - court_order, legal_notice, business_license
    # - university_transcript, diploma, certificate
    # - property_deed, mortgage_document, warranty
    # Return ONLY the specific document type in lowercase with underscores. If uncertain, return "UNKNOWN".""",
    #     "original_language": """Analyze the document text and identify ALL languages present.
    # Document contains sections in multiple languages. List ALL languages found.
    # Examples:
    # - If English only: "English"
    # - If French only: "French"
    # - If both: "English, French"
    # - If trilingual: "English, French, Spanish"
    # Return ONLY language names, comma-separated.""",
    #     "current_language": """Identify the primary language this document is currently written in.
    # Return ONLY the primary language name. If multiple languages, return the dominant one.""",
    #     "confidentiality_level": """Determine the confidentiality or security classification of this document.
    # Look for markings or indicators such as:
    # - CONFIDENTIAL, CLASSIFIED, SECRET, TOP SECRET
    # - INTERNAL USE ONLY, PROPRIETARY, RESTRICTED
    # - PUBLIC, FOR PUBLIC RELEASE
    # - Security stamps, watermarks, or headers
    # Classification levels:
    # - Public: No restrictions, can be freely shared
    # - Internal: Internal use within organization
    # - Confidential: Sensitive information, restricted access
    # - Restricted: Highly sensitive, top-secret information
    # Return ONLY one of these words: Public, Internal, Confidential, Restricted""",
    #     "translator_name": """If this document has been translated, identify the translator's name.
    # Look for:
    # - Translator certifications or signatures
    # - Translation agency information
    # - "Translated by" notices
    # - Official translation stamps or seals
    # - Translator contact information
    # Return ONLY the translator's full name. If this is not a translated document or no translator is identified, return "UNKNOWN".""",
    "issuer_name": """Identify who issued, created, or published this document.

Look for:
- Organization names, agencies, departments
- Company names or letterheads
- Government agencies or institutions
- Individual names (for personally issued documents)
- Official stamps or seals with issuer information

Return ONLY the full official name of the issuer. If unclear, return "UNKNOWN".""",
    #     "officiater_name": """Identify any official authority that validated, certified, witnessed, or authorized this document.
    # Look for:
    # - Notary public names and seals
    # - Certifying agency names
    # - Official witnesses or authorizing bodies
    # - Government officials who signed or stamped
    # - Licensing boards or regulatory authorities
    # Return ONLY the name of the official authority. If no official validation exists, return "UNKNOWN".""",
    #     "date_created": """Find when this document was originally created, written, or authored.
    # Look for:
    # - Creation dates, authored dates, written dates
    # - "Created on", "Date created", "Authored"
    # - Document composition or drafting dates
    # Return the date in YYYY-MM-DD format. If no creation date is found, return "UNKNOWN".""",
    #     "date_of_reception": """Find when this document was received by the current holder.
    # Look for:
    # - "Received", "Date received", "Arrival date"
    # - Postal stamps or delivery confirmations
    # - Filing dates or intake dates
    # - "Delivered on" stamps
    # Return the date in YYYY-MM-DD format. If no reception date is found, return "UNKNOWN".""",
    #     "date_of_issue": """Find the official issue, publication, or release date of this document.
    # Look for:
    # - "Issued", "Date of issue", "Publication date"
    # - "Released", "Effective date"
    # - Official dating stamps or seals
    # - Government or agency issue dates
    # Return the date in YYYY-MM-DD format. If no issue date is found, return "UNKNOWN".""",
    #     "date_of_expiry": """Find when this document expires, becomes invalid, or requires renewal.
    # Look for:
    # - "Expires", "Expiration date", "Valid until"
    # - "Renewal required", "Valid through"
    # - License or certification expiry dates
    # - "Not valid after" dates
    # Return the date in YYYY-MM-DD format. If no expiration date exists, return "UNKNOWN".""",
    #     "tags": """Create comprehensive keywords that describe this document for search and categorization purposes.
    # Include keywords for:
    # - Document category (legal, medical, financial, educational, personal, business, government, technical)
    # - Subject matter (taxes, healthcare, employment, education, property, travel, identification, insurance)
    # - Content type (contract, certificate, statement, report, application, notice, invoice, receipt)
    # - Industry/field (healthcare, legal, finance, education, technology, government, military)
    # - Geographic relevance (federal, state, local, international, specific regions)
    # - Time relevance (annual, quarterly, monthly, historical, current)
    # - Action items (renewal_required, payment_due, action_needed, informational_only)
    # - Format type (official, certified, notarized, electronic, handwritten, typed)
    # Return 15-25 comma-separated keywords. If document content is unclear, return "UNKNOWN".""",
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
    "executive_summary": """Summarize significant administrative, security, or procedural characteristics not covered elsewhere.

Note:
- Security classifications, handling restrictions
- Authentication elements, official markings
- Distribution methods, transmission records
- Document quality, preservation concerns
- Cross-references to related administrative instruments

Present observations in formal, official terminology suitable for administrative records.""",
}

# Password vault for storing encryption passwords securely
PASSWORD_VAULT_PATH = Path("data/password_vault.encrypted")
VAULT_MASTER_KEY: str = "password"

# Define the column mapping from profile data to spreadsheet columns
DATABASE_COLUMN_MAPPING = {
    # Core identification
    "ITEM_ID": "uuid",
    "Title": ["llm_extraction", "extracted_fields", "title"],
    "File_Extension": ["metadata", "extension"],
    # Document classification
    "Document_Type": ["llm_extraction", "extracted_fields", "document_type"],
    "Original_Language": [
        "llm_extraction",
        "extracted_fields",
        "original_language",
    ],
    "Current_Language": [
        "llm_extraction",
        "extracted_fields",
        "current_language",
    ],
    "Confidentiality_Level": [
        "llm_extraction",
        "extracted_fields",
        "confidentiality_level",
    ],
    # Technical data
    "Checksum_SHA256": "checksum",
    "File_Size_MB": ["metadata", "size_mb"],
    # People and organizations
    "Translator_Name": [
        "llm_extraction",
        "extracted_fields",
        "translator_name",
    ],
    "Issuer_Name": ["llm_extraction", "extracted_fields", "issuer_name"],
    "Officiater_Name": [
        "llm_extraction",
        "extracted_fields",
        "officiater_name",
    ],
    # Dates
    "Date_Added": ["stages", "metadata_extraction", "timestamp"],
    "Date_Created": ["llm_extraction", "extracted_fields", "date_created"],
    "Date_of_Reception": [
        "llm_extraction",
        "extracted_fields",
        "date_of_reception",
    ],
    "Date_of_Issue": ["llm_extraction", "extracted_fields", "date_of_issue"],
    "Date_of_Expiry": ["llm_extraction", "extracted_fields", "date_of_expiry"],
    # Content and notes
    "Tags": ["llm_extraction", "extracted_fields", "tags"],
    "Version_Notes": ["llm_extraction", "extracted_fields", "version_notes"],
    "Utility_Notes": ["llm_extraction", "extracted_fields", "utility_notes"],
    "Additional_Notes": [
        "llm_extraction",
        "extracted_fields",
        "additional_notes",
    ],
    # Manual entry fields (will be empty for now)
    "Action_Required": "",
    "Parent_Document_ID": "",
    "Off_Site_Storage_ID": "",
    "On_Site_Storage_ID": "",
    "Backup_Storage_ID": "",
    "Project_ID": "",
    "Version_Number": "",
    # Processing metadata
    "Processing_Status": ["llm_extraction", "success"],
    "OCR_Text_Length": "calculated",
    "Visual_Description_Length": "calculated",
    "Fields_Extracted_Count": "calculated",
    "Processing_Date": ["stages", "llm_field_extraction", "timestamp"],
    "Original_Filename": "original_artifact_name",
    "Encryption_Status": "calculated",
}

USE_PASSPHRASE_ENCRYPTION: bool = False
ENCRYPT_ARTIFACTS: bool = True

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

JAVA_PATH: str = "java"
TIKA_APP_JAR_PATH = r"/assets/tika-app-3.2.3.jar"
TIKA_SERVER_JAR_PATH = r"/assets/tika-app-3.2.3.jar"
TIKA_SERVER_STANDARD_JAR_PATH = r"/assets/tika-app-3.2.3.jar"
