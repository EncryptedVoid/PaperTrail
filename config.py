"""
Configuration Module

Central configuration management for the PaperTrail processing pipeline.
Contains all settings for file processing, security, directory structures,
and system parameters.

Author: Ashiq Gazi
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Set


class ProcessingMode(Enum):
    """
    Processing quality modes that determine the balance between speed and quality.

    FAST: Optimized for speed with lower resolution and simpler processing
    BALANCED: Good balance between speed and quality (default)
    HIGH_QUALITY: Maximum quality with higher resolution and detailed processing
    """

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


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

# Resource directories
BASE_DIR: Path = Path(r"C:\Users\UserX\Desktop\PaperTrail")
UNPROCESSED_ARTIFACTS_DIR: Path = Path(r"C:\Users\UserX\Desktop\PaperTrail-Load")
PASSPHRASE_WORDLIST_PATH: Path = Path("assets/mit_wordlist.txt")

# Immutable/permanent locations
ARTIFACT_PROFILES_DIR: Path = BASE_DIR / "DATA/artifact_profiles"
CHECKSUM_HISTORY_FILE: Path = BASE_DIR / "DATA/checksum_history.txt"
LOG_DIR: Path = BASE_DIR / "DATA/logs"
ARCHIVAL_DIR: Path = BASE_DIR / "DATA/archive"
TEMP_DIR: Path = BASE_DIR / "TEMP"

# Main processing pipeline stages
FAILED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/00_review_failures"
SANITIZED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/01_sanitized"
SCANNED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/02_scanned"
METADATA_EXTRACTED_DIR: Path = BASE_DIR / "PROCESSING/03_metadata_extracted"
CONVERTED_ARTIFACT_DIR: Path = BASE_DIR / "PROCESSING/04_converted"
EMBELLISHED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/05_embellished"
SEMANTICS_EXTRACTED_DIR: Path = BASE_DIR / "PROCESSING/06_semantics_extracted"
TRANSLATED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/07_translated"
PROTECTED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/08_protected"
COMPLETED_ARTIFACTS_DIR: Path = BASE_DIR / "PROCESSING/09_completed"

# System directories collection
SYSTEM_DIRECTORIES: Set[Path] = {
    FAILED_ARTIFACTS_DIR,
    SANITIZED_ARTIFACTS_DIR,
    METADATA_EXTRACTED_DIR,
    CONVERTED_ARTIFACT_DIR,
    EMBELLISHED_ARTIFACTS_DIR,
    SEMANTICS_EXTRACTED_DIR,
    TRANSLATED_ARTIFACTS_DIR,
    PROTECTED_ARTIFACTS_DIR,
    COMPLETED_ARTIFACTS_DIR,
    ARTIFACT_PROFILES_DIR,
    LOG_DIR,
    ARCHIVAL_DIR,
    TEMP_DIR,
    SCANNED_ARTIFACTS_DIR,
}


# ============================================================================
# FILE TYPE DEFINITIONS
# ============================================================================

EMAIL_TYPES = ["eml", "msg", "mbox", "emlx"]

DOCUMENT_TYPES = [
    "pdf",
    "docx",
    "doc",
    "pptx",
    "ppt",
    "odt",
    "odp",
    "ods",
    "rtf",
    "epub",
    "pub",
]

IMAGE_TYPES = [
    "jpeg",
    "jpg",
    "png",
    "bmp",
    "tiff",
    "webp",
    "svg",
    "ico",
    "psd",
    "heic",
    "heif",
]

VIDEO_TYPES = [
    "mp4",
    "avi",
    "mkv",
    "mov",
    "wmv",
    "flv",
    "webm",
    "m4v",
    "mpg",
    "mpeg",
    "3gp",
    "ogv",
]

AUDIO_TYPES = ["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus", "aiff", "ape"]

TEXT_TYPES = [
    "txt",
    "md",
    "markdown",
    "rst",
    "csv",
    "json",
    "xml",
    "yaml",
    "yml",
    "toml",
    "ini",
    "log",
]

ARCHIVE_TYPES = [
    "zip",
    "tar",
    "gz",
    "bz2",
    "xz",
    "7z",
    "rar",
    "tar.gz",
    "tar.bz2",
    "tar.xz",
]

CODE_TYPES = [
    "python",
    "javascript",
    "java",
    "cpp",
    "c",
    "csharp",
    "html",
    "css",
    "php",
    "ruby",
    "go",
    "rust",
    "sql",
]


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
DEFAULT_PROCESSING_MODE: ProcessingMode = ProcessingMode.HIGH_QUALITY

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
    "language": """Analyze the document text and identify the PREDOMINANT language present.
    Document contains sections in multiple languages. List the PREDOMINANT/MAIN language found.
    Examples:
    - If English only: "English"
    - If French only: "French"
    - If both: "English, French", provide the language with more text
    Return ONLY language names.""",
    "issuing_body": """Identify any official authority that validated, certified, witnessed, or authorized this document.
    Look for:
    - Notary public names and seals
    - Certifying agency names
    - Official witnesses or authorizing bodies
    - Government officials who signed or stamped
    - Licensing boards or regulatory authorities
    Return ONLY the name of the official authority. If no official validation exists, return "UNKNOWN".""",
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
MIN_JAVA_VERSION: int = 11
MIN_FILE_TYPE_CONF_SCORE: float = 75.0

SILENCE_THRESH_DB: float = -50.0
MIN_SILENCE_LEN_MS: int = 1000
MIN_NONSILENT_RATIO: float = 0.01
RMS_ENERGY_THRESHOLD_DB: float = -40.0
PREFERRED_AUDIO_MODEL = "large-v3"  # or "medium", "base", "small", "tiny"
MIN_TRANSCRIPTION_SUCCESS_RATE = 0.7  # 70% minimum success rate
MAX_PDF_SIZE_BEFORE_SUBSETTING: int = 8


# Language code mapping (ISO 639-3 to full names for pdf2zh)
LANGUAGE_MAP = {
    "ENG": "English",
    "ZHO": "Simplified Chinese",
    "ZHT": "Traditional Chinese",
    "SPA": "Spanish",
    "FRA": "French",
    "DEU": "German",
    "ITA": "Italian",
    "POR": "Portuguese",
    "RUS": "Russian",
    "JPN": "Japanese",
    "KOR": "Korean",
    "ARA": "Arabic",
    "HIN": "Hindi",
    "BEN": "Bengali",
    "VIE": "Vietnamese",
    "THA": "Thai",
    "TUR": "Turkish",
    "POL": "Polish",
    "NLD": "Dutch",
    "GRE": "Greek",
    "HEB": "Hebrew",
    "SWE": "Swedish",
    "NOR": "Norwegian",
    "DAN": "Danish",
    "FIN": "Finnish",
    "CZE": "Czech",
    "HUN": "Hungarian",
    "ROM": "Romanian",
    "UKR": "Ukrainian",
    "IND": "Indonesian",
    "MAY": "Malay",
    "PER": "Persian",
    "URD": "Urdu",
}

PREFERRED_LANGUAGE_TRANSLATIONS: List[str] = ["ENG", "FRA", "BEN", "DEU", "POL"]
PREFERRED_TRANSLATION_MODEL: str = "ollama:gemma2:9b"

# Watermark translations for different languages
TRANSLATION_WATERMARKS: Dict[str, str] = {
    "DEU": f"Dieses Dokument wurde mit PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL}) aus dem Englischen übersetzt",
    "FRA": f"Ce document a été traduit de l'anglais avec PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "SPA": f"Este documento ha sido traducido del inglés usando PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "ITA": f"Questo documento è stato tradotto dall'inglese usando PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "POR": f"Este documento foi traduzido do inglês usando PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "RUS": f"Этот документ был переведен с английского с помощью PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "JPN": f"この文書はPDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})を使用して英語から翻訳されました",
    "KOR": f"이 문서는 PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})를 사용하여 영어에서 번역되었습니다",
    "CHI": f"本文档已使用 PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL}) 从英文翻译",
    "NLD": f"Dit document is vertaald uit het Engels met PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "POL": f"Ten dokument został przetłumaczony z angielskiego za pomocą PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL})",
    "TUR": f"Bu belge PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL}) kullanılarak İngilizceden çevrilmiştir",
    "HIN": f"यह दस्तावेज़ PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL}) का उपयोग करके अंग्रेजी से अनुवादित किया गया है",
    "ENG": f"This document has been translated using PDFMathTranslate (arxiv.org/pdf/2507.03009) ({PREFERRED_TRANSLATION_MODEL}) from English",
}

INCLUDE_SPECIAL_SYMBOLS_IN_PASSWORD: bool = True
EXCLUDE_VISUALLY_SIMILAR_CHARS_IN_PASSWORD: bool = True
