"""
Configuration Module

Central configuration management for the PaperTrail processing pipeline.
Contains all settings for file processing, security, directory structures,
and system parameters.

Author: Ashiq Gazi
"""

from pathlib import Path
from typing import Dict, List, Set


# Resource directories
TARGET_DRIVE: Path = Path("E\\")
BASE_DIR: Path = Path(f"{TARGET_DRIVE}PAPERTRAIL-PROCESSING")
UNPROCESSED_ARTIFACTS_DIR: Path = Path(r"C:\Users\UserX\Desktop\PaperTrail-Load")

# Immutable/permanent locations
ARTIFACT_PROFILES_DIR: Path = BASE_DIR / "DATA/artifact_profiles"
CHECKSUM_HISTORY_FILE: Path = BASE_DIR / "DATA/checksum_history.txt"
LOG_DIR: Path = BASE_DIR / "LOGS"
ARCHIVAL_DIR: Path = BASE_DIR / "ARCHIVE"
TEMP_DIR: Path = BASE_DIR / "TEMP"

LINKWARDEN_DIR: Path = TARGET_DRIVE / "LINKWARDEN"
ANKI_DIR: Path = TARGET_DRIVE / "ANKI"
BITWARDEN_DIR: Path = TARGET_DRIVE / "BITWARDEN"
FIREFLYIII_DIR: Path = TARGET_DRIVE / "FIREFLY-III"
DIGITAL_ASSET_MANAGEMENT_DIR: Path = TARGET_DRIVE / "RESOURCE-SPACE"
IMMICH_DIR: Path = TARGET_DRIVE / "IMMICH"
AFFINE_DIR: Path = TARGET_DRIVE / "AFFINE"
PERFORMANCE_PORTFOLIO_DIR: Path = TARGET_DRIVE / "PERFORMANCE_PORTFOLIO"
CALIBRE_LIBRARY_DIR: Path = TARGET_DRIVE / "CALIBRE_LIBRARY"
GITLAB_DIR: Path = TARGET_DRIVE / "GITLAB"

# Main processing pipeline stages
EMAIL_OUTPUT_DIR = BASE_DIR / "EMAIL_BACKUP"
IMPORTANT_EMAILS_DIR = UNPROCESSED_ARTIFACTS_DIR
UNIMPORTANT_EMAILS_DIR = ARCHIVAL_DIR
EMAIL_ARTIFACTS_DIR = UNPROCESSED_ARTIFACTS_DIR

DUPLICATE_ARTIFACTS_DIR: Path = BASE_DIR / "REVIEW/DUPLICATES"
CORRUPTED_ARTIFACTS_DIR: Path = BASE_DIR / "REVIEW/CORRUPTED_ARTIFACTS"
UNSUPPORTED_ARTIFACTS_DIR: Path = BASE_DIR / "REVIEW/UNSUPPORTED_ARTIFACTS"
PASSWORD_PROTECTED_ARTIFACTS_DIR: Path = (
    BASE_DIR / "REVIEW/PASSWORD_PROTECTED_ARTIFACTS"
)

# System directories collection
SYSTEM_DIRECTORIES: Set[Path] = {
    TARGET_DRIVE,
    BASE_DIR,
    UNPROCESSED_ARTIFACTS_DIR,
    ARTIFACT_PROFILES_DIR,
    CHECKSUM_HISTORY_FILE,
    LOG_DIR,
    ARCHIVAL_DIR,
    TEMP_DIR,
    LINKWARDEN_DIR,
    ANKI_DIR,
    BITWARDEN_DIR,
    FIREFLYIII_DIR,
    DIGITAL_ASSET_MANAGEMENT_DIR,
    IMMICH_DIR,
    AFFINE_DIR,
    PERFORMANCE_PORTFOLIO_DIR,
    EMAIL_OUTPUT_DIR,
    IMPORTANT_EMAILS_DIR,
    UNIMPORTANT_EMAILS_DIR,
    EMAIL_ARTIFACTS_DIR,
    DUPLICATE_ARTIFACTS_DIR,
    CORRUPTED_ARTIFACTS_DIR,
    UNSUPPORTED_ARTIFACTS_DIR,
    PASSWORD_PROTECTED_ARTIFACTS_DIR,
}

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
    "cr2",
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

CODE_EXTENSIONS = [
    ".py",
    ".js",
    ".java",
    ".cpp",
    ".c",
    ".cs",
    ".html",
    ".css",
    ".php",
    ".rb",
    ".go",
    ".rs",
    ".sql",
    ".swift",
    ".kt",
    ".ts",
    ".sh",
    ".ps1",
    ".pl",
    ".r",
    ".jl",
    ".lua",
    ".scala",
    ".m",
    ".asm",
]

# SHA-2 family - Industry standard, FIPS approved, widely supported
# SHA256 = "sha256"
# SHA384 = "sha384"
# SHA512 = "sha512"
# SHA224 = "sha224"
# SHA3_256 = "sha3_256"
# SHA3_384 = "sha3_384"
# SHA3_512 = "sha3_512"
# SHA3_224 = "sha3_224"
# BLAKE2 family - Modern, fast, and secure
# BLAKE2B = "blake2b"
# BLAKE2S = "blake2s"
# SHAKE family - Extendable output functions
# SHAKE_128 = "shake_128"
# SHAKE_256 = "shake_256"
# Legacy algorithms - Deprecated for security-critical applications
# MD5 = "md5"
# SHA1 = "sha1"

# Checksum algorithm for duplicate detection and integrity verification
CHECKSUM_ALGORITHM = "sha3_512"
CHECKSUM_CHUNK_SIZE_BYTES: int = 8192

ARTIFACT_PREFIX: str = "ARTIFACT"
PROFILE_PREFIX: str = "PROFILE"

# UUID generation settings
UUID_PREFIX: str = ""
INCLUDE_TIMESTAMP_ON_UUID: bool = False
UUID_ENTROPY: int = 16

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SESSION_LOG_FILE_PREFIX: str = "PAPERTRAIL-SESSION"

# Preferred models
PREFERRED_VISUAL_MODEL: str = "Qwen/Qwen2-VL-2B-Instruct"
PREFERRED_LANGUAGE_MODEL: str = "mistral:7b"

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
TIKA_APP_JAR_PATH = (
    r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\assets\tika-app-3.2.3.jar"
)
MIN_JAVA_VERSION: int = 11
MIN_FILE_TYPE_CONF_SCORE: float = 75.0

SILENCE_THRESH_DB: float = -50.0
MIN_SILENCE_LEN_MS: int = 1000
MIN_NONSILENT_RATIO: float = 0.01
RMS_ENERGY_THRESHOLD_DB: float = -40.0
PREFERRED_AUDIO_MODEL = "large-v3"  # or "medium", "base", "small", "tiny"
MIN_TRANSCRIPTION_SUCCESS_RATE = 0.9  # 90% minimum success rate

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

PDF_ARRANGER_EXE_PATH = (
    r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\assets\pdfarranger.exe"
)

GMAIL_ADDRESS = "ashiqarib@gmail.com"
OUTLOOK_ADDRESS = "agazi064@uottawa.ca"
