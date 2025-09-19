# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================

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
    "mp3",
    "aac",
    "ogg",
    "wma",
    "m4a",
    "wav",
    "flac",
    "aiff",
    # Video
    "mp4",
    "avi",
    "mov",
    "mkv",
    "wmv",
    "flv",
    "webm",
    "ogv",
    "m4v",
    # 3D & CAD
    "obj",
    "fbx",
    "dae",
    "3ds",
    "blend",
    "dwg",
    "dxf",
    "step",
    "stl",
    "gcode"
    # Executables & System
    "exe",
    "msi",
    "dmg",
    "pkg",
    "deb",
    "rpm",
    "dll",
    "so",
    "dylib",
    # Databases
    "db",
    "sqlite",
    "mdb",
    "accdb",
    # Proprietary
    "indd",
    "fla",
    "swf",
    "sav",
    "dta",
    "sas7bdat",
    "mat",
    "hdf5",
    # Archives (if extraction added)
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

# File names for permanent storage
CHECKSUM_HISTORY_FILE = "checksum_history.txt"

# Base directory for all processing
BASE_DIR: Path = Path(r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run")

# Directory structure - each directory represents a processing stage
PATHS: Dict[str, Path] = {
    "base_dir": BASE_DIR,
    "unprocessed_dir": BASE_DIR / "unprocessed_artifacts",
    "rename_dir": BASE_DIR / "identified_artifacts",
    "metadata_dir": BASE_DIR / "metadata_extracted",
    "semantics_dir": BASE_DIR / "visually_processed",
    "logs_dir": BASE_DIR / "session_logs",
    "process_completed_dir": BASE_DIR / "processed_artifacts",
    "encrypted_dir": BASE_DIR / "encrypted_artifacts",
    "temp_dir": BASE_DIR / "temp",
    "review_dir": BASE_DIR
    / "review_required",  # For zero-byte files and problematic files
    "profiles_dir": BASE_DIR
    / "artifact_profiles",  # JSON profile files for each artifact
}
