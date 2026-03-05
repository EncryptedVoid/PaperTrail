"""
Configuration Module

Central configuration management for the PaperTrail processing pipeline.
Contains all settings for file processing, security, directory structures,
and system parameters.

Author: Ashiq Gazi
"""

import os
from pathlib import Path
from typing import List , Set

# ── Root Directories ──────────────────────────────────────────────────────────
TARGET_DRIVE: Path = Path( "E:" )
# TARGET_DRIVE: Path = Path( r"C:/Users/UserX/Desktop//" )
UNPROCESSED_ARTIFACTS_DIR: Path = Path( r"E:/PAPERTRAIL-LOAD" )
# UNPROCESSED_ARTIFACTS_DIR: Path = Path( r"C:/Users/UserX/Desktop/PAPERTRAIL-PROCESSING" )

BASE_DIR: Path = Path( TARGET_DRIVE / "PAPERTRAIL-PROCESSING" )
RECURSIVE_SORT_DIR = Path( UNPROCESSED_ARTIFACTS_DIR / "RECURSIVE_SORT" )
PROGRAM_ASSETS_DIR = Path( r"C:/Users/UserX/Desktop/Github-Workspace/PaperTrail/assets" )

# ── Processing Directories ─────────────────────────────────────────────────────
ARCHIVAL_DIR: Path = Path( BASE_DIR / "ARCHIVE" )
ARTIFACT_PROFILES_DIR: Path = Path( BASE_DIR / "ARTIFACT_PROFILES" )
LOG_DIR: Path = Path( BASE_DIR / "PAPERTRAIL-LOGS" )
CHECKSUM_HISTORY_FILE: Path = Path( BASE_DIR / "checksum_history.txt" )
TEMP_DIR: Path = Path( BASE_DIR / "TEMP" )
DELETE_DIR: Path = Path( BASE_DIR / "TO_BE_DELETED" )

# ── Stage Directories ─────────────────────────────────────────────────────
COMPLETED_SANITIZATION_DIR: Path = Path(
		BASE_DIR / "STAGE_STATE_MACHINE" / "[1]COMPLETED_SANITIZATION" )
COMPLETED_FORMAT_CONVERSION_DIR: Path = Path(
		BASE_DIR / "STAGE_STATE_MACHINE" / "[2]COMPLETED_FORMAT_CONVERSION" )

# ── Review Directories ─────────────────────────────────────────────────────────
CORRUPTED_ARTIFACTS_DIR: Path = Path( BASE_DIR / "REVIEW/CORRUPTED_ARTIFACTS" )
DUPLICATE_ARTIFACTS_DIR: Path = Path( BASE_DIR / "REVIEW/DUPLICATE_ARTIFACTS" )
PASSWORD_PROTECTED_ARTIFACTS_DIR: Path = Path( BASE_DIR / "REVIEW/PASSWORD_PROTECTED_ARTIFACTS" )
UNSUPPORTED_ARTIFACTS_DIR: Path = Path( BASE_DIR / "REVIEW/UNSUPPORTED_ARTIFACTS" )
ALTERATIONS_REQUIRED_DIR: Path = Path( BASE_DIR / "REVIEW/ALTERATIONS_REQUIRED" )
SCANNING_REQUIRED_DIR: Path = Path( ALTERATIONS_REQUIRED_DIR / "DOCUMENT_SCANNING_REQUIRED" )
UNESSENTIAL_DIR: Path = Path( BASE_DIR / "UNESSENTIAL_ITEMS" )
ALTERATIONS_CSV: Path = ALTERATIONS_REQUIRED_DIR / "alterations_log.csv"

# ── Application Data Directories ──────────────────────────────────────────────
AFFINE_DIR: Path = Path(
		TARGET_DRIVE / "AFFINE" )  # https://affine.pro/
PERSONAL_LIBRARY_DIR: Path = Path(
		TARGET_DRIVE / "PERSONAL_LIBRARY/BOOKDROP" )  # https://github.com/booklore-app/booklore
DIGITAL_ASSET_MANAGEMENT_DIR: Path = Path(
		TARGET_DRIVE / "DIGITAL_ASSET_MANAGEMENT" )  # https://www.pydio.com/
FIREFLYIII_DIR: Path = Path(
		TARGET_DRIVE / "FIREFLY-III" )  # https://www.firefly-iii.org/
JELLYFIN_DIR: Path = Path(
		TARGET_DRIVE / "JELLYFIN/SELF-HOSTED-APP/media" )  # https://jellyfin.org/
GITLAB_DIR: Path = Path(
		TARGET_DRIVE / "GITLAB" )  # https://about.gitlab.com/
IMMICH_DIR: Path = Path(
		TARGET_DRIVE / "IMMICH" )  # https://immich.app/
LINKWARDEN_DIR: Path = Path(
		TARGET_DRIVE / "LINKWARDEN" )  # https://linkwarden.app/
MONICA_CRM_DIR: Path = Path(
		TARGET_DRIVE / "MONICA_CRM" )  # https://www.monicahq.com/
ODOO_CRM_DIR: Path = Path(
		TARGET_DRIVE / "ODOO_CRM" )  # https://www.odoo.com/app/crm
ODOO_MAINTENANCE_DIR: Path = Path(
		TARGET_DRIVE / "ODOO_MAINTENANCE" )  # https://www.odoo.com/app/maintenance
ODOO_PLM_DIR: Path = Path(
		TARGET_DRIVE / "ODOO_PLM" )  # https://www.odoo.com/app/plm
ODOO_PURCHASE_DIR: Path = Path(
		TARGET_DRIVE / "ODOO_PURCHASE" )  # https://www.odoo.com/app/purchase
ODOO_INVENTORY_DIR: Path = Path(
		TARGET_DRIVE / "ODOO_INVENTORY" )  # https://www.odoo.com/app/inventory
PERFORMANCE_PORTFOLIO_DIR: Path = Path(
		TARGET_DRIVE / "PORTFOLIO_PERFORMANCE" )  # https://www.portfolio-performance.info/en/
ULTIMAKER_CURA_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "ULTIMAKER_CURA" )
GAMES_ARCHIVE_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "GAMES_ARCHIVE" )
PERSONAL_ARCHIVE_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "PERSONAL_ARCHIVE" )
MANUALS_ARCHIVE_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "MANUALS_ARCHIVE" )
SOFTWARE_ARCHIVE_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "SOFTWARE_ARCHIVE" )
ANKI_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "ANKI" )  # https://apps.ankiweb.net/
BITWARDEN_DIR: Path = Path(
		DIGITAL_ASSET_MANAGEMENT_DIR / "BITWARDEN" )  # https://bitwarden.com/

#  ── System Directory Index ─────────────────────────────────────────────────────
# Complete set of all managed directories, excluding loose files
SYSTEM_PROGRAM_TRACKING_FILES: set[ Path ] = {
	CHECKSUM_HISTORY_FILE ,
	ALTERATIONS_CSV ,
}

SYSTEM_DIRECTORIES: set[ Path ] = {
	TARGET_DRIVE ,
	BASE_DIR ,
	UNPROCESSED_ARTIFACTS_DIR ,
	ARCHIVAL_DIR ,
	ARTIFACT_PROFILES_DIR ,
	LOG_DIR ,
	TEMP_DIR ,
	CORRUPTED_ARTIFACTS_DIR ,
	DUPLICATE_ARTIFACTS_DIR ,
	PASSWORD_PROTECTED_ARTIFACTS_DIR ,
	UNSUPPORTED_ARTIFACTS_DIR ,
	ALTERATIONS_REQUIRED_DIR ,
	SCANNING_REQUIRED_DIR ,
	UNESSENTIAL_DIR ,
	AFFINE_DIR ,
	ANKI_DIR ,
	BITWARDEN_DIR ,
	PERSONAL_LIBRARY_DIR ,
	DIGITAL_ASSET_MANAGEMENT_DIR ,
	FIREFLYIII_DIR ,
	JELLYFIN_DIR ,
	GITLAB_DIR ,
	IMMICH_DIR ,
	LINKWARDEN_DIR ,
	MONICA_CRM_DIR ,
	ODOO_CRM_DIR ,
	ODOO_MAINTENANCE_DIR ,
	ODOO_PLM_DIR ,
	ODOO_PURCHASE_DIR ,
	ODOO_INVENTORY_DIR ,
	PERFORMANCE_PORTFOLIO_DIR ,
	ULTIMAKER_CURA_DIR ,
	PERSONAL_ARCHIVE_DIR ,
	GAMES_ARCHIVE_DIR ,
	MANUALS_ARCHIVE_DIR ,
	SOFTWARE_ARCHIVE_DIR ,
	COMPLETED_SANITIZATION_DIR ,
	COMPLETED_FORMAT_CONVERSION_DIR ,
	DELETE_DIR ,
}

APPLICATION_FOLDERS = [
	AFFINE_DIR ,
	ANKI_DIR ,
	BITWARDEN_DIR ,
	PERSONAL_LIBRARY_DIR ,
	DIGITAL_ASSET_MANAGEMENT_DIR ,
	FIREFLYIII_DIR ,
	JELLYFIN_DIR ,
	GITLAB_DIR ,
	IMMICH_DIR ,
	LINKWARDEN_DIR ,
	MONICA_CRM_DIR ,
	ODOO_CRM_DIR ,
	ODOO_MAINTENANCE_DIR ,
	ODOO_PLM_DIR ,
	ODOO_PURCHASE_DIR ,
	ODOO_INVENTORY_DIR ,
	PERFORMANCE_PORTFOLIO_DIR ,
	ULTIMAKER_CURA_DIR ,
	PERSONAL_ARCHIVE_DIR ,
	GAMES_ARCHIVE_DIR ,
	MANUALS_ARCHIVE_DIR ,
	SOFTWARE_ARCHIVE_DIR ,
]

# Checksum algorithm for duplicate detection and integrity verification
CHECKSUM_ALGORITHM = "sha3_512"
CHECKSUM_CHUNK_SIZE_BYTES: int = 8192
FILE_TRIAGE_BATCH_SIZE = 20
FILE_TRIAGE_MAX_PDF_PG = 10
PDF_SUBSET_SIZE = 10

HUGGING_FACE_TOKEN = os.getenv( "HUGGING_FACE_TOKEN" )

ARTIFACT_PREFIX: str = "ARTIFACT"
PROFILE_PREFIX: str = "PROFILE"
LOG_LEVEL: str = "DEBUG"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SESSION_LOG_FILE_PREFIX: str = "PAPERTRAIL-SESSION"

EMAIL_TYPES = [
	"eml" ,
	"msg" ,
]

DOCUMENT_TYPES = [
	"doc" ,
	"docm" ,  # Word macro-enabled
	"docx" ,
	"epub" ,
	"indd" ,  # Adobe InDesign
	"key" ,  # Apple Keynote
	"numbers" ,  # Apple Numbers
	"odp" ,
	"ods" ,
	"odt" ,
	"pages" ,  # Apple Pages
	"pdf" ,
	"ppt" ,
	"pptm" ,  # PowerPoint macro-enabled
	"pptx" ,
	"pub" ,  # Microsoft Publisher
	"rtf" ,
	"xls" ,
	"xlsm" ,  # Excel macro-enabled
	"xlsx" ,
	"xps" ,  # XML Paper Specification
	"chm" ,
]

MICROSOFT_FILE_TYPES = [
	"accdb" ,  # Access database
	"doc" ,
	"docm" ,
	"docx" ,
	"mdb" ,  # Legacy Access database
	"mpp" ,  # Microsoft Project
	"one" ,  # OneNote
	"ppt" ,
	"pptm" ,
	"pptx" ,
	"pub" ,
	"vsd" ,  # Visio (legacy)
	"vsdx" ,  # Visio
	"vdx" ,  # Visio
	"xls" ,
	"xlsm" ,
	"xlsx" ,
	"onepkg" ,
	"pub" ,
]

IMAGE_TYPES = [
	"arw" ,  # Sony RAW
	"avif" ,  # AV1 Image Format
	"bmp" ,
	"cr2" ,  # Canon RAW (gen 2)
	"cr3" ,  # Canon RAW (gen 3)
	"dng" ,  # Adobe Digital Negative
	"eps" ,  # Encapsulated PostScript
	"exr" ,  # OpenEXR HDR
	"gif" ,
	"hdr" ,  # Radiance HDR
	"heic" ,
	"heif" ,
	"ico" ,
	"jxl" ,  # JPEG XL
	"jpeg" ,
	"jpg" ,
	"nef" ,  # Nikon RAW
	"orf" ,  # Olympus RAW
	"png" ,
	"psd" ,
	"raf" ,  # Fujifilm RAW
	"raw" ,  # Generic RAW
	"rw2" ,  # Panasonic RAW
	"svg" ,
	"tif" ,
	"tiff" ,
	"webp" ,
	"xcf" ,  # GIMP native format
]

VIDEO_TYPES = [
	"3gp" ,
	"asf" ,  # Advanced Systems Format
	"avi" ,
	"divx" ,
	"f4v" ,  # Flash MP4
	"flv" ,
	"m2ts" ,  # Blu-ray MPEG-2
	"m4v" ,
	"mkv" ,
	"mov" ,
	"mp4" ,
	"mpeg" ,
	"mpg" ,
	"mts" ,  # AVCHD
	"mxf" ,  # Material Exchange Format (broadcast)
	"ogv" ,
	"rm" ,  # RealMedia
	"rmvb" ,  # RealMedia Variable Bitrate
	"ts" ,  # MPEG Transport Stream
	"vob" ,  # DVD Video Object
	"webm" ,
	"wmv" ,
]

AUDIO_TYPES = [
	"aac" ,
	"ac3" ,  # Dolby Digital
	"aiff" ,
	"amr" ,  # Adaptive Multi-Rate (mobile)
	"ape" ,
	"au" ,  # Sun/NeXT audio
	"dsf" ,  # DSD audio
	"dts" ,  # DTS audio
	"flac" ,
	"m4a" ,
	"mid" ,  # MIDI
	"midi" ,
	"mka" ,  # Matroska audio
	"mp3" ,
	"ogg" ,
	"opus" ,
	"ra" ,  # RealAudio
	"wav" ,
	"wma" ,
]

TEXT_TYPES = [
	"cfg" ,
	"conf" ,
	"csv" ,
	"env" ,
	"htm" ,
	"html" ,
	"ini" ,
	"json" ,
	"jsonl" ,  # JSON Lines
	"log" ,
	"markdown" ,
	"md" ,
	"nfo" ,
	"rst" ,
	"tex" ,
	"toml" ,
	"tsv" ,  # Tab-separated values
	"txt" ,
	"xml" ,
	"yaml" ,
	"yml" ,
]

ARCHIVE_TYPES = {
	".zip" ,
	".7z" ,
	".tar" ,
	".gz" ,
	".tgz" ,
	".bz2" ,
	".tbz2" ,
	".xz" ,
	".txz" ,
	".tar.gz" ,
	".tar.bz2" ,
	".tar.xz" ,
}

CODE_EXTENSIONS = [
	"asm" ,
	"awk" ,
	"bash" ,
	"bat" ,  # Windows batch
	"c" ,
	"clj" ,  # Clojure
	"cmd" ,  # Windows command script
	"cpp" ,
	"cs" ,
	"css" ,
	"dart" ,
	"elm" ,
	"ex" ,  # Elixir
	"exs" ,  # Elixir script
	"f" ,  # Fortran
	"f90" ,  # Fortran 90
	"fish" ,  # Fish shell
	"go" ,
	"graphql" ,
	"hs" ,  # Haskell
	"hcl" ,  # HashiCorp config (Terraform)
	"html" ,
	"java" ,
	"class" ,
	"jl" ,
	"js" ,
	"jsx" ,
	"kt" ,  # Kotlin
	"lua" ,
	"m" ,
	"ml" ,  # OCaml
	"mli" ,  # OCaml interface
	"nim" ,
	"php" ,
	"pl" ,
	"proto" ,  # Protocol Buffers
	"py" ,
	"r" ,
	"rb" ,
	"rs" ,
	"scala" ,
	"sh" ,
	"sql" ,
	"svelte" ,
	"swift" ,
	"tcl" ,
	"tf" ,  # Terraform
	"ts" ,
	"tsx" ,
	"v" ,  # Verilog
	"vhd" ,  # VHDL
	"vhdl" ,
	"vue" ,
	"wasm" ,
	"zig" ,
	"zsh" ,
	"sh" ,  # Shell script (executable)
	"appimage" ,  # Linux portable app
	"bat" ,  # Windows batch script
	"qpf" ,
	"qsf" ,
	"vwf" ,
	"tex" ,  # LaTeX source
]

EXECUTABLE_EXTENSIONS = [
	"app" ,  # macOS application bundle
	"bin" ,  # Generic binary
	"cmd" ,  # Windows command script
	"deb" ,  # Debian installer
	"dmg" ,  # macOS installer image
	"exe" ,
	"ipa" ,  # iOS app package
	"msi" ,  # Windows installer
	"rpm" ,  # Red Hat installer
	"run" ,  # Linux self-executing binary
	"ps1" ,
	"rdp" ,
	"apk" ,
	"jar" ,
]

ANKI_EXTENSIONS = [
	"apkg" ,
	"anki2" ,
	"colpkg" ,
]

CAD_FILES = [
	"gcode" ,
	"bgcode" ,
	"stl" ,
]

# Contact & Address Book File Format Mapping

DIGITAL_CONTACT_EXTENSIONS = [
	"vcf" ,
	"vcard" ,
	"contact" ,
	"contacts" ,
]

# File type mappings
EXTENSION_MAPPING = {
	# Images
	".jpeg" : "image" ,
	".jpg"  : "image" ,
	".png"  : "image" ,
	".heic" : "image" ,
	".cr2"  : "image" ,
	".arw"  : "image" ,
	".nef"  : "image" ,
	".webp" : "image" ,
	# Videos
	".mov"  : "video" ,
	".mp4"  : "video" ,
	".webm" : "video" ,
	".amv"  : "video" ,
	".3gp"  : "video_audio" ,  # Special case - need to probe
	# Audio
	".wav"  : "audio" ,
	".mp3"  : "audio" ,
	".m4a"  : "audio" ,
	".ogg"  : "audio" ,
	# Documents
	".pptx" : "document" ,
	".doc"  : "document" ,
	".docx" : "document" ,
	".rtf"  : "document" ,
	".epub" : "document" ,
	".pub"  : "document" ,
	".djvu" : "document" ,
	".pdf"  : "document" ,
}

# Target formats for each type
TARGET_FORMATS = {
	"image"    : ".png" ,
	"video"    : ".mp4" ,
	"audio"    : ".mp3" ,
	"document" : ".pdf" ,
}

# Alias groups: extensions that should be treated as equivalent
EXTENSION_ALIASES: dict[ str , str ] = {
	"jpeg" : "jpg" ,
	"cr2"  : "jpg" ,
	"arw"  : "jpg" ,
	"nef"  : "jpg" ,
	"vwf"  : "txt" ,
	"qsf"  : "txt" ,
	"qpf"  : "txt" ,
	"ps1"  : "txt" ,
	"vdx"  : "xml" ,
	"tex"  : "m" ,
}

FILETYPE_TRUSTED_EXTENSIONS: Set[ str ] = set( )
FILETYPE_TRUSTED_EXTENSIONS.update( IMAGE_TYPES )
FILETYPE_TRUSTED_EXTENSIONS.update( VIDEO_TYPES )
FILETYPE_TRUSTED_EXTENSIONS.update( AUDIO_TYPES )
FILETYPE_TRUSTED_EXTENSIONS.update( DOCUMENT_TYPES )
FILETYPE_TRUSTED_EXTENSIONS.update( EMAIL_TYPES )
FILETYPE_TRUSTED_EXTENSIONS.update( EXECUTABLE_EXTENSIONS )
FILETYPE_TRUSTED_EXTENSIONS.update( ext.lstrip( "." ) for ext in ARCHIVE_TYPES )

TIKA_SERVER_PORT: int = 9998
JAVA_PATH: Path = Path( PROGRAM_ASSETS_DIR / r"jdk-25.0.2+10-jre/bin/java.exe" )
TIKA_APP_JAR_PATH: Path = Path( PROGRAM_ASSETS_DIR / r"tika-app-3.2.3.jar" )
TIKA_SERVER_JAR_PATH = Path( PROGRAM_ASSETS_DIR / r"tika-server-standard-3.2.3.jar" )
PDF_ARRANGER_EXE_PATH: Path = Path( PROGRAM_ASSETS_DIR / r"pdfarranger.exe" )
OUCH_DECOMPRESSOR_PATH: Path = Path( PROGRAM_ASSETS_DIR / r"ouch-x86_64-pc-windows-msvc/ouch.exe" )
NOTESHRINK_PATH: Path = Path( PROGRAM_ASSETS_DIR / r"noteshrink/noteshrink.py" )

# Apache Tika MIME Type → File Extension Mapping
# Covers: Documents, Spreadsheets, Presentations, Images, Audio, Video,
#         Archives, Code, Data, CAD, eBooks, Fonts, and more.
# ══════════════════════════════════════════════════════════════════════════════
# MIME_TO_EXT_MAP  — Apache Tika MIME Type → preferred file extension
# ══════════════════════════════════════════════════════════════════════════════
MIME_TO_EXT_MAP: dict[ str , List[ str ] ] = {
	# ─────────────────────────────────────────────────────────────────────
	# PLAIN TEXT & GENERIC MARKUP
	# ─────────────────────────────────────────────────────────────────────
	"text/plain"                                                                : [ ".txt" ] ,
	"text/html"                                                                 : [ ".html" ] ,
	"text/xml"                                                                  : [ ".xml" ] ,
	"text/css"                                                                  : [ ".css" ] ,
	"text/csv"                                                                  : [ ".csv" ] ,
	"text/tab-separated-values"                                                 : [ ".tsv" ] ,
	"text/calendar"                                                             : [ ".ics" ] ,
	"text/vcard"                                                                : [ ".vcf" ] ,
	"text/x-vcard"                                                              : [ ".vcf" ] ,
	"text/x-vcalendar"                                                          : [ ".vcs" ] ,
	"text/vtt"                                                                  : [ ".vtt" ] ,
	"text/markdown"                                                             : [ ".md" ] ,
	"text/x-rst"                                                                : [ ".rst" ] ,
	"text/x-asciidoc"                                                           : [ ".adoc" ] ,
	"text/troff"                                                                : [ ".man" ] ,
	"text/x-tex"                                                                : [ ".tex" ] ,
	"text/x-latex"                                                              : [ ".latex" ] ,
	"text/sgml"                                                                 : [ ".sgml" ] ,
	"text/x-log"                                                                : [ ".log" ] ,
	"text/x-ini"                                                                : [ ".ini" ] ,
	"text/x-properties"                                                         : [ ".properties" ] ,
	"text/x-diff"                                                               : [ ".diff" ] ,
	"text/x-patch"                                                              : [ ".patch" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — WEB / SCRIPTING
	# ─────────────────────────────────────────────────────────────────────
	"text/javascript"                                                           : [ ".js" ] ,
	"application/javascript"                                                    : [ ".js" ] ,
	"application/x-javascript"                                                  : [ ".js" ] ,
	"application/typescript"                                                    : [ ".ts" ] ,
	"text/x-typescript"                                                         : [ ".ts" ] ,
	"text/x-python"                                                             : [ ".py" ] ,
	"application/x-python-code"                                                 : [ ".pyc" ] ,
	"text/x-ruby"                                                               : [ ".rb" ] ,
	"text/x-perl"                                                               : [ ".pl" ] ,
	"text/x-php"                                                                : [ ".php" ] ,
	"application/x-httpd-php"                                                   : [ ".php" ] ,
	"text/x-sh"                                                                 : [ ".sh" ] ,
	"text/x-shellscript"                                                        : [ ".sh" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — COMPILED / SYSTEMS
	# ─────────────────────────────────────────────────────────────────────
	"application/onenote; format=package"                                       : [ ".onepkg" ] ,
	"application/java-vm"                                                       : [ ".class" ] ,
	"application/x-sh"                                                          : [ ".sh" ] ,
	"text/x-java-source"                                                        : [ ".java" ] ,
	"text/x-csrc"                                                               : [ ".c" ] ,
	"text/x-chdr"                                                               : [ ".h" ] ,
	"text/x-c++src"                                                             : [ ".cpp" ] ,
	"text/x-c++hdr"                                                             : [ ".hpp" ] ,
	"text/x-csharp"                                                             : [ ".cs" ] ,
	"text/x-go"                                                                 : [ ".go" ] ,
	"text/x-rust"                                                               : [ ".rs" ] ,
	"text/x-swift"                                                              : [ ".swift" ] ,
	"text/x-kotlin"                                                             : [ ".kt" ] ,
	"text/x-scala"                                                              : [ ".scala" ] ,
	"text/x-vb"                                                                 : [ ".vb" ] ,
	"text/x-pascal"                                                             : [ ".pas" ] ,
	"text/x-fortran"                                                            : [ ".f" ] ,
	"text/x-fortran90"                                                          : [ ".f90" ] ,
	"text/x-fortran95"                                                          : [ ".f95" ] ,
	"text/x-asm"                                                                : [ ".asm" ] ,
	"text/x-nasm"                                                               : [ ".asm" ] ,
	"application/x-sharedlib"                                                   : [ ".so" ] ,
	"application/x-mach-binary"                                                 : [ ".dylib" ] ,
	"application/x-java-class"                                                  : [ ".class" ] ,
	"application/x-llvm"                                                        : [ ".bc" ] ,
	"application/wasm"                                                          : [ ".wasm" ] ,
	"text/x-web-markdown"                                                       : [ ".md" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — DATA SCIENCE / STATISTICS / SCIENTIFIC
	# ─────────────────────────────────────────────────────────────────────
	"text/x-r"                                                                  : [ ".r" ] ,
	"text/x-r-source"                                                           : [ ".r" ] ,
	"text/x-matlab"                                                             : [ ".m" ] ,
	"text/x-octave"                                                             : [ ".m" ] ,
	"text/x-julia"                                                              : [ ".jl" ] ,
	"text/x-lua"                                                                : [ ".lua" ] ,
	"text/x-groovy"                                                             : [ ".groovy" ] ,
	"text/x-clojure"                                                            : [ ".clj" ] ,
	"text/x-haskell"                                                            : [ ".hs" ] ,
	"text/x-erlang"                                                             : [ ".erl" ] ,
	"text/x-elixir"                                                             : [ ".ex" ] ,
	"text/x-fsharp"                                                             : [ ".fs" ] ,
	"text/x-ocaml"                                                              : [ ".ml" ] ,
	"text/x-nim"                                                                : [ ".nim" ] ,
	"text/x-zig"                                                                : [ ".zig" ] ,
	"text/x-crystal"                                                            : [ ".cr" ] ,
	"text/x-dart"                                                               : [ ".dart" ] ,
	"text/x-coffeescript"                                                       : [ ".coffee" ] ,
	"application/x-ipynb+json"                                                  : [ ".ipynb" ] ,
	"application/vnd.jupyter"                                                   : [ ".ipynb" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — BUILD / CONFIG / INFRA
	# ─────────────────────────────────────────────────────────────────────
	"text/x-makefile"                                                           : [ ".makefile" ] ,
	"application/x-cmake"                                                       : [ ".cmake" ] ,
	"text/x-dockerfile"                                                         : [ ".dockerfile" ] ,
	"text/x-toml"                                                               : [ ".toml" ] ,
	"application/toml"                                                          : [ ".toml" ] ,
	"application/x-yaml"                                                        : [ ".yaml" ] ,
	"text/x-yaml"                                                               : [ ".yaml" ] ,
	"text/x-powershell"                                                         : [ ".ps1" ] ,
	"application/x-bat"                                                         : [ ".bat" ] ,
	"text/x-nsis"                                                               : [ ".nsi" ] ,
	"text/x-gradle"                                                             : [ ".gradle" ] ,
	"application/x-web-app-manifest+json"                                       : [ ".webapp" ] ,
	"application/manifest+json"                                                 : [ ".webmanifest" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — SHADER / GPU
	# ─────────────────────────────────────────────────────────────────────
	"text/x-glsl"                                                               : [ ".glsl" ] ,
	"text/x-hlsl"                                                               : [ ".hlsl" ] ,
	"text/x-wgsl"                                                               : [ ".wgsl" ] ,
	"text/x-metal"                                                              : [ ".metal" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CODE — QUERY / DATABASE
	# ─────────────────────────────────────────────────────────────────────
	"text/x-sql"                                                                : [ ".sql" ] ,
	"application/x-sqlite3"                                                     : [ ".sqlite" ] ,
	"application/vnd.sqlite3"                                                   : [ ".sqlite" ] ,
	"application/x-dbf"                                                         : [ ".dbf" ] ,
	"application/vnd.ms-access"                                                 : [ ".mdb" ] ,
	"application/x-parquet"                                                     : [ ".parquet" ] ,
	"application/x-avro"                                                        : [ ".avro" ] ,
	"application/x-orc"                                                         : [ ".orc" ] ,
	"application/x-hdf"                                                         : [ ".h5" ] ,
	"application/x-netcdf"                                                      : [ ".nc" ] ,
	"application/x-matlab-data"                                                 : [ ".mat" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# XML VARIANTS
	# ─────────────────────────────────────────────────────────────────────
	"application/xml"                                                           : [ ".xml" ] ,
	"application/xhtml+xml"                                                     : [ ".xhtml" ] ,
	"application/atom+xml"                                                      : [ ".atom" ] ,
	"application/rss+xml"                                                       : [ ".rss" ] ,
	"application/rdf+xml"                                                       : [ ".rdf" ] ,
	"application/soap+xml"                                                      : [ ".xml" ] ,
	"application/wsdl+xml"                                                      : [ ".wsdl" ] ,
	"application/xslt+xml"                                                      : [ ".xslt" ] ,
	"application/mathml+xml"                                                    : [ ".mml" ] ,
	"application/x-dtd"                                                         : [ ".dtd" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# JSON / DATA INTERCHANGE
	# ─────────────────────────────────────────────────────────────────────
	"application/json"                                                          : [ ".json" ] ,
	"application/ld+json"                                                       : [ ".jsonld" ] ,
	"application/geo+json"                                                      : [ ".geojson" ] ,
	"application/schema+json"                                                   : [ ".json" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# PDF & POSTSCRIPT
	# ─────────────────────────────────────────────────────────────────────
	"application/pdf"                                                           : [ ".pdf" ] ,
	"application/postscript"                                                    : [ ".ps" ] ,
	"application/x-eps"                                                         : [ ".eps" ] ,
	"image/x-eps"                                                               : [ ".eps" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# MICROSOFT OFFICE — LEGACY OLE2
	# ─────────────────────────────────────────────────────────────────────
	"application/msword"                                                        : [ ".doc" ] ,
	"application/vnd.ms-excel"                                                  : [ ".xls" ] ,
	"application/vnd.ms-powerpoint"                                             : [ ".ppt" ] ,
	"application/vnd.ms-project"                                                : [ ".mpp" ] ,
	"application/vnd.ms-works"                                                  : [ ".wks" ] ,
	"application/vnd.ms-outlook"                                                : [ ".msg" ] ,
	"application/vnd.ms-publisher"                                              : [ ".pub" ] ,
	"application/x-mspublisher"                                                 : [ ".pub" ] ,
	"application/vnd.visio"                                                     : [ ".vsd" ] ,
	"application/vnd.ms-visio.drawing"                                          : [ ".vsd" ] ,
	"application/x-mswrite"                                                     : [ ".wri" ] ,
	"model/x.stl-ascii"                                                         : [ "stl" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# MICROSOFT OFFICE — OOXML (2007+)
	# ─────────────────────────────────────────────────────────────────────
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document"   : [ ".docx" ] ,
	"application/vnd.openxmlformats-officedocument.wordprocessingml.template"   : [ ".dotx" ] ,
	"application/vnd.ms-word.document.macroEnabled.12"                          : [ ".docm" ] ,
	"application/vnd.ms-word.template.macroEnabled.12"                          : [ ".dotm" ] ,
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"         : [ ".xlsx" ] ,
	"application/vnd.openxmlformats-officedocument.spreadsheetml.template"      : [ ".xltx" ] ,
	"application/vnd.ms-excel.sheet.macroEnabled.12"                            : [ ".xlsm" ] ,
	"application/vnd.ms-excel.template.macroEnabled.12"                         : [ ".xltm" ] ,
	"application/vnd.ms-excel.addin.macroEnabled.12"                            : [ ".xlam" ] ,
	"application/vnd.ms-excel.sheet.binary.macroEnabled.12"                     : [ ".xlsb" ] ,
	"application/vnd.openxmlformats-officedocument.presentationml.presentation" : [ ".pptx" ] ,
	"application/vnd.openxmlformats-officedocument.presentationml.template"     : [ ".potx" ] ,
	"application/vnd.openxmlformats-officedocument.presentationml.slideshow"    : [ ".ppsx" ] ,
	"application/vnd.ms-powerpoint.presentation.macroEnabled.12"                : [ ".pptm" ] ,
	"application/vnd.ms-powerpoint.template.macroEnabled.12"                    : [ ".potm" ] ,
	"application/vnd.ms-powerpoint.slideshow.macroEnabled.12"                   : [ ".ppsm" ] ,
	"application/vnd.ms-powerpoint.addin.macroEnabled.12"                       : [ ".ppam" ] ,
	"application/vnd.openxmlformats-officedocument.presentationml.slide"        : [ ".sldx" ] ,
	"application/vnd.ms-visio.drawing.main+xml"                                 : [ ".vsdx" ] ,
	"application/vnd.ms-visio.stencil.main+xml"                                 : [ ".vssx" ] ,
	"application/vnd.ms-visio.template.main+xml"                                : [ ".vstx" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# OPENDOCUMENT FORMAT (ODF / LibreOffice)
	# ─────────────────────────────────────────────────────────────────────
	"application/vnd.oasis.opendocument.text"                                   : [ ".odt" ] ,
	"application/vnd.oasis.opendocument.text-template"                          : [ ".ott" ] ,
	"application/vnd.oasis.opendocument.text-web"                               : [ ".oth" ] ,
	"application/vnd.oasis.opendocument.text-master"                            : [ ".odm" ] ,
	"application/vnd.oasis.opendocument.spreadsheet"                            : [ ".ods" ] ,
	"application/vnd.oasis.opendocument.spreadsheet-template"                   : [ ".ots" ] ,
	"application/vnd.oasis.opendocument.presentation"                           : [ ".odp" ] ,
	"application/vnd.oasis.opendocument.presentation-template"                  : [ ".otp" ] ,
	"application/vnd.oasis.opendocument.graphics"                               : [ ".odg" ] ,
	"application/vnd.oasis.opendocument.graphics-template"                      : [ ".otg" ] ,
	"application/vnd.oasis.opendocument.chart"                                  : [ ".odc" ] ,
	"application/vnd.oasis.opendocument.formula"                                : [ ".odf" ] ,
	"application/vnd.oasis.opendocument.database"                               : [ ".odb" ] ,
	"application/vnd.oasis.opendocument.image"                                  : [ ".odi" ] ,
	"application/vnd.oasis.opendocument.flat-xml"                               : [ ".fodt" ] ,
	"application/vnd.sun.xml.writer"                                            : [ ".sxw" ] ,
	"application/vnd.sun.xml.calc"                                              : [ ".sxc" ] ,
	"application/vnd.sun.xml.impress"                                           : [ ".sxi" ] ,
	"application/vnd.sun.xml.draw"                                              : [ ".sxd" ] ,
	"application/vnd.sun.xml.math"                                              : [ ".sxm" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# APPLE iWORK
	# ─────────────────────────────────────────────────────────────────────
	"application/vnd.apple.pages"                                               : [ ".pages" ] ,
	"application/vnd.apple.numbers"                                             : [ ".numbers" ] ,
	"application/vnd.apple.keynote"                                             : [ ".key" ] ,
	"multipart/appledouble"                                                     : [ ".pdf" ] ,
	# This is a special Apple mine type for Apple document exports. For now, it is acceptable for this to map to PDF
	# ─────────────────────────────────────────────────────────────────────
	# RICH TEXT / LEGACY WORD PROCESSORS
	# ─────────────────────────────────────────────────────────────────────
	"application/rtf"                                                           : [ ".rtf" ] ,
	"text/rtf"                                                                  : [ ".rtf" ] ,
	"application/x-abiword"                                                     : [ ".abw" ] ,
	"application/x-abiword-compressed"                                          : [ ".zabw" ] ,
	"application/vnd.wordperfect"                                               : [ ".wpd" ] ,
	"application/x-wordperfect"                                                 : [ ".wpd" ] ,
	"application/vnd.lotus-wordpro"                                             : [ ".lwp" ] ,
	"application/clarisworks"                                                   : [ ".cwk" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# DESKTOP PUBLISHING
	# ─────────────────────────────────────────────────────────────────────
	"application/x-indesign"                                                    : [ ".indd" ] ,
	"application/vnd.adobe.indesign"                                            : [ ".indd" ] ,
	"application/x-indesign-template"                                           : [ ".indt" ] ,
	"application/x-quark-xpress"                                                : [ ".qxd" ] ,
	"application/vnd.scribus"                                                   : [ ".sla" ] ,
	"application/x-scribus"                                                     : [ ".sla" ] ,
	"application/x-affinity-publisher"                                          : [ ".afpub" ] ,
	"application/x-affinity-designer"                                           : [ ".afdesign" ] ,
	"application/x-affinity-photo"                                              : [ ".afphoto" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# eBOOK / DIGITAL BOOK
	# ─────────────────────────────────────────────────────────────────────
	"application/epub+zip"                                                      : [ ".epub" ] ,
	"application/epub"                                                          : [ ".epub" ] ,
	"application/x-mobipocket-ebook"                                            : [ ".mobi" ] ,
	"application/vnd.amazon.ebook"                                              : [ ".azw" ] ,
	"application/x-amazon-ebook"                                                : [ ".azw3" ] ,
	"application/x-kindle-book"                                                 : [ ".kfx" ] ,
	"application/x-fictionbook+xml"                                             : [ ".fb2" ] ,
	"application/x-fictionbook2"                                                : [ ".fb2" ] ,
	"application/x-fictionbook2+zip"                                            : [ ".fbz" ] ,
	"application/vnd.lit"                                                       : [ ".lit" ] ,
	"application/x-sony-bbeb"                                                   : [ ".lrf" ] ,
	"application/x-ibooks+zip"                                                  : [ ".ibooks" ] ,
	"application/x-cbr"                                                         : [ ".cbr" ] ,
	"application/x-cbz"                                                         : [ ".cbz" ] ,
	"application/x-cbt"                                                         : [ ".cbt" ] ,
	"application/x-cb7"                                                         : [ ".cb7" ] ,
	"application/x-cbtar"                                                       : [ ".cbt" ] ,
	"application/x-htmlz"                                                       : [ ".htmlz" ] ,
	"application/x-snb"                                                         : [ ".snb" ] ,  # Bambook
	"application/vnd.palm"                                                      : [ ".pdb" ] ,  # Palm eBook
	"application/x-plucker"                                                     : [ ".pdb" ] ,
	"application/vnd.openebook+zip"                                             : [ ".oebzip" ] ,
	"application/x-dtbncx+xml"                                                  : [ ".ncx" ] ,  # EPUB nav
	"application/x-dtbook+xml"                                                  : [ ".xml" ] ,  # DAISY
	# ─────────────────────────────────────────────────────────────────────
	# ARCHIVES & COMPRESSION
	# ─────────────────────────────────────────────────────────────────────
	"application/zip"                                                           : [ ".zip" ] ,
	"application/x-zip-compressed"                                              : [ ".zip" ] ,
	"application/x-tar"                                                         : [ ".tar" ] ,
	"application/gzip"                                                          : [ ".gz" ] ,
	"application/x-gzip"                                                        : [ ".gz" ] ,
	"application/x-bzip2"                                                       : [ ".bz2" ] ,
	"application/x-bzip"                                                        : [ ".bz" ] ,
	"application/x-7z-compressed"                                               : [ ".7z" ] ,
	"application/x-rar-compressed"                                              : [ ".rar" ] ,
	"application/vnd.rar"                                                       : [ ".rar" ] ,
	"application/x-lzma"                                                        : [ ".lzma" ] ,
	"application/x-xz"                                                          : [ ".xz" ] ,
	"application/x-compress"                                                    : [ ".z" ] ,
	"application/x-lzip"                                                        : [ ".lz" ] ,
	"application/x-lzop"                                                        : [ ".lzo" ] ,
	"application/x-zstd"                                                        : [ ".zst" ] ,
	"application/x-snappy-framed"                                               : [ ".sz" ] ,
	"application/x-lha"                                                         : [ ".lha" ] ,
	"application/x-arj"                                                         : [ ".arj" ] ,
	"application/x-ace-compressed"                                              : [ ".ace" ] ,
	"application/x-cpio"                                                        : [ ".cpio" ] ,
	"application/x-ar"                                                          : [ ".ar" ] ,
	"application/java-archive"                                                  : [ ".jar" ] ,
	"application/x-apple-diskimage"                                             : [ ".dmg" ] ,
	"application/x-iso9660-image"                                               : [ ".iso" ] ,
	"application/x-ms-wim"                                                      : [ ".wim" ] ,
	"application/x-cab"                                                         : [ ".cab" ] ,
	"application/vnd.ms-cab-compressed"                                         : [ ".cab" ] ,
	"application/x-stuffit"                                                     : [ ".sit" ] ,
	"application/x-stuffitx"                                                    : [ ".sitx" ] ,
	"application/x-deb"                                                         : [ ".deb" ] ,
	"application/x-rpm"                                                         : [ ".rpm" ] ,
	"application/x-nsis"                                                        : [ ".exe" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# IMAGES — RASTER
	# ─────────────────────────────────────────────────────────────────────
	"image/jpeg"                                                                : [ ".jpg" ] ,
	"image/png"                                                                 : [ ".png" ] ,
	"image/gif"                                                                 : [ ".gif" ] ,
	"image/bmp"                                                                 : [ ".bmp" ] ,
	"image/x-bmp"                                                               : [ ".bmp" ] ,
	"image/tiff"                                                                : [ ".tiff" ] ,
	"image/webp"                                                                : [ ".webp" ] ,
	"image/heic"                                                                : [ ".heic" ] ,
	"image/heif"                                                                : [ ".heif" ] ,
	"image/avif"                                                                : [ ".avif" ] ,
	"image/jxl"                                                                 : [ ".jxl" ] ,
	"image/jp2"                                                                 : [ ".jp2" ] ,
	"image/jpx"                                                                 : [ ".jpx" ] ,
	"image/jpm"                                                                 : [ ".jpm" ] ,
	"image/x-portable-bitmap"                                                   : [ ".pbm" ] ,
	"image/x-portable-graymap"                                                  : [ ".pgm" ] ,
	"image/x-portable-pixmap"                                                   : [ ".ppm" ] ,
	"image/x-portable-anymap"                                                   : [ ".pnm" ] ,
	"image/x-pcx"                                                               : [ ".pcx" ] ,
	"image/vnd.microsoft.icon"                                                  : [ ".ico" ] ,
	"image/x-icon"                                                              : [ ".ico" ] ,
	"image/x-xcf"                                                               : [ ".xcf" ] ,
	"image/x-photoshop"                                                         : [ ".psd" ] ,
	"image/vnd.adobe.photoshop"                                                 : [ ".psd" ] ,
	"image/x-raw-adobe"                                                         : [ ".dng" ] ,
	"image/x-nikon-nef"                                                         : [ ".nef" ] ,
	"image/x-canon-cr2"                                                         : [ ".cr2" ] ,
	"image/x-canon-cr3"                                                         : [ ".cr3" ] ,
	"image/x-canon-crw"                                                         : [ ".crw" ] ,
	"image/x-sony-arw"                                                          : [ ".arw" ] ,
	"image/x-sony-srf"                                                          : [ ".srf" ] ,
	"image/x-olympus-orf"                                                       : [ ".orf" ] ,
	"image/x-fuji-raf"                                                          : [ ".raf" ] ,
	"image/x-panasonic-rw2"                                                     : [ ".rw2" ] ,
	"image/x-sigma-x3f"                                                         : [ ".x3f" ] ,
	"image/x-pentax-pef"                                                        : [ ".pef" ] ,
	"image/x-leica-rwl"                                                         : [ ".rwl" ] ,
	"image/x-hasselblad-3fr"                                                    : [ ".3fr" ] ,
	"image/x-tga"                                                               : [ ".tga" ] ,
	"image/x-sgi"                                                               : [ ".sgi" ] ,
	"image/x-rgb"                                                               : [ ".rgb" ] ,
	"image/x-exr"                                                               : [ ".exr" ] ,
	"image/x-hdr"                                                               : [ ".hdr" ] ,
	"image/x-dds"                                                               : [ ".dds" ] ,
	"image/x-ktx"                                                               : [ ".ktx" ] ,
	"image/x-ktx2"                                                              : [ ".ktx2" ] ,
	"image/x-ilbm"                                                              : [ ".ilbm" ] ,
	"image/x-wmf"                                                               : [ ".wmf" ] ,
	"image/emf"                                                                 : [ ".emf" ] ,
	"image/x-emf"                                                               : [ ".emf" ] ,
	"image/vnd.dwg"                                                             : [ ".dwg" ] ,
	"image/vnd.dxf"                                                             : [ ".dxf" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# IMAGES — VECTOR
	# ─────────────────────────────────────────────────────────────────────
	"image/svg+xml"                                                             : [ ".svg" ] ,
	"image/svg+xml-compressed"                                                  : [ ".svgz" ] ,
	"application/illustrator"                                                   : [ ".ai" ] ,
	"application/vnd.corel-draw"                                                : [ ".cdr" ] ,
	"application/x-corel-draw"                                                  : [ ".cdr" ] ,
	"application/x-xfig"                                                        : [ ".fig" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# AUDIO
	# ─────────────────────────────────────────────────────────────────────
	"audio/mpeg"                                                                : [ ".mp3" ] ,
	"audio/mp3"                                                                 : [ ".mp3" ] ,
	"audio/mp2"                                                                 : [ ".mp2" ] ,
	"audio/mp4"                                                                 : [ ".m4a" ] ,
	"audio/x-m4a"                                                               : [ ".m4a" ] ,
	"audio/ogg"                                                                 : [ ".ogg" ] ,
	"audio/vorbis"                                                              : [ ".oga" ] ,
	"audio/flac"                                                                : [ ".flac" ] ,
	"audio/x-flac"                                                              : [ ".flac" ] ,
	"audio/wav"                                                                 : [ ".wav" ] ,
	"audio/x-wav"                                                               : [ ".wav" ] ,
	"audio/vnd.wave"                                                            : [ ".wav" ] ,
	"audio/aiff"                                                                : [ ".aiff" ] ,
	"audio/x-aiff"                                                              : [ ".aiff" ] ,
	"audio/aac"                                                                 : [ ".aac" ] ,
	"audio/x-aac"                                                               : [ ".aac" ] ,
	"audio/webm"                                                                : [ ".weba" ] ,
	"audio/opus"                                                                : [ ".opus" ] ,
	"audio/speex"                                                               : [ ".spx" ] ,
	"audio/amr"                                                                 : [ ".amr" ] ,
	"audio/x-ms-wma"                                                            : [ ".wma" ] ,
	"audio/x-pn-realaudio"                                                      : [ ".ra" ] ,
	"audio/vnd.rn-realaudio"                                                    : [ ".rm" ] ,
	"audio/midi"                                                                : [ ".midi" ] ,
	"audio/x-midi"                                                              : [ ".midi" ] ,
	"audio/x-mod"                                                               : [ ".mod" ] ,
	"audio/x-s3m"                                                               : [ ".s3m" ] ,
	"audio/x-xm"                                                                : [ ".xm" ] ,
	"audio/x-it"                                                                : [ ".it" ] ,
	"audio/x-mpegurl"                                                           : [ ".m3u" ] ,
	"audio/x-scpls"                                                             : [ ".pls" ] ,
	"audio/x-ape"                                                               : [ ".ape" ] ,
	"audio/x-musepack"                                                          : [ ".mpc" ] ,
	"audio/x-wavpack"                                                           : [ ".wv" ] ,
	"audio/x-tta"                                                               : [ ".tta" ] ,
	"audio/x-caf"                                                               : [ ".caf" ] ,
	"audio/3gpp"                                                                : [ ".3gp" ] ,
	"audio/3gpp2"                                                               : [ ".3g2" ] ,
	"audio/basic"                                                               : [ ".au" ] ,
	"audio/x-au"                                                                : [ ".au" ] ,
	"audio/gsm"                                                                 : [ ".gsm" ] ,
	"audio/vnd.dolby.dd-raw"                                                    : [ ".ac3" ] ,
	"audio/x-dts"                                                               : [ ".dts" ] ,
	"audio/x-dsf"                                                               : [ ".dsf" ] ,  # DSD
	"audio/x-dff"                                                               : [ ".dff" ] ,  # DSD
	"audio/x-tak"                                                               : [ ".tak" ] ,
	"audio/x-la"                                                                : [ ".la" ] ,  # Lossless Audio
	# ─────────────────────────────────────────────────────────────────────
	# VIDEO
	# ─────────────────────────────────────────────────────────────────────
	"video/mp4"                                                                 : [ ".mp4" ] ,
	"video/x-m4v"                                                               : [ ".m4v" ] ,
	"video/mpeg"                                                                : [ ".mpeg" ] ,
	"video/x-mpeg"                                                              : [ ".mpg" ] ,
	"video/ogg"                                                                 : [ ".ogv" ] ,
	"video/webm"                                                                : [ ".webm" ] ,
	"video/x-msvideo"                                                           : [ ".avi" ] ,
	"video/x-ms-wmv"                                                            : [ ".wmv" ] ,
	"video/x-ms-asf"                                                            : [ ".asf" ] ,
	"video/quicktime"                                                           : [ ".mov" ] ,
	"video/x-matroska"                                                          : [ ".mkv" ] ,
	"video/x-flv"                                                               : [ ".flv" ] ,
	"video/x-f4v"                                                               : [ ".f4v" ] ,
	"video/3gpp"                                                                : [ ".3gp" ] ,
	"video/3gpp2"                                                               : [ ".3g2" ] ,
	"video/x-dv"                                                                : [ ".dv" ] ,
	"video/x-ms-vob"                                                            : [ ".vob" ] ,
	"video/dvd"                                                                 : [ ".vob" ] ,
	"video/vnd.rn-realvideo"                                                    : [ ".rm" ] ,
	"video/x-pn-realvideo"                                                      : [ ".rmvb" ] ,
	"video/h264"                                                                : [ ".h264" ] ,
	"video/h265"                                                                : [ ".h265" ] ,
	"video/av1"                                                                 : [ ".av1" ] ,
	"video/x-ogm"                                                               : [ ".ogm" ] ,
	"video/mp2t"                                                                : [ ".ts" ] ,
	"video/x-m2ts"                                                              : [ ".m2ts" ] ,
	"video/mts"                                                                 : [ ".mts" ] ,
	"video/x-mjpeg"                                                             : [ ".mjpeg" ] ,
	"video/x-divx"                                                              : [ ".divx" ] ,
	"video/x-xvid"                                                              : [ ".xvid" ] ,
	"video/x-prores"                                                            : [ ".mov" ] ,  # Apple ProRes in MOV
	"video/x-dnxhd"                                                             : [ ".mxf" ] ,  # Avid DNxHD
	"application/mxf"                                                           : [ ".mxf" ] ,  # MXF broadcast
	"video/x-cinema-dng"                                                        : [ ".dng" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# SUBTITLES / CAPTIONS
	# ─────────────────────────────────────────────────────────────────────
	"application/x-subrip"                                                      : [ ".srt" ] ,
	"text/x-ssa"                                                                : [ ".ssa" ] ,
	"text/x-ass"                                                                : [ ".ass" ] ,
	"application/ttml+xml"                                                      : [ ".ttml" ] ,
	"application/smil+xml"                                                      : [ ".smil" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# 3-D / CAD / SLICER
	# ─────────────────────────────────────────────────────────────────────
	# Slicer
	"text/x-gcode"                                                              : [ ".gcode" ] ,
	"application/x-gcode"                                                       : [ ".gcode" ] ,
	"text/x-bgcode"                                                             : [ ".bgcode" ] ,
	"application/x-bgcode"                                                      : [ ".bgcode" ] ,
	"application/vnd.ms-3mfdocument"                                            : [ ".3mf" ] ,
	"model/3mf"                                                                 : [ ".3mf" ] ,
	"application/x-amf"                                                         : [ ".amf" ] ,
	# Mesh / polygon
	"model/stl"                                                                 : [ ".stl" ] ,
	"model/x.stl-binary"                                                        : [ ".stl" ] ,
	"application/sla"                                                           : [ ".stl" ] ,
	"model/obj"                                                                 : [ ".obj" ] ,
	"text/x-obj"                                                                : [ ".obj" ] ,
	"model/gltf+json"                                                           : [ ".gltf" ] ,
	"model/gltf-binary"                                                         : [ ".glb" ] ,
	"model/vnd.collada+xml"                                                     : [ ".dae" ] ,
	"application/x-3ds"                                                         : [ ".3ds" ] ,
	"application/x-fbx"                                                         : [ ".fbx" ] ,
	"application/vnd.autodesk.fbx"                                              : [ ".fbx" ] ,
	"application/vnd.ms-pki.stl"                                                : [ ".stl" ] ,
	"model/x3d+xml"                                                             : [ ".x3d" ] ,
	"model/x3d+binary"                                                          : [ ".x3db" ] ,
	"model/vrml"                                                                : [ ".wrl" ] ,
	"model/vnd.vrml"                                                            : [ ".wrl" ] ,
	# USD / AR
	"model/vnd.usd"                                                             : [ ".usd" ] ,
	"model/vnd.usda"                                                            : [ ".usda" ] ,
	"model/vnd.usdc"                                                            : [ ".usdc" ] ,
	"model/vnd.usdz+zip"                                                        : [ ".usdz" ] ,
	# Alembic
	"application/x-alembic"                                                     : [ ".abc" ] ,
	# Solid modelling / STEP / IGES
	"application/x-step"                                                        : [ ".step" ] ,
	"model/step"                                                                : [ ".step" ] ,
	"application/x-iges"                                                        : [ ".iges" ] ,
	"model/iges"                                                                : [ ".iges" ] ,
	# AutoCAD
	"application/acad"                                                          : [ ".dwg" ] ,
	"application/x-dwg"                                                         : [ ".dwg" ] ,
	"application/x-dxf"                                                         : [ ".dxf" ] ,
	"application/x-dwf"                                                         : [ ".dwf" ] ,
	# Rhino
	"application/x-rhino3d"                                                     : [ ".3dm" ] ,
	# SketchUp
	"application/x-koan"                                                        : [ ".skp" ] ,
	"application/vnd.sketchup.skp"                                              : [ ".skp" ] ,
	# Blender
	"application/x-blender"                                                     : [ ".blend" ] ,
	# BIM / IFC
	"application/x-step-ifc"                                                    : [ ".ifc" ] ,
	"application/ifc"                                                           : [ ".ifc" ] ,
	# Point cloud
	"application/x-lastools"                                                    : [ ".las" ] ,
	"application/vnd.las"                                                       : [ ".las" ] ,
	"application/x-e57"                                                         : [ ".e57" ] ,
	"application/x-pcd"                                                         : [ ".pcd" ] ,
	# Voxel
	"application/x-magicavoxel"                                                 : [ ".vox" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# GEOSPATIAL
	# ─────────────────────────────────────────────────────────────────────
	"application/vnd.google-earth.kml+xml"                                      : [ ".kml" ] ,
	"application/vnd.google-earth.kmz"                                          : [ ".kmz" ] ,
	"application/x-esri-shape"                                                  : [ ".shp" ] ,
	"application/gml+xml"                                                       : [ ".gml" ] ,
	"application/gpx+xml"                                                       : [ ".gpx" ] ,
	"application/x-filegdb"                                                     : [ ".gdb" ] ,
	"application/x-mapinfo-mif"                                                 : [ ".mif" ] ,
	"application/x-geotiff"                                                     : [ ".tif" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# FONTS
	# ─────────────────────────────────────────────────────────────────────
	"font/ttf"                                                                  : [ ".ttf" ] ,
	"font/otf"                                                                  : [ ".otf" ] ,
	"font/woff"                                                                 : [ ".woff" ] ,
	"font/woff2"                                                                : [ ".woff2" ] ,
	"application/vnd.ms-fontobject"                                             : [ ".eot" ] ,
	"application/x-font-ttf"                                                    : [ ".ttf" ] ,
	"application/x-font-otf"                                                    : [ ".otf" ] ,
	"application/x-font-type1"                                                  : [ ".pfb" ] ,
	"application/x-font-bdf"                                                    : [ ".bdf" ] ,
	"application/x-font-pcf"                                                    : [ ".pcf" ] ,
	"application/x-font-snf"                                                    : [ ".snf" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# EXECUTABLES & BINARIES
	# ─────────────────────────────────────────────────────────────────────/
	"application/x-msdownload"                                                  : [ ".exe" ] ,
	"application/vnd.microsoft.portable-executable"                             : [ ".exe" ] ,
	"application/x-dosexec"                                                     : [ ".exe" ] ,
	"application/x-elf"                                                         : [ ".elf" ] ,
	"application/x-ms-installer"                                                : [ ".msi" ] ,
	"application/x-msi"                                                         : [ ".msi" ] ,
	"application/x-ms-shortcut"                                                 : [ ".lnk" ] ,
	"application/vnd.android.package-archive"                                   : [ ".apk" ] ,
	"application/x-ios-app"                                                     : [ ".ipa" ] ,
	"application/x-ms-pdb"                                                      : [ ".pdb" ] ,
	"application/x-chrome-extension"                                            : [ ".crx" ] ,
	"application/x-xpinstall"                                                   : [ ".xpi" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# EMAIL & MESSAGING
	# ─────────────────────────────────────────────────────────────────────
	"message/rfc822"                                                            : [ ".eml" ] ,
	"application/mbox"                                                          : [ ".mbox" ] ,
	"message/x-emlx"                                                            : [ ".emlx" ] ,
	"application/pkcs7-mime"                                                    : [ ".p7m" ] ,
	"application/vnd.ms-tnef"                                                   : [ ".tnef" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# CRYPTOGRAPHY & SECURITY
	# ─────────────────────────────────────────────────────────────────────
	"application/x-x509-ca-cert"                                                : [ ".cer" ] ,
	"application/x-pem-file"                                                    : [ ".pem" ] ,
	"application/pkcs12"                                                        : [ ".p12" ] ,
	"application/pkcs8"                                                         : [ ".p8" ] ,
	"application/pgp-encrypted"                                                 : [ ".pgp" ] ,
	"application/pgp-signature"                                                 : [ ".sig" ] ,
	"application/pgp-keys"                                                      : [ ".asc" ] ,
	"application/x-ssh-key"                                                     : [ ".pub" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# SCIENTIFIC / CHEMISTRY / BIOLOGY
	# ─────────────────────────────────────────────────────────────────────
	"chemical/x-cdx"                                                            : [ ".cdx" ] ,
	"chemical/x-cif"                                                            : [ ".cif" ] ,
	"chemical/x-cmdf"                                                           : [ ".cmdf" ] ,
	"chemical/x-cml"                                                            : [ ".cml" ] ,
	"chemical/x-csml"                                                           : [ ".csml" ] ,
	"chemical/x-mol2"                                                           : [ ".mol2" ] ,
	"chemical/x-mdl-molfile"                                                    : [ ".mol" ] ,
	"chemical/x-mdl-sdfile"                                                     : [ ".sdf" ] ,
	"chemical/x-pdb"                                                            : [ ".pdb" ] ,
	"chemical/x-xyz"                                                            : [ ".xyz" ] ,
	"application/x-bibtex-text-file"                                            : [ ".bib" ] ,
	"application/x-research-info-systems"                                       : [ ".ris" ] ,
	"application/x-endnote-refer"                                               : [ ".enw" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# DIAGRAMMING
	# ─────────────────────────────────────────────────────────────────────
	"application/vnd.jgraph.mxfile"                                             : [ ".drawio" ] ,
	"application/x-drawio"                                                      : [ ".drawio" ] ,
	"application/x-mermaid"                                                     : [ ".mmd" ] ,
	"application/x-graphviz"                                                    : [ ".dot" ] ,
	"text/vnd.graphviz"                                                         : [ ".dot" ] ,
	"application/x-plantuml"                                                    : [ ".puml" ] ,
	# ─────────────────────────────────────────────────────────────────────
	# MISC APPLICATION
	# ─────────────────────────────────────────────────────────────────────
	"application/octet-stream"                                                  : [ ".bin" ] ,
	"application/x-binary"                                                      : [ ".bin" ] ,
	"application/x-shockwave-flash"                                             : [ ".swf" ] ,
	"application/x-director"                                                    : [ ".dcr" ] ,
	"application/vnd.tcpdump.pcap"                                              : [ ".pcap" ] ,
	"application/x-pcapng"                                                      : [ ".pcapng" ] ,
	"application/x-torrent"                                                     : [ ".torrent" ] ,
}

# Tiers ordered largest→smallest.
# "quantization" can be None, "int8", or "nf4" (4-bit NormalFloat via bitsandbytes).
# VRAM estimates include headroom for activations and KV cache:
#   bf16  raw weight size  ≈ params × 2 bytes
#   int8  raw weight size  ≈ params × 1 byte   (+~15% overhead)
#   nf4   raw weight size  ≈ params × 0.5 byte  (+~20% overhead)
# Adjust thresholds if you observe OOM in practice.

QWEN_VL_CPU_FALLBACK = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_MODEL: str = "qwen2.5:7b"
VISION_MODEL: str = "minicpm-v:8b"

QWEN_VL_MODEL_TIERS = [
	# ── Full precision tiers ──────────────────────────────────────────────
	{
		"model_id"     : "Qwen/Qwen2.5-VL-72B-Instruct" ,
		"min_vram_gb"  : 80 ,
		"quantization" : None ,
	} ,
	{
		"model_id"     : "Qwen/Qwen2.5-VL-32B-Instruct" ,
		"min_vram_gb"  : 40 ,
		"quantization" : None ,
	} ,
	{
		"model_id"     : "Qwen/Qwen2.5-VL-7B-Instruct" ,
		"min_vram_gb"  : 18 ,
		"quantization" : None ,
	} ,
	# ── Quantized 7B tiers (fills the 10 / 12 / 16 GB gaps) ─────────────
	# 16 GB: 7B bf16 is ~14 GB weights; fits with care but no headroom for
	#         anything above the base prompt. Use int8 for a safer margin.
	{
		"model_id"     : "Qwen/Qwen2.5-VL-7B-Instruct" ,
		"min_vram_gb"  : 16 ,
		"quantization" : "int8" ,
	} ,
	# 12 GB: int8 weights ~8 GB + activations fits comfortably.
	{
		"model_id"     : "Qwen/Qwen2.5-VL-7B-Instruct" ,
		"min_vram_gb"  : 12 ,
		"quantization" : "int8" ,
	} ,
	# 10 GB: int8 is marginal; drop to nf4 (4-bit) for safe headroom.
	{
		"model_id"     : "Qwen/Qwen2.5-VL-7B-Instruct" ,
		"min_vram_gb"  : 10 ,
		"quantization" : "nf4" ,
	} ,
	# ── 3B fallback tiers ────────────────────────────────────────────────
	{ "model_id" : "Qwen/Qwen2.5-VL-3B-Instruct" , "min_vram_gb" : 8 , "quantization" : None } ,
	{
		"model_id"     : "Qwen/Qwen2.5-VL-3B-Instruct" ,
		"min_vram_gb"  : 4 ,
		"quantization" : "int8" ,
	} ,
]

FILENAME_SYSTEM_PROMPT = (
	"You are a file-naming tool. You receive document content and return a single "
	"descriptive filename (no extension). Use lowercase words separated by underscores. "
	"Max 8 words. No explanations, no quotes, no extra text — just the filename."
)

VISUAL_DESC_SYSTEM_PROMPT = (
	"You are a visual analysis tool. Describe what you see in the image clearly and "
	"concisely. Focus on: document type, visible text/headings, logos, layout, key visual "
	"elements, colors, and any identifiable information. Two to four sentences max. "
	"No preamble, no 'This image shows...' — just the description."
)

TAGS_SYSTEM_PROMPT = (
	"You are a document classification tool. You receive document content and return "
	"comma-separated tags that categorize and describe the document. Return 5-10 tags. "
	"Tags should cover: document type, subject domain, entities involved, purpose, and "
	"any notable attributes. Lowercase only. No explanations — just the tags."
)

SCAN_SYSTEM_PROMPT = (
	"You are an OCR correction and text extraction tool. You receive a document image "
	"and return ALL readable text exactly as it appears. Preserve structure: headings, "
	"paragraphs, lists, tables. Fix obvious OCR artifacts but do not infer or add content. "
	"No commentary — just the extracted text."
)

FILENAME_PROMPT_TEMPLATE = """Content:
{content}

Return ONLY a descriptive filename (no extension). Examples:
- immigration_board_hearing_notice_2024
- rental_lease_agreement_toronto
- passport_renewal_application_form
- landscape_mountain_sunset_photo"""

VISUAL_DESC_PROMPT = "Describe this image. Be specific about visible text, layout, and content type."

ORGANISATIONAL_TAGS_PROMPT_TEMPLATE = """Content:
{content}

Return ONLY comma-separated tags. Examples:
- legal, immigration, hearing notice, government, irb, refugee claim, official correspondence
- receipt, financial, purchase, retail, electronics, warranty
- medical, lab report, blood work, diagnostic, hospital, patient record"""

ARTIFACT_SCANNING_PROMPT = "Extract ALL readable text from this document image. Preserve the original structure and formatting."

# These describe the OLD container/codec — injecting them would be misleading
EXCLUDED_KEYS_AFTER_CONVERSION: Set[ str ] = {
	# ── MIME / format identity ───────────────────────────────────────
	"Content-Type" ,
	"content-type" ,
	"Content-Encoding" ,
	"mime_type" ,
	"format_name" ,
	"format_long_name" ,

	# ── Container / codec specifics ──────────────────────────────────
	"codec_name" ,
	"codec_long_name" ,
	"codec_type" ,
	"codec_tag" ,
	"codec_tag_string" ,
	"codec_time_base" ,
	"profile" ,
	"level" ,
	"pix_fmt" ,
	"sample_fmt" ,
	"sample_rate" ,
	"channels" ,
	"channel_layout" ,
	"bit_rate" ,
	"max_bit_rate" ,
	"bits_per_raw_sample" ,
	"bits_per_sample" ,
	"nb_frames" ,
	"nb_streams" ,
	"r_frame_rate" ,
	"avg_frame_rate" ,
	"time_base" ,
	"start_time" ,
	"start_pts" ,
	"duration_ts" ,

	# ── Container size / offsets ─────────────────────────────────────
	"File Size" ,
	"File Name" ,
	"size" ,
	"probe_score" ,

	# ── Tika internal / processing fields ────────────────────────────
	"X-Parsed-By" ,
	"X-TIKA:content_handler" ,
	"X-TIKA:embedded_depth" ,
	"X-TIKA:parse_time_millis" ,
	"X-TIKA:Parsed-By" ,
	"X-TIKA:Parsed-By-Full-Set" ,
	"resourceName" ,
	"Content-Length" ,

	# ── PDF-specific structure fields ────────────────────────────────
	"pdf:PDFVersion" ,
	"pdf:PDFExtensionVersion" ,
	"pdf:docinfo:pdf_version" ,
	"pdf:encrypted" ,
	"pdf:hasXFA" ,
	"pdf:hasXMP" ,
	"pdf:hasMarkedContent" ,
	"pdf:hasCollection" ,
	"pdf:totalUnmappedUnicodeChars" ,
	"pdf:unmappedUnicodeCharsPerPage" ,
	"pdf:containsDamagedFont" ,
	"pdf:containsNonEmbeddedFont" ,
	"pdf:overallPercentageUnmappedUnicodeChars" ,
	"pdf:charsPerPage" ,

	# ── Image format specifics that won't apply after conversion ─────
	"Compression Type" ,
	"Compression" ,
	"Data Precision" ,
	"tiff:BitsPerSample" ,
	"tiff:ImageLength" ,
	"tiff:ImageWidth" ,
	"tiff:SamplesPerPixel" ,
	"tiff:Compression" ,
}

# Prefixes that indicate internal/structural keys to skip entirely
EXCLUDED_PREFIXES_AFTER_CONVERSION = (
	"X-TIKA:" ,
	"access_permission:" ,
	"pdf:docinfo:")
