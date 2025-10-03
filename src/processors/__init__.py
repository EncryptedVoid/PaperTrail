"""
Pipelines package for PaperTrail document processing.

Provides pipeline classes for each stage of document processing:
- Sanitization and duplicate detection
- Metadata extraction
- Semantic analysis and field extraction
- Database tabulation and encryption
"""

from .data_extraction import extract_artifact_data
from .doc_translation import translate_multilingual
from .encryption import encrypt
from .file_conversion import convert
from .manual_review import manual_review
from .sanitization import sanitize

__all__ = [
    archive_and_tabulte,
    encrypt,
    extract_artifact_data,
    manual_review,
    manual_backup,
    sanitize,
    convert,
    translate_multilingual,
]

__version__ = "1.2.0"
