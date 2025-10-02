"""
Pipelines package for PaperTrail document processing.

Provides pipeline classes for each stage of document processing:
- Sanitization and duplicate detection
- Metadata extraction
- Semantic analysis and field extraction
- Database tabulation and encryption
"""

from .doc_translation import translate_multilingual
from .embellishment import embellish
from .encryption import password_protect
from .file_conversion import convert
from .metadata_extraction import extract_metadata
from .sanitization import sanitize
from .semantics_extraction import extract_semantics

__all__ = [
    archive_and_tabulte,
    encrypt_and_protect,
    extract_artifact_data,
    manual_review,
    transform,
    manual_backup,
]

__version__ = "1.2.0"
