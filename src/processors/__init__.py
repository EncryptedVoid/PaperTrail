"""
Pipelines package for PaperTrail document processing.

Provides pipeline classes for each stage of document processing:
- Sanitization and duplicate detection
- Metadata extraction
- Semantic analysis and field extraction
- Database tabulation and encryption
"""

from .database_pipeline import tabulate
from .file_conversion import convert
from .metadata_extraction import extract_metadata
from .sanitization import sanitize
from .semantics_metadata_extraction import extract_semantics

__all__ = [
    convert,
    extract_metadata,
    extract_semantics,
    sanitize,
    tabulate,
]

__version__ = "1.2.0"
