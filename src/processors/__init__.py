"""
Pipelines package for PaperTrail document processing.

Provides pipeline classes for each stage of document processing:
- Sanitization and duplicate detection
- Metadata extraction
- Semantic analysis and field extraction
- Database tabulation and encryption
"""

from .file_conversion import convert
from .metadata_extraction import extract_metadata
from .sanitization import sanitize
from .semantics_extraction import extract_semantics
from .tabulation import tabulate

__all__ = [
    convert,
    extract_metadata,
    extract_semantics,
    sanitize,
    tabulate,
]

__version__ = "1.2.0"
