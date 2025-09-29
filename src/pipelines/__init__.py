"""
Pipelines package for PaperTrail document processing.

Provides pipeline classes for each stage of document processing:
- Sanitization and duplicate detection
- Metadata extraction
- Semantic analysis and field extraction
- Database tabulation and encryption
"""

from .conversion_pipeline import ConversionPipeline
from .database_pipeline import DatabasePipeline, TabulationReport
from .metadata_pipeline import MetadataPipeline, MetadataReport
from .sanitizer_pipeline import SanitizationReport, SanitizerPipeline
from .semantics_pipeline import SemanticExtractionReport, SemanticsPipeline

__all__ = [
    # Sanitization
    "SanitizerPipeline",
    "SanitizationReport",
    # Metadata
    "MetadataPipeline",
    "MetadataReport",
    # Semantics
    "SemanticsPipeline",
    "SemanticExtractionReport",
    # Database
    "DatabasePipeline",
    "TabulationReport",
    # File conversion
    "ConversionPipeline",
]

__version__ = "1.2.0"
