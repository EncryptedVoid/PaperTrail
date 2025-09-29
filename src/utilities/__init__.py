"""
Utilities package for PaperTrail.

Provides core utility classes and functions:
- Language model processing (LLM field extraction)
- Visual model processing (OCR and image analysis)
- File type conversion and enhancement
- Common utility functions
"""

from .language_model import LanguageExtractionReport, LanguageProcessor
from .visual_model import (
    HardwareConstraints,
    ProcessingMode,
    ProcessingStats,
    VisualProcessor,
)

# Note: common_utils contains functions but isn't a class-based module
# Import specific functions as needed in other modules directly

__all__ = [
    # Language processing
    "LanguageProcessor",
    "LanguageExtractionReport",
    # Visual processing
    "VisualProcessor",
    "ProcessingMode",
    "HardwareConstraints",
    "ProcessingStats",
]

__version__ = "1.2.0"
