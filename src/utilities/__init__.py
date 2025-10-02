"""
Utilities package for PaperTrail.

Provides core utility classes and functions:
- Language model processing (LLM field extraction)
- Visual model processing (OCR and image analysis)
- File type conversion and enhancement
- Common utility functions
"""

from .audio_model import AudioProcessingStats, AudioProcessor, AudioQualityMode
from .compile_subsets import compile_doc_subset, compile_video_snapshot_subset
from .dependancy_ensurance import (
    ensure_apache_tika,
    ensure_ffmpeg,
    ensure_imagemagick,
    ensure_java,
    ensure_libpff_python,
    ensure_ollama,
    ensure_pandoc,
    ensure_par2,
    ensure_pdfarranger,
)
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
    LanguageProcessor,
    LanguageExtractionReport,
    VisualProcessor,
    AudioProcessor,
    AudioProcessingStats,
    AudioQualityMode,
    ProcessingMode,
    HardwareConstraints,
    ProcessingStats,
    compile_doc_subset,
    compile_video_snapshot_subset,
    ensure_ffmpeg,
    ensure_imagemagick,
    ensure_pandoc,
    ensure_par2,
    ensure_libpff_python,
    ensure_java,
    ensure_apache_tika,
    ensure_pdfarranger,
    ensure_ollama,
]

__version__ = "1.2.0"
