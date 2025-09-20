import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz
from PIL import Image
import io
import logging
import gc
import psutil
import time
from enum import Enum
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from config import (
    DEFAULT_REFRESH_INTERVAL,
    DEFAULT_MEMORY_THRESHOLD,
    DEFAULT_AUTO_MODEL_SELECTION,
    DEFAULT_PROCESSING_MODE,
    RAM_USAGE_RATIO,
    VRAM_USAGE_RATIO,
    ZOOM_FACTOR_FAST,
    ZOOM_FACTOR_BALANCED,
    ZOOM_FACTOR_HIGH_QUALITY,
    MAX_PROCESSING_TIME_PER_DOC,
    MIN_TEXT_SUCCESS_RATE,
    MIN_DESCRIPTION_SUCCESS_RATE,
    MAX_AVERAGE_PROCESSING_TIME,
    MAX_GPU_MEMORY_PERCENT,
    QWEN2VL_2B_MIN_VRAM,
    QWEN2VL_2B_MIN_RAM,
    QWEN2VL_2B_QUALITY,
    QWEN2VL_2B_MAX_TOKENS,
    QWEN2VL_7B_MIN_VRAM,
    QWEN2VL_7B_MIN_RAM,
    QWEN2VL_7B_QUALITY,
    QWEN2VL_7B_MAX_TOKENS,
    QWEN2VL_72B_MIN_VRAM,
    QWEN2VL_72B_MIN_RAM,
    QWEN2VL_72B_QUALITY,
    QWEN2VL_72B_MAX_TOKENS,
    QWEN2VL_7B_CPU_MIN_VRAM,
    QWEN2VL_7B_CPU_MIN_RAM,
    QWEN2VL_7B_CPU_QUALITY,
    QWEN2VL_7B_CPU_MAX_TOKENS,
)


class ProcessingMode(Enum):
    """
    Processing quality modes that determine the balance between speed and quality.

    FAST: Optimized for speed with lower resolution and simpler processing
    BALANCED: Good balance between speed and quality (default)
    HIGH_QUALITY: Maximum quality with higher resolution and detailed processing
    """

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass
class HardwareConstraints:
    """
    Hardware resource constraints for model selection and processing.

    Attributes:
        max_gpu_vram_gb: Maximum available GPU VRAM in gigabytes
        max_ram_gb: Maximum available system RAM in gigabytes
        force_cpu: Whether to force CPU-only processing
    """

    max_gpu_vram_gb: float
    max_ram_gb: float
    force_cpu: bool = False


@dataclass
class ProcessingStats:
    """
    Comprehensive tracking of processing performance and quality metrics.

    This class maintains statistics across all document processing operations
    to enable performance monitoring and optimization decisions.
    """

    documents_processed: int = 0
    pages_processed: int = 0
    text_extractions_successful: int = 0
    descriptions_successful: int = 0
    total_processing_time: float = 0.0
    memory_refreshes: int = 0
    model_switches: int = 0
    average_text_length: float = 0.0
    average_description_length: float = 0.0
    last_refresh_time: float = 0.0


class VisualProcessor:
    """
    Enhanced Visual Processor with hardware optimization and intelligent resource management.

    This processor handles document analysis using vision-language models with automatic
    hardware detection, memory management, and performance optimization. It supports
    both PDF and image file processing with configurable quality modes.

    Key Features:
    - Automatic hardware detection and model selection
    - Memory usage monitoring and automatic model refresh
    - Configurable processing modes (fast, balanced, high-quality)
    - Comprehensive performance tracking and statistics
    - Robust error handling and fallback mechanisms
    """

    def __init__(
        self,
        logger: logging.Logger,
        max_gpu_vram_gb: Optional[float] = None,
        max_ram_gb: Optional[float] = None,
        force_cpu: bool = False,
        processing_mode: Optional[ProcessingMode] = None,
    ) -> None:
        """
        Initialize the Visual Processor with automatic configuration from constants.

        Args:
            logger: Logger instance for tracking operations and debugging
            max_gpu_vram_gb: Override for maximum GPU VRAM (auto-detected if None)
            max_ram_gb: Override for maximum RAM (auto-detected if None)
            force_cpu: Force CPU-only processing regardless of GPU availability
            processing_mode: Processing quality mode (uses DEFAULT_PROCESSING_MODE if None)

        Raises:
            ValueError: If logger is None
            TypeError: If logger is not a logging.Logger instance
            RuntimeError: If no suitable model can be loaded
        """
        # Validate required logger parameter
        if logger is None:
            raise ValueError(
                "Logger is required - VisualProcessor cannot be initialized without a logger"
            )
        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger instance, got {type(logger)}")

        self.logger = logger
        self.logger.info("Initializing VisualProcessor with configuration constants")

        # Apply configuration constants for initialization parameters
        self.processing_mode = processing_mode or ProcessingMode(
            DEFAULT_PROCESSING_MODE
        )
        self.refresh_interval = DEFAULT_REFRESH_INTERVAL
        self.memory_threshold = DEFAULT_MEMORY_THRESHOLD
        self.auto_model_selection = DEFAULT_AUTO_MODEL_SELECTION

        self.logger.info(f"Processing mode set to: {self.processing_mode.value}")
        self.logger.info(f"Auto model selection: {self.auto_model_selection}")
        self.logger.info(f"Memory threshold: {self.memory_threshold}%")
        self.logger.info(f"Refresh interval: {self.refresh_interval} documents")

        # Initialize hardware constraints using configuration ratios
        self.hardware_constraints = self._detect_hardware_constraints(
            max_gpu_vram_gb, max_ram_gb, force_cpu
        )
        self.logger.info(f"Hardware constraints detected: {self.hardware_constraints}")

        # Initialize processing statistics
        self.stats = ProcessingStats()
        self.logger.debug("Processing statistics initialized")

        # Model state variables
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.device = None
        self.max_tokens = QWEN2VL_7B_MAX_TOKENS  # Default to 7B model tokens

        # Select and load optimal model based on hardware constraints
        self._select_and_load_optimal_model()

    def _detect_hardware_constraints(
        self,
        max_gpu_vram_gb: Optional[float],
        max_ram_gb: Optional[float],
        force_cpu: bool,
    ) -> HardwareConstraints:
        """
        Detect available hardware resources or use provided constraints.

        Uses configuration constants RAM_USAGE_RATIO and VRAM_USAGE_RATIO to determine
        safe memory limits for model loading and processing.

        Args:
            max_gpu_vram_gb: Optional override for GPU VRAM limit
            max_ram_gb: Optional override for RAM limit
            force_cpu: Whether to force CPU-only processing

        Returns:
            HardwareConstraints object with detected or provided limits
        """
        self.logger.info("Detecting hardware constraints...")

        # RAM detection using configuration ratio
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            available_ram = total_ram * RAM_USAGE_RATIO  # Use configured ratio
            self.logger.info(
                f"Auto-detected RAM: {total_ram:.1f}GB total, using {available_ram:.1f}GB ({RAM_USAGE_RATIO*100}%)"
            )
        else:
            available_ram = max_ram_gb
            self.logger.info(f"Using provided RAM limit: {available_ram:.1f}GB")

        # GPU detection using configuration ratio
        if max_gpu_vram_gb is None and not force_cpu:
            try:
                if torch.cuda.is_available():
                    # Get GPU memory from torch
                    gpu_props = torch.cuda.get_device_properties(0)
                    total_vram = gpu_props.total_memory / (1024**3)  # Convert to GB
                    available_vram = (
                        total_vram * VRAM_USAGE_RATIO
                    )  # Use configured ratio
                    gpu_name = torch.cuda.get_device_name(0)
                    self.logger.info(
                        f"Auto-detected GPU: {gpu_name} with {total_vram:.1f}GB VRAM, using {available_vram:.1f}GB ({VRAM_USAGE_RATIO*100}%)"
                    )
                else:
                    available_vram = 0
                    self.logger.warning(
                        "CUDA not available, falling back to CPU processing"
                    )
            except Exception as e:
                self.logger.warning(f"Could not detect GPU VRAM: {e}")
                available_vram = 0
        else:
            available_vram = max_gpu_vram_gb or 0
            if max_gpu_vram_gb:
                self.logger.info(
                    f"Using provided GPU VRAM limit: {available_vram:.1f}GB"
                )

        if force_cpu:
            available_vram = 0
            self.logger.info("CPU processing forced, ignoring GPU")

        return HardwareConstraints(
            max_gpu_vram_gb=available_vram,
            max_ram_gb=available_ram,
            force_cpu=force_cpu,
        )

    def _select_and_load_optimal_model(self) -> None:
        """
        Select and load the optimal model based on hardware constraints and configuration.

        Uses configuration constants for model requirements and automatically selects
        the best model that fits within available hardware resources.
        """
        self.logger.info("Selecting optimal model based on hardware constraints...")

        # Define model options with configuration constants
        model_options = [
            {
                "name": "Qwen2-VL-72B",
                "model_id": "Qwen/Qwen2-VL-72B-Instruct",
                "min_vram": QWEN2VL_72B_MIN_VRAM,
                "min_ram": QWEN2VL_72B_MIN_RAM,
                "quality": QWEN2VL_72B_QUALITY,
                "max_tokens": QWEN2VL_72B_MAX_TOKENS,
            },
            {
                "name": "Qwen2-VL-7B",
                "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "min_vram": QWEN2VL_7B_MIN_VRAM,
                "min_ram": QWEN2VL_7B_MIN_RAM,
                "quality": QWEN2VL_7B_QUALITY,
                "max_tokens": QWEN2VL_7B_MAX_TOKENS,
            },
            {
                "name": "Qwen2-VL-2B",
                "model_id": "Qwen/Qwen2-VL-2B-Instruct",
                "min_vram": QWEN2VL_2B_MIN_VRAM,
                "min_ram": QWEN2VL_2B_MIN_RAM,
                "quality": QWEN2VL_2B_QUALITY,
                "max_tokens": QWEN2VL_2B_MAX_TOKENS,
            },
            {
                "name": "Qwen2-VL-7B-CPU",
                "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "min_vram": QWEN2VL_7B_CPU_MIN_VRAM,  # 0.0 for CPU
                "min_ram": QWEN2VL_7B_CPU_MIN_RAM,
                "quality": QWEN2VL_7B_CPU_QUALITY,
                "max_tokens": QWEN2VL_7B_CPU_MAX_TOKENS,
            },
        ]

        # Filter models that fit within hardware constraints
        suitable_models = []
        for model in model_options:
            fits_vram = model["min_vram"] <= self.hardware_constraints.max_gpu_vram_gb
            fits_ram = model["min_ram"] <= self.hardware_constraints.max_ram_gb

            if self.hardware_constraints.force_cpu:
                # Only CPU models when forced
                if model["min_vram"] == 0:
                    suitable_models.append(model)
                    self.logger.debug(f"CPU model fits constraints: {model['name']}")
            else:
                if fits_vram and fits_ram:
                    suitable_models.append(model)
                    self.logger.debug(
                        f"Model fits constraints: {model['name']} (VRAM: {model['min_vram']}GB, RAM: {model['min_ram']}GB)"
                    )

        if not suitable_models:
            self.logger.error(
                "No suitable models found for current hardware constraints"
            )
            # Emergency fallback to CPU model with minimal requirements
            fallback_model = {
                "name": "Qwen2-VL-7B-CPU-Fallback",
                "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "min_vram": 0.0,
                "min_ram": 8.0,  # Reduced RAM requirement for emergency fallback
                "quality": QWEN2VL_7B_CPU_QUALITY,
                "max_tokens": QWEN2VL_7B_CPU_MAX_TOKENS,
            }
            suitable_models = [fallback_model]
            self.logger.warning(
                "Using emergency fallback model with reduced RAM requirements"
            )

        # Select model with highest quality score that fits
        selected_model = max(suitable_models, key=lambda m: m["quality"])
        self.current_model_name = selected_model["name"]
        self.max_tokens = selected_model["max_tokens"]

        self.logger.info(
            f"Selected optimal model: {selected_model['name']} (quality: {selected_model['quality']}/10)"
        )
        self.logger.info(
            f"Model requirements: VRAM {selected_model['min_vram']}GB, RAM {selected_model['min_ram']}GB"
        )
        self.logger.info(f"Max tokens per generation: {self.max_tokens}")

        # Load the selected model
        self._load_model(selected_model["model_id"])

    def _load_model(self, model_id: str) -> None:
        """
        Load or reload the vision model and processor with comprehensive error handling.

        Args:
            model_id: HuggingFace model identifier to load

        Raises:
            RuntimeError: If model loading fails after all fallback attempts
        """
        self.logger.info(f"Loading vision model: {model_id}")
        loading_start_time = time.time()

        try:
            # Clean up existing model if present
            if self.model is not None:
                self.logger.debug("Cleaning up existing model...")
                del self.model
                del self.processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug("GPU memory cache cleared")
                gc.collect()
                self.logger.debug("Garbage collection completed")

            # Determine compute device based on hardware constraints
            if (
                self.hardware_constraints.force_cpu
                or self.hardware_constraints.max_gpu_vram_gb == 0
            ):
                self.device = "cpu"
                self.logger.info("Using CPU for processing")
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.logger.info(f"Using device: {self.device}")

            # Log detailed hardware information
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                compute_capability = torch.cuda.get_device_properties(0).major
                self.logger.info(
                    f"GPU Details: {gpu_name} ({gpu_memory:.1f}GB, Compute {compute_capability}.x)"
                )

            # Configure model loading parameters based on device
            model_kwargs = {
                "torch_dtype": (
                    torch.bfloat16 if self.device == "cuda" else torch.float32
                ),
                "device_map": "auto" if self.device == "cuda" else {"": "cpu"},
            }

            self.logger.debug(f"Model loading kwargs: {model_kwargs}")

            # Load model with progress logging
            self.logger.info("Loading model from transformers library...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, **model_kwargs
            )
            self.logger.info("Model loaded successfully")

            # Load processor
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.logger.info("Processor loaded successfully")

            loading_time = time.time() - loading_start_time
            self.logger.info(f"Model loading completed in {loading_time:.2f} seconds")

            # Log memory usage after loading
            memory_info = self._monitor_memory_usage()
            if self.device == "cuda":
                self.logger.info(
                    f"GPU memory usage after loading: {memory_info.get('gpu_memory_percent', 0):.1f}%"
                )
            self.logger.info(
                f"System RAM usage after loading: {memory_info.get('system_ram_percent', 0):.1f}%"
            )

            self.stats.last_refresh_time = time.time()

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            self.logger.error("Model loading failed, attempting fallback strategies...")
            self._try_fallback_model()

    def _try_fallback_model(self) -> None:
        """
        Attempt to load a fallback model if primary model fails.

        Tries progressively smaller models and eventually forces CPU mode
        if all GPU attempts fail.

        Raises:
            RuntimeError: If all fallback models fail to load
        """
        self.logger.warning("Attempting to load fallback model...")

        # Try CPU mode if not already forced
        if not self.hardware_constraints.force_cpu:
            self.logger.info("Attempting CPU fallback...")
            try:
                # Force CPU mode temporarily
                original_force_cpu = self.hardware_constraints.force_cpu
                self.hardware_constraints.force_cpu = True

                # Try loading with reduced memory requirements
                self._load_model("Qwen/Qwen2-VL-7B-Instruct")
                self.current_model_name = "Qwen2-VL-7B-CPU-Fallback"
                self.max_tokens = QWEN2VL_7B_CPU_MAX_TOKENS

                self.stats.model_switches += 1
                self.logger.info("Successfully loaded CPU fallback model")
                return

            except Exception as e:
                self.logger.error(f"CPU fallback also failed: {e}")
                # Restore original setting
                self.hardware_constraints.force_cpu = original_force_cpu

        # If we reach here, all fallback attempts failed
        self.logger.critical("All fallback models failed to load")
        raise RuntimeError(
            "No models could be loaded - check hardware requirements and model availability"
        )

    def _should_refresh_model(self) -> bool:
        """
        Determine if model should be refreshed based on usage patterns and performance metrics.

        Uses configuration constants to determine refresh triggers:
        - Document processing count (DEFAULT_REFRESH_INTERVAL)
        - Memory usage thresholds (DEFAULT_MEMORY_THRESHOLD, MAX_GPU_MEMORY_PERCENT)
        - Processing time degradation (MAX_AVERAGE_PROCESSING_TIME)

        Returns:
            bool: True if model should be refreshed
        """
        current_time = time.time()

        # Check document count trigger using configuration constant
        if self.stats.documents_processed >= self.refresh_interval:
            self.logger.info(
                f"Refresh triggered: processed {self.stats.documents_processed} documents (limit: {self.refresh_interval})"
            )
            return True

        # Check GPU memory if available, using configuration constants
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                memory_percent = (memory_used / memory_total) * 100

                # Use both default threshold and max GPU memory constant
                if memory_percent > min(self.memory_threshold, MAX_GPU_MEMORY_PERCENT):
                    self.logger.warning(
                        f"Refresh triggered: GPU memory usage {memory_percent:.1f}% exceeds threshold {min(self.memory_threshold, MAX_GPU_MEMORY_PERCENT):.1f}%"
                    )
                    return True

                self.logger.debug(
                    f"GPU memory usage: {memory_percent:.1f}% (threshold: {min(self.memory_threshold, MAX_GPU_MEMORY_PERCENT):.1f}%)"
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not check GPU memory for refresh decision: {e}"
                )

        # Check processing time degradation using configuration constant
        if self.stats.documents_processed > 0:
            avg_time_per_doc = (
                self.stats.total_processing_time / self.stats.documents_processed
            )
            if avg_time_per_doc > MAX_AVERAGE_PROCESSING_TIME:
                self.logger.warning(
                    f"Refresh triggered: slow processing ({avg_time_per_doc:.1f}s per doc, limit: {MAX_AVERAGE_PROCESSING_TIME}s)"
                )
                return True

            self.logger.debug(
                f"Average processing time: {avg_time_per_doc:.1f}s per document"
            )

        # Check overall success rates using configuration constants
        if self.stats.pages_processed > 0:
            text_success_rate = (
                self.stats.text_extractions_successful / self.stats.pages_processed
            )
            desc_success_rate = (
                self.stats.descriptions_successful / self.stats.pages_processed
            )

            if text_success_rate < MIN_TEXT_SUCCESS_RATE:
                self.logger.warning(
                    f"Refresh triggered: low text extraction success rate ({text_success_rate:.2f}, minimum: {MIN_TEXT_SUCCESS_RATE})"
                )
                return True

            if desc_success_rate < MIN_DESCRIPTION_SUCCESS_RATE:
                self.logger.warning(
                    f"Refresh triggered: low description success rate ({desc_success_rate:.2f}, minimum: {MIN_DESCRIPTION_SUCCESS_RATE})"
                )
                return True

        return False

    def refresh_model(self) -> None:
        """
        Refresh the model to clear context and free memory.

        This operation reloads the current model to clear accumulated memory usage
        and reset the processing context. Useful for long-running processes.
        """
        self.logger.info("Refreshing vision model to clear memory and context...")
        refresh_start_time = time.time()

        try:
            # Store current model info for reloading
            current_model_id = "Qwen/Qwen2-VL-7B-Instruct"  # Default fallback
            if hasattr(self, "current_model_name"):
                # Map current model name back to model ID
                model_mapping = {
                    "Qwen2-VL-72B": "Qwen/Qwen2-VL-72B-Instruct",
                    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
                    "Qwen2-VL-2B": "Qwen/Qwen2-VL-2B-Instruct",
                    "Qwen2-VL-7B-CPU": "Qwen/Qwen2-VL-7B-Instruct",
                    "Qwen2-VL-7B-CPU-Fallback": "Qwen/Qwen2-VL-7B-Instruct",
                }
                current_model_id = model_mapping.get(
                    self.current_model_name, current_model_id
                )

            # Reload the model
            self._load_model(current_model_id)

            # Update statistics
            self.stats.memory_refreshes += 1
            processing_count = self.stats.documents_processed
            self.stats.documents_processed = 0  # Reset document counter

            refresh_time = time.time() - refresh_start_time
            self.logger.info(
                f"Model refresh completed in {refresh_time:.2f}s (total processed before refresh: {processing_count})"
            )

        except Exception as e:
            self.logger.error(f"Failed to refresh model: {e}")
            raise

    def _monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage across system RAM and GPU VRAM.

        Provides detailed memory monitoring for performance optimization
        and threshold checking.

        Returns:
            Dict containing memory usage metrics in GB and percentages
        """
        memory_info = {}

        # System RAM monitoring
        try:
            ram = psutil.virtual_memory()
            memory_info["system_ram_used_gb"] = (ram.total - ram.available) / (1024**3)
            memory_info["system_ram_total_gb"] = ram.total / (1024**3)
            memory_info["system_ram_percent"] = ram.percent
            memory_info["system_ram_available_gb"] = ram.available / (1024**3)

            self.logger.debug(
                f"System RAM: {memory_info['system_ram_used_gb']:.1f}GB used / {memory_info['system_ram_total_gb']:.1f}GB total ({memory_info['system_ram_percent']:.1f}%)"
            )
        except Exception as e:
            self.logger.warning(f"Could not get system RAM info: {e}")

        # GPU memory monitoring if available
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (
                    1024**3
                )
                memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (
                    1024**3
                )
                memory_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
                memory_info["gpu_memory_percent"] = (
                    memory_info["gpu_memory_used_gb"]
                    / memory_info["gpu_memory_total_gb"]
                ) * 100
                memory_info["gpu_memory_free_gb"] = (
                    memory_info["gpu_memory_total_gb"]
                    - memory_info["gpu_memory_used_gb"]
                )

                self.logger.debug(
                    f"GPU Memory: {memory_info['gpu_memory_used_gb']:.1f}GB used / {memory_info['gpu_memory_total_gb']:.1f}GB total ({memory_info['gpu_memory_percent']:.1f}%)"
                )
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory info: {e}")

        return memory_info

    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for processing results.

        Analyzes the quality of text extraction and image description results
        to provide feedback for optimization and performance monitoring.

        Args:
            result: Processing result dictionary containing text and description

        Returns:
            Dict containing various quality metrics and scores
        """
        metrics = {}

        # Text extraction quality analysis
        text = result.get("text", "")
        if text and text not in ["NO_TEXT_FOUND", "TEXT_EXTRACTION_FAILED"]:
            metrics["text_extraction_success"] = 1.0
            metrics["text_length"] = len(text)

            # Calculate text quality heuristics
            words = text.split()
            unique_words = set(word.lower() for word in words)
            metrics["text_word_count"] = len(words)
            metrics["text_unique_word_count"] = len(unique_words)
            metrics["text_diversity"] = len(unique_words) / max(len(words), 1)

            # Check for common text quality indicators
            has_punctuation = any(char in text for char in ".,!?;:")
            has_numbers = any(char.isdigit() for char in text)
            has_uppercase = any(char.isupper() for char in text)

            metrics["text_has_punctuation"] = float(has_punctuation)
            metrics["text_has_numbers"] = float(has_numbers)
            metrics["text_has_uppercase"] = float(has_uppercase)

            self.logger.debug(
                f"Text quality: {len(words)} words, {len(unique_words)} unique, diversity: {metrics['text_diversity']:.2f}"
            )
        else:
            metrics["text_extraction_success"] = 0.0
            metrics["text_length"] = 0
            metrics["text_word_count"] = 0
            metrics["text_unique_word_count"] = 0
            metrics["text_diversity"] = 0.0
            metrics["text_has_punctuation"] = 0.0
            metrics["text_has_numbers"] = 0.0
            metrics["text_has_uppercase"] = 0.0

        # Image description quality analysis
        description = result.get("description", "")
        if description and description != "DESCRIPTION_GENERATION_FAILED":
            metrics["description_success"] = 1.0
            metrics["description_length"] = len(description)

            # Check for descriptive quality indicators using configuration-aware keywords
            descriptive_keywords = [
                "color",
                "layout",
                "text",
                "image",
                "diagram",
                "chart",
                "table",
                "document",
                "page",
                "content",
                "visual",
                "format",
                "style",
                "background",
            ]
            keyword_count = sum(
                1 for keyword in descriptive_keywords if keyword in description.lower()
            )
            metrics["description_keyword_count"] = keyword_count
            metrics["description_richness"] = keyword_count / len(descriptive_keywords)

            # Additional description quality metrics
            desc_words = description.split()
            metrics["description_word_count"] = len(desc_words)
            metrics["description_sentence_count"] = (
                description.count(".") + description.count("!") + description.count("?")
            )

            self.logger.debug(
                f"Description quality: {len(desc_words)} words, {keyword_count}/{len(descriptive_keywords)} keywords, richness: {metrics['description_richness']:.2f}"
            )
        else:
            metrics["description_success"] = 0.0
            metrics["description_length"] = 0
            metrics["description_keyword_count"] = 0
            metrics["description_richness"] = 0.0
            metrics["description_word_count"] = 0
            metrics["description_sentence_count"] = 0

        return metrics

    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Image objects with configurable resolution.

        Uses configuration constants for zoom factors based on processing mode
        to balance quality and processing speed.

        Args:
            pdf_path: Path to the PDF file to convert

        Returns:
            List of PIL Image objects, one per PDF page

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF conversion fails
        """
        pdf_path_obj = Path(pdf_path)
        self.logger.info(f"Converting PDF to images: {pdf_path_obj}")

        if not pdf_path_obj.exists():
            self.logger.error(f"PDF file not found: {pdf_path_obj}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        try:
            doc = fitz.open(str(pdf_path_obj))
            images = []
            page_count = len(doc)

            # Use configuration constants for zoom factors based on processing mode
            zoom_factors = {
                ProcessingMode.FAST: ZOOM_FACTOR_FAST,
                ProcessingMode.BALANCED: ZOOM_FACTOR_BALANCED,
                ProcessingMode.HIGH_QUALITY: ZOOM_FACTOR_HIGH_QUALITY,
            }
            zoom = zoom_factors.get(self.processing_mode, ZOOM_FACTOR_BALANCED)

            self.logger.info(
                f"PDF conversion settings: {page_count} pages, {zoom}x zoom ({self.processing_mode.value} mode)"
            )

            for page_num in range(page_count):
                try:
                    page_start_time = time.time()
                    page = doc[page_num]

                    # Create transformation matrix with configured zoom
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)

                    page_time = time.time() - page_start_time
                    self.logger.debug(
                        f"Page {page_num + 1}/{page_count} converted in {page_time:.2f}s - size: {img.size}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    continue

            doc.close()
            conversion_success_rate = len(images) / page_count if page_count > 0 else 0
            self.logger.info(
                f"PDF conversion completed: {len(images)}/{page_count} pages successful ({conversion_success_rate:.1%})"
            )

            if len(images) == 0:
                raise Exception("No pages could be converted from PDF")

            return images

        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def _process_single_image(
        self,
        image: Union[str, Path, Image.Image],
        extract_text: bool = True,
        describe_image: bool = True,
    ) -> Dict[str, Any]:
        """
        Process individual image with comprehensive error handling and quality monitoring.

        Performs both text extraction and image description using the loaded vision model,
        with configurable maximum token limits from configuration constants.

        Args:
            image: Image to process (file path or PIL Image object)
            extract_text: Whether to extract text from the image
            describe_image: Whether to generate image description

        Returns:
            Dict containing processing results, timing, and quality metrics

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image processing fails completely
        """
        start_time = time.time()
        self.logger.debug("Starting single image processing...")

        try:
            # Load and validate image
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    self.logger.error(f"Image file not found: {image_path}")
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                processed_image = Image.open(image_path)
                self.logger.debug(
                    f"Loaded image from file: {image_path} (size: {processed_image.size})"
                )
            else:
                processed_image = image
                self.logger.debug(
                    f"Using provided PIL image (size: {processed_image.size})"
                )

            results = {}

            # Text extraction with detailed logging and error handling
            if extract_text:
                self.logger.debug("Starting text extraction...")
                text_start_time = time.time()

                try:
                    # Prepare text extraction prompt
                    text_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image. Return only the text content, no explanations. If there's no text, return 'NO_TEXT_FOUND'.",
                                },
                            ],
                        }
                    ]

                    # Apply chat template and process vision inputs
                    text_inputs = self.processor.apply_chat_template(
                        text_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(text_messages)

                    # Tokenize inputs
                    inputs = self.processor(
                        text=[text_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)

                    # Generate text with configured max tokens
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,  # Use configured max tokens
                            do_sample=False,  # Deterministic for consistency
                            temperature=0.1,  # Low temperature for factual extraction
                        )

                        # Extract generated tokens
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        # Decode to text
                        text_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["text"] = text_output.strip()

                    text_time = time.time() - text_start_time
                    self.logger.debug(
                        f"Text extraction completed in {text_time:.2f}s, output length: {len(results['text'])} chars"
                    )

                except Exception as e:
                    self.logger.error(f"Text extraction failed: {e}")
                    results["text"] = "TEXT_EXTRACTION_FAILED"

            # Image description with detailed logging and error handling
            if describe_image:
                self.logger.debug("Starting image description generation...")
                desc_start_time = time.time()

                try:
                    # Prepare description prompt
                    desc_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {
                                    "type": "text",
                                    "text": "Describe all visual elements in this image in detail. Include: layout, colors, objects, people, text formatting, charts/graphs, diagrams, symbols, and any other visual content. Be comprehensive but concise.",
                                },
                            ],
                        }
                    ]

                    # Apply chat template and process vision inputs
                    desc_inputs = self.processor.apply_chat_template(
                        desc_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(desc_messages)

                    # Tokenize inputs
                    inputs = self.processor(
                        text=[desc_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)

                    # Generate description with configured max tokens
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,  # Use configured max tokens
                            do_sample=False,  # Deterministic for consistency
                            temperature=0.3,  # Slightly higher temperature for creative description
                        )

                        # Extract generated tokens
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        # Decode to text
                        desc_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["description"] = desc_output.strip()

                    desc_time = time.time() - desc_start_time
                    self.logger.debug(
                        f"Description generation completed in {desc_time:.2f}s, output length: {len(results['description'])} chars"
                    )

                except Exception as e:
                    self.logger.error(f"Image description failed: {e}")
                    results["description"] = "DESCRIPTION_GENERATION_FAILED"

            # Calculate processing metrics
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["quality_metrics"] = self._calculate_quality_metrics(results)

            self.logger.debug(
                f"Single image processing completed in {processing_time:.2f}s"
            )
            return results

        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            raise

    def extract_article_semantics(self, document: Union[str, Path]) -> Dict[str, str]:
        """
        Process document with enhanced monitoring and performance tracking.

        Main entry point for document processing that handles both PDFs and images
        with comprehensive error handling, performance monitoring, and automatic
        model refresh based on configuration thresholds.

        Args:
            document: Path to document file (PDF or image)

        Returns:
            Dict containing extracted text and descriptions

        Raises:
            FileNotFoundError: If document file doesn't exist
            ValueError: If file type is unsupported
            Exception: If processing fails
        """
        document_obj = Path(document)
        self.logger.info(f"Starting document processing: {document_obj.name}")
        self.logger.info(
            f"Document size: {document_obj.stat().st_size / 1024 / 1024:.1f} MB"
        )

        start_time = time.time()

        # Check if model should be refreshed based on configuration thresholds
        if self._should_refresh_model():
            self.logger.info("Auto-refreshing model due to configuration thresholds")
            self.refresh_model()

        # Monitor memory before processing
        memory_before = self._monitor_memory_usage()
        self.logger.info(
            f"Memory before processing - GPU: {memory_before.get('gpu_memory_percent', 0):.1f}%, RAM: {memory_before.get('system_ram_percent', 0):.1f}%"
        )

        # Validate document existence
        if not document_obj.exists():
            self.logger.error(f"Document file not found: {document_obj}")
            raise FileNotFoundError(f"Document file not found: {document_obj}")

        # Determine file type and log processing approach
        file_ext = document_obj.suffix.lower()
        self.logger.info(f"Document type: {file_ext}")

        # Initialize processing containers
        all_text = []
        all_descriptions = []
        page_quality_metrics = []
        processing_errors = []

        try:
            if file_ext == ".pdf":
                # PDF processing with detailed progress tracking
                self.logger.info("Processing PDF document...")
                images = self._pdf_to_images(document_obj)
                total_pages = len(images)
                self.logger.info(
                    f"PDF converted to {total_pages} images, beginning page-by-page processing"
                )

                for i, img in enumerate(images):
                    page_num = i + 1
                    page_start_time = time.time()
                    self.logger.debug(f"Processing page {page_num}/{total_pages}")

                    try:
                        # Process individual page with timeout consideration
                        result = self._process_single_image(img)
                        page_quality_metrics.append(result.get("quality_metrics", {}))

                        # Handle text extraction results
                        if result.get("text") and result["text"] not in [
                            "NO_TEXT_FOUND",
                            "TEXT_EXTRACTION_FAILED",
                        ]:
                            page_text = f"=== Page {page_num} ===\n{result['text']}"
                            all_text.append(page_text)
                            self.stats.text_extractions_successful += 1
                            self.logger.debug(
                                f"Page {page_num}: Text extracted ({len(result['text'])} chars)"
                            )
                        else:
                            self.logger.debug(
                                f"Page {page_num}: No text found or extraction failed"
                            )

                        # Handle description results
                        if (
                            result.get("description")
                            and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                        ):
                            page_desc = f"=== Page {page_num} Visual Description ===\n{result['description']}"
                            all_descriptions.append(page_desc)
                            self.stats.descriptions_successful += 1
                            self.logger.debug(
                                f"Page {page_num}: Description generated ({len(result['description'])} chars)"
                            )
                        else:
                            self.logger.debug(
                                f"Page {page_num}: Description generation failed"
                            )

                        self.stats.pages_processed += 1
                        page_time = time.time() - page_start_time

                        # Check for processing time issues using configuration constant
                        if (
                            page_time > MAX_PROCESSING_TIME_PER_DOC / 10
                        ):  # Scale down for per-page
                            self.logger.warning(
                                f"Page {page_num} took {page_time:.2f}s to process (slower than expected)"
                            )

                    except Exception as e:
                        error_msg = f"Failed to process page {page_num}: {e}"
                        self.logger.error(error_msg)
                        processing_errors.append(error_msg)
                        continue

            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                # Image file processing
                self.logger.info(f"Processing image file: {document_obj.name}")

                result = self._process_single_image(document_obj)
                page_quality_metrics.append(result.get("quality_metrics", {}))

                # Handle text extraction results
                if result.get("text") and result["text"] not in [
                    "NO_TEXT_FOUND",
                    "TEXT_EXTRACTION_FAILED",
                ]:
                    all_text.append(result["text"])
                    self.stats.text_extractions_successful += 1
                    self.logger.info(
                        f"Text extracted from image ({len(result['text'])} chars)"
                    )

                # Handle description results
                if (
                    result.get("description")
                    and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                ):
                    all_descriptions.append(result["description"])
                    self.stats.descriptions_successful += 1
                    self.logger.info(
                        f"Description generated for image ({len(result['description'])} chars)"
                    )

                self.stats.pages_processed += 1

            else:
                # Unsupported file type
                error_msg = f"Unsupported file type: {file_ext}. Supported types: .pdf, .jpg, .jpeg, .png, .bmp, .tiff, .webp"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Compile final results
            final_text = (
                "\n\n".join(all_text) if all_text else "No text found in document."
            )
            final_descriptions = (
                "\n\n".join(all_descriptions)
                if all_descriptions
                else "No visual content described."
            )

            # Update comprehensive statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.documents_processed += 1

            # Update rolling averages for text and description lengths
            if all_text:
                current_avg_text_length = sum(len(text) for text in all_text) / len(
                    all_text
                )
                self.stats.average_text_length = (
                    (
                        self.stats.average_text_length
                        * (self.stats.documents_processed - 1)
                    )
                    + current_avg_text_length
                ) / self.stats.documents_processed

            if all_descriptions:
                current_avg_desc_length = sum(
                    len(desc) for desc in all_descriptions
                ) / len(all_descriptions)
                self.stats.average_description_length = (
                    (
                        self.stats.average_description_length
                        * (self.stats.documents_processed - 1)
                    )
                    + current_avg_desc_length
                ) / self.stats.documents_processed

            # Monitor memory after processing
            memory_after = self._monitor_memory_usage()
            memory_increase = memory_after.get(
                "gpu_memory_percent", 0
            ) - memory_before.get("gpu_memory_percent", 0)

            self.logger.info(
                f"Memory after processing - GPU: {memory_after.get('gpu_memory_percent', 0):.1f}%, RAM: {memory_after.get('system_ram_percent', 0):.1f}%"
            )
            if memory_increase > 5:  # Log significant memory increases
                self.logger.warning(
                    f"GPU memory increased by {memory_increase:.1f}% during processing"
                )

            # Final processing summary
            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            self.logger.info(f"Results summary:")
            self.logger.info(f"  - Text: {len(final_text)} characters")
            self.logger.info(f"  - Descriptions: {len(final_descriptions)} characters")
            self.logger.info(f"  - Pages processed: {self.stats.pages_processed}")
            self.logger.info(
                f"  - Text extractions successful: {self.stats.text_extractions_successful}"
            )
            self.logger.info(
                f"  - Descriptions successful: {self.stats.descriptions_successful}"
            )

            if processing_errors:
                self.logger.warning(
                    f"Processing completed with {len(processing_errors)} errors"
                )
                for error in processing_errors[:3]:  # Log first 3 errors
                    self.logger.warning(f"  - {error}")

            # Check if processing time exceeded configuration threshold
            if processing_time > MAX_PROCESSING_TIME_PER_DOC:
                self.logger.warning(
                    f"Document processing time ({processing_time:.2f}s) exceeded threshold ({MAX_PROCESSING_TIME_PER_DOC}s)"
                )

            return {
                "all_text": final_text,
                "all_imagery": final_descriptions,
            }

        except Exception as e:
            self.logger.error(
                f"Document processing failed for {document_obj.name}: {e}"
            )
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics with configuration context.

        Returns detailed statistics about processing performance, success rates,
        and system resource usage, enhanced with configuration constant context.

        Returns:
            Dict containing comprehensive processing statistics
        """
        self.logger.debug("Generating processing statistics report...")

        # Basic processing statistics
        stats = {
            "documents_processed": self.stats.documents_processed,
            "pages_processed": self.stats.pages_processed,
            "text_extractions_successful": self.stats.text_extractions_successful,
            "descriptions_successful": self.stats.descriptions_successful,
            "total_processing_time": self.stats.total_processing_time,
            "memory_refreshes": self.stats.memory_refreshes,
            "model_switches": self.stats.model_switches,
            # Current configuration
            "current_model": self.current_model_name,
            "device": self.device,
            "processing_mode": self.processing_mode.value,
            "max_tokens": self.max_tokens,
            # Configuration constants context
            "configuration": {
                "refresh_interval": self.refresh_interval,
                "memory_threshold": self.memory_threshold,
                "auto_model_selection": self.auto_model_selection,
                "ram_usage_ratio": RAM_USAGE_RATIO,
                "vram_usage_ratio": VRAM_USAGE_RATIO,
                "max_processing_time_per_doc": MAX_PROCESSING_TIME_PER_DOC,
                "min_text_success_rate": MIN_TEXT_SUCCESS_RATE,
                "min_description_success_rate": MIN_DESCRIPTION_SUCCESS_RATE,
            },
        }

        # Calculate performance rates and averages
        if self.stats.documents_processed > 0:
            stats["avg_processing_time_per_doc"] = (
                self.stats.total_processing_time / self.stats.documents_processed
            )

            # Calculate success rates
            if self.stats.pages_processed > 0:
                stats["text_extraction_success_rate"] = (
                    self.stats.text_extractions_successful / self.stats.pages_processed
                )
                stats["description_success_rate"] = (
                    self.stats.descriptions_successful / self.stats.pages_processed
                )
            else:
                stats["text_extraction_success_rate"] = 0.0
                stats["description_success_rate"] = 0.0

            stats["average_text_length"] = self.stats.average_text_length
            stats["average_description_length"] = self.stats.average_description_length

            # Performance assessment against configuration thresholds
            stats["performance_assessment"] = {
                "text_success_meets_threshold": stats["text_extraction_success_rate"]
                >= MIN_TEXT_SUCCESS_RATE,
                "description_success_meets_threshold": stats["description_success_rate"]
                >= MIN_DESCRIPTION_SUCCESS_RATE,
                "processing_time_within_threshold": stats["avg_processing_time_per_doc"]
                <= MAX_PROCESSING_TIME_PER_DOC,
            }

        # Add current memory usage
        stats["memory_usage"] = self._monitor_memory_usage()

        # Hardware constraints and utilization
        stats["hardware_constraints"] = {
            "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
            "max_ram_gb": self.hardware_constraints.max_ram_gb,
            "force_cpu": self.hardware_constraints.force_cpu,
        }

        # Processing efficiency metrics
        if self.stats.total_processing_time > 0:
            stats["efficiency_metrics"] = {
                "pages_per_minute": (
                    self.stats.pages_processed / self.stats.total_processing_time
                )
                * 60,
                "documents_per_hour": (
                    self.stats.documents_processed / self.stats.total_processing_time
                )
                * 3600,
                "characters_per_second": (
                    (
                        self.stats.average_text_length
                        + self.stats.average_description_length
                    )
                    * self.stats.pages_processed
                )
                / self.stats.total_processing_time,
            }

        self.logger.debug("Processing statistics report generated")
        return stats

    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance against configuration thresholds and suggest optimizations.

        Provides comprehensive performance analysis with specific recommendations
        based on configuration constants and current performance metrics.

        Returns:
            Dict containing performance analysis and optimization recommendations
        """
        self.logger.info(
            "Analyzing performance and generating optimization recommendations..."
        )

        stats = self.get_processing_stats()
        suggestions = []
        performance_issues = []

        # Analyze success rates against configuration thresholds
        text_success_rate = stats.get("text_extraction_success_rate", 0)
        desc_success_rate = stats.get("description_success_rate", 0)

        if text_success_rate < MIN_TEXT_SUCCESS_RATE:
            issue = f"Text extraction success rate ({text_success_rate:.2f}) below threshold ({MIN_TEXT_SUCCESS_RATE})"
            performance_issues.append(issue)
            suggestions.append(
                "Consider using HIGH_QUALITY processing mode for better text extraction"
            )
            suggestions.append(
                "Check if document quality is sufficient for text recognition"
            )
            self.logger.warning(issue)

        if desc_success_rate < MIN_DESCRIPTION_SUCCESS_RATE:
            issue = f"Description success rate ({desc_success_rate:.2f}) below threshold ({MIN_DESCRIPTION_SUCCESS_RATE})"
            performance_issues.append(issue)
            suggestions.append(
                "Consider adjusting processing mode to HIGH_QUALITY for better descriptions"
            )
            suggestions.append(
                "Verify that input images have sufficient visual content to describe"
            )
            self.logger.warning(issue)

        # Analyze processing speed against configuration threshold
        avg_time = stats.get("avg_processing_time_per_doc", 0)
        if avg_time > MAX_PROCESSING_TIME_PER_DOC:
            issue = f"Average processing time ({avg_time:.1f}s) exceeds threshold ({MAX_PROCESSING_TIME_PER_DOC}s)"
            performance_issues.append(issue)
            suggestions.append(
                "Consider switching to FAST processing mode to improve speed"
            )
            suggestions.append(
                "Check if a smaller model variant would be more appropriate"
            )
            suggestions.append(
                "Consider increasing refresh interval to reduce model reloading overhead"
            )
            self.logger.warning(issue)

        # Analyze memory usage against configuration thresholds
        memory_usage = stats.get("memory_usage", {})
        gpu_percent = memory_usage.get("gpu_memory_percent", 0)

        if gpu_percent > MAX_GPU_MEMORY_PERCENT:
            issue = f"GPU memory usage ({gpu_percent:.1f}%) exceeds threshold ({MAX_GPU_MEMORY_PERCENT}%)"
            performance_issues.append(issue)
            suggestions.append(
                "Consider reducing refresh interval for more frequent memory cleanup"
            )
            suggestions.append(
                "Switch to CPU processing if GPU memory is consistently high"
            )
            suggestions.append("Consider using a smaller model variant")
            self.logger.warning(issue)

        # Check refresh frequency optimization
        if self.stats.memory_refreshes > self.stats.documents_processed // 2:
            issue = "Frequent model refreshes detected - may indicate memory pressure"
            performance_issues.append(issue)
            suggestions.append(
                "Consider increasing memory threshold to reduce refresh frequency"
            )
            suggestions.append(
                "Monitor system for memory leaks or resource constraints"
            )

        # Generate hardware utilization recommendations
        if self.device == "cpu" and not self.hardware_constraints.force_cpu:
            if self.hardware_constraints.max_gpu_vram_gb > QWEN2VL_2B_MIN_VRAM:
                suggestions.append(
                    "GPU is available but not being used - consider enabling GPU processing"
                )

        # Performance optimization based on processing mode
        if (
            self.processing_mode == ProcessingMode.HIGH_QUALITY
            and avg_time > MAX_PROCESSING_TIME_PER_DOC
        ):
            suggestions.append(
                "High quality mode is causing slow processing - consider BALANCED mode"
            )
        elif self.processing_mode == ProcessingMode.FAST and (
            text_success_rate < MIN_TEXT_SUCCESS_RATE
            or desc_success_rate < MIN_DESCRIPTION_SUCCESS_RATE
        ):
            suggestions.append(
                "Fast mode may be affecting quality - consider BALANCED or HIGH_QUALITY mode"
            )

        # Generate overall performance score
        performance_score = 0
        total_checks = 0

        # Success rate scoring
        if stats.get("text_extraction_success_rate") is not None:
            performance_score += (
                min(stats["text_extraction_success_rate"] / MIN_TEXT_SUCCESS_RATE, 1.0)
                * 30
            )
            total_checks += 30

        if stats.get("description_success_rate") is not None:
            performance_score += (
                min(
                    stats["description_success_rate"] / MIN_DESCRIPTION_SUCCESS_RATE,
                    1.0,
                )
                * 30
            )
            total_checks += 30

        # Speed scoring (inverse relationship)
        if avg_time > 0:
            speed_score = max(1.0 - (avg_time / MAX_PROCESSING_TIME_PER_DOC), 0.0) * 40
            performance_score += speed_score
            total_checks += 40

        overall_performance_score = (
            (performance_score / total_checks) * 100 if total_checks > 0 else 0
        )

        optimization_report = {
            "current_performance": stats,
            "performance_issues": performance_issues,
            "optimization_suggestions": suggestions,
            "overall_performance_score": overall_performance_score,
            "configuration_thresholds": {
                "min_text_success_rate": MIN_TEXT_SUCCESS_RATE,
                "min_description_success_rate": MIN_DESCRIPTION_SUCCESS_RATE,
                "max_processing_time_per_doc": MAX_PROCESSING_TIME_PER_DOC,
                "max_gpu_memory_percent": MAX_GPU_MEMORY_PERCENT,
            },
            "recommendations_priority": (
                "high"
                if len(performance_issues) > 2
                else "medium" if len(performance_issues) > 0 else "low"
            ),
        }

        self.logger.info(
            f"Performance analysis completed - Overall score: {overall_performance_score:.1f}/100"
        )
        self.logger.info(
            f"Found {len(performance_issues)} performance issues and {len(suggestions)} optimization suggestions"
        )

        return optimization_report
