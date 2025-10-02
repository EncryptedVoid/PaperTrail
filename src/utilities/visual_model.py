import gc
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import fitz
import psutil
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from config import (
    DEFAULT_AUTO_MODEL_SELECTION,
    DEFAULT_MEMORY_THRESHOLD,
    DEFAULT_PROCESSING_MODE,
    DEFAULT_REFRESH_INTERVAL,
    MAX_AVERAGE_PROCESSING_TIME,
    MAX_GPU_MEMORY_PERCENT,
    MAX_PROCESSING_TIME_PER_DOC,
    MIN_DESCRIPTION_SUCCESS_RATE,
    MIN_TEXT_SUCCESS_RATE,
    PREFERRED_VISUAL_MODEL,
    ProcessingMode,
    QWEN2VL_2B_MAX_TOKENS,
    RAM_USAGE_RATIO,
    VRAM_USAGE_RATIO,
    ZOOM_FACTOR_BALANCED,
    ZOOM_FACTOR_FAST,
    ZOOM_FACTOR_HIGH_QUALITY,
)


@dataclass
class ProcessingStats:
    """
    Comprehensive tracking of processing performance and quality metrics.

    This class maintains statistics across all file_path processing operations
    to enable performance monitoring and optimization decisions.
    """

    file_paths_processed: int = 0
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

    This processor handles file_path analysis using vision-language models with automatic
    hardware detection, memory management, and performance optimization. It supports
    both PDF and image file processing with configurable quality modes.

    Key Features:
    - Automatic hardware detection and model selection
    - Memory usage monitoring and automatic model refresh
    - Configurable processing modes (fast, balanced, high-quality)
    - Comprehensive performance tracking and statistics
    - Robust error handling and fallback mechanisms
    - Handwriting detection capabilities
    - Document detection capabilities
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the Visual Processor with automatic configuration from constants.
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

        self.processing_mode: ProcessingMode = DEFAULT_PROCESSING_MODE

        self.refresh_interval = DEFAULT_REFRESH_INTERVAL
        self.memory_threshold = DEFAULT_MEMORY_THRESHOLD
        self.auto_model_selection = DEFAULT_AUTO_MODEL_SELECTION

        self.logger.info(f"Processing mode set to: {self.processing_mode.value}")
        self.logger.info(f"Auto model selection: {self.auto_model_selection}")
        self.logger.info(f"Memory threshold: {self.memory_threshold}%")
        self.logger.info(f"Refresh interval: {self.refresh_interval} file_paths")

        # Initialize processing statistics
        self.stats = ProcessingStats()
        self.logger.debug("Processing statistics initialized")

        # Model state variables - initialize to None until loaded
        self.model = None
        self.processor = None
        self.current_model_name = PREFERRED_VISUAL_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = QWEN2VL_2B_MAX_TOKENS  # Default to 2B model tokens since that's the preferred model

        # Actually load the model
        try:
            self.logger.info("Loading initial visual model...")
            self._load_model(PREFERRED_VISUAL_MODEL)
            self.logger.info("Visual model initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to load initial model: {e}")
            # Don't raise here - let individual processing methods handle it
            self.logger.warning("Model will be loaded on first use")

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
            if self.device == "cuda":
                self.logger.info(f"Using device: {self.device}")
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
                    "dtype": (
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
                self.logger.info(
                    f"Model loading completed in {loading_time:.2f} seconds"
                )

                # Log memory usage after loading
                memory_info = self._monitor_memory_usage()

                self.logger.info(
                    f"GPU memory usage after loading: {memory_info.get('gpu_memory_percent', 0):.1f}%"
                )
            else:
                self.device = "cpu"
                self.logger.info("Using CPU for processing")

                # Log memory usage after loading
                memory_info = self._monitor_memory_usage()

                self.logger.info(
                    f"System RAM usage after loading: {memory_info.get('system_ram_percent', 0):.1f}%"
                )

            self.stats.last_refresh_time = time.time()

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            self.logger.error("Model loading failed, attempting fallback strategies...")
            raise ModuleNotFoundError

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

        # Check file_path count trigger using configuration constant
        if self.stats.file_paths_processed >= self.refresh_interval:
            self.logger.info(
                f"Refresh triggered: processed {self.stats.file_paths_processed} file_paths (limit: {self.refresh_interval})"
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
        if self.stats.file_paths_processed > 0:
            avg_time_per_doc = (
                self.stats.total_processing_time / self.stats.file_paths_processed
            )
            if avg_time_per_doc > MAX_AVERAGE_PROCESSING_TIME:
                self.logger.warning(
                    f"Refresh triggered: slow processing ({avg_time_per_doc:.1f}s per doc, limit: {MAX_AVERAGE_PROCESSING_TIME}s)"
                )
                return True

            self.logger.debug(
                f"Average processing time: {avg_time_per_doc:.1f}s per file_path"
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
            processing_count = self.stats.file_paths_processed
            self.stats.file_paths_processed = 0  # Reset file_path counter

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
                "file_path",
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

        # Check if model is loaded
        if self.model is None or self.processor is None:
            self.logger.error(
                "Model or processor not loaded. Attempting to load now..."
            )
            try:
                self._load_model(self.current_model_name)
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise RuntimeError("Visual model not available for processing")

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

    def detect_handwriting(
        self, image: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """
        Detect whether an image contains handwriting and return a confidence score.

        Args:
            image: Image to analyze (file path or PIL Image object)

        Returns:
            Dict containing:
                - confidence: float (0.0-1.0) indicating confidence that handwriting is present
                - detected: bool indicating if handwriting was detected (confidence > 0.5)
                - analysis: str describing the findings
                - processing_time: float time taken to process

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If model is not available
        """
        self.logger.info("Starting handwriting detection...")
        start_time = time.time()

        # Check if model is loaded
        if self.model is None or self.processor is None:
            self.logger.error("Model not loaded for handwriting detection")
            raise RuntimeError("Visual model not available for processing")

        try:
            # Load and validate image
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                processed_image = Image.open(image_path)
            else:
                processed_image = image

            # Prepare handwriting detection prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {
                            "type": "text",
                            "text": "Analyze this image carefully. Does it contain any handwriting or handwritten text? Respond with a confidence score from 0 to 100 (where 0 means definitely no handwriting, 100 means definitely handwriting present), followed by a brief explanation. Format: CONFIDENCE: [score] | ANALYSIS: [explanation]",
                        },
                    ],
                }
            ]

            # Process and generate response
            text_inputs = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_inputs],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    temperature=0.1,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            # Parse response to extract confidence score
            confidence = 0.0
            analysis = output.strip()

            # Try to extract confidence score from response
            if "CONFIDENCE:" in output.upper():
                try:
                    confidence_part = (
                        output.upper().split("CONFIDENCE:")[1].split("|")[0].strip()
                    )
                    confidence_value = float(
                        "".join(
                            filter(lambda x: x.isdigit() or x == ".", confidence_part)
                        )
                    )
                    confidence = min(max(confidence_value / 100.0, 0.0), 1.0)
                except:
                    # Fallback: analyze keywords in response
                    response_lower = output.lower()
                    if any(
                        word in response_lower
                        for word in [
                            "definitely handwriting",
                            "clearly handwritten",
                            "handwritten text",
                        ]
                    ):
                        confidence = 0.9
                    elif any(
                        word in response_lower
                        for word in ["handwriting", "handwritten"]
                    ):
                        confidence = 0.7
                    elif any(
                        word in response_lower
                        for word in ["possibly handwriting", "might be handwriting"]
                    ):
                        confidence = 0.5
                    elif any(
                        word in response_lower
                        for word in [
                            "no handwriting",
                            "not handwritten",
                            "typed",
                            "printed",
                        ]
                    ):
                        confidence = 0.1

            processing_time = time.time() - start_time

            result = {
                "confidence": confidence,
                "detected": confidence > 0.5,
                "analysis": analysis,
                "processing_time": processing_time,
            }

            self.logger.info(
                f"Handwriting detection completed in {processing_time:.2f}s - "
                f"Confidence: {confidence:.2f}, Detected: {result['detected']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Handwriting detection failed: {e}")
            raise

    def detect_document(self, image: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Detect whether an image is a document and return a confidence score.

        Args:
            image: Image to analyze (file path or PIL Image object)

        Returns:
            Dict containing:
                - confidence: float (0.0-1.0) indicating confidence that this is a document
                - is_document: bool indicating if it's likely a document (confidence > 0.5)
                - document_type: str describing the type of document if detected
                - analysis: str describing the findings
                - processing_time: float time taken to process

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If model is not available
        """
        self.logger.info("Starting document detection...")
        start_time = time.time()

        # Check if model is loaded
        if self.model is None or self.processor is None:
            self.logger.error("Model not loaded for document detection")
            raise RuntimeError("Visual model not available for processing")

        try:
            # Load and validate image
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                processed_image = Image.open(image_path)
            else:
                processed_image = image

            # Prepare document detection prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_image},
                        {
                            "type": "text",
                            "text": "Analyze this image carefully. Is this a document or an image of a document? Documents include: invoices, receipts, forms, letters, reports, contracts, certificates, ID cards, business cards, book pages, articles, spreadsheets, presentations, etc. Respond with a confidence score from 0 to 100 (where 0 means definitely not a document, 100 means definitely a document), the document type if applicable, and a brief explanation. Format: CONFIDENCE: [score] | TYPE: [document type or 'N/A'] | ANALYSIS: [explanation]",
                        },
                    ],
                }
            ]

            # Process and generate response
            text_inputs = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_inputs],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    temperature=0.1,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                output = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            # Parse response to extract confidence score and document type
            confidence = 0.0
            document_type = "Unknown"
            analysis = output.strip()

            # Try to extract confidence score from response
            if "CONFIDENCE:" in output.upper():
                try:
                    confidence_part = (
                        output.upper().split("CONFIDENCE:")[1].split("|")[0].strip()
                    )
                    confidence_value = float(
                        "".join(
                            filter(lambda x: x.isdigit() or x == ".", confidence_part)
                        )
                    )
                    confidence = min(max(confidence_value / 100.0, 0.0), 1.0)
                except:
                    # Fallback: analyze keywords in response
                    response_lower = output.lower()
                    if any(
                        word in response_lower
                        for word in ["definitely a document", "clearly a document"]
                    ):
                        confidence = 0.9
                    elif any(
                        word in response_lower
                        for word in [
                            "document",
                            "invoice",
                            "receipt",
                            "form",
                            "letter",
                            "report",
                        ]
                    ):
                        confidence = 0.7
                    elif any(
                        word in response_lower
                        for word in ["possibly a document", "might be a document"]
                    ):
                        confidence = 0.5
                    elif any(
                        word in response_lower
                        for word in [
                            "not a document",
                            "photograph",
                            "artwork",
                            "screenshot",
                        ]
                    ):
                        confidence = 0.1

            # Try to extract document type
            if "TYPE:" in output.upper():
                try:
                    type_part = output.upper().split("TYPE:")[1].split("|")[0].strip()
                    if type_part and type_part.upper() != "N/A":
                        document_type = type_part
                except:
                    pass

            # Fallback document type detection from keywords
            if document_type == "Unknown" and confidence > 0.5:
                response_lower = output.lower()
                doc_types = {
                    "invoice": ["invoice"],
                    "receipt": ["receipt"],
                    "form": ["form"],
                    "letter": ["letter"],
                    "report": ["report"],
                    "contract": ["contract"],
                    "certificate": ["certificate"],
                    "ID card": ["id card", "identification"],
                    "business card": ["business card"],
                    "spreadsheet": ["spreadsheet", "excel", "table"],
                    "presentation": ["presentation", "slide"],
                    "article": ["article", "page"],
                }
                for doc_type, keywords in doc_types.items():
                    if any(keyword in response_lower for keyword in keywords):
                        document_type = doc_type
                        break

            processing_time = time.time() - start_time

            result = {
                "confidence": confidence,
                "is_document": confidence > 0.5,
                "document_type": document_type,
                "analysis": analysis,
                "processing_time": processing_time,
            }

            self.logger.info(
                f"Document detection completed in {processing_time:.2f}s - "
                f"Confidence: {confidence:.2f}, Is Document: {result['is_document']}, "
                f"Type: {document_type}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Document detection failed: {e}")
            raise

    def extract_text(self, file_path: Path) -> str:
        """
        Extract text from a document file (PDF or image).

        Args:
            file_path: Path to document file (PDF or image)

        Returns:
            str: Extracted text from the document

        Raises:
            FileNotFoundError: If document file doesn't exist
            ValueError: If file type is unsupported
            Exception: If processing fails
        """
        self.logger.info(f"Starting text extraction: {file_path.name}")
        start_time = time.time()

        # Check if model should be refreshed
        if self._should_refresh_model():
            self.logger.info("Auto-refreshing model due to configuration thresholds")
            self.refresh_model()

        # Validate file existence
        if not file_path.exists():
            self.logger.error(f"Document file not found: {file_path}")
            raise FileNotFoundError(f"Document file not found: {file_path}")

        file_ext = file_path.suffix.lower()
        all_text = []

        try:
            if file_ext == ".pdf":
                # PDF processing
                self.logger.info("Processing PDF for text extraction...")
                images = self._pdf_to_images(file_path)
                total_pages = len(images)

                for i, img in enumerate(images):
                    page_num = i + 1
                    self.logger.debug(
                        f"Extracting text from page {page_num}/{total_pages}"
                    )

                    try:
                        result = self._process_single_image(
                            img, extract_text=True, describe_image=False
                        )

                        if result.get("text") and result["text"] not in [
                            "NO_TEXT_FOUND",
                            "TEXT_EXTRACTION_FAILED",
                        ]:
                            page_text = f"=== Page {page_num} ===\n{result['text']}"
                            all_text.append(page_text)
                            self.stats.text_extractions_successful += 1

                        self.stats.pages_processed += 1

                    except Exception as e:
                        self.logger.error(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                        continue

            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                # Image file processing
                self.logger.info(
                    f"Processing image for text extraction: {file_path.name}"
                )
                result = self._process_single_image(
                    file_path, extract_text=True, describe_image=False
                )

                if result.get("text") and result["text"] not in [
                    "NO_TEXT_FOUND",
                    "TEXT_EXTRACTION_FAILED",
                ]:
                    all_text.append(result["text"])
                    self.stats.text_extractions_successful += 1

                self.stats.pages_processed += 1

            else:
                error_msg = f"Unsupported file type: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Compile final results
            final_text = (
                "\n\n".join(all_text) if all_text else "No text found in document."
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.file_paths_processed += 1

            self.logger.info(f"Text extraction completed in {processing_time:.2f}s")
            self.logger.info(f"Extracted text: {len(final_text)} characters")

            return final_text

        except Exception as e:
            self.logger.error(f"Text extraction failed for {file_path.name}: {e}")
            raise

    def extract_visual_description(self, file_path: Path) -> str:
        """
        Extract visual descriptions from a document file (PDF or image).

        Args:
            file_path: Path to document file (PDF or image)

        Returns:
            str: Visual descriptions of the document

        Raises:
            FileNotFoundError: If document file doesn't exist
            ValueError: If file type is unsupported
            Exception: If processing fails
        """
        self.logger.info(f"Starting visual description extraction: {file_path.name}")
        start_time = time.time()

        # Check if model should be refreshed
        if self._should_refresh_model():
            self.logger.info("Auto-refreshing model due to configuration thresholds")
            self.refresh_model()

        # Validate file existence
        if not file_path.exists():
            self.logger.error(f"Document file not found: {file_path}")
            raise FileNotFoundError(f"Document file not found: {file_path}")

        file_ext = file_path.suffix.lower()
        all_descriptions = []

        try:
            if file_ext == ".pdf":
                # PDF processing
                self.logger.info("Processing PDF for visual descriptions...")
                images = self._pdf_to_images(file_path)
                total_pages = len(images)

                for i, img in enumerate(images):
                    page_num = i + 1
                    self.logger.debug(
                        f"Generating description for page {page_num}/{total_pages}"
                    )

                    try:
                        result = self._process_single_image(
                            img, extract_text=False, describe_image=True
                        )

                        if (
                            result.get("description")
                            and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                        ):
                            page_desc = f"=== Page {page_num} Visual Description ===\n{result['description']}"
                            all_descriptions.append(page_desc)
                            self.stats.descriptions_successful += 1

                        self.stats.pages_processed += 1

                    except Exception as e:
                        self.logger.error(
                            f"Failed to generate description for page {page_num}: {e}"
                        )
                        continue

            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                # Image file processing
                self.logger.info(
                    f"Processing image for visual description: {file_path.name}"
                )
                result = self._process_single_image(
                    file_path, extract_text=False, describe_image=True
                )

                if (
                    result.get("description")
                    and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                ):
                    all_descriptions.append(result["description"])
                    self.stats.descriptions_successful += 1

                self.stats.pages_processed += 1

            else:
                error_msg = f"Unsupported file type: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Compile final results
            final_descriptions = (
                "\n\n".join(all_descriptions)
                if all_descriptions
                else "No visual content described."
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.file_paths_processed += 1

            self.logger.info(
                f"Visual description extraction completed in {processing_time:.2f}s"
            )
            self.logger.info(
                f"Generated descriptions: {len(final_descriptions)} characters"
            )

            return final_descriptions

        except Exception as e:
            self.logger.error(
                f"Visual description extraction failed for {file_path.name}: {e}"
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
            "file_paths_processed": self.stats.file_paths_processed,
            "pages_processed": self.stats.pages_processed,
            "text_extractions_successful": self.stats.text_extractions_successful,
            "descriptions_successful": self.stats.descriptions_successful,
            "total_processing_time": self.stats.total_processing_time,
            "memory_refreshes": self.stats.memory_refreshes,
            "model_switches": self.stats.model_switches,
            # Current configuration
            "current_model": self.current_model_name,
            "device": self.device,
            "processing_mode": self.processing_mode,
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
        if self.stats.file_paths_processed > 0:
            stats["avg_processing_time_per_doc"] = (
                self.stats.total_processing_time / self.stats.file_paths_processed
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

        # Processing efficiency metrics
        if self.stats.total_processing_time > 0:
            stats["efficiency_metrics"] = {
                "pages_per_minute": (
                    self.stats.pages_processed / self.stats.total_processing_time
                )
                * 60,
                "file_paths_per_hour": (
                    self.stats.file_paths_processed / self.stats.total_processing_time
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
                "Check if file_path quality is sufficient for text recognition"
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
        if self.stats.memory_refreshes > self.stats.file_paths_processed // 2:
            issue = "Frequent model refreshes detected - may indicate memory pressure"
            performance_issues.append(issue)
            suggestions.append(
                "Consider increasing memory threshold to reduce refresh frequency"
            )
            suggestions.append(
                "Monitor system for memory leaks or resource constraints"
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
