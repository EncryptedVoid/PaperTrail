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


class ProcessingMode(Enum):
    """Processing quality modes"""

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass
class VisionModelSpec:
    """Specification for vision models including hardware requirements"""

    name: str
    model_id: str
    min_gpu_vram_gb: float
    min_ram_gb: float
    quality_score: int  # 1-10, higher is better
    max_tokens: int
    supports_batch: bool = False
    model_type: str = "qwen2vl"  # qwen2vl, llava, blip2, etc.


@dataclass
class HardwareConstraints:
    """Hardware resource constraints"""

    max_gpu_vram_gb: float
    max_ram_gb: float
    force_cpu: bool = False


@dataclass
class ProcessingStats:
    """Track processing performance and quality metrics"""

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
    """Enhanced Visual Processor with hardware optimization and multi-model support"""

    def __init__(
        self,
        logger: logging.Logger,
        max_gpu_vram_gb: Optional[float] = None,
        max_ram_gb: Optional[float] = None,
        force_cpu: bool = False,
        processing_mode: ProcessingMode = ProcessingMode.BALANCED,
        refresh_interval: int = 5,
        memory_threshold: float = 80.0,
        auto_model_selection: bool = True,
        preferred_model: Optional[str] = None,
    ) -> None:
        # Validate required logger parameter
        if logger is None:
            raise ValueError(
                "Logger is required - VisualProcessor cannot be initialized without a logger"
            )
        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger instance, got {type(logger)}")

        # Initialize instance attributes
        self.logger = logger
        self.processing_mode = processing_mode
        self.refresh_interval = refresh_interval
        self.memory_threshold = memory_threshold
        self.auto_model_selection = auto_model_selection

        # Initialize hardware constraints
        self.hardware_constraints = self._detect_hardware_constraints(
            max_gpu_vram_gb, max_ram_gb, force_cpu
        )

        # Initialize model registry
        self.model_registry = self._initialize_model_registry()

        # Initialize processing statistics
        self.stats = ProcessingStats()

        # Model state
        self.model = None
        self.processor = None
        self.current_model_spec = None
        self.device = None

        # Select and load optimal model
        if auto_model_selection:
            self.current_model_spec = self._select_optimal_model(preferred_model)
        else:
            # Use preferred model or default
            model_name = preferred_model or "Qwen/Qwen2-VL-7B-Instruct"
            self.current_model_spec = self._get_model_spec_by_id(model_name)

        self.logger.info(
            f"Initialized VisualProcessor with model: {self.current_model_spec.name}"
        )
        self.logger.info(f"Hardware constraints: {self.hardware_constraints}")

        # Load the selected model
        self._load_model()

    def _detect_hardware_constraints(
        self,
        max_gpu_vram_gb: Optional[float],
        max_ram_gb: Optional[float],
        force_cpu: bool,
    ) -> HardwareConstraints:
        """Detect available hardware or use provided constraints"""

        # RAM detection
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            available_ram = total_ram * 0.7  # Use 70% of total RAM
        else:
            available_ram = max_ram_gb

        # GPU detection
        if max_gpu_vram_gb is None and not force_cpu:
            try:
                if torch.cuda.is_available():
                    # Get GPU memory from torch
                    gpu_props = torch.cuda.get_device_properties(0)
                    total_vram = gpu_props.total_memory / (1024**3)  # Convert to GB
                    available_vram = total_vram * 0.8  # Use 80% of total VRAM
                else:
                    available_vram = 0
            except Exception as e:
                self.logger.warning(f"Could not detect GPU VRAM: {e}")
                available_vram = 0
        else:
            available_vram = max_gpu_vram_gb or 0

        if force_cpu:
            available_vram = 0

        return HardwareConstraints(
            max_gpu_vram_gb=available_vram,
            max_ram_gb=available_ram,
            force_cpu=force_cpu,
        )

    def _initialize_model_registry(self) -> List[VisionModelSpec]:
        """Initialize registry of available vision models with their requirements"""
        return [
            # Qwen2-VL models
            VisionModelSpec(
                name="Qwen2-VL-2B",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                min_gpu_vram_gb=4.0,
                min_ram_gb=8.0,
                quality_score=7,
                max_tokens=512,
                model_type="qwen2vl",
            ),
            VisionModelSpec(
                name="Qwen2-VL-7B",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                min_gpu_vram_gb=14.0,
                min_ram_gb=16.0,
                quality_score=9,
                max_tokens=512,
                model_type="qwen2vl",
            ),
            VisionModelSpec(
                name="Qwen2-VL-72B",
                model_id="Qwen/Qwen2-VL-72B-Instruct",
                min_gpu_vram_gb=144.0,
                min_ram_gb=200.0,
                quality_score=10,
                max_tokens=1024,
                model_type="qwen2vl",
            ),
            # Alternative models for fallback
            VisionModelSpec(
                name="Qwen2-VL-7B-CPU",
                model_id="Qwen/Qwen2-VL-7B-Instruct",
                min_gpu_vram_gb=0.0,  # CPU only
                min_ram_gb=32.0,
                quality_score=8,
                max_tokens=512,
                model_type="qwen2vl",
            ),
        ]

    def _select_optimal_model(
        self, preferred_model: Optional[str] = None
    ) -> VisionModelSpec:
        """Select the best model that fits within hardware constraints"""

        # If a preferred model is specified, try to use it if it fits
        if preferred_model:
            preferred_spec = self._get_model_spec_by_id(preferred_model)
            if preferred_spec and self._model_fits_constraints(preferred_spec):
                self.logger.info(f"Using preferred model: {preferred_spec.name}")
                return preferred_spec
            else:
                self.logger.warning(
                    f"Preferred model {preferred_model} doesn't fit constraints, selecting alternative"
                )

        # Filter models that fit within hardware constraints
        suitable_models = [
            model
            for model in self.model_registry
            if self._model_fits_constraints(model)
        ]

        if not suitable_models:
            # Fallback to CPU-only model
            cpu_models = [m for m in self.model_registry if m.min_gpu_vram_gb == 0]
            if cpu_models:
                best_model = max(cpu_models, key=lambda m: m.quality_score)
                self.logger.warning(
                    f"No GPU models fit constraints, using CPU model: {best_model.name}"
                )
                return best_model
            else:
                raise RuntimeError(
                    "No suitable vision models found for current hardware constraints"
                )

        # Select model with highest quality score that fits
        best_model = max(suitable_models, key=lambda m: m.quality_score)
        self.logger.info(
            f"Selected optimal model: {best_model.name} (quality: {best_model.quality_score})"
        )

        return best_model

    def _model_fits_constraints(self, model_spec: VisionModelSpec) -> bool:
        """Check if a model fits within hardware constraints"""
        if self.hardware_constraints.force_cpu:
            return model_spec.min_gpu_vram_gb == 0

        fits_vram = (
            model_spec.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
        )
        fits_ram = model_spec.min_ram_gb <= self.hardware_constraints.max_ram_gb

        return fits_vram and fits_ram

    def _get_model_spec_by_id(self, model_id: str) -> Optional[VisionModelSpec]:
        """Get model specification by model ID"""
        for spec in self.model_registry:
            if spec.model_id == model_id:
                return spec
        return None

    def _load_model(self) -> None:
        """Load or reload the vision model and processor"""
        self.logger.info(f"Loading vision model: {self.current_model_spec.name}")

        try:
            # Clean up existing model if present
            if self.model is not None:
                self.logger.debug("Cleaning up existing model...")
                del self.model
                del self.processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Determine compute device
            if (
                self.hardware_constraints.force_cpu
                or self.current_model_spec.min_gpu_vram_gb == 0
            ):
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.logger.info(f"Using device: {self.device}")

            # Log hardware info
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": (
                    torch.bfloat16 if self.device == "cuda" else torch.float32
                ),
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.device == "cpu":
                model_kwargs["device_map"] = {"": "cpu"}

            self.logger.debug("Loading vision model from transformers...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.current_model_spec.model_id, **model_kwargs
            )

            self.logger.debug("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.current_model_spec.model_id
            )

            self.logger.info(f"Model loaded successfully on {self.device}")
            self.stats.last_refresh_time = time.time()

        except Exception as e:
            self.logger.error(
                f"Failed to load model {self.current_model_spec.name}: {e}"
            )
            # Try fallback to a smaller model or CPU
            self._try_fallback_model()

    def _try_fallback_model(self) -> None:
        """Try to load a fallback model if primary model fails"""
        self.logger.info("Attempting to load fallback model...")

        # Try smaller models first
        fallback_models = sorted(
            [m for m in self.model_registry if m != self.current_model_spec],
            key=lambda m: m.quality_score,
            reverse=True,
        )

        for fallback_spec in fallback_models:
            if self._model_fits_constraints(fallback_spec):
                try:
                    self.logger.info(f"Trying fallback model: {fallback_spec.name}")
                    self.current_model_spec = fallback_spec
                    self._load_model()
                    self.stats.model_switches += 1
                    return
                except Exception as e:
                    self.logger.warning(
                        f"Fallback model {fallback_spec.name} also failed: {e}"
                    )
                    continue

        raise RuntimeError("All fallback models failed to load")

    def _should_refresh_model(self) -> bool:
        """Check if model should be refreshed based on usage and performance"""
        current_time = time.time()

        # Refresh based on document count
        if self.stats.documents_processed >= self.refresh_interval:
            self.logger.info(
                f"Refresh triggered: processed {self.stats.documents_processed} documents"
            )
            return True

        # Check GPU memory if available
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                memory_percent = (memory_used / memory_total) * 100

                if memory_percent > self.memory_threshold:
                    self.logger.warning(
                        f"Refresh triggered: GPU memory usage {memory_percent:.1f}%"
                    )
                    return True
            except Exception as e:
                self.logger.warning(f"Could not check GPU memory: {e}")

        # Check processing time degradation (refresh if processing is taking too long)
        if self.stats.documents_processed > 0:
            avg_time_per_doc = (
                self.stats.total_processing_time / self.stats.documents_processed
            )
            if avg_time_per_doc > 60.0:  # More than 1 minute per document
                self.logger.warning(
                    f"Refresh triggered: slow processing ({avg_time_per_doc:.1f}s per doc)"
                )
                return True

        return False

    def refresh_model(self) -> None:
        """Refresh the model to clear context and free memory"""
        self.logger.info("Refreshing vision model...")
        try:
            self._load_model()
            self.stats.memory_refreshes += 1
            # Reset some counters but keep overall stats
            processing_count = self.stats.documents_processed
            self.stats.documents_processed = 0
            self.logger.info(
                f"Model refresh completed (total processed: {processing_count})"
            )
        except Exception as e:
            self.logger.error(f"Failed to refresh model: {e}")
            raise

    def switch_model(self, new_model_id: str) -> bool:
        """Switch to a different model"""
        new_spec = self._get_model_spec_by_id(new_model_id)
        if not new_spec:
            self.logger.error(f"Model {new_model_id} not found in registry")
            return False

        if not self._model_fits_constraints(new_spec):
            self.logger.error(f"Model {new_model_id} doesn't fit hardware constraints")
            return False

        try:
            old_model = self.current_model_spec.name
            self.current_model_spec = new_spec
            self._load_model()
            self.stats.model_switches += 1
            self.logger.info(
                f"Successfully switched from {old_model} to {new_spec.name}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to model {new_model_id}: {e}")
            return False

    def _monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        memory_info = {}

        # System RAM
        ram = psutil.virtual_memory()
        memory_info["system_ram_used_gb"] = (ram.total - ram.available) / (1024**3)
        memory_info["system_ram_total_gb"] = ram.total / (1024**3)
        memory_info["system_ram_percent"] = ram.percent

        # GPU memory if available
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
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory info: {e}")

        return memory_info

    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for processing results"""
        metrics = {}

        # Text extraction quality
        text = result.get("text", "")
        if text and text not in ["NO_TEXT_FOUND", "TEXT_EXTRACTION_FAILED"]:
            metrics["text_extraction_success"] = 1.0
            metrics["text_length"] = len(text)
            # Simple quality heuristic based on text length and diversity
            unique_words = len(set(text.lower().split()))
            total_words = len(text.split())
            metrics["text_diversity"] = unique_words / max(total_words, 1)
        else:
            metrics["text_extraction_success"] = 0.0
            metrics["text_length"] = 0
            metrics["text_diversity"] = 0.0

        # Description quality
        description = result.get("description", "")
        if description and description != "DESCRIPTION_GENERATION_FAILED":
            metrics["description_success"] = 1.0
            metrics["description_length"] = len(description)
            # Check for descriptive keywords as quality indicator
            descriptive_keywords = [
                "color",
                "layout",
                "text",
                "image",
                "diagram",
                "chart",
                "table",
            ]
            keyword_count = sum(
                1 for keyword in descriptive_keywords if keyword in description.lower()
            )
            metrics["description_richness"] = keyword_count / len(descriptive_keywords)
        else:
            metrics["description_success"] = 0.0
            metrics["description_length"] = 0
            metrics["description_richness"] = 0.0

        return metrics

    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """Convert PDF pages to PIL Image objects with adaptive resolution"""
        pdf_path_obj = Path(pdf_path)
        self.logger.info(f"Converting PDF to images: {pdf_path_obj}")

        if not pdf_path_obj.exists():
            self.logger.error(f"PDF file not found: {pdf_path_obj}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        try:
            doc = fitz.open(str(pdf_path_obj))
            images = []
            page_count = len(doc)

            # Adjust resolution based on processing mode
            zoom_factors = {
                ProcessingMode.FAST: 1.5,
                ProcessingMode.BALANCED: 2.0,
                ProcessingMode.HIGH_QUALITY: 3.0,
            }
            zoom = zoom_factors.get(self.processing_mode, 2.0)

            self.logger.info(f"PDF has {page_count} pages, using {zoom}x zoom")

            for page_num in range(page_count):
                try:
                    page = doc[page_num]
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)

                    self.logger.debug(
                        f"Page {page_num + 1} converted - size: {img.size}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    continue

            doc.close()
            self.logger.info(f"Successfully converted {len(images)}/{page_count} pages")
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
        """Process individual image with enhanced error handling and quality monitoring"""

        start_time = time.time()

        try:
            # Load image
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                processed_image = Image.open(image_path)
            else:
                processed_image = image

            results = {}

            # Text extraction
            if extract_text:
                try:
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

                    text_inputs = self.processor.apply_chat_template(
                        text_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(text_messages)

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
                            max_new_tokens=self.current_model_spec.max_tokens,
                            do_sample=False,  # Deterministic for consistency
                        )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        text_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["text"] = text_output.strip()

                except Exception as e:
                    self.logger.error(f"Text extraction failed: {e}")
                    results["text"] = "TEXT_EXTRACTION_FAILED"

            # Image description
            if describe_image:
                try:
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

                    desc_inputs = self.processor.apply_chat_template(
                        desc_messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(desc_messages)

                    inputs = self.processor(
                        text=[desc_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.current_model_spec.max_tokens,
                            do_sample=False,
                        )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        desc_output = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["description"] = desc_output.strip()

                except Exception as e:
                    self.logger.error(f"Image description failed: {e}")
                    results["description"] = "DESCRIPTION_GENERATION_FAILED"

            # Calculate processing time and quality metrics
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["quality_metrics"] = self._calculate_quality_metrics(results)

            return results

        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            raise

    def extract_article_semantics(self, document: Union[str, Path]) -> Dict[str, str]:
        """Process document with enhanced monitoring and performance tracking"""
        document_obj = Path(document)
        self.logger.info(f"Starting document processing: {document_obj}")

        start_time = time.time()

        # Check if model should be refreshed
        if self._should_refresh_model():
            self.logger.info(
                "Auto-refreshing model due to usage threshold or memory pressure"
            )
            self.refresh_model()

        # Monitor memory before processing
        memory_before = self._monitor_memory_usage()
        self.logger.debug(
            f"Memory before processing: GPU {memory_before.get('gpu_memory_percent', 0):.1f}%, "
            f"RAM {memory_before.get('system_ram_percent', 0):.1f}%"
        )

        if not document_obj.exists():
            raise FileNotFoundError(f"Document file not found: {document_obj}")

        file_ext = document_obj.suffix.lower()
        self.logger.info(f"Document type: {file_ext}")

        all_text = []
        all_descriptions = []
        page_quality_metrics = []

        try:
            if file_ext == ".pdf":
                images = self._pdf_to_images(document_obj)
                total_pages = len(images)
                self.logger.info(f"Processing {total_pages} pages from PDF")

                for i, img in enumerate(images):
                    page_num = i + 1
                    self.logger.debug(f"Processing page {page_num}/{total_pages}")

                    try:
                        result = self._process_single_image(img)
                        page_quality_metrics.append(result.get("quality_metrics", {}))

                        # Handle text extraction
                        if result.get("text") and result["text"] not in [
                            "NO_TEXT_FOUND",
                            "TEXT_EXTRACTION_FAILED",
                        ]:
                            page_text = f"=== Page {page_num} ===\n{result['text']}"
                            all_text.append(page_text)
                            self.stats.text_extractions_successful += 1

                        # Handle descriptions
                        if (
                            result.get("description")
                            and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                        ):
                            page_desc = f"=== Page {page_num} Visual Description ===\n{result['description']}"
                            all_descriptions.append(page_desc)
                            self.stats.descriptions_successful += 1

                        self.stats.pages_processed += 1

                    except Exception as e:
                        self.logger.error(f"Failed to process page {page_num}: {e}")
                        continue

            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                self.logger.info(f"Processing image file: {document_obj}")

                result = self._process_single_image(document_obj)
                page_quality_metrics.append(result.get("quality_metrics", {}))

                if result.get("text") and result["text"] not in [
                    "NO_TEXT_FOUND",
                    "TEXT_EXTRACTION_FAILED",
                ]:
                    all_text.append(result["text"])
                    self.stats.text_extractions_successful += 1

                if (
                    result.get("description")
                    and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                ):
                    all_descriptions.append(result["description"])
                    self.stats.descriptions_successful += 1

                self.stats.pages_processed += 1

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Compile results
            final_text = (
                "\n\n".join(all_text) if all_text else "No text found in document."
            )
            final_descriptions = (
                "\n\n".join(all_descriptions)
                if all_descriptions
                else "No visual content described."
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.documents_processed += 1

            # Update average metrics
            if all_text:
                avg_text_length = sum(len(text) for text in all_text) / len(all_text)
                self.stats.average_text_length = (
                    self.stats.average_text_length
                    * (self.stats.documents_processed - 1)
                    + avg_text_length
                ) / self.stats.documents_processed

            if all_descriptions:
                avg_desc_length = sum(len(desc) for desc in all_descriptions) / len(
                    all_descriptions
                )
                self.stats.average_description_length = (
                    self.stats.average_description_length
                    * (self.stats.documents_processed - 1)
                    + avg_desc_length
                ) / self.stats.documents_processed

            # Monitor memory after processing
            memory_after = self._monitor_memory_usage()
            self.logger.debug(
                f"Memory after processing: GPU {memory_after.get('gpu_memory_percent', 0):.1f}%, "
                f"RAM {memory_after.get('system_ram_percent', 0):.1f}%"
            )

            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            self.logger.info(
                f"Results - Text: {len(final_text)} chars, Descriptions: {len(final_descriptions)} chars"
            )

            return {
                "all_text": final_text,
                "all_imagery": final_descriptions,
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {document_obj}: {e}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            "documents_processed": self.stats.documents_processed,
            "pages_processed": self.stats.pages_processed,
            "text_extractions_successful": self.stats.text_extractions_successful,
            "descriptions_successful": self.stats.descriptions_successful,
            "total_processing_time": self.stats.total_processing_time,
            "memory_refreshes": self.stats.memory_refreshes,
            "model_switches": self.stats.model_switches,
            "current_model": {
                "name": self.current_model_spec.name,
                "model_id": self.current_model_spec.model_id,
                "quality_score": self.current_model_spec.quality_score,
            },
            "device": self.device,
            "processing_mode": self.processing_mode.value,
        }

        # Calculate rates and averages
        if self.stats.documents_processed > 0:
            stats["avg_processing_time_per_doc"] = (
                self.stats.total_processing_time / self.stats.documents_processed
            )
            stats["text_extraction_success_rate"] = (
                self.stats.text_extractions_successful / self.stats.pages_processed
                if self.stats.pages_processed > 0
                else 0
            )
            stats["description_success_rate"] = (
                self.stats.descriptions_successful / self.stats.pages_processed
                if self.stats.pages_processed > 0
                else 0
            )
            stats["average_text_length"] = self.stats.average_text_length
            stats["average_description_length"] = self.stats.average_description_length

        # Add current memory usage
        stats["memory_usage"] = self._monitor_memory_usage()

        # Hardware constraints
        stats["hardware_constraints"] = {
            "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
            "max_ram_gb": self.hardware_constraints.max_ram_gb,
            "force_cpu": self.hardware_constraints.force_cpu,
        }

        return stats

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of models that fit current hardware constraints"""
        available_models = []
        for model_spec in self.model_registry:
            fits = self._model_fits_constraints(model_spec)
            available_models.append(
                {
                    "name": model_spec.name,
                    "model_id": model_spec.model_id,
                    "quality_score": model_spec.quality_score,
                    "min_gpu_vram_gb": model_spec.min_gpu_vram_gb,
                    "min_ram_gb": model_spec.min_ram_gb,
                    "fits_constraints": fits,
                    "is_current": model_spec == self.current_model_spec,
                }
            )
        return available_models

    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        stats = self.get_processing_stats()
        suggestions = []

        # Check success rates
        text_success_rate = stats.get("text_extraction_success_rate", 0)
        desc_success_rate = stats.get("description_success_rate", 0)

        if text_success_rate < 0.8:
            suggestions.append(
                "Consider switching to a higher quality model for better text extraction"
            )

        if desc_success_rate < 0.8:
            suggestions.append(
                "Consider adjusting processing mode to HIGH_QUALITY for better descriptions"
            )

        # Check processing speed
        avg_time = stats.get("avg_processing_time_per_doc", 0)
        if avg_time > 30:
            suggestions.append(
                "Processing is slow - consider using FAST mode or switching to a smaller model"
            )

        # Check memory usage
        memory_usage = stats.get("memory_usage", {})
        gpu_percent = memory_usage.get("gpu_memory_percent", 0)
        if gpu_percent > 90:
            suggestions.append(
                "GPU memory usage is very high - consider reducing refresh interval or using CPU mode"
            )

        return {
            "current_performance": stats,
            "optimization_suggestions": suggestions,
            "available_models": self.get_available_models(),
        }
