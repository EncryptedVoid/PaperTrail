import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz
from PIL import Image
import io
import logging
import gc
from enum import Enum
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path


class LLM_Models(Enum):
    """Available LLM model configurations"""

    QWEN2_VL_7B = "Qwen/Qwen2-VL-7B-Instruct"


class QwenDocumentProcessor:
    # Class attributes with type hints
    logger: logging.Logger
    device: str
    model: Optional[Qwen2VLForConditionalGeneration]
    processor: Optional[AutoProcessor]
    model_name: LLM_Models
    documents_processed: int
    refresh_interval: int  # Measured in number of files
    memory_threshold: float = 75.0

    def __init__(
        self,
        logger: logging.Logger,
        model_name: LLM_Models = LLM_Models.QWEN2_VL_7B,
        refresh_interval: int = 5,
    ) -> None:
        # Validate required logger parameter
        if logger is None:
            raise ValueError(
                "Logger is required - QwenDocumentProcessor cannot be initialized without a logger"
            )

        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger instance, got {type(logger)}")

        # Initialize instance attributes
        self.logger: logging.Logger = logger
        self.model_name = model_name
        self.documents_processed = 0
        self.model = None
        self.processor = None
        self.refresh_interval = refresh_interval

        self.logger.info(
            f"Initializing QwenDocumentProcessor with model: {model_name.value}"
        )

        # Load the model
        self._load_model()

    def _load_model(self) -> None:
        """Load or reload the Qwen2-VL model and processor"""
        self.logger.info("Loading Qwen2-VL model...")

        try:
            # Clean up existing model if present
            if self.model is not None:
                self.logger.debug("Cleaning up existing model...")
                del self.model
                del self.processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Determine compute device (CUDA vs CPU)
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")

            # Log GPU information if available
            if torch.cuda.is_available():
                gpu_name: str = torch.cuda.get_device_name(0)
                gpu_memory: float = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                self.logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")

            self.logger.debug("Loading Qwen2VL model from transformers...")
            # Load the vision-language model with optimized settings for inference
            self.model: Qwen2VLForConditionalGeneration = (
                Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name.value,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
                    device_map="auto",  # Automatically distribute model across available devices
                )
            )
            self.logger.info("Qwen2VL model loaded successfully")

            self.logger.debug("Loading AutoProcessor...")
            # Load the processor for handling text and image inputs
            self.processor: AutoProcessor = AutoProcessor.from_pretrained(
                self.model_name.value
            )
            self.logger.info("AutoProcessor loaded successfully")

            self.logger.info(
                f"QwenDocumentProcessor model loading complete on {self.device}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load QwenDocumentProcessor model: {e}")
            raise

    def refresh_model(self) -> None:
        """Refresh the model to clear context and free memory"""
        self.logger.info("Refreshing AI model context...")
        try:
            self._load_model()
            self.documents_processed = 0
            self.logger.info("Model refresh completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to refresh model: {e}")
            raise

    def _should_refresh_model(self) -> bool:
        """Check if model should be refreshed based on usage"""
        # Refresh based on config interval
        if self.documents_processed >= self.refresh_interval:
            return True

        # Check GPU memory if available
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_percent = (memory_used / memory_total) * 100

            if memory_percent > self.memory_threshold:
                self.logger.warning(
                    f"High GPU memory usage detected: {memory_percent:.1f}%"
                )
                return True

        return False

    def _pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        # Convert PDF pages to PIL Image objects for processing
        pdf_path_obj: Path = Path(pdf_path)
        self.logger.info(f"Converting PDF to images: {pdf_path_obj}")

        # Validate file existence
        if not pdf_path_obj.exists():
            self.logger.error(f"PDF file not found: {pdf_path_obj}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        try:
            self.logger.debug("Opening PDF with PyMuPDF")
            # Open PDF document using PyMuPDF
            doc = fitz.open(str(pdf_path_obj))  # type: ignore[attr-defined]
            images: List[Image.Image] = []
            page_count: int = len(doc)

            self.logger.info(f"PDF has {page_count} pages to convert")

            # Process each page individually
            for page_num in range(page_count):
                self.logger.debug(f"Converting page {page_num + 1}/{page_count}")

                try:
                    # Get specific page from document
                    page = doc[page_num]  # type: ignore[index]

                    # Create transformation matrix for higher resolution (2x zoom)
                    mat: fitz.Matrix = fitz.Matrix(2.0, 2.0)  # type: ignore[attr-defined]

                    # Render page as pixmap with enhanced resolution
                    pix = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]

                    # Convert pixmap to PNG bytes
                    img_data: bytes = pix.tobytes("png")  # type: ignore[attr-defined]

                    # Create PIL Image from bytes
                    img: Image.Image = Image.open(io.BytesIO(img_data))
                    images.append(img)

                    self.logger.debug(
                        f"Page {page_num + 1} converted successfully - size: {img.size}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    # Continue processing remaining pages even if one fails
                    continue

            # Clean up document resources
            doc.close()  # type: ignore[attr-defined]
            self.logger.info(
                f"Successfully converted {len(images)}/{page_count} pages to images"
            )
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
        # Process individual image for text extraction and/or visual description
        self.logger.debug("Processing single image")

        try:
            # Load image from file path if string/Path provided
            processed_image: Image.Image
            if isinstance(image, (str, Path)):
                image_path: Path = Path(image)
                self.logger.debug(f"Loading image from path: {image_path}")

                # Validate image file exists
                if not image_path.exists():
                    self.logger.error(f"Image file not found: {image_path}")
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                processed_image = Image.open(image_path)
                self.logger.debug(
                    f"Image loaded successfully - size: {processed_image.size}, mode: {processed_image.mode}"
                )
            else:
                # Image is already a PIL Image object
                processed_image = image

            # Initialize results dictionary
            results: Dict[str, Any] = {}

            # Text extraction branch
            if extract_text:
                self.logger.debug("Extracting text from image")
                try:
                    # Prepare messages for text extraction task
                    text_messages: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image. Return only the text content, no explanations or formatting. If there's no text, return 'NO_TEXT_FOUND'.",
                                },
                            ],
                        }
                    ]

                    self.logger.debug("Preparing text extraction inputs")
                    # Apply chat template to format messages properly
                    text_inputs: str = self.processor.apply_chat_template(
                        text_messages, tokenize=False, add_generation_prompt=True
                    )

                    # Process vision information (images/videos)
                    image_inputs: Any
                    video_inputs: Any
                    image_inputs, video_inputs = process_vision_info(text_messages)

                    # Tokenize and prepare inputs for model
                    inputs = self.processor(
                        text=[text_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Move inputs to appropriate device (GPU/CPU)
                    inputs = inputs.to(self.device)

                    self.logger.debug("Running text extraction inference")
                    # Generate text using the model
                    with torch.no_grad():  # Disable gradient computation for inference
                        generated_ids: torch.Tensor = self.model.generate(
                            **inputs, max_new_tokens=512
                        )

                        # Remove input tokens from generated output
                        generated_ids_trimmed: List[torch.Tensor] = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        # Decode generated tokens to text
                        text_output: str = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["text"] = text_output.strip()

                        # Log extraction results
                        if results["text"] == "NO_TEXT_FOUND":
                            self.logger.debug("No text found in image")
                        else:
                            self.logger.info(
                                f"Text extracted successfully - {len(results['text'])} characters"
                            )

                except Exception as e:
                    self.logger.error(f"Text extraction failed: {e}")
                    results["text"] = "TEXT_EXTRACTION_FAILED"

            # Image description branch
            if describe_image:
                self.logger.debug("Generating image description")
                try:
                    # Prepare messages for image description task
                    desc_messages: List[Dict[str, Any]] = [
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

                    self.logger.debug("Preparing image description inputs")
                    # Apply chat template for description task
                    desc_inputs: str = self.processor.apply_chat_template(
                        desc_messages, tokenize=False, add_generation_prompt=True
                    )

                    # Process vision information for description
                    image_inputs, video_inputs = process_vision_info(desc_messages)

                    # Prepare inputs for model inference
                    inputs = self.processor(
                        text=[desc_inputs],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # Move to appropriate device
                    inputs = inputs.to(self.device)

                    self.logger.debug("Running image description inference")
                    # Generate description using the model
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, max_new_tokens=512
                        )

                        # Trim input tokens from output
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        # Decode to human-readable text
                        desc_output: str = self.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        results["description"] = desc_output.strip()
                        self.logger.info(
                            f"Image description generated successfully - {len(results['description'])} characters"
                        )

                except Exception as e:
                    self.logger.error(f"Image description failed: {e}")
                    results["description"] = "DESCRIPTION_GENERATION_FAILED"

            self.logger.debug("Single image processing completed")
            return results

        except Exception as e:
            self.logger.error(f"Failed to process single image: {e}")
            raise

    def extract_article_semantics(self, document: Union[str, Path]) -> Dict[str, str]:
        # Process document file (PDF or image) and extract all text + descriptions
        document_obj: Path = Path(document)
        self.logger.info(f"Starting document processing: {document_obj}")

        # Check if model should be refreshed due to usage
        if self._should_refresh_model():
            self.logger.info(
                "Auto-refreshing model due to usage threshold or memory pressure"
            )
            self.refresh_model()

        # Validate file exists
        if not document_obj.exists():
            self.logger.error(f"Document file not found: {document_obj}")
            raise FileNotFoundError(f"Document file not found: {document_obj}")

        # Determine file type from extension
        file_ext: str = document_obj.suffix.lower()
        self.logger.info(f"Document type detected: {file_ext}")

        # Initialize result containers
        all_text: List[str] = []
        all_descriptions: List[str] = []

        try:
            # Handle PDF documents
            if file_ext == ".pdf":
                self.logger.info(f"Processing PDF document: {document_obj}")

                # Convert PDF pages to images first
                images: List[Image.Image] = self._pdf_to_images(document_obj)
                total_pages: int = len(images)

                self.logger.info(f"Processing {total_pages} pages from PDF")

                # Process each page individually
                for i, img in enumerate(images):
                    page_num: int = i + 1
                    self.logger.info(f"Processing page {page_num}/{total_pages}")

                    try:
                        # Extract text and description from current page
                        result: Dict[str, Any] = self._process_single_image(img)

                        # Handle text extraction results
                        if result.get("text") and result["text"] not in [
                            "NO_TEXT_FOUND",
                            "TEXT_EXTRACTION_FAILED",
                        ]:
                            page_text: str = (
                                f"=== Page {page_num} ===\n{result['text']}"
                            )
                            all_text.append(page_text)
                            self.logger.debug(
                                f"Page {page_num}: Text extracted successfully"
                            )
                        else:
                            self.logger.debug(
                                f"Page {page_num}: No text found or extraction failed"
                            )

                        # Handle description generation results
                        if (
                            result.get("description")
                            and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                        ):
                            page_desc: str = (
                                f"=== Page {page_num} Visual Description ===\n{result['description']}"
                            )
                            all_descriptions.append(page_desc)
                            self.logger.debug(
                                f"Page {page_num}: Description generated successfully"
                            )
                        else:
                            self.logger.debug(
                                f"Page {page_num}: Description generation failed"
                            )

                    except Exception as e:
                        self.logger.error(f"Failed to process page {page_num}: {e}")
                        # Continue with remaining pages even if one fails
                        continue

                self.logger.info(
                    f"PDF processing complete - {len(all_text)} pages with text, {len(all_descriptions)} with descriptions"
                )

            # Handle image files
            elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                self.logger.info(f"Processing image file: {document_obj}")

                try:
                    # Process single image file
                    result: Dict[str, Any] = self._process_single_image(document_obj)

                    # Handle text extraction from image
                    if result.get("text") and result["text"] not in [
                        "NO_TEXT_FOUND",
                        "TEXT_EXTRACTION_FAILED",
                    ]:
                        all_text.append(result["text"])
                        self.logger.info("Image text extraction successful")
                    else:
                        self.logger.info("No text found in image or extraction failed")

                    # Handle image description generation
                    if (
                        result.get("description")
                        and result["description"] != "DESCRIPTION_GENERATION_FAILED"
                    ):
                        all_descriptions.append(result["description"])
                        self.logger.info("Image description generation successful")
                    else:
                        self.logger.warning("Image description generation failed")

                except Exception as e:
                    self.logger.error(f"Failed to process image: {e}")
                    raise

            # Handle unsupported file types
            else:
                error_msg: str = f"Unsupported file type: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Compile final results into strings
            final_text: str = (
                "\n\n".join(all_text) if all_text else "No text found in document."
            )
            final_descriptions: str = (
                "\n\n".join(all_descriptions)
                if all_descriptions
                else "No visual content described."
            )

            # Increment processing counter
            self.documents_processed += 1

            self.logger.info(f"Document processing completed successfully")
            self.logger.info(
                f"Final results - Text: {len(final_text)} chars, Descriptions: {len(final_descriptions)} chars"
            )

            # Return structured results
            return {
                "all_text": final_text,
                "all_imagery": final_descriptions,
            }

        except Exception as e:
            self.logger.error(f"Document processing failed for {document_obj}: {e}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = {
            "documents_processed": self.documents_processed,
            "model_loaded": self.model is not None,
            "device": self.device,
        }

        if torch.cuda.is_available():
            stats["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_memory_total_gb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**3)
            stats["gpu_memory_percent"] = (
                stats["gpu_memory_used_gb"] / stats["gpu_memory_total_gb"]
            ) * 100

        return stats
