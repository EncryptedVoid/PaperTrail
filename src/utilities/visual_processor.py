import logging
import random
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from utilities.dependancy_ensurance import ensure_poppler


class VisualProcessor:
    """
    Qwen2.5-VL processor for PDF and image text extraction and description.

    Processes documents with intelligent page sampling for large PDFs.
    Sends multiple images in a single model call for better context.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_id: str = "Qwen/Qwen2.5-VL-2B-Instruct",
        max_pages: int = 6,
        max_tokens: int = 512,
    ) -> None:
        """
        Initialize processor.

        Args:
                        logger: Logger instance
                        model_id: HuggingFace model identifier
                        max_pages: Max pages to process (first, last, + random middle pages)
                        max_tokens: Max tokens to generate per inference
        """
        ensure_poppler()

        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger, got {type(logger)}")

        self.logger = logger
        self.max_pages = max_pages
        self.max_tokens = max_tokens

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Load model
        self.logger.info(f"Loading {model_id}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.logger.info("Model loaded")

    def _load_images(
        self, file_path: Path, subset_size: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Load images from PDF or image file with page sampling.

        For PDFs exceeding max_pages:
        - Always includes first and last page
        - Randomly samples remaining pages from middle

        Args:
                        file_path: Path to file
                        subset_size: Override max_pages for this file

        Returns:
                        List of PIL Images
        """
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            # Convert PDF to images
            self.logger.info(f"Converting PDF: {file_path.name}")
            images = convert_from_path(str(file_path))
            total = len(images)
            max_allowed = subset_size if subset_size else self.max_pages

            # Sample if needed
            if total > max_allowed:
                # First and last pages
                indices = [0, total - 1]

                # Random middle pages
                middle_count = max_allowed - 2
                if middle_count > 0 and total > 2:
                    middle = list(range(1, total - 1))
                    indices.extend(
                        random.sample(middle, min(middle_count, len(middle)))
                    )

                indices.sort()
                images = [images[i] for i in indices]
                self.logger.info(f"Sampled {len(images)} of {total} pages: {indices}")
            else:
                self.logger.info(f"Processing all {total} pages")

            return images

        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return [Image.open(file_path)]

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _generate(self, images: List[Image.Image], prompt: str) -> str:
        """
        Generate text from images using Qwen2.5-VL.

        Args:
                        images: List of PIL Images
                        prompt: Text prompt

        Returns:
                        Generated text
        """
        # Build message with all images
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_tokens
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output.strip()

    def extract_text(self, file_path: Path, subset_size: Optional[int] = None) -> str:
        """
        Extract text from document.

        Args:
                        file_path: Path to PDF or image
                        subset_size: Override max_pages

        Returns:
                        Extracted text
        """
        self.logger.info(f"Extracting text: {file_path.name}")
        images = self._load_images(file_path, subset_size)

        prompt = (
            "Extract all text from these images. "
            "If multiple pages, extract text from each page separately. "
            "Return only the text content with clear page separations."
        )

        return self._generate(images, prompt)

    def extract_visual_description(
        self, file_path: Path, subset_size: Optional[int] = None
    ) -> str:
        """
        Generate visual descriptions of document.

        Args:
                        file_path: Path to PDF or image
                        subset_size: Override max_pages

        Returns:
                        Visual descriptions
        """
        self.logger.info(f"Describing: {file_path.name}")
        images = self._load_images(file_path, subset_size)

        prompt = (
            "Describe the visual elements in these images. "
            "Include layout, colors, text formatting, charts, diagrams. "
            "If multiple pages, describe each separately."
        )

        return self._generate(images, prompt)

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, "model"):
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Cleaned up")
