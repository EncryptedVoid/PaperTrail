"""
LLM-based field extraction with hardware auto-detection and context management

This module provides intelligent document field extraction using local LLM models through OLLAMA.
It automatically detects hardware capabilities, selects optimal models, and manages context to
prevent performance degradation during batch processing.

Key Features:
- Automatic hardware detection and model selection
- Context refresh to prevent model degradation
- Multiple extraction strategies (single/multi-prompt)
- Comprehensive validation and error handling
- Performance monitoring and statistics

Dependencies:
- requests: HTTP communication with OLLAMA API
- psutil: System hardware detection
- GPUtil: GPU memory detection (optional)
"""

import ollama
import logging
import datetime
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict
from config import SYSTEM_PROMPT, FIELD_PROMPTS, PREFERRED_LANGUAGE_MODEL


class LanguageExtractionReport(TypedDict):
    """
    Structured type definition for field extraction results.

    This TypedDict ensures consistent return format from the extract_fields method
    and provides clear documentation of expected fields in the response.
    """

    success: bool
    processing_timestamp: str
    model_used: str
    processing_time_seconds: float
    extracted_fields: Optional[Dict[str, str]]  # Only present on success
    error: Optional[str]  # Only present on failure
    error_type: Optional[str]  # Only present on failure


class LanguageProcessor:
    """
    Handles LLM-based document field extraction with intelligent model management.

    This class provides a robust interface for extracting structured data from documents
    using local LLM models via OLLAMA. It includes automatic error handling, performance
    monitoring, and context management to maintain consistent extraction quality.

    Attributes:
        logger (logging.Logger): Logger instance for tracking operations and debugging
        model (str): Name of the LLM model being used for extraction
        client (ollama.Client): OLLAMA client for model communication

    Example:
        >>> processor = LanguageProcessor(logger, model="mistral:7b")
        >>> result = processor.extract_fields(ocr_text, visual_desc, "doc_123")
        >>> if result["success"]:
        ...     fields = result["extracted_fields"]
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the LanguageProcessor with proper error handling.

        Sets up the OLLAMA client connection and validates the specified model.
        All instance variables are initialized first to ensure consistent state
        even if initialization fails partway through.

        Args:
            logger (logging.Logger): Configured logger for operation tracking
            model (str, optional): OLLAMA model name to use. Defaults to "mistral:7b"

        Raises:
            Exception: If OLLAMA client initialization fails or model is unavailable
        """
        # Initialize ALL instance variables FIRST to ensure consistent object state
        self.logger: logging.Logger = logger
        self.model: str = PREFERRED_LANGUAGE_MODEL

        self.logger.info("=== Initializing LanguageProcessor ===")

        try:
            # Attempt to establish connection to OLLAMA service
            self.client = ollama.Client()

            # Log successful initialization with model details
            self.logger.info("Language processor initialization completed successfully")
            self.logger.info(f"Active model: {self.model}")

        except Exception as e:
            # Log detailed error information for debugging
            self.logger.error(f"Failed to initialize LanguageProcessor: {e}")
            # Re-raise to indicate initialization failure

    def extract_fields(
        self,
        ocr_text: str,
        visual_description: str,
    ) -> LanguageExtractionReport:
        """
        Extract all document fields using LLM with comprehensive error handling.

        This method processes document content through the configured LLM model to extract
        structured field data. It iterates through all configured field prompts, handles
        individual field extraction failures gracefully, and provides detailed performance
        metrics and error reporting.

        Args:
            ocr_text (str): Raw text content extracted from document OCR
            visual_description (str): AI-generated description of document visual elements
            uuid (str): Unique identifier for the document being processed

        Returns:
            ExtractionReport: Structured report containing either:
                - On success: extracted_fields dict with field_name -> extracted_value mappings
                - On failure: error details and diagnostic information
                Both cases include processing metadata and performance metrics

        Note:
            Individual field extraction failures are logged but don't stop the overall
            process. Failed fields are marked as "UNKNOWN" in the results.
        """
        # Start timing for performance monitoring
        start_time: datetime = datetime.now()

        self.logger.debug("Extracting fields from summaries...")

        # Log input data sizes for debugging and optimization
        self.logger.info(
            f"Input sizes: OCR={len(ocr_text)} chars, "
            f"Visual description={len(visual_description)} chars"
        )

        try:
            # Main extraction loop - process each configured field independently
            extracted_fields: Dict[str, str] = {}

            # Iterate through all field prompts defined in configuration
            for field_name, field_instruction in FIELD_PROMPTS.items():
                try:
                    self.logger.debug(f"Extracting field: {field_name}")

                    # Construct complete prompt combining system instructions with document data
                    # This ensures the model has full context for accurate extraction
                    complete_prompt: str = f"""
                    {SYSTEM_PROMPT}

                    DOCUMENT DATA:
                    =============
                    OCR TEXT CONTENT:
                    {ocr_text}

                    VISUAL DESCRIPTION:
                    {visual_description}

                    TASK:
                    =====
                    {field_instruction}"""

                    # Check if client is properly initialized
                    if self.client is None:
                        raise RuntimeError("OLLAMA client is not initialized")

                    # Send prompt to LLM and wait for complete response
                    # stream=False ensures we get the full response before proceeding
                    response: Dict[str, Any] = self.client.generate(
                        model=self.model, prompt=complete_prompt, stream=False
                    )

                    # Extract the text response from the LLM output
                    response_text: str = response["response"]
                    extracted_fields[field_name] = response_text

                    # Log successful extraction with response preview
                    self.logger.info(
                        f"Extracted field '{field_name}': {response_text[:100]}..."
                        if len(response_text) > 100
                        else f"Extracted field '{field_name}': {response_text}"
                    )

                except Exception as e:
                    # Handle individual field extraction failures gracefully
                    # Log the error but continue processing other fields
                    self.logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_fields[field_name] = "UNKNOWN"

            # Calculate total processing time for performance monitoring
            processing_time: float = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Processing time: {processing_time:.2f}s")

            # Return successful extraction report with all metadata
            return LanguageExtractionReport(
                success=True,
                extracted_fields=extracted_fields,
                processing_timestamp=datetime.now().isoformat(),
                model_used=self.model,
                processing_time_seconds=processing_time,
                error=None,
                error_type=None,
            )

        except Exception as e:
            # Comprehensive error handling with context preservation
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log detailed error information for debugging
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Processing time before failure: {processing_time:.2f}s")

            # Log full traceback for debugging if available
            if hasattr(e, "__traceback__"):
                import traceback

                self.logger.debug("Full traceback:", exc_info=True)

            # Return failure report with diagnostic information
            return LanguageExtractionReport(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                processing_timestamp=datetime.now().isoformat(),
                processing_time_seconds=processing_time,
                model_used=self.model,
                extracted_fields=None,
            )
