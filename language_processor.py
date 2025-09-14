"""
Enhanced LLM-based field extraction with hardware auto-detection and context management
"""

import requests
import json
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class ExtractionMode(Enum):
    SINGLE_PROMPT = "single"
    MULTI_PROMPT = "multi"
    AUTO = "auto"


@dataclass
class ModelSpec:
    name: str
    min_ram_gb: float
    min_gpu_vram_gb: float
    min_cpu_cores: int
    context_window: int
    quality_score: int  # 1-10, higher is better


@dataclass
class HardwareConstraints:
    max_ram_gb: float
    max_gpu_vram_gb: float
    max_cpu_cores: int


class LanguageProcessor:
    def __init__(
        self,
        logger: logging.Logger,
        max_ram_gb: Optional[float] = None,
        max_gpu_vram_gb: Optional[float] = None,
        max_cpu_cores: Optional[int] = None,
        context_refresh_interval: int = 50,
        host: str = "http://localhost:11434",
        timeout: int = 300,
        extraction_mode: ExtractionMode = ExtractionMode.SINGLE_PROMPT,
        auto_model_selection: bool = True,
    ):
        self.logger = logger
        self.host = host
        self.timeout = timeout
        self.extraction_mode = extraction_mode
        self.context_refresh_interval = context_refresh_interval
        self.processed_count = 0
        self.prompts_since_refresh = 0

        # Initialize hardware constraints
        self.hardware_constraints = self._detect_hardware_constraints(
            max_ram_gb, max_gpu_vram_gb, max_cpu_cores
        )

        # Initialize model registry
        self.model_registry = self._initialize_model_registry()

        # Select and load optimal model
        if auto_model_selection:
            self.model = self._select_optimal_model()
            self._load_model()
        else:
            self.model = "llama2"  # Default fallback

        self.logger.info(f"Initialized with model: {self.model}")
        self.logger.info(f"Hardware constraints: {self.hardware_constraints}")

    def _detect_hardware_constraints(
        self,
        max_ram_gb: Optional[float],
        max_gpu_vram_gb: Optional[float],
        max_cpu_cores: Optional[int],
    ) -> HardwareConstraints:
        """Detect available hardware or use provided constraints"""

        # RAM detection
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            available_ram = total_ram * 0.7  # Use 70% of total RAM
        else:
            available_ram = max_ram_gb

        # CPU detection
        if max_cpu_cores is None:
            available_cores = (
                psutil.cpu_count(logical=False) or 1
            )  # Physical cores, default to 1
        else:
            available_cores = max_cpu_cores

        # GPU detection (simplified - would need more sophisticated detection)
        if max_gpu_vram_gb is None:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                available_gpu = (
                    max([gpu.memoryTotal / 1024 for gpu in gpus]) if gpus else 0
                )
            except ImportError:
                self.logger.warning("GPUtil not available, assuming no GPU")
                available_gpu = 0
        else:
            available_gpu = max_gpu_vram_gb

        return HardwareConstraints(
            max_ram_gb=available_ram,
            max_gpu_vram_gb=available_gpu,
            max_cpu_cores=available_cores,
        )

    def _initialize_model_registry(self) -> List[ModelSpec]:
        """Initialize registry of available models with their requirements"""
        return [
            ModelSpec("llama3.2:1b", 2, 0, 2, 2048, 6),
            ModelSpec("llama3.2:3b", 4, 0, 4, 2048, 7),
            ModelSpec("llama3.1:8b", 8, 0, 6, 4096, 8),
            ModelSpec("llama3.1:70b", 32, 24, 16, 8192, 10),
            ModelSpec("qwen2.5:7b", 7, 0, 4, 4096, 8),
            ModelSpec("qwen2.5:14b", 14, 8, 8, 8192, 9),
            ModelSpec("mistral:7b", 7, 0, 4, 4096, 7),
            ModelSpec("gemma2:9b", 9, 6, 6, 8192, 8),
            ModelSpec("codellama:13b", 13, 8, 8, 4096, 8),
            # Add more models as needed
        ]

    def _select_optimal_model(self) -> str:
        """Select the best model that fits within hardware constraints"""
        suitable_models = []

        for model in self.model_registry:
            if (
                model.min_ram_gb <= self.hardware_constraints.max_ram_gb
                and model.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
                and model.min_cpu_cores <= self.hardware_constraints.max_cpu_cores
            ):
                suitable_models.append(model)

        if not suitable_models:
            self.logger.warning("No models fit hardware constraints, using llama3.2:1b")
            return "llama3.2:1b"

        # Select model with highest quality score
        best_model = max(suitable_models, key=lambda m: m.quality_score)
        self.logger.info(
            f"Selected optimal model: {best_model.name} (quality: {best_model.quality_score})"
        )

        return best_model.name

    def _load_model(self):
        """Load the selected model in OLLAMA"""
        try:
            # Check if model is already loaded
            response = requests.get(f"{self.host}/api/tags", timeout=30)
            if response.status_code == 200:
                loaded_models = [
                    model["name"] for model in response.json().get("models", [])
                ]
                if self.model in loaded_models:
                    self.logger.info(f"Model {self.model} already available")
                    return

            # Pull model if not available
            self.logger.info(f"Loading model {self.model}...")
            pull_payload = {"name": self.model}
            response = requests.post(
                f"{self.host}/api/pull", json=pull_payload, timeout=600
            )

            if response.status_code == 200:
                self.logger.info(f"Successfully loaded model {self.model}")
            else:
                self.logger.error(f"Failed to load model {self.model}: {response.text}")

        except Exception as e:
            self.logger.error(f"Error loading model {self.model}: {e}")

    def _refresh_context(self):
        """Refresh the model context to prevent degradation"""
        try:
            self.logger.info("Refreshing model context...")

            # Send a context reset prompt
            reset_payload = {
                "model": self.model,
                "prompt": "Reset context. Ready for new document analysis.",
                "stream": False,
                "options": {"num_predict": 10},
            }

            response = requests.post(
                f"{self.host}/api/generate", json=reset_payload, timeout=30
            )
            if response.status_code == 200:
                self.prompts_since_refresh = 0
                self.logger.info("Context refreshed successfully")
            else:
                self.logger.warning(f"Context refresh failed: {response.text}")

        except Exception as e:
            self.logger.error(f"Error refreshing context: {e}")

    def extract_fields(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        uuid: str,
    ) -> Dict[str, Any]:
        """
        Extract all document fields using LLM with JSON template

        Args:
            ocr_text: Text extracted from OCR
            visual_description: Visual description from image processing
            metadata: Technical metadata (file size, type, etc.)
            uuid: Document UUID for logging

        Returns:
            Dictionary with extracted fields or error info
        """
        self.logger.info(f"Extracting fields for document {uuid}")

        # Check if context refresh is needed
        if self.prompts_since_refresh >= self.context_refresh_interval:
            self._refresh_context()

        try:
            if self.extraction_mode == ExtractionMode.MULTI_PROMPT:
                extracted_fields = self._extract_multi_prompt(
                    ocr_text, visual_description, metadata, uuid
                )
            elif self.extraction_mode == ExtractionMode.AUTO:
                # Try single prompt first, fallback to multi if it fails
                try:
                    extracted_fields = self._extract_single_prompt(
                        ocr_text, visual_description, metadata, uuid
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Single prompt failed for {uuid}, trying multi-prompt: {e}"
                    )
                    extracted_fields = self._extract_multi_prompt(
                        ocr_text, visual_description, metadata, uuid
                    )
            else:  # SINGLE_PROMPT
                extracted_fields = self._extract_single_prompt(
                    ocr_text, visual_description, metadata, uuid
                )

            # Validate and clean extracted fields
            validated_fields = self._validate_extracted_fields(extracted_fields, uuid)

            self.processed_count += 1
            self.prompts_since_refresh += 1
            self.logger.info(f"Successfully extracted fields for {uuid}")

            return {
                "success": True,
                "extracted_fields": validated_fields,
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "extraction_mode": self.extraction_mode.value,
            }

        except Exception as e:
            self.logger.error(f"Field extraction failed for {uuid}: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_fields": self._get_default_template(),
                "processing_timestamp": datetime.now().isoformat(),
            }

    def _extract_single_prompt(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        uuid: str,
    ) -> Dict[str, str]:
        """Extract all fields in a single prompt"""
        template = self._get_default_template()
        prompt = self._build_extraction_prompt(
            ocr_text, visual_description, metadata, template
        )
        response = self._send_to_ollama(prompt)
        return self._parse_llm_response(response, uuid)

    def _extract_multi_prompt(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        uuid: str,
    ) -> Dict[str, str]:
        """Extract fields using multiple focused prompts"""
        extracted_fields = {}

        # Group related fields for focused extraction
        field_groups = {
            "basic_info": ["title", "document_type", "current_language"],
            "people": ["translator_name", "issuer_name", "officiater_name"],
            "dates": [
                "date_created",
                "date_of_reception",
                "date_of_issue",
                "date_of_expiry",
            ],
            "classification": ["confidentiality_level", "tags"],
            "notes": ["version_notes", "utility_notes", "additional_notes"],
        }

        for group_name, fields in field_groups.items():
            try:
                group_template = {field: "UNKNOWN" for field in fields}
                prompt = self._build_focused_extraction_prompt(
                    ocr_text, visual_description, metadata, group_template, group_name
                )
                response = self._send_to_ollama(prompt)
                group_results = self._parse_llm_response(
                    response, f"{uuid}_{group_name}"
                )
                extracted_fields.update(group_results)

            except Exception as e:
                self.logger.warning(f"Failed to extract {group_name} for {uuid}: {e}")
                for field in fields:
                    extracted_fields[field] = "UNKNOWN"

        return extracted_fields

    def _build_focused_extraction_prompt(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        template: Dict[str, str],
        group_name: str,
    ) -> str:
        """Build a focused prompt for extracting a specific group of fields"""

        group_descriptions = {
            "basic_info": "basic document information like title, type, and language",
            "people": "names of people or organizations involved",
            "dates": "all dates mentioned in the document",
            "classification": "document classification and categorization",
            "notes": "additional notes and version information",
        }

        prompt = f"""You are a document analysis expert. Extract {group_descriptions.get(group_name, 'specific information')} from this document.

DOCUMENT INFORMATION:
===================

OCR TEXT:
{ocr_text[:2000]}

VISUAL DESCRIPTION:
{visual_description[:500]}

FILE METADATA:
- Filename: {metadata.get('filename', 'unknown')}
- File type: {metadata.get('extension', 'unknown')}

INSTRUCTIONS:
============
Focus only on extracting the following fields and return ONLY valid JSON:

{json.dumps(template, indent=2)}

Return ONLY the JSON object, no explanations. Use "UNKNOWN" for any field you cannot determine.

JSON:"""

        return prompt

    def _get_default_template(self) -> Dict[str, str]:
        """Get the default field template"""
        return {
            "title": "UNKNOWN",
            "document_type": "UNKNOWN",
            "current_language": "UNKNOWN",
            "confidentiality_level": "UNKNOWN",
            "translator_name": "UNKNOWN",
            "issuer_name": "UNKNOWN",
            "officiater_name": "UNKNOWN",
            "date_created": "UNKNOWN",
            "date_of_reception": "UNKNOWN",
            "date_of_issue": "UNKNOWN",
            "date_of_expiry": "UNKNOWN",
            "tags": "UNKNOWN",
            "version_notes": "UNKNOWN",
            "utility_notes": "UNKNOWN",
            "additional_notes": "UNKNOWN",
        }

    def _validate_extracted_fields(
        self, fields: Dict[str, str], uuid: str
    ) -> Dict[str, str]:
        """Validate and clean extracted fields"""
        validated = {}

        for key, value in fields.items():
            # Clean empty values
            if not value or value.strip() == "":
                validated[key] = "UNKNOWN"
            else:
                validated[key] = str(value).strip()

        # Validate date formats
        date_fields = [
            "date_created",
            "date_of_reception",
            "date_of_issue",
            "date_of_expiry",
        ]
        for date_field in date_fields:
            if date_field in validated and validated[date_field] != "UNKNOWN":
                validated[date_field] = self._validate_date_format(
                    validated[date_field]
                )

        # Ensure all required fields exist
        template = self._get_default_template()
        for required_field in template.keys():
            if required_field not in validated:
                validated[required_field] = "UNKNOWN"
                self.logger.warning(f"Missing field '{required_field}' for {uuid}")

        return validated

    def _validate_date_format(self, date_str: str) -> str:
        """Validate and normalize date format to YYYY-MM-DD"""
        if date_str == "UNKNOWN":
            return date_str

        # Try to parse various date formats
        import re
        from datetime import datetime

        # Common date patterns
        patterns = [
            r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
            r"(\d{1,2})-(\d{1,2})-(\d{4})",  # MM-DD-YYYY
        ]

        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if len(match.group(1)) == 4:  # Year first
                        year, month, day = match.groups()
                    else:  # Month/day first
                        month, day, year = match.groups()

                    # Validate date
                    datetime(int(year), int(month), int(day))
                    return f"{year.zfill(4)}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    continue

        # If no valid date found, return original or UNKNOWN
        return "UNKNOWN"

    def _build_extraction_prompt(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        template: Dict[str, str],
    ) -> str:
        """Build the extraction prompt with all context"""

        prompt = f"""You are a document analysis expert. Extract specific information from this document and return it as JSON.

DOCUMENT INFORMATION:
===================

OCR TEXT:
{ocr_text[:2000]}  # Limit text to prevent token overflow

VISUAL DESCRIPTION:
{visual_description[:500]}

FILE METADATA:
- Filename: {metadata.get('filename', 'unknown')}
- File type: {metadata.get('extension', 'unknown')}
- File size: {metadata.get('size_mb', 0)}MB

INSTRUCTIONS:
============
Extract the following information and return ONLY valid JSON in exactly this format:

{json.dumps(template, indent=2)}

FIELD DEFINITIONS:
- title: The main title or name of the document
- document_type: Type like "certificate", "invoice", "contract", "letter", "report", etc.
- current_language: Language the document is written in (e.g., "English", "Spanish")
- confidentiality_level: "Public", "Internal", "Confidential", "Restricted", or "UNKNOWN"
- translator_name: Name of translator if this is a translated document
- issuer_name: Organization or person who issued/created this document
- officiater_name: Official body or authority that validated/certified this document
- date_created: When document was originally created (YYYY-MM-DD format)
- date_of_reception: When you/organization received this document (YYYY-MM-DD format)
- date_of_issue: Official issue/publication date (YYYY-MM-DD format)
- date_of_expiry: Expiration date if applicable (YYYY-MM-DD format)
- tags: Comma-separated keywords describing the document content
- version_notes: Any version information or revision notes
- utility_notes: How this document is typically used or its purpose
- additional_notes: Any other relevant information

IMPORTANT RULES:
- Return ONLY the JSON object, no explanations
- Use "UNKNOWN" for any field you cannot determine
- For dates, use YYYY-MM-DD format or "UNKNOWN"
- Be specific and accurate - don't guess
- Extract actual names and organizations, not generic terms

JSON:"""

        return prompt

    def _send_to_ollama(self, prompt: str) -> str:
        """Send prompt to OLLAMA and get response"""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
                "num_predict": 1000,  # Limit response length
            },
        }

        self.logger.debug(f"Sending request to OLLAMA: {url}")

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _parse_llm_response(self, response: str, uuid: str) -> Dict[str, str]:
        """Parse and validate the LLM JSON response"""

        # Clean the response - remove any markdown formatting
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            # Parse JSON
            parsed = json.loads(cleaned)

            # Validate all required fields are present
            template = self._get_default_template()

            # Ensure all fields exist (add UNKNOWN for missing ones)
            for field in template.keys():
                if field not in parsed:
                    parsed[field] = "UNKNOWN"
                    self.logger.warning(
                        f"Missing field '{field}' for {uuid}, set to UNKNOWN"
                    )

            # Clean up any empty strings
            for key, value in parsed.items():
                if value == "" or value is None:
                    parsed[key] = "UNKNOWN"

            self.logger.debug(f"Successfully parsed {len(parsed)} fields for {uuid}")
            return parsed

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response for {uuid}: {e}")
            self.logger.error(f"Raw response: {response[:200]}...")

            # Return template with UNKNOWN values if parsing fails
            return self._get_default_template()

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_processed": self.processed_count,
            "prompts_since_refresh": self.prompts_since_refresh,
            "model": self.model,
            "host": self.host,
            "hardware_constraints": {
                "max_ram_gb": self.hardware_constraints.max_ram_gb,
                "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
                "max_cpu_cores": self.hardware_constraints.max_cpu_cores,
            },
            "context_refresh_interval": self.context_refresh_interval,
            "extraction_mode": self.extraction_mode.value,
        }

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            old_model = self.model
            self.model = model_name
            self._load_model()
            self._refresh_context()
            self.logger.info(f"Switched from {old_model} to {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to switch to model {model_name}: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of models that fit current hardware constraints"""
        suitable_models = []
        for model in self.model_registry:
            if (
                model.min_ram_gb <= self.hardware_constraints.max_ram_gb
                and model.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
                and model.min_cpu_cores <= self.hardware_constraints.max_cpu_cores
            ):
                suitable_models.append(model.name)
        return suitable_models
