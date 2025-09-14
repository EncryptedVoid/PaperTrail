#!/usr/bin/env python3
"""
LLM-based field extraction using JSON template approach
Takes OCR text + metadata and extracts all document fields at once
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class LanguageProcessor:
    def __init__(
        self,
        logger: logging.Logger,
        model: str = "llama2",
        host: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.logger = logger
        self.model = model
        self.host = host
        self.timeout = timeout
        self.processed_count = 0

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

        # Create the JSON template the LLM should fill
        template = {
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

        # Build the prompt with all available information
        prompt = self._build_extraction_prompt(
            ocr_text, visual_description, metadata, template
        )

        try:
            # Send to OLLAMA
            response = self._send_to_ollama(prompt)

            # Parse the JSON response
            extracted_fields = self._parse_llm_response(response, uuid)

            self.processed_count += 1
            self.logger.info(f"Successfully extracted fields for {uuid}")

            return {
                "success": True,
                "extracted_fields": extracted_fields,
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
            }

        except Exception as e:
            self.logger.error(f"Field extraction failed for {uuid}: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_fields": template,  # Return template with UNKNOWN values
                "processing_timestamp": datetime.now().isoformat(),
            }

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
            required_fields = [
                "title",
                "document_type",
                "current_language",
                "confidentiality_level",
                "translator_name",
                "issuer_name",
                "officiater_name",
                "date_created",
                "date_of_reception",
                "date_of_issue",
                "date_of_expiry",
                "tags",
                "version_notes",
                "utility_notes",
                "additional_notes",
            ]

            # Ensure all fields exist (add UNKNOWN for missing ones)
            for field in required_fields:
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

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_processed": self.processed_count,
            "model": self.model,
            "host": self.host,
        }
