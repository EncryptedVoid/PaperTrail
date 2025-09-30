"""
PDF Multi-Language Translator Module
Translates PDFs to multiple languages while preserving formatting using PDFMathTranslate
"""

import os
from pathlib import Path
from typing import List , Optional

import fitz  # PyMuPDF
from pdf2zh import translate

# Language code mapping (ISO 639-3 to full names for pdf2zh)
LANGUAGE_MAP = {
    "ENG": "English",
    "ZHO": "Simplified Chinese",
    "ZHT": "Traditional Chinese",
    "SPA": "Spanish",
    "FRA": "French",
    "DEU": "German",
    "ITA": "Italian",
    "POR": "Portuguese",
    "RUS": "Russian",
    "JPN": "Japanese",
    "KOR": "Korean",
    "ARA": "Arabic",
    "HIN": "Hindi",
    "BEN": "Bengali",
    "VIE": "Vietnamese",
    "THA": "Thai",
    "TUR": "Turkish",
    "POL": "Polish",
    "NLD": "Dutch",
    "GRE": "Greek",
    "HEB": "Hebrew",
    "SWE": "Swedish",
    "NOR": "Norwegian",
    "DAN": "Danish",
    "FIN": "Finnish",
    "CZE": "Czech",
    "HUN": "Hungarian",
    "ROM": "Romanian",
    "UKR": "Ukrainian",
    "IND": "Indonesian",
    "MAY": "Malay",
    "PER": "Persian",
    "URD": "Urdu",
}


def translate_pdf_multilingual(
    pdf_path: str,
    target_languages: List[str],
    source_language: str = "ENG",
    translation_service: str = "google",
    output_dir: Optional[str] = None,
    num_threads: int = 4,
    ollama_model: Optional[str] = None,
    ollama_host: str = "http://127.0.0.1:11434",
) -> bool:
    """
    Translate a PDF to multiple languages while preserving formatting.

    Args:
                    pdf_path: Path to the input PDF file
                    target_languages: List of 3-letter language codes (e.g., ['DEU', 'FRA', 'SPA'])
                    source_language: 3-letter code for source language (default: 'ENG')
                    translation_service: Translation service to use
                                                                                             Options: 'google', 'deepl', 'ollama', 'openai', etc.
                    output_dir: Directory to save translated files (default: same as input)
                    num_threads: Number of threads for parallel translation (default: 4)
                    ollama_model: Model name if using Ollama (e.g., 'gemma2', 'llama3')
                    ollama_host: Ollama server URL (default: http://127.0.0.1:11434)

    Returns:
                    bool: True if ALL translations succeeded, False otherwise

    Example:
                    >>> success = translate_pdf_multilingual(
                    ...     'document.pdf',
                    ...     ['DEU', 'FRA', 'SPA'],
                    ...     source_language='ENG',
                    ...     translation_service='ollama',
                    ...     ollama_model='gemma2'
                    ... )
                    >>> print(f"All translations successful: {success}")
    """

    # Validate input file
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found")
        return False

    # Setup paths
    input_path = Path(pdf_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Validate source language
    if source_language not in LANGUAGE_MAP:
        print(f"Error: Source language '{source_language}' not supported")
        print(f"Supported languages: {', '.join(LANGUAGE_MAP.keys())}")
        return False

    # Validate target languages
    invalid_langs = [lang for lang in target_languages if lang not in LANGUAGE_MAP]
    if invalid_langs:
        print(f"Error: Unsupported target languages: {', '.join(invalid_langs)}")
        print(f"Supported languages: {', '.join(LANGUAGE_MAP.keys())}")
        return False

    # Setup environment for Ollama if used
    if translation_service == "ollama":
        if ollama_model is None:
            print("Error: ollama_model must be specified when using Ollama service")
            return False
        os.environ["OLLAMA_HOST"] = ollama_host
        os.environ["OLLAMA_MODEL"] = ollama_model
        print(f"Using Ollama with model: {ollama_model} at {ollama_host}")

    # Get source language name
    source_lang_name = LANGUAGE_MAP[source_language]

    # Track success for all translations
    all_successful = True
    successful_translations = []
    failed_translations = []

    # Translate to each target language
    for target_lang_code in target_languages:
        target_lang_name = LANGUAGE_MAP[target_lang_code]

        # Create output filename: XXX_originalname.pdf
        output_filename = f"{target_lang_code}_{input_path.name}"
        output_path = output_dir / output_filename

        print(f"\n{'='*60}")
        print(f"Translating: {source_lang_name} -> {target_lang_name}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        try:
            # Translation parameters
            params = {
                "lang_in": source_lang_name,
                "lang_out": target_lang_name,
                "service": translation_service,
                "thread": num_threads,
                "output": str(output_path.with_suffix("")),  # pdf2zh adds .pdf
            }

            # Execute translation
            result = translate(files=[str(input_path)], **params)

            # Check if translation was successful
            if result and len(result) > 0:
                # pdf2zh returns tuple (mono_file, dual_file)
                mono_file, dual_file = result[0]

                # Rename the mono file to our desired output
                if os.path.exists(mono_file):
                    final_output = output_dir / output_filename
                    if mono_file != str(final_output):
                        os.rename(mono_file, final_output)

                    # Clean up dual file if not needed
                    if dual_file and os.path.exists(dual_file):
                        os.remove(dual_file)

                    print(f"✓ Successfully translated to {target_lang_name}")
                    successful_translations.append(target_lang_code)
                else:
                    print(f"✗ Translation failed: Output file not created")
                    failed_translations.append(target_lang_code)
                    all_successful = False
            else:
                print(f"✗ Translation failed: No result returned")
                failed_translations.append(target_lang_code)
                all_successful = False

        except Exception as e:
            print(f"✗ Translation failed with error: {str(e)}")
            failed_translations.append(target_lang_code)
            all_successful = False

    # Print summary
    print(f"\n{'='*60}")
    print("TRANSLATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total languages: {len(target_languages)}")
    print(f"Successful: {len(successful_translations)} - {successful_translations}")
    print(f"Failed: {len(failed_translations)} - {failed_translations}")
    print(f"Overall success: {all_successful}")
    print(f"{'='*60}\n")

    return all_successful


def _add_footer_to_pdf(
    input_pdf, output_pdf, footer_text, logo_path, link_url, footer_height=50
):
    doc = fitz.open(input_pdf)

    for page in doc:
        # Get current page dimensions
        rect = page.rect

        # Extend the page height
        new_rect = fitz.Rect(0, 0, rect.width, rect.height + footer_height)
        page.set_mediabox(new_rect)

        # Footer area starts at old page height
        footer_y_start = rect.height

        # Add logo
        if logo_path:
            logo_rect = fitz.Rect(
                10, footer_y_start + 10, 60, footer_y_start + footer_height - 10
            )
            page.insert_image(logo_rect, filename=logo_path)

        # Add text
        text_rect = fitz.Rect(
            70,
            footer_y_start + 10,
            rect.width - 10,
            footer_y_start + footer_height - 10,
        )
        page.insert_textbox(text_rect, footer_text, fontsize=10, color=(0, 0, 0))

        # Add clickable link
        link_rect = fitz.Rect(
            70,
            footer_y_start + 10,
            rect.width - 10,
            footer_y_start + footer_height - 10,
        )
        page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": link_url})

    doc.save(output_pdf)
    doc.close()
