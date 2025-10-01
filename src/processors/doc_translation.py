"""
PDF Multi-Language Translator Module
Translates PDFs to multiple languages while preserving formatting using pdf2zh
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any , Dict , List

import fitz  # PyMuPDF
from pdf2zh import set_service , translate
from tqdm import tqdm

from config import (
	ARTIFACT_PREFIX ,
	ARTIFACT_PROFILES_DIR ,
	LANGUAGE_MAP ,
	PREFERRED_LANGUAGE_TRANSLATIONS ,
	PREFERRED_TRANSLATION_MODEL ,
	PROFILE_PREFIX ,
	TRANSLATION_WATERMARKS ,
)


def _add_watermark(pdf_path: str, watermark_text: str, font_size: int = 8):
    """
    Add a watermark footer to all pages of a PDF.

    Args:
                    pdf_path: Path to the PDF file to watermark
                    watermark_text: Text to display in the watermark footer
                    font_size: Size of the watermark text (default: 8)
    """
    doc = fitz.open(pdf_path)

    for page in doc:
        rect = page.rect
        # Position watermark at bottom center of page
        footer_rect = fitz.Rect(50, rect.height - 30, rect.width - 50, rect.height - 10)
        page.insert_textbox(
            footer_rect,
            watermark_text,
            fontsize=font_size,
            align=fitz.TEXT_ALIGN_CENTER,
            color=(0.5, 0.5, 0.5),  # Gray text for subtle appearance
        )

    # Save with incremental update to preserve PDF structure
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()


def translate_multilingual(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:
    """
    Translate artifact PDFs to multiple languages while preserving formatting.

    This method performs multi-language translation by:
    - Discovering all artifact files in the source directory
    - Loading artifact profiles to extract source language metadata
    - Creating copies of the original file for each target language
    - Translating each copy using pdf2zh with the configured translation service
    - Adding language-specific watermarks to translated PDFs
    - Tracking translation success/failure for each language
    - Updating artifact profiles with translation metadata
    - Moving processed artifacts to success or failure directories

    The translation process includes:
    - File discovery and artifact ID extraction
    - Profile validation and metadata loading
    - Source language validation against supported languages
    - Per-language translation with individual error handling
    - Watermark application to distinguish translations
    - Profile updates with translation statistics
    - File movement with conflict resolution

    Args:
                    logger: Logger instance for tracking operations and errors
                    source_dir: Directory containing artifact files to translate. All files
                                                             starting with ARTIFACT_PREFIX will be processed
                    failure_dir: Directory to move artifacts that fail processing.
                                                                    Files are moved here when profile loading or validation fails
                    success_dir: Directory to move successfully processed artifacts after
                                                                    all translations complete

    Returns:
                    None. This method processes files in-place, creates translated copies,
                    and moves originals to appropriate directories based on results.

    Raises:
                    FileNotFoundError: If artifact profile does not exist
                    ValueError: If profile JSON is corrupted or source language is unsupported
                    OSError: If file operations fail due to permissions or disk issues

    Note:
                    - Original artifacts are preserved and copied for each translation
                    - Each target language gets a separate file: {lang_code}_{original_name}
                    - Translation failures for individual languages don't stop other translations
                    - Profile updates include counts of successful and failed translations
                    - Naming conflicts in destination directories are automatically resolved
                    - All translation errors are logged with full exception details
                    - The original artifact is only moved after ALL translations are attempted
    """

    # Log stage header for clear progress tracking in logs
    logger.info("=" * 80)
    logger.info("MULTI-LANGUAGE TRANSLATION STAGE")
    logger.info("=" * 80)

    # Discover all artifact files in the source directory
    # Only process files that match the expected ARTIFACT- prefix
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]

    # Handle empty directory case - nothing to process
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

    # Sort files by size for consistent processing order (smaller files first)
    # This provides faster initial feedback during processing
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to translate")
    logger.info(f"Target languages: {', '.join(PREFERRED_LANGUAGE_TRANSLATIONS)}")
    logger.info(f"Translation service: {PREFERRED_TRANSLATION_MODEL}")

    # Process each artifact file individually
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Translating to multiple languages",
        unit="artifacts",
    ):
        try:
            logger.info("-" * 80)
            logger.info(f"Processing artifact: {artifact.name}")
            logger.info(f"File size: {artifact.stat().st_size:,} bytes")

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )
            logger.debug(f"Looking for profile: {artifact_profile_path.name}")

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            logger.info(f"Profile found: {artifact_profile_path.name}")

            # Load existing profile data from JSON
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
                logger.debug("Profile loaded successfully")
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Extract source language from profile metadata
            # Changed from artifact_profile_path to artifact_profile_data
            source_language = artifact_profile_data["extracted_semantics"]["language"]
            logger.info(f"Source language detected: {source_language}")

            # Validate source language against supported languages
            # Changed from UnsupportedOperatorError to ValueError
            if source_language not in LANGUAGE_MAP:
                error_msg = f"Source language '{source_language}' not supported"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Get human-readable source language name
            source_lang_name = LANGUAGE_MAP[source_language]
            logger.info(f"Source language name: {source_lang_name}")

            # Track translation results for each target language
            # Initialize tracking variables before loop
            successful_translations = []
            failed_translations = []

            # Translate to each configured target language
            for target_lang_code in PREFERRED_LANGUAGE_TRANSLATIONS:
                # Skip if source and target are the same language
                if target_lang_code == source_language:
                    logger.info(
                        f"Skipping translation to {target_lang_code} (same as source)"
                    )
                    continue

                # Get human-readable target language name
                target_lang_name = LANGUAGE_MAP[target_lang_code]

                # Define output filename with language prefix
                # Format: {lang_code}_{original_name} (e.g., es_ARTIFACT-123.pdf)
                output_filename = (
                    f"{artifact.stem}-{target_lang_code.upper()}-{artifact.suffix}"
                )
                output_path = source_dir / output_filename

                # Log translation attempt
                logger.info(f"\n{'='*60}")
                logger.info(f"Translating: {source_lang_name} → {target_lang_name}")
                logger.info(f"Output file: {output_filename}")
                logger.info(f"{'='*60}")

                try:
                    # Create a copy of the original file for translation
                    # This ensures we keep the original intact and have separate files per language
                    doc_copy_for_translation = source_dir / f"temp_{output_filename}"
                    logger.debug(
                        f"Creating temporary copy: {doc_copy_for_translation.name}"
                    )
                    shutil.copy2(str(artifact), str(doc_copy_for_translation))
                    logger.debug("Temporary copy created successfully")

                    # Perform translation using pdf2zh
                    # Proper handling of translate() return value which is [(mono_path, dual_path)]
                    logger.info("Starting pdf2zh translation...")
                    result = translate(
                        files=[
                            str(doc_copy_for_translation)
                        ],  # Translate the copy, not the original
                        lang_in=source_lang_name,
                        lang_out=target_lang_name,
                        service=PREFERRED_TRANSLATION_MODEL,
                        thread=1,  # Single thread for local Ollama models
                    )
                    logger.debug(f"Translation result: {result}")

                    # Properly unpack the tuple result from translate()
                    # Result format: [(mono_path, dual_path)] where mono is translation-only
                    if result:
                        mono_path, dual_path = result[0]
                        logger.debug(f"Mono path: {mono_path}")
                        logger.debug(f"Dual path: {dual_path}")

                        # Use the monolingual (translation-only) version
                        # Rename it to our desired output filename
                        if os.path.exists(mono_path):
                            logger.debug(f"Renaming {mono_path} to {output_path}")
                            shutil.move(mono_path, str(output_path))

                            # Clean up the dual version if it exists (we only keep mono)
                            if os.path.exists(dual_path):
                                logger.debug(f"Removing dual version: {dual_path}")
                                os.remove(dual_path)

                            # Add language-specific watermark to translated PDF
                            watermark_text = TRANSLATION_WATERMARKS[target_lang_code]
                            logger.debug(f"Adding watermark: {watermark_text}")
                            _add_watermark(str(output_path), watermark_text)

                            # Clean up temporary copy
                            if os.path.exists(doc_copy_for_translation):
                                logger.debug("Removing temporary copy")
                                os.remove(doc_copy_for_translation)

                            logger.info(
                                f"✓ Successfully translated to {target_lang_name}"
                            )
                            successful_translations.append(target_lang_code)
                        else:
                            logger.warning(
                                f"Translation output file not found: {mono_path}"
                            )
                            failed_translations.append(target_lang_code)

                            # Clean up temporary copy on failure
                            if os.path.exists(doc_copy_for_translation):
                                os.remove(doc_copy_for_translation)
                    else:
                        logger.error("Translation returned empty result")
                        failed_translations.append(target_lang_code)

                        # Clean up temporary copy on failure
                        if os.path.exists(doc_copy_for_translation):
                            os.remove(doc_copy_for_translation)

                except Exception as e:
                    logger.error(
                        f"✗ Translation to {target_lang_name} failed: {str(e)}",
                        exc_info=True,
                    )
                    failed_translations.append(target_lang_code)

            # Log translation summary for this artifact
            logger.info(f"\n{'='*60}")
            logger.info(f"Translation Summary for {artifact.name}")
            logger.info(
                f"Successful: {len(successful_translations)}/{len(PREFERRED_LANGUAGE_TRANSLATIONS)}"
            )
            logger.info(
                f"Failed: {len(failed_translations)}/{len(PREFERRED_LANGUAGE_TRANSLATIONS)}"
            )
            if successful_translations:
                logger.info(f"Success languages: {', '.join(successful_translations)}")
            if failed_translations:
                logger.warning(f"Failed languages: {', '.join(failed_translations)}")
            logger.info(f"{'='*60}\n")

            # Update artifact profile with translation stage metadata
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            artifact_profile_data["stage_progression_data"]["translation"] = {
                "status": "completed",
                "generated_translations": len(successful_translations),
                "failed_translations": len(failed_translations),
                "successful_languages": successful_translations,
                "failed_languages": failed_translations,
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile back to disk
            try:
                logger.debug(f"Updating profile: {artifact_profile_path.name}")
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move original artifact to success directory after all translations complete
            success_location: Path = success_dir / artifact.name

            # Handle naming conflicts in success directory
            if success_location.exists():
                logger.warning(
                    f"File already exists in success dir: {success_location.name}"
                )
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                while success_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / new_name
                    counter += 1
                logger.info(f"Using alternative name: {success_location.name}")

            logger.info(f"Moving original artifact to success directory")
            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

        except Exception as e:
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                while failure_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue

    logger.info("=" * 80)
    logger.info("Multi-language translation stage completed")
    logger.info("=" * 80)
