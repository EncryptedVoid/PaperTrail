import logging
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import (
    ANKI_DIR,
    BITWARDEN_DIR,
    CALIBRE_LIBRARY_DIR,
    DIGITAL_ASSET_MANAGEMENT_DIR,
    FIREFLYIII_DIR,
    GITLAB_DIR,
    LINKWARDEN_DIR,
    PERFORMANCE_PORTFOLIO_DIR,
    UNSUPPORTED_ARTIFACTS_DIR,
)
from utilities.automatic_sorting import (
    is_anki_deck,
    is_backup_codes_file,
    is_book,
    is_bookmark_file,
    is_code,
    is_financial_document,
    is_supported,
)
from utilities.visual_processor import VisualProcessor


def automatically_sorting(
    logger: logging.Logger, visual_processor: VisualProcessor, source_dir: Path
):
    # Discover all artifact files in the source directory
    # Filter for files that start with the ARTIFACT_PREFIX to ensure we only process valid artifacts
    unprocessed_artifacts: List[Path] = [item for item in source_dir.iterdir()]

    # Handle empty directory case - exit early if no artifacts found
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

        # Sort files by size for consistent processing order (smaller files first)
        # This provides faster initial feedback and helps identify issues early
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Process each artifact file with progress tracking
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Auto-sorting artifacts",
        unit="artifacts",
    ):
        if artifact.suffix == ".html" and is_bookmark_file(file_path=artifact):
            shutil.move(src=artifact, dst=LINKWARDEN_DIR)
        elif is_code:
            shutil.move(src=artifact, dst=GITLAB_DIR)
        elif is_anki_deck(file_path=artifact):
            shutil.move(src=artifact, dst=ANKI_DIR)
        elif is_backup_codes_file(file_path=artifact):
            shutil.move(src=artifact, dst=BITWARDEN_DIR)
        elif is_book(file_path=artifact):
            shutil.move(src=artifact, dst=CALIBRE_LIBRARY_DIR)
        elif is_financial_document(
            file_path=artifact, visual_processor=visual_processor, logger=logger
        ):
            shutil.copy2(src=artifact, dst=FIREFLYIII_DIR)
            shutil.copy2(src=artifact, dst=PERFORMANCE_PORTFOLIO_DIR)
            shutil.move(src=artifact, dst=DIGITAL_ASSET_MANAGEMENT_DIR)
        elif not is_supported(file_path=artifact):
            shutil.move(src=artifact, dst=UNSUPPORTED_ARTIFACTS_DIR)

    return None
