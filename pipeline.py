import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from checksum_processor import HashAlgorithm, generate_checksum
from id_gen import generate_uuid4plus
from article_prop_extraction import MetadataExtractor
from visual_processor import QwenDocumentProcessor
from tqdm import tqdm


def load_config(
    config_file: str = r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\config.json",
) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_path = Path(config_file)

    if not config_path.exists():
        print(f"Configuration file not found: {config_file}")
        print("Please create processing_config.json with your settings")
        exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"Configuration loaded from: {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)


def setup_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create all necessary directories and return paths"""
    base_dir = Path(config["base_dir"])

    paths = {
        "base_dir": base_dir,
        "metadata_dir": base_dir / config["directories"]["metadata"],
        "semantics_dir": base_dir / config["directories"]["semantics"],
        "logs_dir": base_dir / config["directories"]["logs"],
        "completed_dir": base_dir / config["directories"]["completed"],
        "temp_dir": base_dir / config["directories"]["temp"],
        "duplicate_dir": base_dir / config["directories"]["duplicate"],
        "review_dir": base_dir / config["directories"]["review"],
        "processed_checksums_file": base_dir / config["files"]["processed_checksums"],
        "completed_checksums_file": base_dir / config["files"]["completed_checksums"],
    }

    # Create directories
    for name, path in paths.items():
        if name.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)

    return paths


def load_checksum_sets(paths: Dict[str, Path]) -> tuple[Set[str], Set[str]]:
    """Load processed and completed checksum sets from disk"""

    processed_checksums: Set[str] = set()
    completed_checksums: Set[str] = set()

    # Load processed checksums
    if paths["processed_checksums_file"].exists():
        try:
            with open(paths["processed_checksums_file"], "r", encoding="utf-8") as f:
                processed_list = json.load(f)
                processed_checksums = set(processed_list)
                logger.info(
                    f"Loaded {len(processed_checksums)} processed checksums from disk"
                )
        except Exception as e:
            logger.warning(f"Failed to load processed checksums: {e}")

    # Load completed checksums
    if paths["completed_checksums_file"].exists():
        try:
            with open(paths["completed_checksums_file"], "r", encoding="utf-8") as f:
                completed_list = json.load(f)
                completed_checksums = set(completed_list)
                logger.info(
                    f"Loaded {len(completed_checksums)} completed checksums from disk"
                )
        except Exception as e:
            logger.warning(f"Failed to load completed checksums: {e}")

    return processed_checksums, completed_checksums


def save_checksum_sets(
    processed_checksums: Set[str], completed_checksums: Set[str], paths: Dict[str, Path]
) -> None:
    """Save processed and completed checksum sets to disk"""

    try:
        # Save processed checksums
        with open(paths["processed_checksums_file"], "w", encoding="utf-8") as f:
            json.dump(list(processed_checksums), f)

        # Save completed checksums
        with open(paths["completed_checksums_file"], "w", encoding="utf-8") as f:
            json.dump(list(completed_checksums), f)

        logger.debug("Checksum sets saved to disk")

    except Exception as e:
        logger.error(f"Failed to save checksum sets: {e}")


def setup_logging(config: Dict[str, Any], paths: Dict[str, Path]):
    """Set up logging configuration"""
    log_filename = f"SESSION-{datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")}.log"

    handlers = [
        logging.FileHandler(
            f"{str(paths['logs_dir'])}/{log_filename}", encoding="utf-8"
        )
    ]

    if config["logging"]["console_logging"]:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"]),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
    return logger


def main():
    """Main processing function"""

    # Load configuration
    config = load_config()

    # Setup directories
    paths = setup_directories(config)

    # Setup logging
    global logger
    logger = setup_logging(config, paths)
    logger.info(f"Logging initialized. Log file: {paths['logs_dir']}")

    # Load session limits
    max_files_per_session = config["session_limits"]["max_files_per_session"]
    logger.info(f"Session limit: {max_files_per_session} files per session")

    # Load existing checksum tracking data
    processed_checksum, completed_checksum = load_checksum_sets(paths)
    logger.info(
        f"Resume state: {len(processed_checksum)} processed, {len(completed_checksum)} completed"
    )

    # Initialize processors
    extractor = MetadataExtractor(logger=logger)
    logger.info(f"Document Metadata Extractor initialized.")

    processor = QwenDocumentProcessor(logger=logger, config=config)
    logger.info(f"LLM-based Document Semantics and Text Extractor initialized.")

    logger.info(f"Starting document processing in: {paths['base_dir']}")
    supported_extensions = set(config["supported_extensions"])
    logger.info(f"Supported extensions: {supported_extensions}")

    # Get all documents in base directory (non-recursive)
    documents: List[Path] = [
        item
        for item in paths["base_dir"].iterdir()
        if item.is_file() and item.suffix.lower() in supported_extensions
    ]

    num_of_docs: int = len(documents)
    logger.info(f"Found {num_of_docs} documents to process")

    # Filter out already processed documents for resumability
    unprocessed_documents = []
    skipped_completed = 0
    skipped_duplicates = 0

    logger.info("Scanning for already processed documents...")
    checksum_algorithm = HashAlgorithm[
        config["processing_options"]["checksum_algorithm"]
    ]

    for doc in tqdm(documents, desc="Scanning documents", unit="files"):
        try:
            # Generate checksum to check if already processed
            doc_checksum = generate_checksum(doc, algorithm=checksum_algorithm)

            if doc_checksum in completed_checksum:
                skipped_completed += 1
                continue
            elif doc_checksum in processed_checksum:
                # This document was started but not completed - send to review
                logger.warning(
                    f"Found incomplete processing for {doc.name}, moving to review"
                )
                try:
                    review_location = paths["review_dir"] / doc.name
                    if not review_location.exists():  # Avoid overwriting
                        doc.rename(review_location)
                        logger.info(f"Moved incomplete document to review: {doc.name}")
                    else:
                        logger.warning(
                            f"Review file already exists: {review_location.name}"
                        )
                except Exception as e:
                    logger.error(f"Failed to move {doc.name} to review: {e}")
                continue
            else:
                unprocessed_documents.append(doc)

        except Exception as e:
            logger.error(f"Failed to check processing status for {doc.name}: {e}")
            unprocessed_documents.append(doc)  # Include it to be safe

    logger.info(
        f"Processing summary: {len(unprocessed_documents)} new, {skipped_completed} already completed"
    )

    if not unprocessed_documents:
        logger.info("No documents to process! All files have been completed.")
        return

    # Limit session to max_files_per_session
    if len(unprocessed_documents) > max_files_per_session:
        session_documents = unprocessed_documents[:max_files_per_session]
        remaining_count = len(unprocessed_documents) - max_files_per_session
        logger.info(
            f"SESSION LIMIT: Processing only {max_files_per_session} files this session"
        )
        logger.info(f"Remaining files for next session: {remaining_count}")
    else:
        session_documents = unprocessed_documents
        logger.info(
            f"Processing all {len(session_documents)} remaining files this session"
        )

    # Process each document with progress tracking
    failed_documents = []

    for i, doc in enumerate(
        tqdm(session_documents, desc="Processing documents", unit="files"), 1
    ):
        logger.info(f"Processing document {i}/{len(session_documents)}: {doc.name}")

        try:
            # Generate unique identifier for this document
            doc_id: str = generate_uuid4plus()
            logger.debug(f"Generated doc_id: {doc_id} for {doc.name}")

            # Rename document with unique ID (keeps original extension)
            old_name = doc.name
            target_doc = doc.rename(paths["base_dir"] / f"DOC-{doc_id}{doc.suffix}")
            logger.info(f"Renamed: {old_name} -> {target_doc.name}")

            # Generate checksum for document integrity verification
            logger.info(f"Generating checksum for: {target_doc.name}")
            doc_checksum: str = generate_checksum(
                file_path=target_doc, algorithm=checksum_algorithm
            )
            logger.info(f"Checksum generated: {doc_checksum[:32]}...")

            # Mark as being processed
            processed_checksum.add(doc_checksum)
            save_checksum_sets(processed_checksum, completed_checksum, paths)

            # Create metadata file
            metadata_file: Path = paths["metadata_dir"] / f"METADATA-{doc_id}.json"
            metadata_file.touch()
            logger.debug(f"Created metadata file: {metadata_file.name}")

            # Create semantics file
            semantics_file: Path = paths["semantics_dir"] / f"SEMANTICS-{doc_id}.txt"
            semantics_file.touch()
            logger.debug(f"Created semantics file: {semantics_file.name}")

            # Extract metadata
            logger.info("Extracting document metadata...")
            extracted_doc_props: Dict[str, Any] = extractor.extract(target_doc)

            # Write metadata
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(extracted_doc_props, f, indent=2)
            logger.debug(f"Wrote metadata to: {metadata_file.name}")

            # Extract semantics using AI
            logger.info("Extracting document semantics with AI...")
            extracted_semantic_details: Dict[str, str] = (
                processor.extract_article_semantics(document=target_doc)
            )

            with open(semantics_file, "w", encoding="utf-8") as f:
                json.dump(extracted_semantic_details, f, indent=2)
            logger.debug(f"Wrote semantics to: {semantics_file.name}")

            # Move processed document to completed directory
            final_doc_location = target_doc.rename(
                paths["completed_dir"] / target_doc.name
            )
            logger.info(f"Moved to completed: {final_doc_location.name}")

            # Mark as completed
            completed_checksum.add(doc_checksum)
            save_checksum_sets(processed_checksum, completed_checksum, paths)

            logger.info(f"✓ Successfully processed: {old_name}")

        except Exception as e:
            logger.error(f"✗ Failed to process {doc.name}: {str(e)}")
            logger.exception("Full error details:")
            failed_documents.append((doc.name, str(e)))

            # Remove from processed set if it failed
            if "doc_checksum" in locals():
                processed_checksum.discard(doc_checksum)
                save_checksum_sets(processed_checksum, completed_checksum, paths)

    # Final summary
    logger.info("=" * 60)
    logger.info("SESSION PROCESSING COMPLETED!")
    logger.info("=" * 60)
    logger.info(
        f"Successfully processed: {len(session_documents) - len(failed_documents)}"
    )
    logger.info(f"Failed documents: {len(failed_documents)}")
    logger.info(f"Previously completed: {skipped_completed}")

    # Check if more files remain
    total_remaining = len(unprocessed_documents) - len(session_documents)
    if total_remaining > 0:
        logger.info(
            f"REMAINING FILES: {total_remaining} files need processing in future sessions"
        )
        if config["session_limits"]["restart_recommended_after_session"]:
            logger.info(
                "RECOMMENDATION: Restart your PC before next processing session for optimal memory management"
            )

    if failed_documents:
        logger.info("\nFailed documents:")
        for doc_name, error in failed_documents:
            logger.info(f"  - {doc_name}: {error}")

    logger.info(f"\nProcessed files location: {paths['completed_dir']}")
    logger.info(f"Metadata location: {paths['metadata_dir']}")
    logger.info(f"Semantics location: {paths['semantics_dir']}")


if __name__ == "__main__":
    main()
