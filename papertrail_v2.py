#!C:/Users/UserX/AppData/Local/Programs/Python/Python313/python.exe

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

from checksum_utils import HashAlgorithm, generate_checksum
from language_processor import LanguageProcessor
from metadata_processor import MetadataExtractor
from uuid_utils import generate_uuid4plus
from visual_processor import QwenDocumentProcessor


@dataclass
class ProcessingConfig:
    gpu_memory_limit_percent: float = 75.0
    model_refresh_interval: int = 20
    max_memory_threshold_percent: float = 75.0
    checksum_algorithm: HashAlgorithm = HashAlgorithm.SHA3_512
    base_dir: Path = Path(
        r"C:\Users\UserX\Desktop\Github-Workspace\PaperTrail\test_run"
    )
    checksum_history_filename: str = "checksum_history.txt"


@dataclass
class FileTypeConfig:
    supported_extensions: Set[str] = field(
        default_factory=lambda: {
            # Text & Documents
            ".txt",
            ".md",
            ".rtf",
            ".doc",
            ".docx",
            ".pdf",
            ".odt",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".ods",
            ".odp",
            ".odg",
            # Data formats
            ".csv",
            ".tsv",
            ".json",
            ".xml",
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
            ".heic",
            ".heif",
            ".raw",
            ".cr2",
            ".nef",
            ".arw",
            ".dng",
            ".orf",
            ".rw2",
            ".tga",
            ".psd",
            # Email & Communication
            ".eml",
            ".msg",
            ".mbox",
            ".ics",
            ".vcs",
        }
    )

    unsupported_extensions: Set[str] = field(
        default_factory=lambda: {
            # Audio
            "mp3",
            "aac",
            "ogg",
            "wma",
            "m4a",
            "wav",
            "flac",
            "aiff",
            # Video
            "mp4",
            "avi",
            "mov",
            "mkv",
            "wmv",
            "flv",
            "webm",
            "ogv",
            "m4v",
            # 3D & CAD
            "obj",
            "fbx",
            "dae",
            "3ds",
            "blend",
            "dwg",
            "dxf",
            "step",
            # Executables & System
            "exe",
            "msi",
            "dmg",
            "pkg",
            "deb",
            "rpm",
            "dll",
            "so",
            "dylib",
            # Databases
            "db",
            "sqlite",
            "mdb",
            "accdb",
            # Proprietary
            "indd",
            "fla",
            "swf",
            "sav",
            "dta",
            "sas7bdat",
            "mat",
            "hdf5",
            # Archives
            ".zip",
            ".rar",
            ".7z",
            ".tar",
            ".gz",
        }
    )


@dataclass
class ProcessingPaths:
    base_dir: Path
    unprocessed_dir: Path
    rename_dir: Path
    metadata_dir: Path
    semantics_dir: Path
    logs_dir: Path
    completed_dir: Path
    temp_dir: Path
    review_dir: Path
    profiles_dir: Path
    llm_processed_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.llm_processed_dir = self.base_dir / "llm_processed"


@dataclass
class SessionData:
    session_id: str
    start_time: str
    status: str = "running"
    total_files: int = 0
    processed_files: int = 0
    file_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stage_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "unprocessed": 0,
            "rename": 0,
            "metadata": 0,
            "semantics": 0,
            "completed": 0,
            "failed": 0,
            "review": 0,
        }
    )
    performance: Dict[str, float] = field(
        default_factory=lambda: {"files_per_minute": 0.0, "total_runtime_seconds": 0}
    )
    errors: List[Dict[str, str]] = field(default_factory=list)
    end_time: str = ""


class PaperTrailProcessor:
    def __init__(self, config: ProcessingConfig, file_config: FileTypeConfig) -> None:
        self.config = config
        self.file_config = file_config
        self.paths = self._create_paths()
        self.session_start_time = datetime.now()
        self.session_data = self._initialize_session_data()
        self.session_json_path = (
            self.paths.logs_dir / f"SESSION-{self.session_data.session_id}.json"
        )
        self.checksum_history: Set[str] = set()
        self.logger = self._setup_logging()

    def _create_paths(self) -> ProcessingPaths:
        base = self.config.base_dir
        return ProcessingPaths(
            base_dir=base,
            unprocessed_dir=base / "unprocessed_artifacts",
            rename_dir=base / "identified_artifacts",
            metadata_dir=base / "metadata_extracted",
            semantics_dir=base / "visually_processed",
            logs_dir=base / "session_logs",
            completed_dir=base / "completed_artifacts",
            temp_dir=base / "temp",
            review_dir=base / "review_required",
            profiles_dir=base / "artifact_profiles",
        )

    def _initialize_session_data(self) -> SessionData:
        session_timestamp = self.session_start_time.strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")
        return SessionData(
            session_id=session_timestamp, start_time=self.session_start_time.isoformat()
        )

    def _setup_logging(self) -> logging.Logger:
        self._ensure_directories_exist()

        log_filename = f"PAPERTRAIL-SESSION-{self.session_data.session_id}.log"
        handlers = [
            logging.FileHandler(self.paths.logs_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

        logger = logging.getLogger(__name__)
        self._log_session_start(log_filename)
        return logger

    def _ensure_directories_exist(self) -> None:
        for path_attr in [attr for attr in dir(self.paths) if attr.endswith("_dir")]:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)

    def _log_session_start(self, log_filename: str) -> None:
        self.logger.info("=" * 80)
        self.logger.info("PAPERTRAIL DOCUMENT PROCESSING PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {self.session_data.session_id}")
        self.logger.info(f"Logging initialized. Log file: {log_filename}")
        self.logger.info(f"Session JSON: {self.session_json_path.name}")
        self.logger.info(f"Base directory: {self.paths.base_dir}")
        self.logger.info(
            f"Supported extensions: {len(self.file_config.supported_extensions)} types"
        )
        self.logger.info(
            f"Unsupported extensions: {len(self.file_config.unsupported_extensions)} types"
        )

    def _load_checksum_history(self) -> None:
        checksum_path = self.paths.base_dir / self.config.checksum_history_filename

        if checksum_path.exists():
            try:
                with open(checksum_path, "r", encoding="utf-8") as f:
                    self.checksum_history = {line.strip() for line in f if line.strip()}
                self.logger.info(
                    f"Loaded {len(self.checksum_history)} permanent checksums from history"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load permanent checksums: {e}")
                raise e
        else:
            self.logger.info("No permanent checksum history found - creating new file")
            checksum_path.touch()

    def _update_session_json(self) -> None:
        try:
            with open(self.session_json_path, "w", encoding="utf-8") as f:
                json.dump(self.session_data.__dict__, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to update SESSION JSON: {e}")

    def _update_stage_counts(self, from_stage: str, to_stage: str) -> None:
        if from_stage in self.session_data.stage_counts:
            self.session_data.stage_counts[from_stage] = max(
                0, self.session_data.stage_counts[from_stage] - 1
            )

        self.session_data.stage_counts[to_stage] = (
            self.session_data.stage_counts.get(to_stage, 0) + 1
        )

    def _log_detailed_progress(self, filename: str, file_size: int, stage: str) -> None:
        elapsed_seconds = (datetime.now() - self.session_start_time).total_seconds()
        files_per_minute = (
            (self.session_data.processed_files / elapsed_seconds * 60)
            if elapsed_seconds > 0
            else 0
        )

        self.session_data.performance.update(
            {
                "files_per_minute": round(files_per_minute, 2),
                "total_runtime_seconds": int(elapsed_seconds),
            }
        )

        size_str = self._format_file_size(file_size)
        file_type_summary = ", ".join(
            f"{count} {ext}" for ext, count in self.session_data.file_types.items()
        )
        stage_summary = ", ".join(
            f"{stage}({count})"
            for stage, count in self.session_data.stage_counts.items()
            if count > 0
        )

        self.logger.info("=" * 60)
        self.logger.info(
            f"[File {self.session_data.processed_files}/{self.session_data.total_files}] Completed: {filename} ({size_str})"
        )
        self.logger.info(f"Stage: {stage}")
        self.logger.info(
            f"Session Runtime: {elapsed_seconds//3600:02.0f}:{(elapsed_seconds%3600)//60:02.0f}:{elapsed_seconds%60:02.0f} | Speed: {files_per_minute:.1f} files/min"
        )
        self.logger.info(f"File Types: {file_type_summary}")
        self.logger.info(f"Stage Status: {stage_summary}")
        self.logger.info("=" * 60)

    @staticmethod
    def _format_file_size(file_size: int) -> str:
        if file_size < 1024:
            return f"{file_size}B"
        elif file_size < 1024 * 1024:
            return f"{file_size/1024:.1f}KB"
        else:
            return f"{file_size/(1024*1024):.1f}MB"

    def _save_checksum(self, checksum: str) -> None:
        checksum_path = self.paths.base_dir / self.config.checksum_history_filename
        with open(checksum_path, "a", encoding="utf-8") as f:
            f.write(f"{checksum}\n")

    def _move_to_review(self, artifact_path: Path, reason: str) -> None:
        destination = self.paths.review_dir / artifact_path.name
        artifact_path.rename(destination)
        self.logger.debug(f"Moved {artifact_path.name} to review folder: {reason}")

    def _move_to_failed(
        self, artifact_path: Path, source_dir: Path, stage: str
    ) -> None:
        failed_dir = source_dir / "failed"
        failed_dir.mkdir(exist_ok=True)

        try:
            artifact_path.rename(failed_dir / artifact_path.name)
            self._update_stage_counts(stage.replace("_failed", ""), "failed")
            self.logger.info(f"Moved failed file to: {failed_dir / artifact_path.name}")
        except Exception as move_error:
            self.logger.error(
                f"Failed to move {artifact_path.name} to failed directory: {move_error}"
            )

    def process_duplicate_detection(self) -> List[Path]:
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            "STAGE 1: DUPLICATE DETECTION, UNSUPPORTED FILES, AND CHECKSUM VERIFICATION"
        )
        self.logger.info("=" * 80)

        artifacts = sorted(
            self.paths.unprocessed_dir.iterdir(), key=lambda p: p.stat().st_size
        )

        if not artifacts:
            self.logger.info("No artifacts found in unprocessed directory")
            return []

        self.session_data.total_files = len(artifacts)
        self._count_file_types(artifacts)
        self._update_session_json()

        stats = {"duplicates": 0, "zero_byte": 0, "unsupported": 0}
        remaining_artifacts: List[Path] = []

        for artifact in tqdm(
            artifacts, desc="Checking for duplicates, zeros, and unsupported artifacts"
        ):
            try:
                if self._handle_unsupported_file(artifact, stats):
                    continue
                if self._handle_zero_byte_file(artifact, stats):
                    continue
                if self._handle_duplicate_file(artifact, stats):
                    continue

                remaining_artifacts.append(artifact)
                self.session_data.stage_counts["unprocessed"] += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to process {artifact.name} in duplicate detection: {e}"
                )
                self.session_data.errors.append(
                    {
                        "file": artifact.name,
                        "stage": "duplicate_detection",
                        "error": str(e),
                    }
                )

        self._log_duplicate_detection_results(remaining_artifacts, stats)
        self.session_data.total_files = len(remaining_artifacts)
        self._update_session_json()

        return remaining_artifacts

    def _count_file_types(self, artifacts: List[Path]) -> None:
        for artifact in artifacts:
            self.session_data.file_types[artifact.suffix.lower()] += 1

        file_summary = ", ".join(
            f"{count} {ext}" for ext, count in self.session_data.file_types.items()
        )
        self.logger.info(f"File type breakdown: {file_summary}")

    def _handle_unsupported_file(self, artifact: Path, stats: Dict[str, int]) -> bool:
        if artifact.suffix in self.file_config.unsupported_extensions:
            self._move_to_review(artifact, "unsupported file type")
            stats["unsupported"] += 1
            return True
        return False

    def _handle_zero_byte_file(self, artifact: Path, stats: Dict[str, int]) -> bool:
        if artifact.stat().st_size == 0:
            self._move_to_review(artifact, "zero-size file")
            stats["zero_byte"] += 1
            return True
        return False

    def _handle_duplicate_file(self, artifact: Path, stats: Dict[str, int]) -> bool:
        checksum = generate_checksum(artifact, algorithm=self.config.checksum_algorithm)
        self.logger.debug(f"Generated checksum for {artifact.name}: {checksum[:16]}...")

        if checksum in self.checksum_history:
            self._move_to_review(artifact, "duplicate file")
            stats["duplicates"] += 1
            return True

        self.checksum_history.add(checksum)
        self._save_checksum(checksum)
        return False

    def _log_duplicate_detection_results(
        self, remaining: List[Path], stats: Dict[str, int]
    ) -> None:
        self.logger.info("Duplicate detection complete:")
        self.logger.info(f"  - {len(remaining)} new files to process")
        self.logger.info(f"  - {stats['duplicates']} duplicates skipped")
        self.logger.info(f"  - {stats['zero_byte']} zero-byte files moved to review")
        self.logger.info(
            f"  - {stats['unsupported']} unsupported files moved to review"
        )

    def process_uuid_renaming(self) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 2: UUID RENAMING AND PROFILE CREATION")
        self.logger.info("=" * 80)

        artifacts = list(self.paths.unprocessed_dir.iterdir())

        for artifact in tqdm(artifacts, desc="Preparing artifact profiles"):
            try:
                self._process_single_rename(artifact)
            except Exception as e:
                self.logger.error(f"Failed to rename {artifact.name}: {e}")
                self.session_data.errors.append(
                    {"file": artifact.name, "stage": "rename", "error": str(e)}
                )

        self.logger.info(
            f"Renaming stage complete - {self.session_data.stage_counts['rename']} files renamed"
        )

    def _process_single_rename(self, artifact: Path) -> None:
        artifact_id = generate_uuid4plus()
        original_name = artifact.name
        file_size = artifact.stat().st_size

        profile_data = {
            "uuid": artifact_id,
            "original_filename": original_name,
            "file_size": file_size,
            "file_extension": artifact.suffix.lower(),
            "stages": {
                "renamed": {
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }
            },
        }

        new_name = f"ARTIFACT-{artifact_id}{artifact.suffix}"
        renamed_artifact = artifact.rename(self.paths.rename_dir / new_name)

        profile_path = self.paths.profiles_dir / f"PROFILE-{artifact_id}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        self.session_data.processed_files += 1
        self._update_stage_counts("unprocessed", "rename")
        self._log_detailed_progress(new_name, file_size, "renamed")
        self._update_session_json()

        self.logger.debug(f"Renamed: {original_name} -> {new_name}")

    def process_metadata_extraction(self) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 3: METADATA EXTRACTION")
        self.logger.info("=" * 80)

        artifacts = sorted(
            self.paths.rename_dir.iterdir(), key=lambda p: p.stat().st_size
        )

        if not artifacts:
            self.logger.info("No artifacts found for metadata extraction")
            return

        extractor = MetadataExtractor(logger=self.logger)
        self.logger.info("Document Metadata Extractor initialized")

        for artifact in tqdm(artifacts, desc="Extracting metadata"):
            try:
                self._process_single_metadata_extraction(artifact, extractor)
            except Exception as e:
                self.logger.error(
                    f"Failed to extract metadata for {artifact.name}: {e}"
                )
                self._move_to_failed(artifact, self.paths.rename_dir, "metadata_failed")
                self.session_data.errors.append(
                    {
                        "file": artifact.name,
                        "stage": "metadata_extraction",
                        "error": str(e),
                    }
                )

        self.logger.info(
            f"Metadata extraction complete - {self.session_data.stage_counts['metadata']} files processed"
        )

    def _process_single_metadata_extraction(
        self, artifact: Path, extractor: MetadataExtractor
    ) -> None:
        artifact_id = artifact.stem[9:]  # Remove "ARTIFACT-" prefix
        profile_path = self.paths.profiles_dir / f"PROFILE-{artifact_id}.json"

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        self.logger.info(f"Extracting metadata for: {artifact.name}")
        extracted_metadata = extractor.extract(artifact)

        profile_data["metadata"] = extracted_metadata
        profile_data["stages"]["metadata_extraction"] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        moved_artifact = artifact.rename(self.paths.metadata_dir / artifact.name)
        self._update_stage_counts("rename", "metadata")

        file_size = moved_artifact.stat().st_size
        self._log_detailed_progress(artifact.name, file_size, "metadata_extraction")
        self._update_session_json()

    def process_semantic_extraction(self) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 4: SEMANTIC EXTRACTION (VISUAL PROCESSING)")
        self.logger.info("=" * 80)

        artifacts = sorted(
            self.paths.metadata_dir.iterdir(), key=lambda p: p.stat().st_size
        )

        if not artifacts:
            self.logger.info("No artifacts found for semantic extraction")
            return

        processor = QwenDocumentProcessor(logger=self.logger)
        self.logger.info("LLM-based Document Semantics and Text Extractor initialized")

        for artifact in tqdm(artifacts, desc="Extracting semantic descriptions"):
            try:
                self._process_single_semantic_extraction(artifact, processor)
            except Exception as e:
                self.logger.error(
                    f"Failed to extract semantics for {artifact.name}: {e}"
                )
                self._move_to_failed(
                    artifact, self.paths.metadata_dir, "semantic_failed"
                )
                self.session_data.errors.append(
                    {
                        "file": artifact.name,
                        "stage": "semantic_extraction",
                        "error": str(e),
                    }
                )

        self.logger.info(
            f"Semantic extraction complete - {self.session_data.stage_counts['semantics']} files processed"
        )

    def _process_single_semantic_extraction(
        self, artifact: Path, processor: QwenDocumentProcessor
    ) -> None:
        artifact_id = artifact.stem[9:]
        profile_path = self.paths.profiles_dir / f"PROFILE-{artifact_id}.json"

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        self.logger.info(f"Extracting semantics for: {artifact.name}")
        extracted_semantics = processor.extract_article_semantics(document=artifact)

        profile_data["semantics"] = extracted_semantics
        profile_data["stages"]["semantic_extraction"] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        moved_artifact = artifact.rename(self.paths.semantics_dir / artifact.name)
        self._update_stage_counts("metadata", "semantics")

        file_size = moved_artifact.stat().st_size
        self._log_detailed_progress(artifact.name, file_size, "semantic_extraction")
        self._update_session_json()

    def process_llm_field_extraction(self) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 5: LLM FIELD EXTRACTION (SEMANTIC METADATA)")
        self.logger.info("=" * 80)

        artifacts = sorted(
            self.paths.semantics_dir.iterdir(), key=lambda p: p.stat().st_size
        )

        if not artifacts:
            self.logger.info("No artifacts found for LLM field extraction")
            return

        # Ensure LLM processed directory exists
        self.paths.llm_processed_dir.mkdir(parents=True, exist_ok=True)

        field_extractor = LanguageProcessor(logger=self.logger)
        self.logger.info("LLM Document Field Extractor initialized")

        for artifact in tqdm(
            artifacts, desc="Extracting semantic metadata with LLM processing"
        ):
            try:
                self._process_single_llm_extraction(artifact, field_extractor)
            except Exception as e:
                self.logger.error(
                    f"Failed to process LLM extraction for {artifact.name}: {e}"
                )
                self._move_to_failed(artifact, self.paths.semantics_dir, "llm_failed")
                self.session_data.errors.append(
                    {
                        "file": artifact.name,
                        "stage": "llm_field_extraction",
                        "error": str(e),
                    }
                )

        extractor_stats = field_extractor.get_stats()
        self.logger.info(
            f"LLM field extraction complete - {self.session_data.stage_counts.get('llm_processed', 0)} files processed successfully"
        )
        self.logger.info(
            f"LLM model used: {extractor_stats['model']} on {extractor_stats['host']}"
        )
        self.logger.info(
            f"Total LLM API calls made: {extractor_stats['total_processed']}"
        )

    def _process_single_llm_extraction(
        self, artifact: Path, field_extractor: LanguageProcessor
    ) -> None:
        artifact_id = artifact.stem[9:]
        profile_path = self.paths.profiles_dir / f"PROFILE-{artifact_id}.json"

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        ocr_text = profile_data.get("semantics", {}).get("all_text", "")
        visual_description = profile_data.get("semantics", {}).get("all_imagery", "")
        metadata = profile_data.get("metadata", {})

        if self._should_skip_llm_processing(ocr_text, visual_description, artifact):
            return

        self.logger.info(f"Extracting structured fields for: {artifact.name}")
        self.logger.debug(
            f"OCR text length: {len(ocr_text)} chars, Visual desc length: {len(visual_description)} chars"
        )

        extraction_result = field_extractor.extract_fields(
            ocr_text=ocr_text,
            visual_description=visual_description,
            metadata=metadata,
            uuid=artifact_id,
        )

        profile_data["llm_extraction"] = extraction_result
        profile_data["stages"]["llm_field_extraction"] = {
            "status": "completed" if extraction_result["success"] else "failed",
            "timestamp": datetime.now().isoformat(),
            "model_used": field_extractor.model,
            "fields_extracted": len(extraction_result.get("extracted_fields", {})),
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        moved_artifact = artifact.rename(self.paths.llm_processed_dir / artifact.name)

        stage_target = "llm_processed" if extraction_result["success"] else "failed"
        self._update_stage_counts("semantics", stage_target)

        file_size = moved_artifact.stat().st_size
        extraction_status = "successful" if extraction_result["success"] else "failed"
        self._log_detailed_progress(
            artifact.name, file_size, f"llm_extraction_{extraction_status}"
        )
        self._update_session_json()

        self._log_llm_extraction_results(artifact.name, extraction_result)

    def _should_skip_llm_processing(
        self, ocr_text: str, visual_description: str, artifact: Path
    ) -> bool:
        empty_ocr = not ocr_text or ocr_text.strip() in [
            "No text found in document.",
            "",
        ]
        empty_visual = not visual_description or visual_description.strip() in [
            "No visual content described.",
            "",
        ]

        if empty_ocr and empty_visual:
            self.logger.error(
                f"Both OCR text extraction and visual processing failed for {artifact.name} - moving to review"
            )

            review_dir = self.paths.semantics_dir / "ocr_failed"
            review_dir.mkdir(exist_ok=True)
            artifact.rename(review_dir / artifact.name)
            return True

        return False

    def _log_llm_extraction_results(
        self, artifact_name: str, extraction_result: Dict[str, Any]
    ) -> None:
        if extraction_result["success"]:
            extracted_fields = extraction_result["extracted_fields"]
            non_unknown_fields = sum(
                1 for v in extracted_fields.values() if v != "UNKNOWN"
            )
            total_fields = len(extracted_fields)
            self.logger.info(
                f"LLM extraction successful for {artifact_name}: {non_unknown_fields}/{total_fields} fields extracted"
            )

            key_fields = ["title", "document_type", "issuer_name", "date_of_issue"]
            extracted_sample = {
                k: extracted_fields.get(k, "UNKNOWN") for k in key_fields
            }
            self.logger.debug(f"Key fields extracted: {extracted_sample}")
        else:
            self.logger.warning(
                f"LLM extraction failed for {artifact_name}: {extraction_result.get('error', 'Unknown error')}"
            )

    def process_completion(self) -> None:
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 6: COMPLETION AND FINAL PROCESSING")
        self.logger.info("=" * 80)

        artifacts = sorted(
            self.paths.llm_processed_dir.iterdir(), key=lambda p: p.stat().st_size
        )

        if not artifacts:
            self.logger.info("No artifacts found for completion")
            return

        for artifact in tqdm(artifacts, desc="Completing processing"):
            try:
                self._process_single_completion(artifact)
            except Exception as e:
                self.logger.error(
                    f"Failed to complete processing for {artifact.name}: {e}"
                )
                self.session_data.errors.append(
                    {"file": artifact.name, "stage": "completion", "error": str(e)}
                )

    def _process_single_completion(self, artifact: Path) -> None:
        artifact_id = artifact.stem[9:]
        profile_path = self.paths.profiles_dir / f"PROFILE-{artifact_id}.json"

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        completed_stages = [
            s for s in profile_data["stages"].values() if s.get("status") == "completed"
        ]

        profile_data["stages"]["completed"] = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_stages_completed": len(completed_stages),
        }

        profile_data["processing_summary"] = self._create_processing_summary(
            profile_data
        )

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        final_artifact = artifact.rename(self.paths.completed_dir / artifact.name)
        self._update_stage_counts("llm_processed", "completed")

        file_size = final_artifact.stat().st_size
        self._log_detailed_progress(artifact.name, file_size, "completed")
        self._update_session_json()

        self.logger.debug(f"Processing completed for: {artifact.name}")

    def _create_processing_summary(
        self, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "total_file_size_mb": profile_data.get("file_size", 0) / (1024 * 1024),
            "stages_completed": list(profile_data["stages"].keys()),
            "has_ocr_text": bool(
                profile_data.get("semantics", {}).get("all_text", "").strip()
            ),
            "has_visual_description": bool(
                profile_data.get("semantics", {}).get("all_imagery", "").strip()
            ),
            "llm_extraction_success": profile_data.get("llm_extraction", {}).get(
                "success", False
            ),
            "fields_with_data": self._count_extracted_fields(profile_data),
        }

    def _count_extracted_fields(self, profile_data: Dict[str, Any]) -> int:
        if not profile_data.get("llm_extraction", {}).get("success"):
            return 0

        extracted_fields = profile_data.get("llm_extraction", {}).get(
            "extracted_fields", {}
        )
        return len([v for v in extracted_fields.values() if v != "UNKNOWN"])

    def finalize_session(self) -> None:
        self.session_data.status = "completed"
        self.session_data.end_time = datetime.now().isoformat()
        self._update_session_json()

        total_elapsed = (datetime.now() - self.session_start_time).total_seconds()
        self._log_final_summary(total_elapsed)

    def _log_final_summary(self, total_elapsed: float) -> None:
        successful_files = self.session_data.stage_counts["completed"]
        failed_files = self.session_data.stage_counts["failed"]

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PAPERTRAIL SESSION PROCESSING COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"Session ID: {self.session_data.session_id}")
        self.logger.info(
            f"Total Runtime: {total_elapsed//3600:02.0f}:{(total_elapsed%3600)//60:02.0f}:{total_elapsed%60:02.0f}"
        )
        self.logger.info(f"Successfully Processed: {successful_files} files")
        self.logger.info(f"Failed Processing: {failed_files} files")
        self.logger.info(
            f"Average Speed: {self.session_data.performance['files_per_minute']:.1f} files/min"
        )

        final_file_summary = ", ".join(
            f"{count} {ext}" for ext, count in self.session_data.file_types.items()
        )
        self.logger.info(f"File Types Processed: {final_file_summary}")

        self.logger.info("\nOutput Locations:")
        self.logger.info(f"  Completed Files: {self.paths.completed_dir}")
        self.logger.info(f"  Profile Data: {self.paths.profiles_dir}")
        self.logger.info(f"  Session Logs: {self.paths.logs_dir}")
        self.logger.info(f"  Files for Review: {self.paths.review_dir}")

        if self.session_data.errors:
            self.logger.info(f"\nErrors Encountered ({len(self.session_data.errors)}):")
            for error in self.session_data.errors:
                self.logger.info(
                    f"  {error['file']} ({error['stage']}): {error['error']}"
                )

        self.logger.info("=" * 80)

    def run_full_pipeline(self) -> None:
        try:
            self._load_checksum_history()

            remaining_artifacts = self.process_duplicate_detection()
            if not remaining_artifacts:
                self.logger.info("No artifacts to process. Pipeline complete.")
                return

            self.process_uuid_renaming()
            self.process_metadata_extraction()
            self.process_semantic_extraction()
            self.process_llm_field_extraction()
            self.process_completion()

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            self.session_data.status = "failed"
            self.session_data.errors.append(
                {"file": "pipeline", "stage": "main", "error": str(e)}
            )
        finally:
            self.finalize_session()


def main() -> None:
    config = ProcessingConfig()
    file_config = FileTypeConfig()

    processor = PaperTrailProcessor(config, file_config)
    processor.run_full_pipeline()


if __name__ == "__main__":
    main()
