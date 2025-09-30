import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import psutil
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from config import (
    DEFAULT_AUTO_MODEL_SELECTION,
    DEFAULT_MEMORY_THRESHOLD,
    DEFAULT_PROCESSING_MODE,
    DEFAULT_REFRESH_INTERVAL,
    MAX_AVERAGE_PROCESSING_TIME,
    MAX_GPU_MEMORY_PERCENT,
    MAX_PROCESSING_TIME_PER_DOC,
    MIN_NONSILENT_RATIO,
    MIN_SILENCE_LEN_MS,
    MIN_TRANSCRIPTION_SUCCESS_RATE,
    PREFERRED_AUDIO_MODEL,
    RMS_ENERGY_THRESHOLD_DB,
    SILENCE_THRESH_DB,
)


class AudioQualityMode(Enum):
    """
    Audio processing quality modes for transcription accuracy vs speed tradeoff.

    FAST: Quick processing with beam_size=1, suitable for simple audio
    BALANCED: Good balance with beam_size=3 (default)
    HIGH_QUALITY: Maximum accuracy with beam_size=5, best for complex audio
    """

    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass
class AudioProcessingStats:
    """
    Comprehensive tracking of audio processing performance and quality metrics.

    This class maintains statistics across all audio processing operations
    to enable performance monitoring and optimization decisions.
    """

    files_processed: int = 0
    files_with_audio: int = 0
    files_transcribed: int = 0
    transcriptions_successful: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    memory_refreshes: int = 0
    model_switches: int = 0
    average_transcription_length: float = 0.0
    last_refresh_time: float = 0.0


class AudioProcessor:
    """
    Enhanced Audio Processor with hardware optimization and intelligent resource management.

    This processor handles audio transcription and detection using Whisper models with
    automatic hardware detection, memory management, and performance optimization.
    It supports various audio and video file formats with configurable quality modes.

    Key Features:
    - Automatic hardware detection and model selection
    - Robust audio presence detection using multiple methods
    - Memory usage monitoring and automatic model refresh
    - Configurable processing modes (fast, balanced, high-quality)
    - Comprehensive performance tracking and statistics
    - Robust error handling and fallback mechanisms
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the Audio Processor with automatic configuration.

        Args:
                        logger: Logger instance for tracking operations and errors

        Raises:
                        ValueError: If logger is None
                        TypeError: If logger is not a logging.Logger instance
        """
        # Validate required logger parameter
        if logger is None:
            raise ValueError(
                "Logger is required - AudioProcessor cannot be initialized without a logger"
            )
        if not isinstance(logger, logging.Logger):
            raise TypeError(f"Expected logging.Logger instance, got {type(logger)}")

        self.logger = logger
        self.logger.info("Initializing AudioProcessor with configuration constants")

        # Apply configuration constants for initialization parameters
        if DEFAULT_PROCESSING_MODE == "fast":
            self.quality_mode = AudioQualityMode.FAST
        elif DEFAULT_PROCESSING_MODE == "high_quality":
            self.quality_mode = AudioQualityMode.HIGH_QUALITY
        else:
            self.quality_mode = AudioQualityMode.BALANCED

        self.logger.info(f"Audio quality mode set to: {self.quality_mode.value}")
        self.logger.info(f"Auto model selection: {DEFAULT_AUTO_MODEL_SELECTION}")
        self.logger.info(f"Memory threshold: {DEFAULT_MEMORY_THRESHOLD}%")
        self.logger.info(f"Refresh interval: {DEFAULT_REFRESH_INTERVAL} files")

        # Initialize processing statistics
        self.stats = AudioProcessingStats()
        self.logger.debug("Processing statistics initialized")

        # Audio detection parameters
        self.logger.debug("Audio detection parameters initialized")

        # Model state variables - initialize to None until loaded
        self.model = None
        self.current_model_name = PREFERRED_AUDIO_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Beam size configuration based on quality mode
        self.beam_size_map = {
            AudioQualityMode.FAST: 1,
            AudioQualityMode.BALANCED: 3,
            AudioQualityMode.HIGH_QUALITY: 5,
        }

        # Load the model
        try:
            self.logger.info("Loading initial audio model...")
            self._load_model(PREFERRED_AUDIO_MODEL)
            self.logger.info("Audio model initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to load initial model: {e}")
            self.logger.warning("Model will be loaded on first use")

    def _load_model(self, model_id: str) -> None:
        """
        Load or reload the Whisper model with comprehensive error handling.

        Args:
                        model_id: Whisper model identifier (e.g., "large-v3", "medium", "base")

        Raises:
                        RuntimeError: If model loading fails after all fallback attempts
        """
        self.logger.info(f"Loading Whisper model: {model_id}")
        loading_start_time = time.time()

        try:
            # Clean up existing model if present
            if self.model is not None:
                self.logger.debug("Cleaning up existing model...")
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug("GPU memory cache cleared")
                gc.collect()
                self.logger.debug("Garbage collection completed")

            # Log hardware details
            if self.device == "cuda":
                self.logger.info(f"Using device: {self.device}")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                self.logger.info(f"GPU Details: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.logger.info("Using CPU for processing")

            # Load Whisper model
            self.logger.info(f"Loading model with compute_type: {self.compute_type}")
            self.model = WhisperModel(
                model_id, device=self.device, compute_type=self.compute_type
            )
            self.current_model_name = model_id

            loading_time = time.time() - loading_start_time
            self.logger.info(f"Model loading completed in {loading_time:.2f} seconds")

            # Log memory usage after loading
            memory_info = self._monitor_memory_usage()
            if self.device == "cuda":
                self.logger.info(
                    f"GPU memory usage after loading: {memory_info.get('gpu_memory_percent', 0):.1f}%"
                )
            else:
                self.logger.info(
                    f"System RAM usage after loading: {memory_info.get('system_ram_percent', 0):.1f}%"
                )

            self.stats.last_refresh_time = time.time()

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            self.logger.error("Model loading failed, attempting fallback strategies...")
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def _should_refresh_model(self) -> bool:
        """
        Determine if model should be refreshed based on usage patterns and performance.

        Returns:
                        bool: True if model should be refreshed
        """
        # Check file count trigger
        if self.stats.files_processed >= DEFAULT_REFRESH_INTERVAL:
            self.logger.info(
                f"Refresh triggered: processed {self.stats.files_processed} files (limit: {DEFAULT_REFRESH_INTERVAL})"
            )
            return True

        # Check GPU memory if available
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                memory_percent = (memory_used / memory_total) * 100

                if memory_percent > min(
                    DEFAULT_MEMORY_THRESHOLD, MAX_GPU_MEMORY_PERCENT
                ):
                    self.logger.warning(
                        f"Refresh triggered: GPU memory usage {memory_percent:.1f}% exceeds threshold"
                    )
                    return True
            except Exception as e:
                self.logger.warning(f"Could not check GPU memory: {e}")

        # Check processing time degradation
        if self.stats.files_processed > 0:
            avg_time_per_file = (
                self.stats.total_processing_time / self.stats.files_processed
            )
            if avg_time_per_file > MAX_AVERAGE_PROCESSING_TIME:
                self.logger.warning(
                    f"Refresh triggered: slow processing ({avg_time_per_file:.1f}s per file)"
                )
                return True

        # Check transcription success rate
        if self.stats.files_transcribed > 0:
            success_rate = (
                self.stats.transcriptions_successful / self.stats.files_transcribed
            )
            if success_rate < MIN_TRANSCRIPTION_SUCCESS_RATE:
                self.logger.warning(
                    f"Refresh triggered: low success rate ({success_rate:.2f})"
                )
                return True

        return False

    def refresh_model(self) -> None:
        """
        Refresh the model to clear context and free memory.
        """
        self.logger.info("Refreshing audio model to clear memory and context...")
        refresh_start_time = time.time()

        try:
            current_model_id = self.current_model_name
            self._load_model(current_model_id)

            self.stats.memory_refreshes += 1
            processing_count = self.stats.files_processed
            self.stats.files_processed = 0

            refresh_time = time.time() - refresh_start_time
            self.logger.info(
                f"Model refresh completed in {refresh_time:.2f}s (processed before: {processing_count})"
            )
        except Exception as e:
            self.logger.error(f"Failed to refresh model: {e}")
            raise

    def _monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage across system RAM and GPU VRAM.

        Returns:
                        Dict containing memory usage metrics in GB and percentages
        """
        memory_info = {}

        # System RAM monitoring
        try:
            ram = psutil.virtual_memory()
            memory_info["system_ram_used_gb"] = (ram.total - ram.available) / (1024**3)
            memory_info["system_ram_total_gb"] = ram.total / (1024**3)
            memory_info["system_ram_percent"] = ram.percent

            self.logger.debug(
                f"System RAM: {memory_info['system_ram_used_gb']:.1f}GB / "
                f"{memory_info['system_ram_total_gb']:.1f}GB ({memory_info['system_ram_percent']:.1f}%)"
            )
        except Exception as e:
            self.logger.warning(f"Could not get system RAM info: {e}")

        # GPU memory monitoring
        if torch.cuda.is_available() and self.device == "cuda":
            try:
                memory_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (
                    1024**3
                )
                memory_info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
                memory_info["gpu_memory_percent"] = (
                    memory_info["gpu_memory_used_gb"]
                    / memory_info["gpu_memory_total_gb"]
                ) * 100

                self.logger.debug(
                    f"GPU Memory: {memory_info['gpu_memory_used_gb']:.1f}GB / "
                    f"{memory_info['gpu_memory_total_gb']:.1f}GB ({memory_info['gpu_memory_percent']:.1f}%)"
                )
            except Exception as e:
                self.logger.warning(f"Could not get GPU memory info: {e}")

        return memory_info

    def has_audio(self, file_path: Path) -> bool:
        """
        Reliably detect if a file contains meaningful audio content.

        Uses multiple detection methods for robustness:
        1. Silence detection using pydub (primary method)
        2. RMS energy analysis using librosa (fallback/validation)

        Args:
                        file_path: Path to audio/video file to analyze

        Returns:
                        bool: True if file contains meaningful audio, False otherwise
        """
        self.logger.debug(f"Checking for audio presence in: {file_path.name}")
        detection_start = time.time()

        try:
            # Method 1: Silence detection with pydub (fast and reliable)
            try:
                audio = AudioSegment.from_file(str(file_path))
                audio_duration_ms = len(audio)

                self.logger.debug(
                    f"Audio file loaded: duration={audio_duration_ms/1000:.2f}s"
                )

                # Detect non-silent segments
                nonsilent = detect_nonsilent(
                    audio,
                    min_silence_len=MIN_SILENCE_LEN_MS,
                    silence_thresh=SILENCE_THRESH_DB,
                )

                if not nonsilent:
                    self.logger.debug("No non-silent segments detected")
                    return False

                # Calculate ratio of non-silent audio
                nonsilent_duration = sum(end - start for start, end in nonsilent)
                nonsilent_ratio = nonsilent_duration / audio_duration_ms

                self.logger.debug(
                    f"Non-silent ratio: {nonsilent_ratio:.2%} "
                    f"(threshold: {MIN_NONSILENT_RATIO:.2%})"
                )

                has_audio_result = nonsilent_ratio > MIN_NONSILENT_RATIO

            except Exception as e:
                self.logger.warning(
                    f"Pydub detection failed: {e}, trying librosa fallback"
                )

                # Method 2: RMS energy analysis with librosa (fallback)
                y, sr = librosa.load(str(file_path), sr=None)

                # Calculate RMS energy
                rms = librosa.feature.rms(y=y)[0]
                db = librosa.amplitude_to_db(rms)
                mean_db = np.mean(db)

                self.logger.debug(
                    f"RMS energy: {mean_db:.2f}dB "
                    f"(threshold: {RMS_ENERGY_THRESHOLD_DB}dB)"
                )

                has_audio_result = mean_db > RMS_ENERGY_THRESHOLD_DB

            detection_time = time.time() - detection_start
            self.logger.info(
                f"Audio detection completed in {detection_time:.2f}s: "
                f"{'AUDIO FOUND' if has_audio_result else 'NO AUDIO'}"
            )

            return has_audio_result

        except Exception as e:
            self.logger.error(f"Audio detection failed for {file_path.name}: {e}")
            # Conservative approach: assume no audio on error
            return False

    def transcribe_audio(self, file_path: Path) -> str:
        """
        Transcribe audio from file with comprehensive error handling and monitoring.

        Args:
                        file_path: Path to audio/video file to transcribe

        Returns:
                        str: Transcribed text, or error message if transcription fails

        Raises:
                        FileNotFoundError: If file_path doesn't exist
                        RuntimeError: If model is not loaded
        """
        self.logger.info(f"Starting transcription: {file_path.name}")
        self.logger.info(f"File size: {file_path.stat().st_size / 1024 / 1024:.1f} MB")

        start_time = time.time()

        # Check if model should be refreshed
        if self._should_refresh_model():
            self.logger.info("Auto-refreshing model due to configuration thresholds")
            self.refresh_model()

        # Monitor memory before processing
        memory_before = self._monitor_memory_usage()

        # Validate file_path existence
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if model is loaded
        if self.model is None:
            self.logger.error("Model not loaded. Attempting to load now...")
            try:
                self._load_model(self.current_model_name)
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise RuntimeError("Audio model not available for transcription")

        try:
            # First check if file has audio
            if not self.has_audio(file_path):
                self.logger.info("No meaningful audio detected, skipping transcription")
                self.stats.files_processed += 1
                return "NO_AUDIO_FOUND"

            self.stats.files_with_audio += 1
            self.stats.files_transcribed += 1

            # Get beam size based on quality mode
            beam_size = self.beam_size_map[self.quality_mode]

            self.logger.info(
                f"Transcribing with beam_size={beam_size} ({self.quality_mode.value} mode)"
            )

            # Perform transcription
            transcription_start = time.time()
            segments, info = self.model.transcribe(
                str(file_path),
                beam_size=beam_size,
                language=None,  # Auto-detect language
                vad_filter=True,  # Use voice activity detection
            )

            # Extract audio duration from info
            audio_duration = info.duration
            self.stats.total_audio_duration += audio_duration

            self.logger.info(
                f"Detected language: {info.language} "
                f"(probability: {info.language_probability:.2f})"
            )
            self.logger.info(f"Audio duration: {audio_duration:.2f}s")

            # Collect all segments
            all_segments = []
            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:
                    all_segments.append(segment_text)
                    self.logger.debug(
                        f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment_text}"
                    )

            # Join segments into full transcription
            full_transcription = " ".join(all_segments)

            transcription_time = time.time() - transcription_start

            if not full_transcription:
                self.logger.warning("Transcription produced no text")
                return "TRANSCRIPTION_FAILED"

            # Update statistics
            self.stats.transcriptions_successful += 1
            self.stats.average_transcription_length = (
                (
                    self.stats.average_transcription_length
                    * (self.stats.transcriptions_successful - 1)
                )
                + len(full_transcription)
            ) / self.stats.transcriptions_successful

            # Calculate processing metrics
            processing_time = time.time() - start_time
            self.stats.total_processing_time += processing_time
            self.stats.files_processed += 1

            # Monitor memory after processing
            memory_after = self._monitor_memory_usage()

            # Log completion summary
            self.logger.info(f"Transcription completed in {transcription_time:.2f}s")
            self.logger.info(f"Total processing time: {processing_time:.2f}s")
            self.logger.info(
                f"Transcription length: {len(full_transcription)} characters"
            )
            self.logger.info(
                f"Real-time factor: {transcription_time/audio_duration:.2f}x "
                f"({transcription_time:.1f}s / {audio_duration:.1f}s)"
            )

            if processing_time > MAX_PROCESSING_TIME_PER_DOC:
                self.logger.warning(
                    f"Processing time ({processing_time:.2f}s) exceeded threshold ({MAX_PROCESSING_TIME_PER_DOC}s)"
                )

            return full_transcription

        except Exception as e:
            self.logger.error(
                f"Transcription failed for {file_path.name}: {e}", exc_info=True
            )
            self.stats.files_processed += 1
            return "TRANSCRIPTION_FAILED"

    # def get_processing_stats(self) -> Dict[str, Any]:
    #     """
    #     Get comprehensive processing statistics with configuration context.


#
#     Returns:
#                     Dict containing comprehensive processing statistics
#     """
#     self.logger.debug("Generating processing statistics report...")
#
#     stats = {
#         "files_processed": self.stats.files_processed,
#         "files_with_audio": self.stats.files_with_audio,
#         "files_transcribed": self.stats.files_transcribed,
#         "transcriptions_successful": self.stats.transcriptions_successful,
#         "total_processing_time": self.stats.total_processing_time,
#         "total_audio_duration": self.stats.total_audio_duration,
#         "memory_refreshes": self.stats.memory_refreshes,
#         "model_switches": self.stats.model_switches,
#         "current_model": self.current_model_name,
#         "device": self.device,
#         "quality_mode": self.quality_mode.value,
#         "configuration": {
#             "refresh_interval": DEFAULT_REFRESH_INTERVAL,
#             "memory_threshold": DEFAULT_MEMORY_THRESHOLD,
#             "auto_model_selection": DEFAULT_AUTO_MODEL_SELECTION,
#             "detection_params": {
#                 "SILENCE_THRESH_DB": SILENCE_THRESH_DB,
#                 "MIN_SILENCE_LEN_MS": MIN_SILENCE_LEN_MS,
#                 "MIN_NONSILENT_RATIO": MIN_NONSILENT_RATIO,
#                 "RMS_ENERGY_THRESHOLD_DB": RMS_ENERGY_THRESHOLD_DB,
#             },
#         },
#     }
#
#     # Calculate rates and averages
#     if self.stats.files_processed > 0:
#         stats["avg_processing_time_per_file"] = (
#             self.stats.total_processing_time / self.stats.files_processed
#         )
#         stats["audio_detection_rate"] = (
#             self.stats.files_with_audio / self.stats.files_processed
#         )
#
#     if self.stats.files_transcribed > 0:
#         stats["transcription_success_rate"] = (
#             self.stats.transcriptions_successful / self.stats.files_transcribed
#         )
#
#     if self.stats.total_audio_duration > 0:
#         stats["avg_audio_duration"] = (
#             self.stats.total_audio_duration / self.stats.files_with_audio
#             if self.stats.files_with_audio > 0
#             else 0
#         )
#         stats["processing_real_time_factor"] = (
#             self.stats.total_processing_time / self.stats.total_audio_duration
#         )
#
#     stats["average_transcription_length"] = self.stats.average_transcription_length
#     stats["memory_usage"] = self._monitor_memory_usage()
#
#     self.logger.debug("Processing statistics report generated")
#     return stats
#
# def optimize_performance(self) -> Dict[str, Any]:
#     """
#     Analyze performance and suggest optimizations.
#
#     Returns:
#                     Dict containing performance analysis and recommendations
#     """
#     self.logger.info("Analyzing performance and generating recommendations...")
#
#     stats = self.get_processing_stats()
#     suggestions = []
#     performance_issues = []
#
#     # Analyze transcription success rate
#     success_rate = stats.get("transcription_success_rate", 0)
#     if success_rate < MIN_TRANSCRIPTION_SUCCESS_RATE:
#         issue = f"Transcription success rate ({success_rate:.2f}) below threshold"
#         performance_issues.append(issue)
#         suggestions.append("Consider using HIGH_QUALITY mode for better accuracy")
#         suggestions.append("Check audio quality of input files")
#
#     # Analyze processing speed
#     avg_time = stats.get("avg_processing_time_per_file", 0)
#     if avg_time > MAX_PROCESSING_TIME_PER_DOC:
#         issue = f"Average processing time ({avg_time:.1f}s) exceeds threshold"
#         performance_issues.append(issue)
#         suggestions.append("Consider switching to FAST mode for better speed")
#         suggestions.append("Check if using GPU acceleration")
#
#     # Analyze real-time factor
#     rtf = stats.get("processing_real_time_factor", 0)
#     if rtf > 0:
#         if rtf > 0.5:
#             suggestions.append(
#                 f"Real-time factor is {rtf:.2f}x - consider faster model or GPU"
#             )
#
#     # Memory usage analysis
#     memory_usage = stats.get("memory_usage", {})
#     gpu_percent = memory_usage.get("gpu_memory_percent", 0)
#     if gpu_percent > MAX_GPU_MEMORY_PERCENT:
#         issue = f"GPU memory usage ({gpu_percent:.1f}%) exceeds threshold"
#         performance_issues.append(issue)
#         suggestions.append("Consider reducing refresh interval")
#         suggestions.append("Switch to CPU if GPU memory is insufficient")
#
#     # Calculate performance score
#     performance_score = 0
#     if stats.get("transcription_success_rate") is not None:
#         performance_score += (
#             min(success_rate / MIN_TRANSCRIPTION_SUCCESS_RATE, 1.0) * 50
#         )
#     if avg_time > 0:
#         performance_score += (
#             max(1.0 - (avg_time / MAX_PROCESSING_TIME_PER_DOC), 0.0) * 50
#         )
#
#     return {
#         "current_performance": stats,
#         "performance_issues": performance_issues,
#         "optimization_suggestions": suggestions,
#         "overall_performance_score": performance_score,
#         "recommendations_priority": (
#             "high"
#             if len(performance_issues) > 2
#             else "medium" if len(performance_issues) > 0 else "low"
#         ),
#     }
