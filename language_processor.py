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

import requests
import json
import re
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import subprocess
import requests
import time
from requests.exceptions import RequestException, ConnectionError, Timeout
import os
import signal


@dataclass
class ModelSpec:
    """
    Specification for an LLM model including resource requirements and capabilities.

    Attributes:
        name (str): OLLAMA model identifier (e.g., "llama3.1:8b")
        min_ram_gb (float): Minimum system RAM required in gigabytes
        min_gpu_vram_gb (float): Minimum GPU VRAM required in gigabytes (0 for CPU-only)
        min_cpu_cores (int): Minimum CPU cores required for reasonable performance
        context_window (int): Maximum context length in tokens the model supports
        quality_score (int): Subjective quality rating from 1-10 (10 being best)
                           Based on accuracy, coherence, and instruction following
    """

    name: str
    min_ram_gb: float
    min_gpu_vram_gb: float
    min_cpu_cores: int
    context_window: int
    quality_score: int  # 1-10 scale, higher indicates better extraction quality


@dataclass
class HardwareConstraints:
    """
    Available system hardware resources for model execution.

    Attributes:
        max_ram_gb (float): Available system RAM in gigabytes
        max_gpu_vram_gb (float): Available GPU memory in gigabytes (0 if no GPU)
        max_cpu_cores (int): Number of available CPU cores for processing
    """

    max_ram_gb: float
    max_gpu_vram_gb: float
    max_cpu_cores: int


class LanguageProcessor:
    """
    Enhanced LLM-based document field extractor with intelligent model management.

    This class handles document field extraction using local LLM models via OLLAMA.
    It automatically detects system capabilities, selects appropriate models, and
    manages extraction context to maintain consistent performance.

    Key Features:
    - Hardware-aware model selection
    - Context refresh to prevent degradation
    - Multiple extraction strategies
    - Comprehensive error handling and logging
    - Performance monitoring

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> processor = LanguageProcessor(
        ...     logger=logger,
        ...     auto_model_selection=True
        ... )
        >>> result = processor.extract_fields(
        ...     ocr_text="Document content...",
        ...     visual_description="Visual analysis...",
        ...     metadata={"filename": "doc.pdf"},
        ...     uuid="doc-123"
        ... )
    """

    # ====================================================================================
    # CLASS CONSTANTS - Configuration parameters that control system behavior
    # ====================================================================================

    # MEMORY MANAGEMENT
    # ==================
    DEFAULT_RAM_USAGE_RATIO = (
        0.6  # Use 60% of available system RAM for model allocation
    )
    # How it works: When auto-detecting RAM, multiply total system RAM by this ratio
    # Example: 16GB system RAM * 0.6 = 9.6GB available for LLM models
    # Why 60%: Leaves 40% headroom for:
    #   - Operating system (typically needs 2-4GB)
    #   - Other applications and background processes
    #   - Python interpreter and our application overhead
    #   - Buffer for memory spikes during processing
    # Impact:
    #   - Higher values (0.7-0.8): Can run larger/better models but risk OOM crashes
    #   - Lower values (0.4-0.5): More stable but limits model selection
    #   - Too high (>0.8): System becomes unstable, potential crashes
    #   - Too low (<0.4): Unnecessarily restricts model options

    # CONTEXT MANAGEMENT
    # ===================
    DEFAULT_CONTEXT_REFRESH_INTERVAL = 50  # Refresh model context every 50 prompts
    # How it works: Counter tracks prompts sent to LLM, triggers reset at this limit
    # Why needed: LLM models accumulate context pollution over many requests:
    #   - Gradual drift in response quality and consistency
    #   - Potential hallucinations from accumulated noise
    #   - Slower response times as context window fills
    #   - Less reliable JSON formatting in responses
    # Process: Sends lightweight "reset context" prompt to clear accumulated state
    # Impact:
    #   - Lower values (20-30): More frequent refreshes, better consistency, higher overhead
    #   - Higher values (80-100): Less overhead but potential quality degradation
    #   - Too low (<10): Excessive overhead, minimal benefit
    #   - Too high (>100): Noticeable quality drift, inconsistent outputs
    # Typical signs refresh is needed: Malformed JSON, verbose responses, off-topic content

    # NETWORK AND TIMEOUT CONFIGURATION
    # ==================================
    DEFAULT_TIMEOUT_SECONDS = 300  # 5 minute timeout for LLM API requests
    # How it works: Maximum time to wait for OLLAMA API response before giving up
    # Why 5 minutes: Accounts for various processing scenarios:
    #   - Small models (1-3B params): Usually respond in 5-30 seconds
    #   - Large models (13B+ params): Can take 60-180 seconds for complex prompts
    #   - First request after model load: Additional 10-30 seconds for initialization
    #   - System under load: Response times can increase 2-3x
    #   - GPU vs CPU processing: CPU can be 5-10x slower than GPU
    # Impact:
    #   - Shorter timeouts (60-120s): Faster failure detection but may timeout valid requests
    #   - Longer timeouts (600s+): Less chance of premature timeouts but slower error detection
    #   - Network factors: Local OLLAMA (fast), remote API (add network latency)

    # LLM GENERATION PARAMETERS
    # ==========================
    DEFAULT_TEMPERATURE = (
        0.1  # Low temperature for consistent, deterministic extraction
    )
    # How it works: Controls randomness in token selection (0.0 = deterministic, 1.0 = very random)
    # Temperature effect on token probability distribution:
    #   - 0.0: Always picks highest probability token (completely deterministic)
    #   - 0.1: Slight randomness, heavily favors high-probability tokens
    #   - 0.5: Balanced randomness, good for creative tasks
    #   - 1.0: High randomness, can produce unexpected/creative outputs
    # Why 0.1 for extraction: We want consistent, predictable field extraction:
    #   - Same document should produce same field values across runs
    #   - Reduces hallucinations and invented content
    #   - Improves JSON formatting consistency
    #   - Better adherence to "UNKNOWN" for missing fields
    # Trade-offs:
    #   - Lower (0.0-0.2): More consistent but less flexible with edge cases
    #   - Higher (0.3-0.7): More adaptive but less predictable/reliable

    DEFAULT_TOP_P = 0.9  # Nucleus sampling: consider top 90% probability mass
    # How it works: Alternative to temperature - limits token selection to top N% of probability
    # Process: Sort all possible tokens by probability, select from top 90% cumulative mass
    # Example: If "invoice" has 40% probability, "receipt" 30%, "document" 20%, "paper" 10%
    #   - Cumulative: 40% + 30% + 20% = 90%, so only consider first 3 tokens
    #   - Excludes low-probability tokens that might be noise or hallucinations
    # Why 0.9: Balances quality and flexibility:
    #   - Allows model to choose from most likely options
    #   - Filters out improbable/nonsensical tokens
    #   - Works well with low temperature for consistent extraction
    # Impact:
    #   - Higher values (0.95-1.0): More token options, potentially more creative but less focused
    #   - Lower values (0.7-0.8): More restrictive, very conservative outputs
    #   - Combined with low temperature: Produces highly consistent, reliable extractions

    # TOKEN LIMITS AND TRUNCATION
    # =============================
    MAX_RESPONSE_TOKENS = (
        2000  # Limit LLM response length to prevent runaway generation
    )
    # How it works: Tells OLLAMA to stop generating after N tokens
    # Why needed: Prevents several issues:
    #   - Runaway generation (model keeps talking beyond useful content)
    #   - Excessive token usage and costs
    #   - Very long response times for verbose outputs
    #   - Potential context window overflow in long conversations
    # What 2000 tokens represents:
    #   - ~1500-2000 words of text
    #   - Complete JSON response with all 15+ fields
    #   - Room for explanatory text if model is verbose
    # Typical field extraction needs: 200-800 tokens for clean JSON response
    # Impact:
    #   - Too low (500-1000): Risk of truncated JSON responses
    #   - Too high (5000+): Allows verbose, unfocused responses

    MAX_OCR_TOKENS = 2000  # Truncate OCR text input to prevent context overflow
    # How it works: Limits OCR text to approximately 2000 tokens before sending to LLM
    # Token estimation: ~4 characters per token (rough approximation)
    # So 2000 tokens ≈ 8000 characters of OCR text
    # Why truncation needed:
    #   - LLM context windows are limited (2K-8K tokens for most models)
    #   - Need room for prompt instructions, visual description, and response
    #   - Very long documents would overflow context and cause failures
    # What gets truncated: Only the beginning of document is kept
    # Trade-offs:
    #   - Shorter limits (1000): More room for instructions but less document content
    #   - Longer limits (4000): More content but risk of context overflow
    # Note: Important info is usually at document beginning (titles, headers, key fields)

    MAX_VISUAL_TOKENS = 750  # Truncate visual description to preserve context space
    # How it works: Limits visual analysis text to ~750 tokens (~3000 characters)
    # Purpose of visual description:
    #   - Describes document layout, formatting, visual elements
    #   - Helps identify document type and structure
    #   - Provides context when OCR text is unclear or incomplete
    #   - Examples: "Header with logo", "table with 3 columns", "official seal"
    # Why smaller than OCR limit:
    #   - Visual descriptions are typically shorter and more structured
    #   - OCR text contains the actual field values we need
    #   - Visual info is supplementary context, not primary content
    # Impact:
    #   - Shorter (500): Very brief visual context, might miss layout clues
    #   - Longer (1500): More visual detail but less room for OCR content

    OLLAMA_SERVER = None

    def __init__(
        self,
        logger: logging.Logger,
        max_ram_gb: Optional[float] = None,
        max_gpu_vram_gb: Optional[float] = None,
        max_cpu_cores: Optional[int] = None,
        context_refresh_interval: int = DEFAULT_CONTEXT_REFRESH_INTERVAL,
        host: str = "http://localhost:11434",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        auto_model_selection: bool = True,
    ):
        """Initialize the LanguageProcessor with proper error handling."""

        # Initialize ALL instance variables FIRST
        self.logger = logger
        self.host = host
        self.timeout = timeout
        self.context_refresh_interval = context_refresh_interval

        # Initialize tracking counters BEFORE any method calls
        self.processed_count = 0
        self.prompts_since_refresh = 0
        self.total_processing_time = 0.0
        self.successful_extractions = 0
        self.failed_extractions = 0

        self.logger.info("=== Initializing LanguageProcessor ===")
        self.logger.info(
            f"refresh_interval={context_refresh_interval}, "
            f"host={host}, timeout={timeout}s"
        )

        try:
            # Now do hardware detection
            self.hardware_constraints = self._detect_hardware_constraints(
                max_ram_gb, max_gpu_vram_gb, max_cpu_cores
            )

            # Initialize model registry
            self.model_registry = self._initialize_model_registry()
            self.logger.info(f"Loaded {len(self.model_registry)} model specifications")

            # Select and load optimal model
            if auto_model_selection:
                self.model = self._select_optimal_model()
            else:
                self.model = "mistral-nemo:12b"
                self.logger.warning(f"Using specified fallback model: {self.model}")

            # Load the model
            self._load_model()

            self.logger.info("Language processor initialization completed successfully")
            self.logger.info(f"Active model: {self.model}")
            self.logger.info(
                f"Hardware constraints: RAM={self.hardware_constraints.max_ram_gb:.1f}GB, "
                f"GPU={self.hardware_constraints.max_gpu_vram_gb:.1f}GB, "
                f"CPU={self.hardware_constraints.max_cpu_cores} cores"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LanguageProcessor: {e}")
            # Set safe fallback values
            if not hasattr(self, "hardware_constraints"):
                self.hardware_constraints = HardwareConstraints(
                    max_ram_gb=8.0, max_gpu_vram_gb=0.0, max_cpu_cores=4
                )
            if not hasattr(self, "model_registry"):
                self.model_registry = self._initialize_model_registry()
            if not hasattr(self, "model"):
                self.model = "llama3.2:3b"
            raise

    def _detect_hardware_constraints(
        self,
        max_ram_gb: Optional[float],
        max_gpu_vram_gb: Optional[float],
        max_cpu_cores: Optional[int],
    ) -> HardwareConstraints:
        """
        Detect available hardware resources or use provided constraints.

        This method intelligently detects system capabilities to inform model selection.
        It uses conservative estimates to ensure stable operation under load.

        Args:
            max_ram_gb: Override for maximum RAM usage (None for auto-detection)
            max_gpu_vram_gb: Override for maximum GPU memory (None for auto-detection)
            max_cpu_cores: Override for CPU core count (None for auto-detection)

        Returns:
            HardwareConstraints object with detected or specified limits

        Note:
            Auto-detected RAM is limited to 70% of total to leave headroom for OS and
            other applications. GPU detection requires GPUtil package.
        """
        self.logger.info("--- Detecting Hardware Constraints ---")

        # RAM detection with conservative allocation
        if max_ram_gb is None:
            total_ram = psutil.virtual_memory().total / (
                1024**3
            )  # Bytes to GB conversion
            available_ram = total_ram * self.DEFAULT_RAM_USAGE_RATIO
            self.logger.info(
                f"RAM detection: total={total_ram:.1f}GB, "
                f"allocated={available_ram:.1f}GB "
                f"({self.DEFAULT_RAM_USAGE_RATIO*100:.0f}% of total)"
            )
        else:
            available_ram = max_ram_gb
            self.logger.info(f"RAM override: using specified {available_ram:.1f}GB")

        # CPU detection with logical vs physical core consideration
        if max_cpu_cores is None:
            # Use physical cores for more accurate performance estimation
            physical_cores = psutil.cpu_count(logical=False) or 1
            logical_cores = psutil.cpu_count(logical=True) or 1
            available_cores = physical_cores  # Conservative estimate
            self.logger.info(
                f"CPU detection: physical={physical_cores}, "
                f"logical={logical_cores}, allocated={available_cores}"
            )
        else:
            available_cores = max_cpu_cores
            self.logger.info(f"CPU override: using specified {available_cores} cores")

        # GPU detection with error handling for missing dependencies
        if max_gpu_vram_gb is None:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    # Find GPU with most memory (GB conversion from MB)
                    available_gpu = max([gpu.memoryTotal / 1024 for gpu in gpus])
                    gpu_names = [gpu.name for gpu in gpus]
                    self.logger.info(
                        f"GPU detection: found {len(gpus)} GPU(s), "
                        f"max_memory={available_gpu:.1f}GB"
                    )
                    self.logger.debug(f"GPU details: {gpu_names}")
                else:
                    available_gpu = 0
                    self.logger.info("GPU detection: no GPUs found")
            except ImportError:
                available_gpu = 0
                self.logger.warning(
                    "GPU detection: GPUtil not available, assuming CPU-only mode"
                )
            except Exception as e:
                available_gpu = 0
                self.logger.error(f"GPU detection failed: {e}, assuming CPU-only mode")
        else:
            available_gpu = max_gpu_vram_gb
            self.logger.info(f"GPU override: using specified {available_gpu:.1f}GB")

        constraints = HardwareConstraints(
            max_ram_gb=available_ram,
            max_gpu_vram_gb=available_gpu,
            max_cpu_cores=available_cores,
        )

        self.logger.info(f"Hardware constraints finalized: {constraints}")
        return constraints

    def _initialize_model_registry(self) -> List[ModelSpec]:
        """
        Initialize registry of available models with their resource requirements.

        This registry contains empirically determined resource requirements for various
        LLM models. Requirements are conservative estimates to ensure stable operation.

        Returns:
            List of ModelSpec objects with requirements and quality ratings

        Note:
            Quality scores are subjective ratings (1-10) based on:
            - Accuracy of field extraction
            - Instruction following capability
            - Consistency across document types
            - Speed vs accuracy trade-offs
        """
        self.logger.debug("Initializing model registry with resource requirements")

        models = [
            # Small, fast models for resource-constrained environments
            ModelSpec("llama3.2:1b", 2, 0, 2, 2048, 6),  # Basic accuracy, very fast
            ModelSpec("llama3.2:3b", 4, 0, 4, 2048, 7),  # Good balance for small docs
            # Medium models for balanced performance
            ModelSpec("llama3.1:8b", 8, 0, 6, 4096, 8),  # Excellent general purpose
            ModelSpec("qwen2.5:7b", 7, 0, 4, 4096, 8),  # Strong multilingual support
            ModelSpec("mistral:7b", 7, 0, 4, 4096, 7),  # Fast and reliable
            ModelSpec("gemma2:9b", 9, 6, 6, 8192, 8),  # Good with structured data
            ModelSpec(
                "codellama:13b", 13, 8, 8, 4096, 8
            ),  # Excellent for technical docs
            # Large models for maximum accuracy
            ModelSpec("qwen2.5:14b", 14, 8, 8, 8192, 9),  # Superior multilingual
            ModelSpec("llama3.1:70b", 32, 24, 16, 8192, 10),  # Highest quality, slow
        ]

        self.logger.info(f"Model registry initialized with {len(models)} models")
        self.logger.debug(
            "Model quality scores: "
            + ", ".join([f"{m.name}={m.quality_score}" for m in models])
        )

        return models

    def _select_optimal_model(self) -> str:
        """
        Select the best model that fits within hardware constraints.

        Selection algorithm:
        1. Filter models that fit within hardware limits
        2. Rank by quality score (accuracy/capability metric)
        3. Select highest-scoring model
        4. Fallback to smallest model if none fit

        Returns:
            String identifier of the selected model

        Note:
            Quality scores range from 1-10 where:
            - 1-3: Basic functionality, limited accuracy
            - 4-6: Adequate for simple documents
            - 7-8: Good for most document types
            - 9-10: Excellent accuracy, handles complex documents
        """
        self.logger.info("--- Selecting Optimal Model ---")
        self.logger.debug(f"Evaluating {len(self.model_registry)} candidate models")

        suitable_models = []
        rejected_models = []

        for model in self.model_registry:
            fits_constraints = (
                model.min_ram_gb <= self.hardware_constraints.max_ram_gb
                and model.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
                and model.min_cpu_cores <= self.hardware_constraints.max_cpu_cores
            )

            if fits_constraints:
                suitable_models.append(model)
                self.logger.debug(
                    f"✓ {model.name}: quality={model.quality_score}, "
                    f"ram={model.min_ram_gb}GB, gpu={model.min_gpu_vram_gb}GB, "
                    f"cores={model.min_cpu_cores}"
                )
            else:
                rejected_models.append(model)
                reasons = []
                if model.min_ram_gb > self.hardware_constraints.max_ram_gb:
                    reasons.append(
                        f"RAM: need {model.min_ram_gb}GB, have {self.hardware_constraints.max_ram_gb:.1f}GB"
                    )
                if model.min_gpu_vram_gb > self.hardware_constraints.max_gpu_vram_gb:
                    reasons.append(
                        f"GPU: need {model.min_gpu_vram_gb}GB, have {self.hardware_constraints.max_gpu_vram_gb:.1f}GB"
                    )
                if model.min_cpu_cores > self.hardware_constraints.max_cpu_cores:
                    reasons.append(
                        f"CPU: need {model.min_cpu_cores} cores, have {self.hardware_constraints.max_cpu_cores}"
                    )

                self.logger.debug(f"✗ {model.name}: {'; '.join(reasons)}")

        if not suitable_models:
            self.logger.warning(
                f"No models fit hardware constraints! "
                f"Rejected {len(rejected_models)} models. "
                f"Using minimal fallback model."
            )
            return "llama3.2:1b"  # Smallest available model

        # Select model with highest quality score
        best_model = max(suitable_models, key=lambda m: m.quality_score)

        self.logger.info(f"Model selection complete:")
        self.logger.info(f"  Selected: {best_model.name}")
        self.logger.info(f"  Quality score: {best_model.quality_score}/10")
        self.logger.info(
            f"  Requirements: {best_model.min_ram_gb}GB RAM, "
            f"{best_model.min_gpu_vram_gb}GB GPU, {best_model.min_cpu_cores} cores"
        )
        self.logger.info(f"  Context window: {best_model.context_window:,} tokens")
        self.logger.info(f"  Alternatives considered: {len(suitable_models)-1}")

        return best_model.name

    def _load_model(self):
        """
        Load the selected model in OLLAMA with comprehensive status checking.

        This method handles:
        - Checking if model is already loaded
        - Pulling model from registry if needed
        - Verifying successful loading
        - Detailed progress logging

        Raises:
            ConnectionError: If OLLAMA server is not accessible
            TimeoutError: If model loading exceeds timeout
            RuntimeError: If model loading fails
        """
        self.logger.info(f"--- Loading Model: {self.model} ---")

        try:

            # Start OLLAMA server automatically
            self.logger.info("Starting OLLAMA server...")
            try:
                # Get the PID when you start it
                ollama_process: subprocess.Popen = subprocess.Popen(
                    ["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                pid: int = ollama_process.pid
                self.OLLAMA_SERVER = ollama_process
                self.pid = pid
                self.logger.info(f"OLLAMA server started with PID: {pid}")
            except FileNotFoundError:
                self.logger.error(
                    "OLLAMA executable not found. Please install OLLAMA first."
                )
                raise
            except Exception as e:
                self.logger.error(f"Failed to start OLLAMA process: {e}")
                raise

            # Wait for server to start up
            self.logger.info("Waiting for server to become responsive...")
            server_ready: bool = False
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response: requests.Response = requests.get(
                        "http://localhost:11434/api/tags", timeout=2
                    )
                    if response.status_code == 200:
                        self.logger.info(
                            f"OLLAMA server is responsive after {i+1} seconds"
                        )
                        server_ready = True
                        break
                except (ConnectionError, Timeout, RequestException):
                    time.sleep(1)
                    self.logger.debug(f"Server not ready, attempt {i+1}/30")
                except Exception as e:
                    self.logger.warning(f"Unexpected error checking server: {e}")
                    time.sleep(1)

            if not server_ready:
                self.logger.error(
                    "OLLAMA server failed to become responsive within 30 seconds"
                )
                ollama_process.terminate()
                raise RuntimeError("OLLAMA server startup failed")

            self.logger.debug("OLLAMA server is accessible")

            # Check if model is already loaded
            loaded_models = [
                model["name"] for model in response.json().get("models", [])
            ]
            self.logger.debug(f"Currently loaded models: {loaded_models}")

            if self.model in loaded_models:
                self.logger.info(f"Model {self.model} already loaded and ready")
                return

            # Pull model if not available
            self.logger.info(f"Model {self.model} not found locally, downloading...")
            self.logger.warning(
                "Model download may take several minutes depending on size and connection"
            )

            pull_payload = {"name": self.model}

            # Use extended timeout for model downloads (10 minutes)
            download_timeout = 600
            self.logger.debug(
                f"Starting model download with {download_timeout}s timeout..."
            )

            response = requests.post(
                f"{self.host}/api/pull", json=pull_payload, timeout=download_timeout
            )

            if response.status_code == 200:
                self.logger.info(f"Successfully loaded model {self.model}")

                # Verify model is now available
                verify_response = requests.get(f"{self.host}/api/tags", timeout=30)
                if verify_response.status_code == 200:
                    current_models = [
                        m["name"] for m in verify_response.json().get("models", [])
                    ]
                    if self.model in current_models:
                        self.logger.info(f"Model {self.model} verified as loaded")
                    else:
                        self.logger.warning(
                            f"Model {self.model} not found in verification check"
                        )
            else:
                error_msg = (
                    f"Failed to load model {self.model}: HTTP {response.status_code}"
                )
                if response.text:
                    error_msg += f" - {response.text}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

        except requests.Timeout:
            error_msg = (
                f"Timeout loading model {self.model} (exceeded {download_timeout}s)"
            )
            self.logger.error(error_msg)
            raise TimeoutError(error_msg)
        except requests.ConnectionError as e:
            error_msg = f"Cannot connect to OLLAMA server at {self.host}: {e}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error loading model {self.model}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _refresh_context(self):
        """
        Refresh the model context to prevent performance degradation.

        During long processing sessions, LLM models can experience context drift
        and performance degradation. This method sends a lightweight reset prompt
        to clear accumulated context and restore consistent performance.

        The refresh is performed automatically based on context_refresh_interval
        but can also be called manually when performance issues are detected.

        Note:
            Context refresh adds a small latency overhead (~1-2 seconds) but
            significantly improves consistency for batch processing.
        """
        self.logger.info("--- Refreshing Model Context ---")
        self.logger.debug(f"Prompts since last refresh: {self.prompts_since_refresh}")

        start_time = datetime.now()

        try:
            # Send a lightweight context reset prompt
            reset_payload = {
                "model": self.model,
                "prompt": "Reset context. Ready for new document analysis.",
                "stream": False,
                "options": {
                    "num_predict": 10,  # Minimal response to save time
                    "temperature": 0.1,
                },
            }

            self.logger.debug("Sending context reset prompt...")
            response = requests.post(
                f"{self.host}/api/generate",
                json=reset_payload,
                timeout=30,  # Short timeout for reset operation
            )

            refresh_time = (datetime.now() - start_time).total_seconds()

            if response.status_code == 200:
                self.prompts_since_refresh = 0
                self.logger.info(
                    f"Context refreshed successfully in {refresh_time:.2f}s"
                )
                self.logger.debug(
                    f"Reset response: {response.json().get('response', '')[:100]}..."
                )
            else:
                self.logger.warning(
                    f"Context refresh failed: HTTP {response.status_code} - {response.text}"
                )
                self.logger.warning("Continuing with potentially degraded context")

        except requests.Timeout:
            refresh_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Context refresh timed out after {refresh_time:.2f}s")
        except Exception as e:
            refresh_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Error refreshing context after {refresh_time:.2f}s: {e}"
            )

    def _validate_and_clean_fields(self, fields: Dict[str, str]) -> Dict[str, str]:
        """Basic validation and cleaning of extracted field values"""
        cleaned_fields = {}

        for field_name, value in fields.items():
            # Handle empty or None values
            if not value or not value.strip():
                cleaned_fields[field_name] = "UNKNOWN"
                continue

            # Basic cleanup
            cleaned_value = value.strip()

            # Remove common LLM response prefixes
            prefixes_to_remove = [
                "the ",
                "a ",
                "an ",
                "this is ",
                "this appears to be ",
                "the document is ",
                "the document shows ",
                "based on the text ",
            ]

            for prefix in prefixes_to_remove:
                if cleaned_value.lower().startswith(prefix):
                    cleaned_value = cleaned_value[len(prefix) :].strip()
                    break

            # Check for variations of "unknown"
            if cleaned_value.lower() in [
                "unknown",
                "n/a",
                "none",
                "not found",
                "not available",
                "unclear",
            ]:
                cleaned_fields[field_name] = "UNKNOWN"
                continue

            # Basic field-specific cleaning
            if "date" in field_name:
                # Extract date patterns if they exist
                import re

                date_match = re.search(
                    r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}",
                    cleaned_value,
                )
                cleaned_value = date_match.group(0) if date_match else "UNKNOWN"

            elif field_name == "confidentiality_level":
                # Normalize confidentiality levels
                conf_lower = cleaned_value.lower()
                if "public" in conf_lower:
                    cleaned_value = "Public"
                elif "internal" in conf_lower:
                    cleaned_value = "Internal"
                elif "confidential" in conf_lower or "restricted" in conf_lower:
                    cleaned_value = "Confidential"
                else:
                    cleaned_value = "Public"  # Default

            elif field_name == "document_type":
                # Lowercase with underscores for consistency
                cleaned_value = (
                    cleaned_value.lower().replace(" ", "_").replace("-", "_")
                )

            # Remove excessive whitespace and limit length
            cleaned_value = " ".join(cleaned_value.split())
            if len(cleaned_value) > 1000:
                cleaned_value = cleaned_value[:1000].strip()

            # Final assignment
            cleaned_fields[field_name] = cleaned_value if cleaned_value else "UNKNOWN"

        return cleaned_fields

    def _clean_response(self, response: str):
        # Strip common LLM artifacts
        cleaned = response.strip()

        # Remove markdown formatting
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)  # Remove **bold**
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)  # Remove *italic*
        cleaned = re.sub(r"`(.*?)`", r"\1", cleaned)  # Remove `code`

        # Remove explanatory phrases
        artifacts = [
            "Based on the provided document",
            "After reviewing the content",
            "The document appears to be",
            "I can identify that",
            "Upon analysis,",
        ]
        for artifact in artifacts:
            if cleaned.startswith(artifact):
                # Find the actual answer after the explanation
                cleaned = cleaned.split(":")[-1].strip()

        return cleaned.strip('"').strip("'")

    def extract_fields(
        self,
        ocr_text: str,
        visual_description: str,
        metadata: Dict[str, Any],
        uuid: str,
    ) -> Dict[str, Any]:
        """
        Extract all document fields using LLM with comprehensive error handling.
        """
        start_time = datetime.now()
        self.logger.info(f"=== Starting field extraction for document {uuid} ===")
        self.logger.info(
            f"Input sizes: OCR={len(ocr_text)} chars, "
            f"d{len(visual_description)} chars, "
            f"Metadata keys={list(metadata.keys())}"
        )

        # Check if context refresh is needed
        if self.prompts_since_refresh >= self.context_refresh_interval:
            self.logger.info(
                f"Context refresh triggered "
                f"({self.prompts_since_refresh}/{self.context_refresh_interval} prompts)"
            )
            self._refresh_context()

        try:

            # Field extraction with full data and system prompts
            field_instructions = {
                "title": """Create a descriptive title that captures the document's purpose and content.

                If the document has an official title, use it. If not, synthesize a descriptive title based on:
                - Document type and purpose
                - Issuing organization
                - Main subject matter

                Examples:
                - "Immigration and Refugee Board Virtual Hearing Notice"
                - "Refugee Protection Division Hearing Preparation Guide"
                - "IRB Document Submission Requirements and Procedures"

                Return ONLY the title. No quotes, no explanations.""",
                "document_type": """Identify the specific type of document this is.

            Examples of document types:
            - birth_certificate, passport_us, driver_license_ca
            - invoice, receipt, bank_statement, credit_report
            - contract, lease_agreement, employment_contract
            - medical_record, prescription, insurance_policy
            - w2_tax_form, 1099, tax_return
            - court_order, legal_notice, business_license
            - university_transcript, diploma, certificate
            - property_deed, mortgage_document, warranty

            Return ONLY the specific document type in lowercase with underscores. If uncertain, return "UNKNOWN".""",
                "language": """Analyze the document text and identify ALL languages present.

            Document contains sections in multiple languages. List ALL languages found.
            Examples:
            - If English only: "English"
            - If French only: "French"
            - If both: "English, French"
            - If trilingual: "English, French, Spanish"

            Return ONLY language names, comma-separated.""",
                "confidentiality_level": """Determine the confidentiality or security classification of this document.

            Look for markings or indicators such as:
            - CONFIDENTIAL, CLASSIFIED, SECRET, TOP SECRET
            - INTERNAL USE ONLY, PROPRIETARY, RESTRICTED
            - PUBLIC, FOR PUBLIC RELEASE
            - Security stamps, watermarks, or headers

            Classification levels:
            - Public: No restrictions, can be freely shared
            - Internal: Internal use within organization
            - Confidential: Sensitive information, restricted access
            - Restricted: Highly sensitive, top-secret information

            Return ONLY one of these words: Public, Internal, Confidential, Restricted""",
                "translator_name": """If this document has been translated, identify the translator's name.

            Look for:
            - Translator certifications or signatures
            - Translation agency information
            - "Translated by" notices
            - Official translation stamps or seals
            - Translator contact information

            Return ONLY the translator's full name. If this is not a translated document or no translator is identified, return "UNKNOWN".""",
                "issuer_name": """Identify who issued, created, or published this document.

            Look for:
            - Organization names, agencies, departments
            - Company names or letterheads
            - Government agencies or institutions
            - Individual names (for personally issued documents)
            - Official stamps or seals with issuer information

            Return ONLY the full official name of the issuer. If unclear, return "UNKNOWN".""",
                "officiater_name": """Identify any official authority that validated, certified, witnessed, or authorized this document.

            Look for:
            - Notary public names and seals
            - Certifying agency names
            - Official witnesses or authorizing bodies
            - Government officials who signed or stamped
            - Licensing boards or regulatory authorities

            Return ONLY the name of the official authority. If no official validation exists, return "UNKNOWN".""",
                "date_created": """Find when this document was originally created, written, or authored.

            Look for:
            - Creation dates, authored dates, written dates
            - "Created on", "Date created", "Authored"
            - Document composition or drafting dates

            Return the date in YYYY-MM-DD format. If no creation date is found, return "UNKNOWN".""",
                "date_of_reception": """Find when this document was received by the current holder.

            Look for:
            - "Received", "Date received", "Arrival date"
            - Postal stamps or delivery confirmations
            - Filing dates or intake dates
            - "Delivered on" stamps

            Return the date in YYYY-MM-DD format. If no reception date is found, return "UNKNOWN".""",
                "date_of_issue": """Find the official issue, publication, or release date of this document.

            Look for:
            - "Issued", "Date of issue", "Publication date"
            - "Released", "Effective date"
            - Official dating stamps or seals
            - Government or agency issue dates

            Return the date in YYYY-MM-DD format. If no issue date is found, return "UNKNOWN".""",
                "date_of_expiry": """Find when this document expires, becomes invalid, or requires renewal.

            Look for:
            - "Expires", "Expiration date", "Valid until"
            - "Renewal required", "Valid through"
            - License or certification expiry dates
            - "Not valid after" dates

            Return the date in YYYY-MM-DD format. If no expiration date exists, return "UNKNOWN".""",
                "tags": """Create comprehensive keywords that describe this document for search and categorization purposes.

            Include keywords for:
            - Document category (legal, medical, financial, educational, personal, business, government, technical)
            - Subject matter (taxes, healthcare, employment, education, property, travel, identification, insurance)
            - Content type (contract, certificate, statement, report, application, notice, invoice, receipt)
            - Industry/field (healthcare, legal, finance, education, technology, government, military)
            - Geographic relevance (federal, state, local, international, specific regions)
            - Time relevance (annual, quarterly, monthly, historical, current)
            - Action items (renewal_required, payment_due, action_needed, informational_only)
            - Format type (official, certified, notarized, electronic, handwritten, typed)

            Return 15-25 comma-separated keywords. If document content is unclear, return "UNKNOWN".""",
                "version_notes": """Analyze document versioning, revision history, and administrative metadata.

            Look for:
            - Version numbers, revision dates, edition information
            - Document control numbers, form numbers
            - "Supersedes" notices, amendment references
            - Administrative tracking information

            Provide a professional assessment of document currency and version status.
            If no versioning found, state: "No explicit version control information identified."

            Use formal, administrative language.""",
                "utility_notes": """Provide a professional analysis of this document's administrative function and legal purpose.

            Analyze:
            - Regulatory or statutory requirements this document fulfills
            - Administrative processes it initiates or supports
            - Legal obligations or rights it establishes
            - Institutional workflows it facilitates
            - Compliance requirements it addresses

            Write in formal, bureaucratic language appropriate for government documentation.""",
                "additional_notes": """Document significant administrative, security, or procedural characteristics not covered elsewhere.

            Note:
            - Security classifications, handling restrictions
            - Authentication elements, official markings
            - Distribution methods, transmission records
            - Document quality, preservation concerns
            - Cross-references to related administrative instruments

            Present observations in formal, official terminology suitable for administrative records.""",
            }

            SYSTEM_PROMPT = """You are a document extraction tool. Extract ONLY the requested information.

            CRITICAL RULES:
            - Return ONLY the answer, nothing else
            - NO explanations, NO reasoning, NO "based on", NO "therefore"
            - NO sentences, just the raw information
            - If not found, return exactly: UNKNOWN

            Examples:
            GOOD: "Immigration and Refugee Board of Canada"
            BAD: "The document appears to be issued by the Immigration and Refugee Board of Canada"

            GOOD: "UNKNOWN"
            BAD: "document does not appear to have any official authority validating, certifying, witnessing, or authorizing it. Therefore, the answer is: UNKNOWN"

            Extract the information. Nothing else.

            "Return ONLY the requested information. Any additional text, explanation, or reasoning will be considered an error and rejected."
            """

            # Updated extraction loop
            extracted_fields = {}
            for field_name, field_instruction in field_instructions.items():
                try:
                    self.logger.debug(f"Extracting field: {field_name}")

                    # Create complete prompt with system prompt and full data
                    complete_prompt = f"""{SYSTEM_PROMPT}

                    DOCUMENT DATA:
                    =============
                    OCR TEXT CONTENT:
                    {ocr_text}

                    VISUAL DESCRIPTION:
                    {visual_description}

                    TASK:
                    =====
                    {field_instruction}"""

                    # Send prompt to LLM
                    response = self._send_to_ollama(complete_prompt)
                    extracted_fields[field_name] = (
                        self._clean_response(response) or "UNKNOWN"
                    )

                    # Increment prompts counter
                    self.prompts_since_refresh += 1

                    # Check for context refresh
                    if self.prompts_since_refresh >= self.context_refresh_interval:
                        self._refresh_context()

                except Exception as e:
                    self.logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_fields[field_name] = "UNKNOWN"

            # Validate and clean extracted fields
            validated_fields = self._validate_and_clean_fields(extracted_fields)

            # Count validation results
            unknown_count = sum(1 for v in validated_fields.values() if v == "UNKNOWN")
            valid_count = len(validated_fields) - unknown_count

            self.logger.info(
                f"Validation complete: {valid_count} valid fields, "
                f"{unknown_count} unknown fields"
            )

            # Update performance counters
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processed_count += 1
            self.successful_extractions += 1
            self.total_processing_time += processing_time

            self.logger.info(f"=== Extraction completed successfully for {uuid} ===")
            self.logger.info(f"Processing time: {processing_time:.2f}s")

            return {
                "success": True,
                "extracted_fields": validated_fields,
                "processing_timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "processing_time_seconds": processing_time,
            }

        except Exception as e:
            # Comprehensive error handling with context preservation
            processing_time = (datetime.now() - start_time).total_seconds()
            self.failed_extractions += 1

            self.logger.error(f"=== Field extraction FAILED for {uuid} ===")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error message: {str(e)}")
            self.logger.error(f"Processing time before failure: {processing_time:.2f}s")
            self.logger.error(
                f"Session stats: {self.processed_count} total, "
                f"{self.failed_extractions} failed"
            )

            # Log additional context for debugging
            if hasattr(e, "__traceback__"):
                import traceback

                self.logger.debug("Full traceback:", exc_info=True)

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "model_used": self.model,
            }

    def _send_to_ollama(self, prompt: str) -> str:
        """Send prompt to OLLAMA and get response"""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Higher for more varied/detailed responses
                "top_p": 0.95,
                "num_predict": 4000,  # Much higher token limit
                "repeat_penalty": 1.1,  # Prevents repetitive responses
            },
        }

        self.logger.debug(f"Sending request to OLLAMA: {url}")

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics and system status.

        Returns detailed information about processing performance, hardware
        utilization, and system configuration for monitoring and optimization.

        Returns:
            Dictionary containing:
            - Processing counters and success rates
            - Performance metrics (timing, throughput)
            - Hardware configuration and constraints
            - Model information and settings
            - Context management status

        Note:
            Statistics are cumulative since processor initialization.
            Call this method regularly for performance monitoring.
        """
        # Calculate derived metrics
        total_attempts = self.successful_extractions + self.failed_extractions
        success_rate = (
            (self.successful_extractions / total_attempts * 100)
            if total_attempts > 0
            else 0
        )

        avg_processing_time = (
            (self.total_processing_time / self.processed_count)
            if self.processed_count > 0
            else 0
        )

        # Estimate throughput (documents per hour)
        if self.total_processing_time > 0:
            throughput_per_hour = (
                self.processed_count / self.total_processing_time
            ) * 3600
        else:
            throughput_per_hour = 0

        stats = {
            # Processing Statistics
            "processing": {
                "total_processed": self.processed_count,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "success_rate_percent": round(success_rate, 2),
                "total_processing_time_seconds": round(self.total_processing_time, 2),
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "estimated_throughput_per_hour": round(throughput_per_hour, 1),
            },
            # Context Management
            "context": {
                "prompts_since_refresh": self.prompts_since_refresh,
                "context_refresh_interval": self.context_refresh_interval,
                "refresh_usage_percent": round(
                    (self.prompts_since_refresh / self.context_refresh_interval) * 100,
                    1,
                ),
            },
            # Model Configuration
            "model": {
                "current_model": self.model,
                "available_models": len(self.model_registry),
                "suitable_models": len(self.get_available_models()),
            },
            # Hardware Configuration
            "hardware": {
                "max_ram_gb": self.hardware_constraints.max_ram_gb,
                "max_gpu_vram_gb": self.hardware_constraints.max_gpu_vram_gb,
                "max_cpu_cores": self.hardware_constraints.max_cpu_cores,
                "gpu_available": self.hardware_constraints.max_gpu_vram_gb > 0,
            },
            # System Configuration
            "configuration": {
                "host": self.host,
                "timeout_seconds": self.timeout,
                "temperature": self.DEFAULT_TEMPERATURE,
                "max_response_tokens": self.MAX_RESPONSE_TOKENS,
                "max_ocr_tokens": self.MAX_OCR_TOKENS,
            },
            # Timestamp
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.debug(
            f"Generated stats: {stats['processing']['total_processed']} processed, "
            f"{stats['processing']['success_rate_percent']}% success rate"
        )

        return stats

    def get_available_models(self) -> List[str]:
        """
        Get list of models compatible with current hardware constraints.

        This method filters the model registry to show only models that can
        run on the current system based on detected hardware capabilities.

        Returns:
            List of model names that fit within hardware constraints

        Note:
            The list is sorted by quality score (best first) for easy selection.
        """
        self.logger.debug("Filtering models by hardware compatibility")

        suitable_models = []

        for model in self.model_registry:
            compatible = (
                model.min_ram_gb <= self.hardware_constraints.max_ram_gb
                and model.min_gpu_vram_gb <= self.hardware_constraints.max_gpu_vram_gb
                and model.min_cpu_cores <= self.hardware_constraints.max_cpu_cores
            )

            if compatible:
                suitable_models.append(model)
                self.logger.debug(
                    f"Compatible: {model.name} (quality: {model.quality_score})"
                )
            else:
                self.logger.debug(
                    f"Incompatible: {model.name} - resource requirements exceed limits"
                )

        # Sort by quality score (best first)
        suitable_models.sort(key=lambda m: m.quality_score, reverse=True)
        model_names = [model.name for model in suitable_models]

        self.logger.info(
            f"Found {len(model_names)} compatible models out of {len(self.model_registry)} total"
        )

        return model_names
