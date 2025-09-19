#!C:/Users/UserX/AppData/Local/Programs/Python/Python313/python.exe

from tqdm import tqdm
import logging
import json
import subprocess
import requests
import time
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Any
from collections import defaultdict
from checksum_utils import HashAlgorithm, generate_checksum
from uuid_utils import generate_uuid4plus
from encryption_utils import (
    generate_passphrase,
    generate_password,
    encrypt_file,
    decrypt_file,
)
from metadata_processor import MetadataExtractor
from visual_processor import (
    VisualProcessor,
    ProcessingMode,
    VisionModelSpec,
    HardwareConstraints,
    ProcessingStats,
)
from language_processor import LanguageProcessor
from database_processor import create_final_spreadsheet

# =====================================================================
# INITIAL SETUP - DIRECTORIES, LOGGING, SESSION TRACKING
# =====================================================================

# Create all required directories - but preserve existing ones
try:
    for name, path in PATHS.items():
        if name.endswith("_dir"):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    # Setup comprehensive logging (both console and file)
    session_timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S_%Z%z")
    log_filename = f"PAPERTRAIL-SESSION-{session_timestamp}.log"

    handlers = [
        logging.FileHandler(
            f"{str(PATHS['logs_dir'])}/{log_filename}", encoding="utf-8"
        ),
    ]

    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
except Exception as e:
    raise e


logger.info("=" * 80)
logger.info("PAPERTRAIL DOCUMENT PROCESSING PIPELINE STARTED")
logger.info("=" * 80)
logger.info(f"Session ID: {session_timestamp}")
logger.info(f"Logging initialized. Log file: {log_filename}")
logger.info(f"Session JSON: {session_json_path.name}")
logger.info(f"Base directory: {PATHS['base_dir']}")
logger.info(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
logger.info(f"Unsupported extensions: {UNSUPPORTED_EXTENSIONS}")
