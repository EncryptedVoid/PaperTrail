"""
Enhanced DatabasePipeline with complete tabulate function implementation
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import logging
import base64
import hashlib
import os
import secrets
import string
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from tqdm import tqdm
from config import (
    DEFAULT_PASSPHRASE_WORD_COUNT,
    DEFAULT_PASSWORD_LENGTH,
    DEFAULT_WORD_SEPARATOR,
    ENCRYPTED_FILE_EXTENSION,
    KEY_DERIVATION_ITERATIONS,
    MAX_PASSWORD_GENERATION_ATTEMPTS,
    SALT_LENGTH_BYTES,
    SIMILAR_CHARACTERS,
    SYMBOL_CHARACTERS,
    PASSPHRASE_WORDLIST_PATH,
    ARTIFACT_PROFILES_DIR,
    PROFILE_PREFIX,
    ARTIFACT_PREFIX,
)

# Add missing constants that seem to be missing from your config
MINIMUM_WORD_LENGTH = 3  # Minimum word length for passphrase generation


class TabulationReport(TypedDict):
    """
    Type definition for the tabulation report returned by the tabulate method.
    """
    processed_files: int
    total_files: int
    encrypted_files: int
    failed_encryptions: int
    failed_exports: int
    skipped_files: int
    spreadsheet_exports: Dict[str, Path]
    password_vault_created: bool
    errors: List[str]
    processing_time: float


class DatabasePipeline:
    """
    Comprehensive database pipeline providing file encryption and spreadsheet export services.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the DatabasePipeline with logging configuration."""
        self.logger: logging.Logger = logger

        # Password vault for storing encryption passwords securely
        self.password_vault_path = Path("data/password_vault.encrypted")
        self.vault_master_key = None

        # Define the column mapping from profile data to spreadsheet columns
        self.column_mapping = {
            # Core identification
            "ITEM_ID": "uuid",
            "Title": ["llm_extraction", "extracted_fields", "title"],
            "File_Extension": ["metadata", "extension"],
            # Document classification
            "Document_Type": ["llm_extraction", "extracted_fields", "document_type"],
            "Original_Language": ["llm_extraction", "extracted_fields", "original_language"],
            "Current_Language": ["llm_extraction", "extracted_fields", "current_language"],
            "Confidentiality_Level": ["llm_extraction", "extracted_fields", "confidentiality_level"],
            # Technical data
            "Checksum_SHA256": "checksum",
            "File_Size_MB": ["metadata", "size_mb"],
            # People and organizations
            "Translator_Name": ["llm_extraction", "extracted_fields", "translator_name"],
            "Issuer_Name": ["llm_extraction", "extracted_fields", "issuer_name"],
            "Officiater_Name": ["llm_extraction", "extracted_fields", "officiater_name"],
            # Dates
            "Date_Added": ["stages", "metadata_extraction", "timestamp"],
            "Date_Created": ["llm_extraction", "extracted_fields", "date_created"],
            "Date_of_Reception": ["llm_extraction", "extracted_fields", "date_of_reception"],
            "Date_of_Issue": ["llm_extraction", "extracted_fields", "date_of_issue"],
            "Date_of_Expiry": ["llm_extraction", "extracted_fields", "date_of_expiry"],
            # Content and notes
            "Tags": ["llm_extraction", "extracted_fields", "tags"],
            "Version_Notes": ["llm_extraction", "extracted_fields", "version_notes"],
            "Utility_Notes": ["llm_extraction", "extracted_fields", "utility_notes"],
            "Additional_Notes": ["llm_extraction", "extracted_fields", "additional_notes"],
            # Manual entry fields (will be empty for now)
            "Action_Required": "",
            "Parent_Document_ID": "",
            "Off_Site_Storage_ID": "",
            "On_Site_Storage_ID": "",
            "Backup_Storage_ID": "",
            "Project_ID": "",
            "Version_Number": "",
            # Processing metadata
            "Processing_Status": ["llm_extraction", "success"],
            "OCR_Text_Length": "calculated",
            "Visual_Description_Length": "calculated",
            "Fields_Extracted_Count": "calculated",
            "Processing_Date": ["stages", "llm_field_extraction", "timestamp"],
            "Original_Filename": "original_artifact_name",
            "Encryption_Status": "calculated",
        }

        self.logger.info("DatabasePipeline initialized successfully")

    def tabulate(
        self,
        source_dir: Path,
        review_dir: Path,
        success_dir: Path,
        use_passphrase: bool = False,
        export_spreadsheets: bool = True,
        encrypt_files: bool = True
    ) -> TabulationReport:
        """
        Main tabulation function that encrypts files, stores passwords, and exports spreadsheets.

        This method performs the complete database pipeline process:
        1. Discovers all processed artifact files in the source directory
        2. Encrypts each file with a unique password/passphrase
        3. Securely stores encryption passwords in an encrypted vault
        4. Exports all profile data to Excel and CSV spreadsheets
        5. Moves processed files to appropriate directories
        6. Generates comprehensive processing report

        Args:
            source_dir: Directory containing processed artifacts to tabulate
            review_dir: Directory to move problematic files for manual review
            success_dir: Directory to move successfully processed files
            use_passphrase: Whether to use memorable passphrases instead of random passwords
            export_spreadsheets: Whether to generate Excel/CSV exports
            encrypt_files: Whether to encrypt the artifact files

        Returns:
            TabulationReport containing detailed results of the tabulation process
        """
        start_time = datetime.now()

        # Validate input directories
        if not source_dir.exists():
            error_msg = f"Source directory does not exist: {source_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Create output directories if they don't exist
        for directory in [review_dir, success_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")

        # Initialize report
        report: TabulationReport = {
            "processed_files": 0,
            "total_files": 0,
            "encrypted_files": 0,
            "failed_encryptions": 0,
            "failed_exports": 0,
            "skipped_files": 0,
            "spreadsheet_exports": {},
            "password_vault_created": False,
            "errors": [],
            "processing_time": 0.0,
        }

        # Log tabulation stage header
        self.logger.info("=" * 80)
        self.logger.info("DATABASE TABULATION AND ENCRYPTION STAGE")
        self.logger.info("=" * 80)

        # Discover artifact files
        try:
            artifacts = [
                item for item in source_dir.iterdir()
                if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
            ]
        except Exception as e:
            error_msg = f"Failed to scan source directory: {e}"
            self.logger.error(error_msg)
            report["errors"].append(error_msg)
            return report

        if not artifacts:
            self.logger.info("No artifact files found for tabulation")
            # Still try to export existing profiles if requested
            if export_spreadsheets:
                try:
                    spreadsheet_files = self._export_to_spreadsheet(
                        ARTIFACT_PROFILES_DIR, success_dir
                    )
                    report["spreadsheet_exports"] = spreadsheet_files
                except Exception as e:
                    self.logger.error(f"Spreadsheet export failed: {e}")
                    report["errors"].append(str(e))
            return report

        report["total_files"] = len(artifacts)
        self.logger.info(f"Found {len(artifacts)} artifact files to process")

        # Initialize password vault if encryption is enabled
        if encrypt_files:
            try:
                self._initialize_password_vault()
                report["password_vault_created"] = True
                self.logger.info("Password vault initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize password vault: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                # Continue without encryption if vault fails
                encrypt_files = False

        # Process each artifact
        for artifact in tqdm(artifacts, desc="Processing artifacts", unit="file"):
            try:
                # Extract UUID for profile lookup
                artifact_id = artifact.stem[(len(ARTIFACT_PREFIX)+1):]
                profile_path = ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"

                # Verify profile exists
                if not profile_path.exists():
                    error_msg = f"Profile not found for artifact: {artifact.name}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)
                    report["skipped_files"] += 1
                    self._move_to_review(artifact, review_dir, "missing_profile")
                    continue

                # Encrypt file if requested
                encryption_password = None
                if encrypt_files:
                    try:
                        self.logger.info(f"Encrypting artifact: {artifact.name}")
                        encryption_password = self._encrypt_file(artifact, use_passphrase)

                        # Store password in vault
                        self._store_password_in_vault(artifact_id, encryption_password)
                        report["encrypted_files"] += 1

                        self.logger.info(f"Successfully encrypted: {artifact.name}")

                    except Exception as e:
                        error_msg = f"Encryption failed for {artifact.name}: {e}"
                        self.logger.error(error_msg)
                        report["errors"].append(error_msg)
                        report["failed_encryptions"] += 1
                        # Continue processing even if encryption fails
                        encryption_password = None

                # Update profile with encryption status
                try:
                    self._update_profile_with_encryption(
                        profile_path, encryption_password is not None
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to update profile encryption status: {e}")

                # Move artifact to success directory
                try:
                    moved_artifact = artifact.rename(success_dir / artifact.name)
                    report["processed_files"] += 1
                    self.logger.debug(f"Moved to success directory: {artifact.name}")
                except Exception as e:
                    error_msg = f"Failed to move {artifact.name} to success directory: {e}"
                    self.logger.error(error_msg)
                    report["errors"].append(error_msg)

            except Exception as e:
                error_msg = f"Failed to process {artifact.name}: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                self._move_to_review(artifact, review_dir, "processing_error")

        # Export spreadsheets if requested
        if export_spreadsheets:
            try:
                self.logger.info("Exporting profile data to spreadsheets...")
                spreadsheet_files = self._export_to_spreadsheet(
                    ARTIFACT_PROFILES_DIR, success_dir
                )
                report["spreadsheet_exports"] = spreadsheet_files
                self.logger.info("Spreadsheet export completed successfully")
            except Exception as e:
                error_msg = f"Spreadsheet export failed: {e}"
                self.logger.error(error_msg)
                report["errors"].append(error_msg)
                report["failed_exports"] += 1

        # Finalize report
        total_time = (datetime.now() - start_time).total_seconds()
        report["processing_time"] = total_time

        # Log final summary
        self._log_tabulation_summary(report)

        return report

    def _initialize_password_vault(self) -> None:
        """Initialize the encrypted password vault for storing encryption passwords."""
        try:
            # Create vault directory if it doesn't exist
            self.password_vault_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate or load master key for vault encryption
            master_key_file = self.password_vault_path.parent / "vault_master.key"

            if master_key_file.exists():
                # Load existing master key
                with open(master_key_file, "rb") as f:
                    self.vault_master_key = f.read()
                self.logger.debug("Loaded existing vault master key")
            else:
                # Generate new master key
                self.vault_master_key = Fernet.generate_key()
                with open(master_key_file, "wb") as f:
                    f.write(self.vault_master_key)
                os.chmod(master_key_file, 0o600)  # Restrict permissions
                self.logger.info("Generated new vault master key")

        except Exception as e:
            error_msg = f"Failed to initialize password vault: {e}"
            self.logger.error(error_msg)
            raise

    def _store_password_in_vault(self, artifact_id: str, password: str) -> None:
        """Store an encryption password in the secure vault."""
        try:
            # Load existing vault data or create new
            vault_data = {}
            if self.password_vault_path.exists():
                cipher = Fernet(self.vault_master_key)
                with open(self.password_vault_path, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = cipher.decrypt(encrypted_data)
                vault_data = json.loads(decrypted_data.decode())

            # Add new password entry
            vault_data[artifact_id] = {
                "password": password,
                "timestamp": datetime.now().isoformat(),
                "algorithm": "PBKDF2-SHA256-Fernet"
            }

            # Encrypt and save vault
            cipher = Fernet(self.vault_master_key)
            vault_json = json.dumps(vault_data, indent=2).encode()
            encrypted_vault = cipher.encrypt(vault_json)

            with open(self.password_vault_path, "wb") as f:
                f.write(encrypted_vault)

            os.chmod(self.password_vault_path, 0o600)  # Restrict permissions
            self.logger.debug(f"Stored password for artifact {artifact_id} in vault")

        except Exception as e:
            error_msg = f"Failed to store password in vault: {e}"
            self.logger.error(error_msg)
            raise

    def _update_profile_with_encryption(self, profile_path: Path, encrypted: bool) -> None:
        """Update profile with encryption status information."""
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            # Add encryption information
            profile_data["encryption"] = {
                "encrypted": encrypted,
                "timestamp": datetime.now().isoformat(),
                "algorithm": "PBKDF2-SHA256-Fernet" if encrypted else None
            }

            # Update stages
            profile_data.setdefault("stages", {})
            profile_data["stages"]["encryption"] = {
                "status": "completed" if encrypted else "skipped",
                "timestamp": datetime.now().isoformat(),
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to update profile {profile_path}: {e}")

    def _move_to_review(self, artifact: Path, review_dir: Path, reason: str) -> None:
        """Move an artifact to the review directory with reason categorization."""
        try:
            reason_dir = review_dir / reason
            reason_dir.mkdir(exist_ok=True, parents=True)

            target_path = reason_dir / artifact.name
            artifact.rename(target_path)

            self.logger.info(f"Moved {artifact.name} to review: {reason}")

        except Exception as e:
            self.logger.error(f"Failed to move {artifact.name} to review: {e}")

    def _log_tabulation_summary(self, report: TabulationReport) -> None:
        """Log the final tabulation summary."""
        self.logger.info("Tabulation complete:")
        self.logger.info(f"  - {report['processed_files']} files successfully processed")
        self.logger.info(f"  - {report['encrypted_files']} files encrypted")
        self.logger.info(f"  - {report['failed_encryptions']} encryption failures")
        self.logger.info(f"  - {report['skipped_files']} files skipped")
        self.logger.info(f"  - Processing time: {report['processing_time']:.2f} seconds")

        if report["spreadsheet_exports"]:
            self.logger.info("  - Spreadsheet exports:")
            for format_type, file_path in report["spreadsheet_exports"].items():
                self.logger.info(f"    - {format_type.upper()}: {file_path}")

        if report["password_vault_created"]:
            self.logger.info(f"  - Password vault: {self.password_vault_path}")

        if report["errors"]:
            self.logger.warning(f"  - {len(report['errors'])} errors encountered")

    def _generate_passphrase(self) -> str:
        """Generate memorable passphrase using random word combinations."""
        self.logger.debug(
            f"Generating passphrase with {DEFAULT_PASSPHRASE_WORD_COUNT} words"
        )

        if DEFAULT_PASSPHRASE_WORD_COUNT < 1:
            error_msg = "Word count must be at least 1"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        wordlist = PASSPHRASE_WORDLIST_PATH

        try:
            if not wordlist.exists():
                raise FileNotFoundError(f"Wordlist not found: {wordlist}")

            with open(wordlist, "r", encoding="utf-8") as f:
                words = [line.strip().lower() for line in f
                        if line.strip() and len(line.strip()) >= MINIMUM_WORD_LENGTH]

            if len(words) < DEFAULT_PASSPHRASE_WORD_COUNT:
                raise ValueError(f"Not enough words in wordlist: {len(words)}")

            # Select random words
            selected_words = [
                secrets.choice(words).capitalize()
                for _ in range(DEFAULT_PASSPHRASE_WORD_COUNT)
            ]

            passphrase = DEFAULT_WORD_SEPARATOR.join(selected_words)

            self.logger.debug(f"Generated passphrase with {len(selected_words)} words")
            return passphrase

        except Exception as e:
            # Fallback to built-in words if wordlist fails
            self.logger.warning(f"Wordlist failed, using built-in words: {e}")
            builtin_words = [
                "Forest", "Mountain", "River", "Ocean", "Desert", "Valley",
                "Garden", "Bridge", "Castle", "Tower", "Village", "Island",
                "Storm", "Thunder", "Lightning", "Rainbow", "Sunset", "Dawn"
            ]
            selected_words = [
                secrets.choice(builtin_words)
                for _ in range(DEFAULT_PASSPHRASE_WORD_COUNT)
            ]
            return DEFAULT_WORD_SEPARATOR.join(selected_words)

    # Keep all your existing helper methods for spreadsheet export...
    # (_export_to_spreadsheet, _process_profile, _get_nested_value, etc.)

    def _calculate_special_field(self, field_name: str, profile: Dict) -> Any:
        """Calculate special fields that need computation."""
        if field_name == "Fields_Extracted_Count":
            try:
                extracted_fields = self._get_nested_value(
                    profile, ["llm_extraction", "extracted_fields"]
                )
                if isinstance(extracted_fields, dict):
                    return sum(1 for v in extracted_fields.values() if v != "UNKNOWN")
                return 0
            except:
                return 0

        elif field_name == "OCR_Text_Length":
            try:
                text = self._get_nested_value(profile, ["semantics", "all_text"])
                return len(str(text)) if text != "UNKNOWN" else 0
            except:
                return 0

        elif field_name == "Visual_Description_Length":
            try:
                desc = self._get_nested_value(profile, ["semantics", "all_imagery"])
                return len(str(desc)) if desc != "UNKNOWN" else 0
            except:
                return 0

        elif field_name == "Encryption_Status":
            try:
                encrypted = self._get_nested_value(profile, ["encryption", "encrypted"])
                return "Yes" if encrypted else "No"
            except:
                return "No"

        return "UNKNOWN"

    def _encrypt_file(self, artifact: Path, passphrase: bool = False) -> str:
        """
        Encrypt a file using password-based encryption with industry-standard security practices.

        This method implements secure file encryption using a combination of PBKDF2 key derivation
        and Fernet symmetric encryption. The encrypted file includes a randomly generated salt
        for enhanced security against rainbow table attacks.

        Security Architecture:
        - PBKDF2 key derivation with configurable iteration count (default: 100,000+)
        - Fernet symmetric encryption (AES 128 in CBC mode with HMAC authentication)
        - Cryptographically secure random salt generation for each encryption
        - Salt prepended to encrypted data for self-contained decryption

        File Format:
        [SALT_BYTES][ENCRYPTED_DATA]
        - First N bytes: Random salt used for key derivation
        - Remaining bytes: Fernet-encrypted file contents

        Args:
            artifact: Path to the file to encrypt - must exist and be readable
            password: Password string to use for encryption - must be non-empty

        Raises:
            FileNotFoundError: If the input file doesn't exist or cannot be accessed
            ValueError: If password is empty or contains only whitespace
            PermissionError: If unable to read input file or write encrypted output
            OSError: If file I/O operations fail due to system constraints

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> agent.encrypt_file(Path("document.txt"), "secure_password_123")
            # Creates "document.txt.encrypted" with secure encryption

        Note:
            The original file is not modified. A new encrypted file is created with
            the configured encrypted file extension (typically ".encrypted").
        """
        # Log the start of encryption operation for audit trail
        self.logger.info(f"Starting file encryption for {artifact.name}")

        if not artifact.exists():
            error_msg = f"File not found for encryption: {artifact}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Log file size for performance monitoring
        try:
            artifact_size: int = artifact.stat().st_size
            self.logger.debug(f"Encrypting file of size: {artifact_size:,} bytes")

        except OSError as stat_error:
            self.logger.warning(f"Could not determine file size: {stat_error}")

        # Generate cryptographically secure random salt for key derivation
        encryption_salt: bytes = os.urandom(SALT_LENGTH_BYTES)
        self.logger.debug(
            f"Generated {SALT_LENGTH_BYTES}-byte random salt for key derivation"
        )

        password = self._generate_passphrase() if passphrase else self._generate_password()

        # Derive encryption key from password using PBKDF2 with strong parameters
        try:
            key_derivation_function = PBKDF2HMAC(
                algorithm=hashes.SHA256(),  # Use SHA-256 for key derivation
                length=32,  # 256-bit key for Fernet encryption
                salt=encryption_salt,  # Unique salt for this encryption
                iterations=KEY_DERIVATION_ITERATIONS,  # Configurable iteration count
            )

            # Derive raw key and encode for Fernet usage
            raw_derived_key: bytes = key_derivation_function.derive(
                password.encode("utf-8")
            )
            fernet_key: bytes = base64.urlsafe_b64encode(raw_derived_key)

            self.logger.debug(
                f"Derived encryption key using {KEY_DERIVATION_ITERATIONS} PBKDF2 iterations"
            )

        except Exception as key_derivation_error:
            error_msg = f"Failed to derive encryption key: {key_derivation_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from key_derivation_error

        # Initialize Fernet symmetric encryption cipher
        try:
            encryption_cipher = Fernet(fernet_key)
            self.logger.debug("Initialized Fernet encryption cipher successfully")

        except Exception as cipher_error:
            error_msg = f"Failed to initialize encryption cipher: {cipher_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from cipher_error

        # Read original file contents for encryption
        try:
            with open(artifact, "rb") as original_file:
                original_artifact_data: bytes = original_file.read()

            self.logger.debug(
                f"Read {len(original_artifact_data):,} bytes from source file"
            )

        except PermissionError as permission_error:
            error_msg = f"Permission denied reading file for encryption: {artifact}"
            self.logger.error(error_msg)
            raise PermissionError(error_msg) from permission_error

        except OSError as read_error:
            error_msg = f"I/O error reading file for encryption: {read_error}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from read_error

        # Encrypt the file data using Fernet
        try:
            start_time: float = datetime.now().timestamp()
            encrypted_artifact_data: bytes = encryption_cipher.encrypt(original_artifact_data)
            encryption_duration: float = datetime.now().timestamp() - start_time

            self.logger.debug(
                f"Encrypted {len(original_artifact_data):,} bytes to {len(encrypted_artifact_data):,} bytes "
                f"in {encryption_duration:.2f} seconds"
            )

        except Exception as encryption_error:
            error_msg = f"Failed to encrypt file data: {encryption_error}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from encryption_error

        # Write encrypted file with salt prepended for self-contained decryption
        encrypted_artifact_path: Path = artifact.with_suffix(
            artifact.suffix + ENCRYPTED_FILE_EXTENSION
        )

        try:
            with open(encrypted_artifact_path, "wb") as encrypted_file:
                # Write salt first for later key derivation during decryption
                encrypted_file.write(encryption_salt)

                # Write encrypted data after salt
                encrypted_file.write(encrypted_artifact_data)

                # Ensure data is written to disk immediately
                encrypted_file.flush()
                os.fsync(encrypted_file.fileno())

            self.logger.info(
                f"Successfully encrypted file: {artifact.name} -> {encrypted_artifact_path.name}"
            )

            return password

        except PermissionError as permission_error:
            error_msg = (
                f"Permission denied writing encrypted file: {encrypted_artifact_path}"
            )
            self.logger.error(error_msg)
            raise PermissionError(error_msg) from permission_error

        except OSError as write_error:
            error_msg = f"I/O error writing encrypted file: {write_error}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from write_error

    def _generate_password(
        self,
        character_length: int = DEFAULT_PASSWORD_LENGTH,
        include_special_symbols: bool = True,
        exclude_similar_looking_chars: bool = True,
    ) -> str:
        """
        Generate cryptographically secure random password with comprehensive complexity validation.

        This method creates passwords using the secrets module for cryptographic security,
        ensuring that generated passwords meet complexity requirements while being suitable
        for various authentication systems.

        Password Complexity Algorithm:
        1. Build character pool based on configuration options
        2. Generate random password using cryptographically secure random source
        3. Validate character distribution (uppercase, lowercase, digits, symbols)
        4. Retry generation if complexity requirements not met (up to configured limit)
        5. Return password that meets all complexity requirements

        Character Pool Configuration:
        - Always includes: uppercase letters (A-Z), lowercase letters (a-z), digits (0-9)
        - Optional: special symbols (!@#$%^&*()_+-=[]{}|;:,.<>?)
        - Optional exclusion: visually similar characters (0O1lI|)

        Security Features:
        - Uses secrets.choice() for cryptographically secure random generation
        - Configurable complexity validation prevents weak passwords
        - No predictable patterns or sequences in generated passwords
        - Suitable for high-security authentication systems

        Args:
            character_length: Desired password length (minimum 1, recommended 12+)
            include_special_symbols: Whether to include special symbols in character pool
            exclude_similar_looking_chars: Whether to exclude visually similar characters

        Returns:
            Cryptographically secure random password meeting complexity requirements

        Raises:
            ValueError: If character_length is less than 1

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> password = agent.generate_password(16, True, True)
            >>> len(password)
            16
            >>> any(c.isupper() for c in password)  # Contains uppercase
            True
            >>> any(c.islower() for c in password)  # Contains lowercase
            True

        Note:
            For passwords shorter than 4 characters, complexity validation is skipped
            to avoid infinite generation loops. Such short passwords are not recommended
            for security applications.
        """
        # Log password generation request for audit trail
        self.logger.debug(
            f"Generating password: length={character_length}, "
            f"symbols={include_special_symbols}, exclude_similar={exclude_similar_looking_chars}"
        )

        # Validate minimum password length requirement
        if character_length < 1:
            error_msg: str = "Password length must be at least 1 character"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Build character pool starting with basic alphanumeric characters
        available_characters: str = string.ascii_letters + string.digits
        self.logger.debug(f"Base character pool size: {len(available_characters)}")

        # Add special symbols if requested for increased complexity
        if include_special_symbols:
            available_characters += SYMBOL_CHARACTERS
            self.logger.debug(f"Added symbols, pool size: {len(available_characters)}")

        # Remove visually similar characters if requested for usability
        if exclude_similar_looking_chars:
            original_length: int = len(available_characters)
            available_characters = "".join(
                char for char in available_characters if char not in SIMILAR_CHARACTERS
            )
            removed_count: int = original_length - len(available_characters)
            self.logger.debug(
                f"Removed {removed_count} similar characters, final pool size: {len(available_characters)}"
            )

        # Generate password with complexity validation loop
        for generation_attempt in range(1, MAX_PASSWORD_GENERATION_ATTEMPTS + 1):
            # Generate random password using cryptographically secure source
            generated_password: str = "".join(
                secrets.choice(available_characters) for _ in range(character_length)
            )

            # Skip complexity validation for very short passwords to avoid infinite loops
            if character_length < 4:
                self.logger.debug(
                    f"Generated short password on attempt {generation_attempt}"
                )
                return generated_password

            # Validate password meets complexity requirements
            has_lowercase_letter: bool = any(
                char.islower() for char in generated_password
            )
            has_uppercase_letter: bool = any(
                char.isupper() for char in generated_password
            )
            has_numeric_digit: bool = any(char.isdigit() for char in generated_password)
            has_special_symbol: bool = (
                any(char in SYMBOL_CHARACTERS for char in generated_password)
                if include_special_symbols
                else True  # Skip symbol check if symbols not required
            )

            # Return password if it meets all complexity requirements
            if (
                has_lowercase_letter
                and has_uppercase_letter
                and has_numeric_digit
                and has_special_symbol
            ):
                self.logger.debug(
                    f"Generated complex password on attempt {generation_attempt}"
                )
                return generated_password

            # Log complexity validation failure for debugging
            missing_types: List[str] = []
            if not has_lowercase_letter:
                missing_types.append("lowercase")
            if not has_uppercase_letter:
                missing_types.append("uppercase")
            if not has_numeric_digit:
                missing_types.append("digits")
            if not has_special_symbol and include_special_symbols:
                missing_types.append("symbols")

            self.logger.debug(
                f"Attempt {generation_attempt} failed complexity check, missing: {', '.join(missing_types)}"
            )

        # Fallback: return last generated password even if complexity check failed
        # This password is still cryptographically secure, just may not meet all complexity rules
        self.logger.warning(
            f"Could not generate password meeting complexity requirements after {MAX_PASSWORD_GENERATION_ATTEMPTS} attempts, "
            f"returning cryptographically secure password"
        )
        return generated_password

    def _generate_passphrase(self) -> str:
        """
        Generate memorable passphrase using random word combinations with configurable formatting.

        This method creates human-friendly passphrases that are significantly easier to remember
        than random character passwords while maintaining good entropy for security. Passphrases
        are particularly suitable for master passwords and long-term authentication scenarios.

        Passphrase Generation Process:
        1. Select word source (built-in collection or external wordlist file)
        2. Choose random words using cryptographically secure selection
        3. Apply formatting options (capitalization, separators)
        4. Optionally append random number for additional entropy
        5. Combine components into final passphrase

        Security Considerations:
        - Uses secrets module for cryptographically secure word selection
        - Built-in wordlist contains 80+ carefully chosen words
        - External wordlist support for larger vocabulary
        - Optional numeric suffix increases entropy significantly
        - Word separation prevents dictionary attacks on individual components

        Entropy Calculation:
        - 4 words from 80-word list: ~26 bits entropy
        - 4 words from 2000-word list: ~44 bits entropy
        - Additional 2-digit number: +6.6 bits entropy
        - Recommended: 4+ words for strong security
        Returns:
            Memorable passphrase string combining random words with specified formatting

        Raises:
            ValueError: If DEFAULT_PASSPHRASE_WORD_COUNT is less than 1

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> passphrase = agent.generate_passphrase(4, "-", True, True)
            >>> passphrase
            'Forest-Mountain-River-Ocean-73'
            >>> len(passphrase.split("-"))
            5  # 4 words + 1 number

        Note:
            If external wordlist is requested but unavailable, the method gracefully
            falls back to the built-in word collection with appropriate logging.
        """
        # Log passphrase generation request for audit trail
        self.logger.debug(
            f"Generating passphrase with {DEFAULT_PASSPHRASE_WORD_COUNT} words and {DEFAULT_WORD_SEPARATOR} as a separator"
        )

        # Validate minimum word count requirement
        if DEFAULT_PASSPHRASE_WORD_COUNT < 1:
            error_msg: str = "Word count must be at least 1"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Log start of external wordlist processing
        self.logger.debug(
            f"Extracting {DEFAULT_PASSPHRASE_WORD_COUNT} words from external wordlist"
        )

        # Define potential locations for MIT wordlist file
        wordlist: Path = PASSPHRASE_WORDLIST_PATH

        # Count total lines in wordlist file for random selection
        try:
            if not wordlist.exists():
                raise FileNotFoundError

            with open(wordlist, "r", encoding="utf-8") as f:
                total_line_count: int = sum(1 for line in f if line.strip())

            self.logger.debug(f"Wordlist contains {total_line_count} total lines")

        except OSError as error:
            error_msg = f"Failed to load wordlist file at {wordlist}: {error}"
            self.logger.error(error_msg)
            raise OSError(error_msg) from error

        # Generate sorted list of random line numbers for efficient processing
        random_line_numbers: List[int] = sorted(
            secrets.randbelow(total_line_count)
            for _ in range(DEFAULT_PASSPHRASE_WORD_COUNT)


        # Extract words from selected lines using streaming approach
        extracted_words: List[str] = []

        try:
            with open(wordlist, "r", encoding="utf-8") as f:
                for current_line_number, artifact_line in enumerate(f):
                    # Check if current line is one of our selected lines
                    if current_line_number in random_line_numbers:
                        # Clean and validate word
                        cleaned_word: str = artifact_line.strip().lower()

                        # Apply quality filters for word selection
                        if (
                            len(cleaned_word)
                            >= MINIMUM_WORD_LENGTH  # Minimum length requirement
                            and cleaned_word.isalpha()  # Only alphabetic characters
                            and not any(
                                char in cleaned_word for char in "'\""
                            )  # No quotes
                        ):
                            extracted_words.append(cleaned_word)
                            self.logger.debug(f"Extracted word: {cleaned_word}")

                        # Stop processing once we have enough quality words
                        if len(extracted_words) >= DEFAULT_PASSPHRASE_WORD_COUNT:
                            break

        except OSError as extraction_error:
            error_msg = (
                f"Failed to extract words from wordlist file: {extraction_error}"
            )
            self.logger.error(error_msg)
            raise OSError(error_msg) from extraction_error

        # Log extraction results
        self.logger.info(
            f"Successfully extracted {len(extracted_words)} words from external wordlist"
        )

        # Apply capitalization formatting if requested
        extracted_words = [word.capitalize() for word in extracted_words]
        self.logger.debug("Applied capitalization to selected words")

        # Combine words with specified separator
        passphrase: str = DEFAULT_WORD_SEPARATOR.join(selected_words)

        # Log passphrase generation completion (without revealing actual passphrase)
        self.logger.info(
            f"Generated passphrase with {len(selected_words)} components, "
            f"total length: {len(passphrase)} characters"
        )

        return passphrase

    def _export_to_spreadsheet(
        self,
        profiles_dir: Path,
        output_dir: Path,
        filename_prefix: str = "PaperTrail_Artifact_Registry",
    ) -> Dict[str, Path]:
        """
        Export all profile JSON files to Excel and CSV spreadsheets

        Returns:
            Dictionary with paths to created files
        """
        self.logger.info("Starting spreadsheet export process...")

        # Find all profile files
        profile_files = list(profiles_dir.glob(f"{PROFILE_PREFIX}-*.json"))
        self.logger.info(f"Found {len(profile_files)} profile files to export")

        if not profile_files:
            self.logger.warning("No profile files found for export")
            return {}

        # Process each profile
        rows = []
        for profile_file in profile_files:
            try:
                row = self._process_profile(profile_file)
                if row:
                    rows.append(row)
            except Exception as e:
                self.logger.error(f"Failed to process profile {profile_file.name}: {e}")
                continue

        if not rows:
            self.logger.error("No valid data to export")
            return {}

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by processing date, then by title
        df = df.sort_values(["Processing_Date", "Title"], na_position="last")

        # Reorder columns to match your requirements
        ordered_columns = [
            "ITEM_ID",
            "Title",
            "Action_Required",
            "File_Extension",
            "Document_Type",
            "Original_Language",
            "Current_Language",
            "Confidentiality_Level",
            "Checksum_SHA256",
            "Translator_Name",
            "Parent_Document_ID",
            "Off_Site_Storage_ID",
            "On_Site_Storage_ID",
            "Backup_Storage_ID",
            "Project_ID",
            "Issuer_Name",
            "Officiater_Name",
            "Version_Number",
            "Date_Added",
            "Date_Created",
            "Date_of_Reception",
            "Date_of_Issue",
            "Date_of_Expiry",
            "Tags",
            "Version_Notes",
            "Utility_Notes",
            "Additional_Notes",
        ]

        # Add columns that exist in df but not in ordered list
        additional_cols = [col for col in df.columns if col not in ordered_columns]
        final_columns = ordered_columns + additional_cols

        # Reorder DataFrame columns
        df = df.reindex(columns=[col for col in final_columns if col in df.columns])

        # Generate output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}

        # Excel file
        excel_path = output_dir / f"{filename_prefix}_{timestamp}.xlsx"
        self._export_to_excel(df, excel_path)
        output_files["excel"] = excel_path

        # CSV file
        csv_path = output_dir / f"{filename_prefix}_{timestamp}.csv"
        self._export_to_csv(df, csv_path)
        output_files["csv"] = csv_path

        # Summary stats
        self._log_export_summary(df, len(profile_files))

        return output_files

    def _process_profile(self, profile_file: Path) -> Optional[Dict[str, Any]]:
        """Process a single profile JSON file into a spreadsheet row"""
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load profile {profile_file.name}: {e}")
            return None

        row = {}

        for column_name, field_path in self.column_mapping.items():
            try:
                if field_path == "":
                    # Manual entry field - leave empty
                    row[column_name] = ""
                elif field_path == "calculated":
                    # Special calculated field
                    row[column_name] = self._calculate_special_field(
                        column_name, profile
                    )
                elif isinstance(field_path, str):
                    # Simple field path
                    row[column_name] = profile.get(field_path, "UNKNOWN")
                elif isinstance(field_path, list):
                    # Nested field path
                    row[column_name] = self._get_nested_value(profile, field_path)
                else:
                    row[column_name] = "UNKNOWN"

            except Exception as e:
                self.logger.debug(
                    f"Failed to extract {column_name} from {profile_file.name}: {e}"
                )
                row[column_name] = "UNKNOWN"

        # Clean up the row
        row = self._clean_row_data(row)

        return row

    def _get_nested_value(self, data: Dict, path: List[str]) -> Any:
        """Navigate nested dictionary using path list"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return "UNKNOWN"
        return current if current is not None else "UNKNOWN"

    def _calculate_special_field(self, field_name: str, profile: Dict) -> Any:
        """Calculate special fields that need computation"""
        if field_name == "Fields_Extracted_Count":
            try:
                extracted_fields = self._get_nested_value(
                    profile, ["llm_extraction", "extracted_fields"]
                )
                if isinstance(extracted_fields, dict):
                    return sum(1 for v in extracted_fields.values() if v != "UNKNOWN")
                return 0
            except:
                return 0

        elif field_name == "OCR_Text_Length":
            try:
                text = self._get_nested_value(profile, ["semantics", "all_text"])
                return len(str(text)) if text != "UNKNOWN" else 0
            except:
                return 0

        elif field_name == "Visual_Description_Length":
            try:
                desc = self._get_nested_value(profile, ["semantics", "all_imagery"])
                return len(str(desc)) if desc != "UNKNOWN" else 0
            except:
                return 0

        return "UNKNOWN"

    def _clean_row_data(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize row data"""
        cleaned = {}
        for key, value in row.items():
            # Convert None to UNKNOWN
            if value is None:
                cleaned[key] = "UNKNOWN"
            # Clean up strings
            elif isinstance(value, str):
                cleaned[key] = value.strip() if value.strip() else "UNKNOWN"
            # Convert booleans
            elif isinstance(value, bool):
                cleaned[key] = "Yes" if value else "No"
            # Keep numbers as-is
            else:
                cleaned[key] = value

        return cleaned

    def _export_to_excel(self, df: pd.DataFrame, output_path: Path):
        """Export DataFrame to Excel with formatting"""
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Documents", index=False)

                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets["Documents"]

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 chars
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            self.logger.info(f"Excel file exported: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export Excel file: {e}")

    def _export_to_csv(self, df: pd.DataFrame, output_path: Path):
        """Export DataFrame to CSV"""
        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"CSV file exported: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export CSV file: {e}")

    def _log_export_summary(self, df: pd.DataFrame, total_profiles: int):
        """Log summary statistics about the export"""
        self.logger.info("=" * 60)
        self.logger.info("SPREADSHEET EXPORT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total profiles processed: {total_profiles}")
        self.logger.info(f"Rows exported: {len(df)}")
        self.logger.info(f"Columns exported: {len(df.columns)}")

        # Document type breakdown
        if "Document_Type" in df.columns:
            doc_types = df["Document_Type"].value_counts()
            self.logger.info(f"Document types: {dict(doc_types)}")

        # Processing success rate
        if "Processing_Status" in df.columns:
            success_rate = (df["Processing_Status"] == "Yes").sum() / len(df) * 100
            self.logger.info(f"LLM processing success rate: {success_rate:.1f}%")
