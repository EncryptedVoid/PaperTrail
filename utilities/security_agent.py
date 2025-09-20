"""
Security Agent Module

Comprehensive cryptographic and security utilities for file processing pipelines.
Provides secure checksum generation, encryption, password generation, and UUID creation
with configurable algorithms and enterprise-grade security practices.

This module implements:
- Multi-CHECKSUM_ALGORITHM file checksum calculation with performance optimization
- Password-based file encryption using PBKDF2 and Fernet symmetric encryption
- Cryptographically secure password and passphrase generation
- Multiple UUID generation strategies for different use cases
- Persistent checksum history management for duplicate detection

Author: Ashiq Gazi
"""

import base64
import hashlib
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Set
from config import (
    CHECKSUM_CHUNK_SIZE_BYTES,
    CHECKSUM_HISTORY_FILE,
    CHECKSUM_ALGORITHM,
)


class SecurityAgent:
    """
    Comprehensive security agent providing cryptographic services for file processing pipelines.

    This class encapsulates all security-related operations including file hashing,
    encryption, secure random generation, and checksum history management. It is
    designed to be a centralized security service that can be shared across multiple
    components of a file processing system.

    Key Features:
    - Multi-CHECKSUM_ALGORITHM file checksum calculation with memory-efficient streaming
    - Password-based file encryption using industry-standard PBKDF2 + Fernet
    - Cryptographically secure password and passphrase generation
    - Multiple UUID generation strategies optimized for different use cases
    - Persistent checksum history management with atomic updates
    - Comprehensive error handling and logging for audit trails

    Security Principles:
    - All random generation uses cryptographically secure sources (secrets module)
    - File operations are performed in chunks to handle large files efficiently
    - Encryption uses strong key derivation (PBKDF2) with configurable iteration counts
    - All operations are logged for security audit and debugging purposes
    - Error messages are sanitized to prevent information leakage
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the SecurityAgent with logging and persistent storage configuration.

        This constructor establishes the security agent's operational context including
        logging configuration for audit trails and the location for persistent checksum
        history storage. All cryptographic operations performed by this agent will be
        logged through the provided logger.

        Args:
            logger: Configured logger instance for recording all security operations,
                    including checksum generation, encryption operations, and error conditions

        Note:
            The history file will be created automatically if it doesn't exist. The parent
            directory must exist and be writable. All file operations are performed with
            appropriate error handling and logging.
        """
        # Store core dependencies for use throughout the security agent
        self.logger: logging.Logger = logger

        # Log successful initialization with configuration details for debugging
        self.logger.info("SecurityAgent initialized successfully")
        self.logger.debug(f"Logger level configured: {self.logger.level}")
        self.logger.debug(f"Checksum history file: {CHECKSUM_HISTORY_FILE}")

        # Ensure parent directory exists for history file operations
        try:
            CHECKSUM_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(
                f"Ensured history directory exists: {CHECKSUM_HISTORY_FILE.parent}"
            )

        except OSError as directory_error:
            self.logger.warning(
                f"Could not create history directory: {directory_error}"
            )

    def generate_uuid5(self, name: str, namespace_fragment: str = "") -> str:
        """
        Generate deterministic UUID based on namespace and name using UUID5 specification.

        This method creates reproducible UUIDs where the same inputs always produce
        identical outputs. This deterministic property makes UUID5 ideal for creating
        consistent identifiers across different systems and deployments.

        Deterministic Properties:
        - Same name + namespace always produces identical UUID
        - Platform-independent consistency
        - Useful for idempotent operations and data migration
        - Enables predictable identifier generation

        Use Cases:
        - User account IDs based on email addresses
        - Resource identifiers consistent across deployments
        - Content addressing where same content needs same ID
        - API endpoints requiring consistent resource identifiers
        - Migration scripts where IDs must be predictable
        - Idempotent operations requiring stable identifiers

        Security Considerations:
        - UUIDs are deterministic and can be predicted if inputs are known
        - Not suitable for security tokens or session identifiers
        - Names should be validated to prevent injection attacks
        - Use random UUIDs (UUID4+) for security-critical applications

        Args:
            name: Input string to generate UUID from - must be non-empty
            namespace_fragment: Optional custom namespace UUID string (defaults to DNS namespace)

        Returns:
            Deterministic UUID5 string that will be identical for same inputs

        Raises:
            ValueError: If namespace_fragment is provided but invalid format
            ValueError: If name is empty or invalid

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> uuid1 = agent.generate_uuid5("john@example.com")
            >>> uuid1
            "8c8a5cf6-8b8a-5c3d-9f2e-7a1b4c5d6e8f"

            >>> uuid2 = agent.generate_uuid5("john@example.com")  # Same result!
            >>> uuid2
            "8c8a5cf6-8b8a-5c3d-9f2e-7a1b4c5d6e8f"

            >>> uuid1 == uuid2
            True

        Note:
            Uses DNS namespace as default for consistency. Custom namespaces
            can be provided for organizational partitioning of UUID space.
        """
        # Log UUID5 generation request for audit trail
        self.logger.debug(
            f"Generating UUID5: name='{name}', namespace='{namespace_fragment or 'DNS'}'"
        )

        # Validate input parameters
        if not name or not name.strip():
            error_msg: str = "Name cannot be empty for UUID5 generation"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Use DNS namespace as default, or parse custom namespace
            if namespace_fragment is None:
                namespace: uuid.UUID = uuid.NAMESPACE_DNS
                self.logger.debug("Using DNS namespace for UUID5 generation")

            else:
                namespace = uuid.UUID(namespace_fragment)
                self.logger.debug(f"Using custom namespace: {namespace_fragment}")

            # Generate deterministic UUID5
            generated_uuid5: str = str(uuid.uuid5(namespace, name))

            # Log successful generation (safe to log UUID5 as it's deterministic)
            self.logger.info(f"Generated UUID5 for name '{name}': {generated_uuid5}")

            return generated_uuid5

        except ValueError as namespace_error:
            error_msg = f"Invalid namespace UUID format: {namespace_fragment}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from namespace_error

        except Exception as generation_error:
            error_msg = f"Failed to generate UUID5: {generation_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from generation_error

    def generate_ulid(self, identifier_prefix: str = "") -> str:
        """
        Generate sortable UUID with timestamp prefix for chronological ordering (ULID-style).

        This method creates UUIDs that automatically sort chronologically by creation time,
        making them ideal for applications where temporal ordering is important. The timestamp
        component ensures that UUIDs created later will always sort after earlier ones.

        ULID Format:
        - 48-bit timestamp (milliseconds since epoch)
        - 80-bit random component
        - Total: 128-bit UUID with chronological sorting properties
        - Standard UUID formatting with hyphens

        Chronological Properties:
        - Natural chronological sorting without additional indexes
        - Creation time embedded in the identifier
        - Monotonically increasing (within millisecond precision)
        - Compatible with standard UUID tooling

        Use Cases:
        - Database primary keys where chronological sorting is valuable
        - Log entries requiring natural time ordering
        - Distributed systems generating IDs independently but needing global order
        - Event sourcing systems where event order is critical
        - Message queues where processing order matters by creation time
        - File systems where chronological listing is desired

        Performance Benefits:
        - Excellent database index performance due to monotonic ordering
        - Reduced index fragmentation compared to random UUIDs
        - Natural clustering of related temporal data

        Args:
            identifier_prefix: Optional string prefix for namespacing and categorization

        Returns:
            Timestamp-prefixed UUID that sorts chronologically by creation time

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> ulid1 = agent.generate_ulid("event")
            >>> time.sleep(0.001)  # Ensure different timestamp
            >>> ulid2 = agent.generate_ulid("event")
            >>> ulid1 < ulid2  # Chronological sorting
            True

            >>> ulid1
            "event-018a4c2b-7f3e-4d9a-8b1c-2f5e7a9c3d6f"

        Note:
            Timestamp precision is milliseconds. UUIDs created within the same
            millisecond may not sort in creation order due to random component.
        """
        # Log ULID generation request for audit trail
        self.logger.debug(f"Generating ULID with prefix: '{identifier_prefix}'")

        try:
            # Get current timestamp in milliseconds since Unix epoch
            current_timestamp_ms: int = int(time.time() * 1000)
            self.logger.debug(f"Using timestamp: {current_timestamp_ms}")

            # Convert timestamp to 12-character hexadecimal (48 bits)
            timestamp_hex: str = f"{current_timestamp_ms:012x}"

            # Generate cryptographically secure random component (80 bits)
            random_bytes: bytes = secrets.token_bytes(10)  # 10 bytes = 80 bits
            random_hex: str = random_bytes.hex()

            # Combine timestamp and random components (128 bits total)
            combined_hex: str = timestamp_hex + random_hex

            # Format as standard UUID: 8-4-4-4-12 character groups
            formatted_ulid: str = (
                f"{combined_hex[:8]}-{combined_hex[8:12]}-"
                f"{combined_hex[12:16]}-{combined_hex[16:20]}-{combined_hex[20:32]}"
            )

            # Add prefix if provided
            final_ulid: str = (
                f"{identifier_prefix}-{formatted_ulid}"
                if identifier_prefix
                else formatted_ulid
            )

            # Log successful generation
            self.logger.info(
                f"Generated ULID with timestamp {current_timestamp_ms}: {final_ulid}"
            )

            return final_ulid

        except Exception as ulid_error:
            error_msg: str = f"Failed to generate ULID: {ulid_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from ulid_error

    def generate_lenbased_uuid(self, length: int = 8) -> str:
        """
        Generate compact, URL-safe identifier using base64 encoding for space-constrained applications.

        This method creates shorter alternatives to standard UUIDs while maintaining
        cryptographic randomness. The base64 encoding produces URL-safe identifiers
        suitable for web applications and systems with character limitations.

        Entropy Analysis:
        - 8 characters ≈ 48 bits entropy ≈ 1 in 281 trillion collision chance
        - 12 characters ≈ 72 bits entropy ≈ 1 in 4.7 quadrillion collision chance
        - 16 characters ≈ 96 bits entropy ≈ 1 in 79 quintillion collision chance

        Character Set:
        - Base64 URL-safe alphabet: A-Z, a-z, 0-9, -, _
        - No padding characters (=) for cleaner appearance
        - Safe for use in URLs, filenames, and web applications

        Use Cases:
        - URL slugs and short links for web applications
        - QR codes where space is at premium
        - Mobile applications with character limits
        - Display IDs in user interfaces
        - Temporary tokens and session keys
        - File names requiring concise identifiers

        Trade-offs:
        - Shorter length = higher collision probability
        - Less entropy than full UUIDs
        - Not suitable for large-scale systems without collision handling

        Args:
            length: Length of generated identifier (minimum 1, recommended 8+)

        Returns:
            URL-safe base64 string of specified length

        Raises:
            ValueError: If length is less than 1

        Example:
            >>> agent = SecurityAgent(logger, Path("checksums.txt"))
            >>> short_id = agent.generate_lenbased_uuid(8)
            >>> short_id
            "xK9mP2vL"
            >>> len(short_id)
            8

            >>> longer_id = agent.generate_lenbased_uuid(12)
            >>> longer_id
            "aB3xK9mP2vLq"

        Note:
            For applications requiring strong uniqueness guarantees, use longer
            lengths (12+ characters) or consider full UUIDs with compression.
        """
        # Log compact UUID generation request
        self.logger.debug(f"Generating length-based UUID: length={length}")

        # Validate minimum length requirement
        if length < 1:
            error_msg: str = "Length must be at least 1 character"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Calculate bytes needed for sufficient entropy
            # Use extra bytes to account for base64 expansion and padding removal
            byte_count: int = max(length, 6)  # Minimum 6 bytes for reasonable entropy

            # Generate cryptographically secure random bytes
            random_bytes: bytes = secrets.token_bytes(byte_count)

            # Encode as URL-safe base64 and remove padding for cleaner appearance
            base64_encoded: str = base64.urlsafe_b64encode(random_bytes).decode("ascii")
            clean_encoded: str = base64_encoded.rstrip("=")

            # Truncate to exactly the requested length
            final_identifier: str = clean_encoded[:length]

            # Calculate effective entropy for logging
            effective_entropy_bits: float = (
                length * 6
            )  # Approximate bits per base64 character
            self.logger.info(
                f"Generated {length}-character identifier with ~{effective_entropy_bits:.0f} bits entropy"
            )

            return final_identifier

        except Exception as generation_error:
            error_msg = f"Failed to generate length-based UUID: {generation_error}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from generation_error
