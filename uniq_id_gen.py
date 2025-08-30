"""
UUID generation utilities with cryptographic security and various ID formats.

This module provides multiple UUID generation strategies for different use cases,
from simple random IDs to time-sortable and deterministic identifiers.
"""

from datetime import datetime, timezone
from typing import Optional
import uuid
import secrets
import time
import base64


def generate_uuid4plus(
    prefix: str = "", include_timestamp: bool = False, entropy: int = 16
) -> str:
    """
    Enhanced UUID4 with guaranteed cryptographic security.
    Your go-to function for secure, random UUIDs.

    Use Cases:
    - Any time you'd use uuid.uuid4() but want guaranteed security
    - User IDs, session IDs, transaction IDs, API keys
    - General purpose unique identifiers for production systems
    - Default choice for most applications requiring unique IDs

    Args:
        prefix: Optional string prefix to prepend to the UUID

    Returns:
        String in format "{prefix}-{secure_uuid}" or just "{secure_uuid}" if no prefix

    Example:
        >>> generate_uuid4plus()
        "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        >>> generate_uuid4plus("user")
        "user-a3f5c2d1-8b94-4f6e-9a72-1e3d5c7b9f2a"

    Security:
        Uses secrets.token_bytes() - cryptographically secure on all platforms.
        This is more secure than standard uuid.uuid4() which may vary by system.
    """
    # Use secrets for guaranteed cryptographic security instead of uuid.uuid4()
    secure_uuid = str(uuid.UUID(bytes=secrets.token_bytes(entropy), version=4))

    if include_timestamp:
        # Add ISO timestamp for traceability
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        result = f"{timestamp}-{secure_uuid}"
    else:
        result = secure_uuid

    return f"{prefix}-{result}" if prefix else result


def generate_uuid5(name: str, namespace_uuid: Optional[str] = None) -> str:
    """
    Generate a deterministic UUID based on namespace and name (UUID5).
    Same inputs ALWAYS produce the exact same UUID - fully reproducible.

    Use Cases:
    - User accounts based on email addresses
    - Resource IDs that must be consistent across deployments
    - Idempotent operations (same request = same ID)
    - Content addressing (same content = same ID)
    - Migration scripts where IDs must be predictable
    - API endpoints that need consistent resource identifiers

    Args:
        name: Input string to generate UUID from
        namespace_uuid: Optional custom namespace UUID string (defaults to DNS namespace)

    Returns:
        Deterministic UUID5 string

    Example:
        >>> generate_uuid5("john@example.com")
        "8c8a5cf6-8b8a-5c3d-9f2e-7a1b4c5d6e8f"
        >>> generate_uuid5("john@example.com")  # Same result!
        "8c8a5cf6-8b8a-5c3d-9f2e-7a1b4c5d6e8f"

    Raises:
        ValueError: If namespace_uuid is provided but invalid
    """
    try:
        # Use DNS namespace as default, or parse custom namespace
        namespace = (
            uuid.NAMESPACE_DNS if namespace_uuid is None else uuid.UUID(namespace_uuid)
        )
        return str(uuid.uuid5(namespace, name))
    except ValueError as e:
        raise ValueError(f"Invalid namespace UUID: {namespace_uuid}") from e


def generate_ulid(identifier_prefix: str = "") -> str:
    """
    Generate a sortable UUID with timestamp prefix (ULID-style).
    UUIDs automatically sort chronologically by creation time.

    Use Cases:
    - Database primary keys where chronological sorting is important
    - Log entries that need natural time ordering
    - Distributed systems where nodes generate IDs independently but need ordering
    - Event sourcing systems
    - Message queues where processing order matters
    - File systems where chronological listing is desired

    Args:
        identifier_prefix: Optional string prefix

    Returns:
        Timestamp-prefixed UUID that sorts chronologically

    Example:
        >>> generate_ulid("msg")
        "msg-018a4c2b-7f3e-4d9a-8b1c-2f5e7a9c3d6f"
        >>> generate_ulid()
        "018a4c2c-1a2b-4c3d-8f7e-9a1b2c3d4e5f"

    Benefits:
        Natural chronological sorting, good database index performance.
    """
    # Get current timestamp in milliseconds since epoch
    timestamp_ms = int(time.time() * 1000)

    # Convert to hex and pad to 12 characters (48 bits)
    timestamp_hex = f"{timestamp_ms:012x}"

    # Generate random component (80 bits)
    random_hex = secrets.token_hex(10)  # 20 hex chars = 80 bits

    # Combine timestamp + random = 32 hex chars total (128 bits)
    sortable_uuid = f"{timestamp_hex}{random_hex}"

    # Format as standard UUID: 8-4-4-4-12
    formatted = (
        f"{sortable_uuid[:8]}-{sortable_uuid[8:12]}-"
        f"{sortable_uuid[12:16]}-{sortable_uuid[16:20]}-{sortable_uuid[20:32]}"
    )

    return f"{identifier_prefix}-{formatted}" if identifier_prefix else formatted


def generate_lenbased_uuid(length: int = 8) -> str:
    """
    Generate a compact, URL-safe identifier using base64 encoding.
    Much shorter than standard UUIDs while maintaining randomness.

    Use Cases:
    - URL slugs and short links
    - QR codes where space is limited
    - Mobile apps with character limits
    - Display IDs in user interfaces
    - Temporary tokens and session keys
    - File names that need to be concise

    Args:
        length: Length of the generated ID (default: 8, minimum: 1)

    Returns:
        URL-safe base64 string of specified length

    Example:
        >>> generate_lenbased_uuid(8)
        "xK9mP2vL"
        >>> generate_lenbased_uuid(12)
        "aB3xK9mP2vLq"

    Note:
        Shorter IDs have higher collision probability. 8 chars ≈ 1 in 281 trillion.

    Raises:
        ValueError: If length is less than 1
    """
    if length < 1:
        raise ValueError("Length must be at least 1")

    # Generate enough random bytes to ensure we have sufficient entropy
    # Use length + 2 to account for base64 padding removal
    byte_count = max(length, 6)  # Minimum bytes for good entropy
    random_bytes = secrets.token_bytes(byte_count)

    # Encode as URL-safe base64 and remove padding
    short_id = base64.urlsafe_b64encode(random_bytes).decode("ascii").rstrip("=")

    # Return exactly the requested length
    return short_id[:length]
