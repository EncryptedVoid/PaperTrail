#!/usr/bin/env python3
"""
Ultra-secure password generator with cryptographic guarantees.

This module provides cryptographically secure password generation using Python's
secrets module. Follows PEP 8 style guidelines and PEP 484 type annotations.

Exports:
    - PasswordConfig: Configuration dataclass for password parameters
    - PasswordType: Enum for password generation modes
    - SecurityProfile: Predefined security configurations
    - generate_password: Main generation function

Security Design:
    - Uses secrets module for cryptographically secure randomness
    - Implements configurable entropy requirements
    - Prevents common password weaknesses
    - Provides mathematical security guarantees

Standards References:
    - NIST SP 800-63B: https://pages.nist.gov/800-63-3/sp800-63b.html
    - RFC 4086 (Randomness): https://tools.ietf.org/html/rfc4086
    - OWASP Password Guidelines: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
    - PEP 8: https://peps.python.org/pep-0008/
    - PEP 484: https://peps.python.org/pep-0484/
    - PEP 257: https://peps.python.org/pep-0257/
"""


import math
import re
import secrets
import string
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Set


# Export list for clean module interface
__all__ = ["PasswordConfig", "PasswordType", "SecurityProfile", "generate_password"]

# Character set constants - treating as strings throughout
_LOWERCASE = string.ascii_lowercase
_UPPERCASE = string.ascii_uppercase
_DIGITS = string.digits
_SYMBOLS_BASIC = "!@#$%^&*"
_SYMBOLS_EXTENDED = "!@#$%^&*()_+-=[]{}|;:,.<>?"
_SYMBOLS_FULL = string.punctuation
_WORDS = "words"  # Special marker for word-based generation
_SEPARATORS = "-_"
_AMBIGUOUS_CHARS = "0O1lI|"
_SEQUENTIAL_PATTERNS = [
    "abc",
    "bcd",
    "cde",
    "def",
    "efg",
    "fgh",
    "ghi",
    "hij",
    "ijk",
    "jkl",
    "klm",
    "lmn",
    "mno",
    "nop",
    "opq",
    "pqr",
    "qrs",
    "rst",
    "stu",
    "tuv",
    "uvw",
    "vwx",
    "wxy",
    "xyz",
    "123",
    "234",
    "345",
    "456",
    "567",
    "678",
    "789",
    "890",
    "qwe",
    "wer",
    "ert",
    "rty",
    "tyu",
    "yui",
    "uio",
    "iop",
    "asd",
    "sdf",
    "dfg",
    "fgh",
    "ghj",
    "hjk",
    "jkl",
    "zxc",
    "xcv",
    "cvb",
    "vbn",
    "bnm",
]


class PasswordType(Enum):
    """Password generation modes."""

    CHARACTER_BASED = "character_based"
    PASSPHRASE = "passphrase"


@dataclass(frozen=True, slots=True)
class PasswordConfig:
    """
    Immutable configuration for secure password generation.

    This dataclass encapsulates all password generation parameters following
    modern security standards and defensive programming principles.

    Attributes:
        password_type: Generation mode (character-based vs passphrase)
        length: Total password length (minimum 12 for character-based, 4 words for passphrase)
        min_lowercase: Minimum required lowercase letters
        min_uppercase: Minimum required uppercase letters
        min_digits: Minimum required numeric digits
        min_symbols: Minimum required symbol characters
        exclude_ambiguous: Remove visually similar characters (0O1lI|)
        exclude_sequential: Prevent keyboard patterns (qwerty, 123abc)
        character_sets: Set of strings defining allowed characters
        max_repeated: Maximum consecutive identical characters allowed
        ensure_pronounceable: Attempt to make password somewhat pronounceable
        word_count: Number of words (passphrase mode only)
        word_separator: Separator between words (passphrase mode only)
        min_word_length: Minimum length per word (passphrase mode only)
        max_word_length: Maximum length per word (passphrase mode only)
        max_gen_attempts: Maximum generation attempts before failure

    Raises:
        ValueError: If configuration is invalid or creates weak passwords
        TypeError: If incompatible parameters are provided

    Security Notes:
        - Character-based: Minimum 12 chars follows NIST SP 800-63B
        - Passphrase: Minimum 4 words provides ~52+ bits entropy
        - All validation ensures cryptographically strong passwords
        - Entropy calculations based on NIST and security research
    """

    # Core parameters (required for all types)
    password_type: PasswordType
    length: int
    exclude_ambiguous: bool = True
    exclude_sequential: bool = True
    max_repeated: int = 2
    max_gen_attempts: int = 1000

    # Character-based password parameters
    min_lowercase: int = 1
    min_uppercase: int = 1
    min_digits: int = 1
    min_symbols: int = 1
    character_sets: Set[str] = None
    ensure_pronounceable: bool = False

    # Passphrase-specific parameters
    word_count: int = 0
    word_separator: str = ""
    min_word_length: int = 0
    max_word_length: int = 0

    def __post_init__(self) -> None:
        """
        Comprehensive validation of configuration parameters.

        Validates parameters based on password type and ensures
        cryptographically secure configurations.
        """
        self._validate_core_parameters()

        if self.password_type == PasswordType.CHARACTER_BASED:
            self._validate_character_based_parameters()
        elif self.password_type == PasswordType.PASSPHRASE:
            self._validate_passphrase_parameters()
        else:
            raise ValueError(f"Unsupported password type: {self.password_type}")

    def _validate_core_parameters(self) -> None:
        """Validate core parameters common to all password types."""
        if self.length < 1:
            raise ValueError("Password length must be positive")

        if self.max_repeated < 1:
            raise ValueError("max_repeated must be at least 1")

        if not isinstance(self.password_type, PasswordType):
            raise TypeError("password_type must be a PasswordType enum value")

    def _validate_character_based_parameters(self) -> None:
        """Validate parameters specific to character-based passwords."""
        # Enforce NIST minimum for character-based passwords
        if self.length < 12:
            raise ValueError(
                "Character-based passwords must be at least 12 characters "
                "(NIST SP 800-63B recommendation)"
            )

        # Validate minimum character requirements
        for attr_name, min_val in [
            ("min_lowercase", self.min_lowercase),
            ("min_uppercase", self.min_uppercase),
            ("min_digits", self.min_digits),
            ("min_symbols", self.min_symbols),
        ]:
            if min_val < 0:
                raise ValueError(f"{attr_name} cannot be negative")

        # Check if requirements can be satisfied
        min_required = (
            self.min_lowercase + self.min_uppercase + self.min_digits + self.min_symbols
        )

        if self.length < min_required:
            raise ValueError(
                f"Password length ({self.length}) cannot satisfy minimum "
                f"character requirements (total: {min_required})"
            )

        # Set default character sets if none provided
        if self.character_sets is None:
            raise ValueError(
                "character_sets cannot be empty for custom numerical & symbolic passwords"
            )

        # Validate character sets
        if not self.character_sets:
            raise ValueError(
                "character_sets cannot be empty for character-based passwords"
            )

        if not isinstance(self.character_sets, set):
            raise TypeError("character_sets must be a set of strings")

        # Validate max_repeated is reasonable
        if self.max_repeated > self.length // 2:
            raise ValueError(
                f"max_repeated ({self.max_repeated}) is too high for "
                f"password length ({self.length})"
            )

        # Ensure required character types are available in character_sets
        requirements = [
            (self.min_lowercase > 0, _LOWERCASE, "lowercase"),
            (self.min_uppercase > 0, _UPPERCASE, "uppercase"),
            (self.min_digits > 0, _DIGITS, "digits"),
        ]

        for required, charset, name in requirements:
            if required and charset not in self.character_sets:
                raise ValueError(
                    f"min_{name} > 0 but {name} charset not in character_sets"
                )

        # Check for symbol requirements
        if self.min_symbols > 0:
            symbol_sets = {
                _SYMBOLS_BASIC,
                _SYMBOLS_EXTENDED,
                _SYMBOLS_FULL,
            }
            if not any(s in self.character_sets for s in symbol_sets):
                raise ValueError(
                    "min_symbols > 0 but no symbol character sets provided"
                )

    def _validate_passphrase_parameters(self) -> None:
        """Validate parameters specific to passphrase generation."""
        # For passphrases, length represents minimum total character count
        if self.length < 15:  # Reasonable minimum for 4+ words
            raise ValueError(
                "Passphrase minimum length should be at least 15 characters"
            )

        if self.word_count < 3:
            raise ValueError("Passphrases must contain at least 3 words for security")

        if self.word_count > 12:
            raise ValueError("Passphrases with more than 12 words become unwieldy")

        if self.min_word_length < 3:
            raise ValueError("Words must be at least 3 characters long")

        if self.max_word_length < self.min_word_length:
            raise ValueError("max_word_length must be >= min_word_length")

        if self.max_word_length > 15:
            raise ValueError("Words longer than 15 characters are impractical")

        # Validate separator
        if not self.word_separator:
            raise ValueError("word_separator cannot be empty")

        if len(self.word_separator) > 3:
            raise ValueError("word_separator should be 1-3 characters")

        # Check separator is safe
        if re.search(r"[a-zA-Z0-9]", self.word_separator):
            raise ValueError("word_separator should not contain letters or digits")

        # For passphrases, character-based minimums should be 0 or small
        char_minimums = [
            self.min_lowercase,
            self.min_uppercase,
            self.min_digits,
            self.min_symbols,
        ]
        if any(m > 2 for m in char_minimums):
            raise ValueError(
                "Character minimums should be low (≤2) for passphrase mode "
                "as they're handled through word selection"
            )

    def estimate_entropy(self) -> float:
        """
        Calculate estimated entropy in bits for this configuration.

        Returns:
            float: Estimated entropy in bits

        References:
            - NIST SP 800-63B Section 5.1.1: https://pages.nist.gov/800-63-3/sp800-63b.html#sec5
            - Shannon Information Theory: https://en.wikipedia.org/wiki/Entropy_(information_theory)
        """
        if self.password_type == PasswordType.CHARACTER_BASED:
            # Calculate character pool size
            pool_size = 0
            for charset in self.character_sets:
                if charset == _WORDS or charset == _SEPARATORS:
                    continue  # Skip word-based sets for character passwords
                pool_size += len(charset)

            # Adjust for excluded ambiguous characters
            if self.exclude_ambiguous:
                ambiguous_chars = "0O1lI|"
                pool_size -= len(
                    [
                        c
                        for c in ambiguous_chars
                        if any(c in cs for cs in self.character_sets)
                    ]
                )

            return self.length * math.log2(max(pool_size, 1))

        else:  # PASSPHRASE
            # Assume ~7776 word dictionary (standard Diceware)
            # Plus entropy from separators and capitalization
            word_entropy = self.word_count * math.log2(7776)
            separator_entropy = (
                math.log2(len(_SEPARATORS)) if len(self.word_separator) == 1 else 0
            )
            return word_entropy + separator_entropy

    def get_security_level(self) -> str:
        """
        Get human-readable security level based on entropy.

        Returns:
            str: Security level description

        References:
            - NIST Entropy Requirements: https://pages.nist.gov/800-63-3/sp800-63b.html#sec5
        """
        entropy = self.estimate_entropy()

        if entropy < 50:
            return "WEAK"
        elif entropy < 70:
            return "MODERATE"
        elif entropy < 100:
            return "STRONG"
        elif entropy < 150:
            return "VERY_STRONG"
        else:
            return "ULTRA_STRONG"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.password_type == PasswordType.CHARACTER_BASED:
            return (
                f"Password Config using STANDARD PASSWORD method with len={self.length}, "
                f"min=[{self.min_lowercase}L,{self.min_uppercase}U,"
                f"{self.min_digits}D,{self.min_symbols}S], "
                f"entropy=~{self.estimate_entropy():.0f}bits, "
                f"security={self.get_security_level()})"
            )
        else:
            return (
                f"Password Config using PASSPHRASE method with {self.word_count} words, "
                f"sep='{self.word_separator}', min_len={self.length}, "
                f"entropy=~{self.estimate_entropy():.0f}bits, "
                f"security={self.get_security_level()})"
            )


class SecurityProfile(Enum):
    """
    Predefined security profiles for password generation.

    Includes both character-based and passphrase options for different
    security needs and usability requirements.

    References:
        - NIST Authentication Guidelines: https://pages.nist.gov/800-63-3/sp800-63b.html
        - OWASP Password Guidelines: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
    """

    BASIC = "basic"
    SUPER = "super"
    ULTRA = "ultra"
    PASSPHRASE = "passphrase"

    def get_config(self) -> PasswordConfig:
        """
        Returns a PasswordConfig instance for the selected security profile.

        Returns:
            PasswordConfig: Configured instance for the security level
        """
        if self == SecurityProfile.BASIC:
            # User-friendly but secure
            return PasswordConfig(
                password_type=PasswordType.CHARACTER_BASED,
                length=12,
                min_lowercase=1,
                min_uppercase=1,
                min_digits=1,
                min_symbols=1,
                exclude_ambiguous=True,
                exclude_sequential=False,  # Allow some patterns for memorability
                character_sets={
                    _LOWERCASE,
                    _UPPERCASE,
                    _DIGITS,
                    _SYMBOLS_BASIC,  # Only common symbols
                },
                max_repeated=3,  # More lenient
                ensure_pronounceable=False,
            )

        elif self == SecurityProfile.SUPER:
            # Strong security with good usability
            return PasswordConfig(
                password_type=PasswordType.CHARACTER_BASED,
                length=18,
                min_lowercase=2,
                min_uppercase=2,
                min_digits=2,
                min_symbols=2,
                exclude_ambiguous=True,
                exclude_sequential=True,
                character_sets={
                    _LOWERCASE,
                    _UPPERCASE,
                    _DIGITS,
                    _SYMBOLS_EXTENDED,  # More symbols available
                },
                max_repeated=2,
                ensure_pronounceable=False,
            )

        elif self == SecurityProfile.ULTRA:
            # Maximum security for high-value targets
            return PasswordConfig(
                password_type=PasswordType.CHARACTER_BASED,
                length=28,
                min_lowercase=4,
                min_uppercase=4,
                min_digits=4,
                min_symbols=4,
                exclude_ambiguous=True,
                exclude_sequential=True,
                character_sets={
                    _LOWERCASE,
                    _UPPERCASE,
                    _DIGITS,
                    _SYMBOLS_FULL,  # All available symbols
                },
                max_repeated=1,  # Minimal repetition
                ensure_pronounceable=False,
            )

        elif self == SecurityProfile.PASSPHRASE:
            # Memorable but secure word-based passwords
            return PasswordConfig(
                password_type=PasswordType.PASSPHRASE,
                length=25,  # Minimum total characters
                min_lowercase=0,  # Handled through word selection
                min_uppercase=1,  # At least one capitalized word
                min_digits=1,  # At least one number
                min_symbols=1,  # Separators count as symbols
                exclude_ambiguous=True,
                exclude_sequential=False,  # Words can have sequential letters
                character_sets={
                    _WORDS,  # Enable word-based generation
                    _SEPARATORS,  # Word separators
                    _DIGITS,  # For adding numbers
                },
                max_repeated=2,  # Allow some repetition in words
                ensure_pronounceable=True,  # Words are inherently pronounceable
                word_count=4,  # Good balance of security/memorability
                word_separator="-",  # Clean, readable separator
                min_word_length=4,  # Avoid very short words
                max_word_length=8,  # Avoid very long words
            )

    @classmethod
    def get_profile_info(cls) -> dict:
        """
        Returns detailed information about each security profile.

        Returns:
            dict: Profile descriptions, entropy estimates, and use cases
        """
        return {
            cls.BASIC: {
                "entropy_bits": "~79 bits",
                "use_case": "Standard websites, social media, low-risk accounts",
                "description": "Meets basic security requirements with user-friendly character sets",
                "example_pattern": "Xy7$kLm9pQ!z",
                "character_pool": "62 chars (letters + digits + 8 symbols)",
            },
            cls.SUPER: {
                "entropy_bits": "~119 bits",
                "use_case": "Banking, work accounts, email, important services",
                "description": "Strong security following modern best practices",
                "example_pattern": "Kp9#Xm2$vL8!qR3@Nt",
                "character_pool": "78+ chars (letters + digits + extended symbols)",
            },
            cls.ULTRA: {
                "entropy_bits": "~185+ bits",
                "use_case": "Root passwords, crypto wallets, admin accounts",
                "description": "Maximum security for the most sensitive accounts",
                "example_pattern": "mK9$pX2#vL8!qR3@Nt7%gF4^zA1&",
                "character_pool": "94+ chars (all printable ASCII)",
            },
            cls.PASSPHRASE: {
                "entropy_bits": "~52+ bits",
                "use_case": "Human-memorable passwords, manual entry scenarios",
                "description": "Word-based passwords balancing security and memorability",
                "example_pattern": "Mountain-Tiger-River-Moon7",
                "character_pool": "7776+ word dictionary + separators + numbers",
            },
        }


def _build_character_pool(config: PasswordConfig) -> str:
    """
    Build available character pool from configuration.

    Args:
        config: Password configuration specifying character requirements

    Returns:
        String containing all valid characters for password generation

    Notes:
        - Removes ambiguous characters if requested
        - Eliminates duplicates while preserving character order
        - Applies filtering rules consistently
    """
    pool = ""

    # Combine all requested character sets
    for char_set in config.character_sets:
        pool += char_set

    # Remove ambiguous characters if requested
    if config.exclude_ambiguous:
        pool = "".join(c for c in pool if c not in _AMBIGUOUS_CHARS)

    # Remove duplicates while preserving order (uses dict.fromkeys trick)
    pool = "".join(dict.fromkeys(pool))

    return pool


def _has_sequential_pattern(password: str, exclude_sequential: bool) -> bool:
    """
    Check if password contains common sequential patterns.

    Args:
        password: Password string to validate
        exclude_sequential: Whether to check for patterns

    Returns:
        True if sequential patterns found, False otherwise

    Notes:
        - Checks against keyboard layouts and alphabetic sequences
        - Case-insensitive comparison for broader pattern detection

    References:
        - OWASP Password Guidelines: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
    """
    if not exclude_sequential:
        return False

    password_lower = password.lower()

    for pattern in _SEQUENTIAL_PATTERNS:
        if pattern in password_lower:
            return True

    return False


def _meets_character_requirements(
    password: str,
    min_lowercase: int,
    min_uppercase: int,
    min_digits: int,
    min_symbols: int,
) -> bool:
    """
    Verify password meets minimum character type requirements.

    Args:
        password: Password string to validate
        min_lowercase: Minimum lowercase letters required
        min_uppercase: Minimum uppercase letters required
        min_digits: Minimum numeric digits required
        min_symbols: Minimum symbol characters required

    Returns:
        True if all requirements met, False otherwise

    Notes:
        - Uses Counter for efficient character classification
        - Symbols include all punctuation characters
    """
    char_counts: Counter[str] = Counter()

    for char in password:
        if char in string.ascii_lowercase:
            char_counts["lowercase"] += 1
        elif char in string.ascii_uppercase:
            char_counts["uppercase"] += 1
        elif char in string.digits:
            char_counts["digits"] += 1
        elif char in string.punctuation:
            char_counts["symbols"] += 1

    return (
        char_counts["lowercase"] >= min_lowercase
        and char_counts["uppercase"] >= min_uppercase
        and char_counts["digits"] >= min_digits
        and char_counts["symbols"] >= min_symbols
    )


def _has_excessive_repetition(password: str, max_repeated: int) -> bool:
    """
    Check for excessive consecutive character repetition.

    Args:
        password: Password string to validate
        max_repeated: Maximum allowed consecutive identical characters

    Returns:
        True if excessive repetition found, False otherwise

    Notes:
        - Prevents patterns like "aaa" or "111" that weaken entropy
        - Uses sliding window approach for efficient detection
    """
    consecutive_count = 1

    for i in range(1, len(password)):
        if password[i] == password[i - 1]:
            consecutive_count += 1
            if consecutive_count > max_repeated:
                return True
        else:
            consecutive_count = 1

    return False


def generate_password(config: PasswordConfig) -> str:
    """
    Generate a cryptographically secure password from configuration.

    Args:
        config: PasswordConfig object specifying all generation parameters

    Returns:
        Cryptographically secure password meeting all specified requirements

    Raises:
        ValueError: If configuration creates impossible or insecure requirements
        RuntimeError: If unable to generate compliant password within max_gen_attempts
        TypeError: If config is not a PasswordConfig instance

    Security Guarantees:
        - Uses secrets.choice() for cryptographically secure randomness
        - Minimum entropy based on NIST SP 800-63B guidelines
        - 16-char password: ~99+ bits entropy (computationally infeasible)
        - 20-char password: ~124+ bits entropy (exceeds AES-128 strength)
        - Validates against common password weakness patterns

    Examples:
        >>> from password_generator import generate_password, SecurityProfile
        >>>
        >>> # Using predefined security profile
        >>> config = SecurityProfile.SUPER.get_config()
        >>> password = generate_password(config)
        >>>
        >>> # Using custom configuration
        >>> config = PasswordConfig(
        ...     password_type=PasswordType.CHARACTER_BASED,
        ...     length=20,
        ...     min_symbols=3,
        ...     character_sets={_LOWERCASE, _UPPERCASE, _DIGITS, _SYMBOLS_EXTENDED}
        ... )
        >>> password = generate_password(config)

    Performance:
        - O(n) time complexity where n is password length
        - Expected attempts: 1-10 for reasonable configurations
        - Worst case: max_gen_attempts before RuntimeError

    References:
        - NIST SP 800-63B: https://pages.nist.gov/800-63-3/sp800-63b.html
        - RFC 4086 Randomness: https://tools.ietf.org/html/rfc4086
        - Python secrets module: https://docs.python.org/3/library/secrets.html
    """
    if not isinstance(config, PasswordConfig):
        raise TypeError("config must be a PasswordConfig instance")

    # Note: Character-based only for now (passphrase implementation would need word dictionary)
    if config.password_type != PasswordType.CHARACTER_BASED:
        raise NotImplementedError("Only CHARACTER_BASED passwords currently supported")

    # Build character pool from configuration
    character_pool = _build_character_pool(config)

    if len(character_pool) < 4:
        raise ValueError("Character pool too small - need at least 4 unique characters")

    # Generate password with validation loop
    for _ in range(config.max_gen_attempts):
        # Use cryptographically secure random generation
        password = "".join(secrets.choice(character_pool) for _ in range(config.length))

        # Validate against all security requirements
        if (
            _meets_character_requirements(
                password,
                config.min_lowercase,
                config.min_uppercase,
                config.min_digits,
                config.min_symbols,
            )
            and not _has_sequential_pattern(password, config.exclude_sequential)
            and not _has_excessive_repetition(password, config.max_repeated)
        ):
            return password

    # Failed to generate compliant password
    raise RuntimeError(
        f"Failed to generate compliant password after {config.max_gen_attempts} attempts. "
        f"Current password configuration: {config}"
    )


# Example usage and testing
if __name__ == "__main__":

    print("\nGenerating Sample Passwords:")
    print("=" * 30)

    # Test each security profile
    for profile in SecurityProfile:
        if profile == SecurityProfile.PASSPHRASE:
            print(f"{profile.value.upper()}: [Not implemented - needs word dictionary]")
            continue

        try:
            config = profile.get_config()
            password = generate_password(config)

            print(f"{profile.value.upper()}: {password}")
            print(f"  Config: {config}")
            print(f"  Entropy: ~{config.estimate_entropy():.1f} bits")
            print(f"  Security: {config.get_security_level()}")
            print()

        except Exception as e:
            print(f"{profile.value.upper()}: ERROR - {e}")
            print()

    print("=" * 30)
