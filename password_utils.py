#!/usr/bin/env python3
"""
Simple, secure password generator.
Generates random passwords or memorable passphrases.
"""

import secrets
import string


def generate_password(length=16, include_symbols=True, exclude_similar=True):
    """
    Generate a secure random password.

    Args:
        length: Password length (default 16)
        include_symbols: Include symbols (default True)
        exclude_similar: Remove 0O1lI| to avoid confusion (default True)

    Returns:
        Secure random password
    """
    chars = string.ascii_letters + string.digits

    if include_symbols:
        chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

    if exclude_similar:
        similar_chars = "0O1lI|"
        chars = "".join(c for c in chars if c not in similar_chars)

    # Generate and ensure we have variety
    for _ in range(50):  # Try up to 50 times
        password = "".join(secrets.choice(chars) for _ in range(length))

        # Check we have at least one of each type (if long enough)
        if length >= 4:
            has_lower = any(c.islower() for c in password)
            has_upper = any(c.isupper() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_symbol = (
                any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
                if include_symbols
                else True
            )

            if has_lower and has_upper and has_digit and has_symbol:
                return password

    # If we can't meet requirements, just return random (still secure)
    return "".join(secrets.choice(chars) for _ in range(length))


def generate_passphrase(
    words=6, separator="-", add_number=False, capitalize=False, use_wordlist=False
):
    """
    Generate a memorable passphrase.

    Args:
        words: Number of words (default 4)
        separator: Character between words (default "-")
        add_number: Add random 2-digit number (default True)
        capitalize: Capitalize each word (default True)
        use_wordlist: Try to use MIT wordlist file if available (default True)

    Returns:
        Memorable passphrase like "Forest-Mountain-River-Ocean-73"
    """
    word_list = [
        "apple",
        "beach",
        "cloud",
        "dance",
        "eagle",
        "flame",
        "grape",
        "house",
        "island",
        "jungle",
        "kite",
        "lemon",
        "moon",
        "night",
        "ocean",
        "piano",
        "quiet",
        "river",
        "stone",
        "tower",
        "voice",
        "water",
        "yellow",
        "zebra",
        "magic",
        "forest",
        "mountain",
        "thunder",
        "crystal",
        "dragon",
        "castle",
        "wizard",
        "tiger",
        "phoenix",
        "storm",
        "garden",
        "bridge",
        "rocket",
        "compass",
        "diamond",
        "emerald",
        "falcon",
        "harbor",
        "knight",
        "library",
        "marble",
        "anchor",
        "bronze",
        "copper",
        "desert",
        "engine",
        "feather",
        "golden",
        "helmet",
        "ivory",
        "jasper",
        "keystone",
        "lantern",
        "mirror",
        "nectar",
        "opal",
        "palace",
        "quartz",
        "ruby",
        "silver",
        "topaz",
        "umbrella",
        "valley",
        "willow",
        "cosmic",
        "shadow",
        "bright",
        "winter",
        "summer",
        "spring",
        "autumn",
        "plasma",
        "neutron",
        "galaxy",
        "stellar",
    ]

    if use_wordlist:
        # Try the optimized approach - read random lines instead of whole file
        chosen = _get_random_words_from_file(words)
    else:
        chosen = [secrets.choice(word_list) for _ in range(words)]

    # Capitalize if requested
    if capitalize:
        chosen = [word.capitalize() for word in chosen]

    # Add random number if requested
    if add_number:
        chosen.append(str(secrets.randbelow(90) + 10))  # 10-99

    return separator.join(chosen)


def _get_random_words_from_file(word_count):
    """Memory-efficient: read only random lines from MIT wordlist."""
    from pathlib import Path

    # Try to find the MIT wordlist file
    possible_paths = [
        Path("assets/mit_wordlist.txt"),
        Path("mit_wordlist.txt"),
        Path(__file__).parent / "assets" / "mit_wordlist.txt",
    ]

    wordlist_path = None
    for path in possible_paths:
        if path.exists():
            wordlist_path = path
            break

    if not wordlist_path:
        raise FileNotFoundError("MIT wordlist not found")

    # First, count total lines in file
    with open(wordlist_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())

    # Generate random line numbers
    random_lines = sorted(secrets.randbelow(line_count) for _ in range(word_count))

    # Read only the specific lines we need
    words = []
    with open(wordlist_path, "r", encoding="utf-8") as f:
        for current_line, line in enumerate(f):
            if current_line in random_lines:
                word = line.strip().lower()
                if len(word) >= 3:  # Only use words 3+ characters
                    words.append(word)
                if len(words) >= word_count:
                    break

    return words[:word_count]


def demo():
    """Demo the password generator."""
    print("Password Generator")
    print("=" * 40)

    print("\nRandom Passwords:")
    print(f"Standard (16):    {generate_password()}")
    print(f"Long (24):        {generate_password(24)}")
    print(f"Ultra (32):       {generate_password(32)}")
    print(f"No symbols:       {generate_password(16, include_symbols=False)}")
    print(f"Allow similar:    {generate_password(16, exclude_similar=False)}")

    print("\nPassphrases:")
    print(f"Standard:         {generate_passphrase()}")
    print(f"5 words:          {generate_passphrase(5)}")
    print(f"Underscore:       {generate_passphrase(4, '_')}")
    print(f"Number suffixes:  {generate_passphrase(4, add_number=True)}")
    print(f"Capitalize:       {generate_passphrase(4, capitalize=True)}")
