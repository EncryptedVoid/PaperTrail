#!/usr/bin/env python3
"""
Test UUID extraction logic
"""

from pathlib import Path

# Test the current (broken) logic vs fixed logic
ARTIFACT_PREFIX = "ARTIFACT"
PROFILE_PREFIX = "PROFILE"


def test_uuid_extraction():
    """Test both extraction methods"""

    # Example filenames from your logs
    test_files = [
        "ARTIFACT-24931e49-4775-43db-be8e-a07c082293e1.pdf",
        "ARTIFACT-a62ca4fb-b297-45d5-b93b-6943724c8f2d.pdf",
        "ARTIFACT-0616350a-e3a6-4f0c-b8a0-a230c32b5853.pdf",
    ]

    for filename in test_files:
        filepath = Path(filename)

        print(f"\nTesting: {filename}")
        print(f"  stem: {filepath.stem}")

        # Current (broken) logic
        broken_id = filepath.stem[len(ARTIFACT_PREFIX) :]
        print(f"  Broken extraction: '{broken_id}'")

        # Fixed logic
        fixed_id = filepath.stem[len(ARTIFACT_PREFIX) + 1 :]
        print(f"  Fixed extraction:  '{fixed_id}'")

        # Expected profile path
        profile_name = f"{PROFILE_PREFIX}-{fixed_id}.json"
        print(f"  Expected profile:  {profile_name}")

        # Check what the broken logic would look for
        broken_profile = f"{PROFILE_PREFIX}-{broken_id}.json"
        print(f"  Broken would look for: {broken_profile}")


if __name__ == "__main__":
    test_uuid_extraction()
