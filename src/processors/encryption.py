import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Cryptography imports for secure file encryption operations
from tqdm import tqdm

# Import all security-related constants from centralized configuration
from config import (
    ARTIFACT_PREFIX,
    ARTIFACT_PROFILES_DIR,
    PROFILE_PREFIX,
    VAULT_MASTER_KEY,
)
from utilities.dependancy_ensurance import ensure_7zip, ensure_keepassxc


def encrypt(
    logger: logging.Logger,
    source_dir: Path,
    failure_dir: Path,
    success_dir: Path,
) -> None:

    ensure_7zip()
    ensure_keepassxc()

    # Log metadata extraction stage header for clear progress tracking
    logger.info("=" * 80)
    logger.info("ENCRYPTION STAGE")
    logger.info("=" * 80)

    # Discover all artifact files in the source directory
    unprocessed_artifacts: List[Path] = [
        item
        for item in source_dir.iterdir()
        if item.is_file() and item.name.startswith(f"{ARTIFACT_PREFIX}-")
    ]

    # Handle empty directory case
    if not unprocessed_artifacts:
        logger.info("No artifact files found in source directory")
        return None

    # Sort files by size for consistent processing order (smaller files first)
    unprocessed_artifacts.sort(key=lambda p: p.stat().st_size)
    logger.info(f"Found {len(unprocessed_artifacts)} artifact files to process")

    # Process each artifact file
    for artifact in tqdm(
        unprocessed_artifacts,
        desc="Encrypting with passwords",
        unit="artifacts",
    ):
        try:
            logger.info(f"Processing artifact: {artifact.name}")

            # Extract UUID from filename for profile lookup
            # Expected format: ARTIFACT-{uuid}.ext
            artifact_id: str = artifact.stem[len(ARTIFACT_PREFIX) + 1 :]
            artifact_profile_path: Path = (
                ARTIFACT_PROFILES_DIR / f"{PROFILE_PREFIX}-{artifact_id}.json"
            )

            # Validate that profile exists for this artifact
            if not artifact_profile_path.exists():
                error_msg: str = f"Profile not found for artifact: {artifact.name}"
                logger.error(error_msg, exc_info=True)
                raise FileNotFoundError(error_msg)

            # Load existing profile data
            artifact_profile_data: Dict[str, Any]
            try:
                with open(artifact_profile_path, "r", encoding="utf-8") as f:
                    artifact_profile_data = json.load(f)
            except json.JSONDecodeError as e:
                error_msg: str = f"Corrupted profile JSON for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg: str = f"Failed to load profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Extract filename from path
            artifact_id = artifact.name

            # Generate password or passphrase
            if (
                artifact_profile_data["semantics"]["confidentiality_level"].lower()
                == "confidential"
            ):
                generated_password = subprocess.run(
                    ["keepassxc-cli", "diceware", "--words", "6"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            else:
                generated_password = subprocess.run(
                    ["keepassxc-cli", "generate", "-L", "20", "-U", "-l", "-n", "-s"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()

            # Prepare input: master password + generated password (twice for confirmation)
            input_data = f"{generated_password}\n{generated_password}\n"
            input_data = f"{VAULT_MASTER_KEY}\n" + input_data

            # Add entry to database
            add_process = subprocess.run(
                [
                    "keepassxc-cli",
                    "add",
                    VAULT_MASTER_KEY,
                    artifact_id,
                    "--username",
                    artifact,
                    "--url",
                    artifact,
                ],
                input=input_data,
                text=True,
                capture_output=True,
            )

            if add_process.returncode != 0:
                raise Exception(f"Failed to add entry: {add_process.stderr}")

            password = _encrypt_file(logger=logger, artifact=artifact)

            # Store extracted text if available
            if password and len(password) > 0:
                artifact_profile_data["password_protection_encryption"] = {
                    "success": True,
                    "password": password,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                artifact_profile_data["password_protection_encryption"] = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            # Update stage tracking
            if "stage_progression_data" not in artifact_profile_data:
                artifact_profile_data["stage_progression_data"] = {}

            artifact_profile_data["stage_progression_data"][
                "password_protection_encryption"
            ] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }

            # Save updated profile back to disk
            try:
                with open(artifact_profile_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_profile_data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Profile updated successfully for: {artifact.name}")
            except Exception as e:
                error_msg: str = f"Failed to save profile for {artifact.name}: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            # Move artifact to success directory
            success_location: Path = success_dir / artifact.name

            # Handle naming conflicts
            if success_location.exists():
                base_name: str = success_location.stem
                extension: str = success_location.suffix
                counter: int = 1
                while success_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    success_location = success_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(success_location))
            logger.info(f"Moved processed artifact to: {success_location}")

        except Exception as e:
            error_msg: str = f"Error processing {artifact.name}: {e}"
            logger.error(error_msg, exc_info=True)

            # Move failed artifact to failure directory
            failure_location: Path = failure_dir / artifact.name

            # Handle naming conflicts in failure directory
            if failure_location.exists():
                base_name: str = failure_location.stem
                extension: str = failure_location.suffix
                counter: int = 1
                while failure_location.exists():
                    new_name: str = f"{base_name}_{counter}{extension}"
                    failure_location = failure_dir / new_name
                    counter += 1

            shutil.move(str(artifact), str(failure_location))
            logger.info(f"Moved failed artifact to: {failure_location}")
            continue

    logger.info("Encryption stage completed")
