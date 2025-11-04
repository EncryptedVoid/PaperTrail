#!/usr/bin/env python3
"""
Email Backup and Classification System

This module handles the complete email backup and classification workflow:
1. Backs up emails from Gmail and Outlook accounts to local storage
2. Classifies each email using an LLM (Large Language Model) as important, unimportant, or containing attachments
3. Organizes emails into categorized directories for easy retrieval

The classification uses Ollama with llama3.2:3b model to determine email importance
based on content analysis. Emails with attachments are automatically separated.
"""

import email
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import (
	EMAIL_ARTIFACTS_DIR ,
	EMAIL_OUTPUT_DIR ,
	GMAIL_ADDRESS ,
	IMPORTANT_EMAILS_DIR ,
	OUTLOOK_ADDRESS ,
	UNIMPORTANT_EMAILS_DIR ,
)

logger = logging.getLogger(__name__)


def run_cmd(cmd, description, shell=True, check=False, capture=True):
    """
    Execute a shell command and capture its output.

    Args:
                    cmd (str): The command to execute
                    description (str): Human-readable description of the command for logging
                    shell (bool): Whether to run command through the shell (default: True)
                    check (bool): Whether to raise exception on non-zero exit code (default: False)
                    capture (bool): Whether to capture stdout/stderr (default: True)

    Returns:
                    tuple: (success: bool, output: str) - Success status and command output

    This function wraps subprocess.run() to provide consistent error handling
    and logging for all command-line operations in the system.
    """
    logger.debug(f"Executing command: {cmd}")
    try:
        # subprocess.run() executes the command and waits for completion
        # text=True ensures output is returned as strings, not bytes
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            text=True,
            capture_output=capture,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        success = result.returncode == 0
        if success:
            logger.debug(f"Command succeeded: {description}")
        else:
            logger.warning(
                f"Command failed with code {result.returncode}: {description}"
            )
            if result.stderr:
                logger.debug(f"Command stderr: {result.stderr}")
        return success, result.stdout if capture else ""
    except Exception as e:
        logger.error(f"Command execution error for '{description}': {e}")
        return False, str(e)


def check_command_exists(cmd):
    """
    Check if a command-line tool is available in the system PATH.

    Args:
                    cmd (str): The command name to check (e.g., 'git', 'python')

    Returns:
                    bool: True if command exists and is executable, False otherwise

    Uses 'which' on Unix-like systems (Linux, macOS) and 'where' on Windows
    to locate executables in the system PATH.
    """
    logger.debug(f"Checking if command exists: {cmd}")
    # platform.system() returns 'Windows', 'Linux', 'Darwin' (macOS), etc.
    success, _ = run_cmd(
        f"which {cmd}" if platform.system() != "Windows" else f"where {cmd}",
        "",
        capture=True,
    )
    return success


def classify_email(eml_file):
    """
    Classify an email as important, unimportant, or containing attachments.

    Args:
                    eml_file (Path): Path to the .eml email file to classify

    Returns:
                    str: Classification result - 'important', 'unimportant', or 'attachments'

    Classification logic:
    1. If email has attachments -> 'attachments'
    2. Otherwise, uses LLM to analyze subject and body content
    3. LLM determines if email is important based on keywords like:
             finances, law, identity, immigration, payments, invoices, reports, work, freelancing

    If any step fails, defaults to 'important' to avoid losing potentially critical emails.
    """
    logger.debug(f"Classifying email: {eml_file.name}")

    try:
        # email.message_from_bytes() parses the raw .eml file into a Message object
        # This gives us access to headers, body, attachments, etc.
        msg = email.message_from_bytes(eml_file.read_bytes())
    except Exception as e:
        logger.warning(
            f"Failed to parse email {eml_file.name}, defaulting to important: {e}"
        )
        return "important"

    # Check if email has any attachments
    # msg.walk() iterates through all parts of a multipart email
    # get_content_disposition() returns 'attachment' for attached files
    has_attachments = any(
        part.get_content_disposition() == "attachment" for part in msg.walk()
    )

    if has_attachments:
        logger.debug(
            f"Email {eml_file.name} has attachments, categorizing as artifacts"
        )
        return "attachments"

    # Extract subject and body for LLM analysis
    # [:100] limits subject to 100 characters to avoid token limits
    subject = msg.get("Subject", "")[:100]
    body = ""

    # Handle multipart emails (HTML + text, attachments, etc.)
    if msg.is_multipart():
        for part in msg.walk():
            # Only extract plain text content, skip HTML/images/etc.
            if part.get_content_type() == "text/plain":
                try:
                    # get_payload(decode=True) gets the raw bytes
                    # .decode() converts bytes to string, ignoring invalid characters
                    body += part.get_payload(decode=True).decode(errors="ignore")
                except Exception as e:
                    logger.debug(f"Failed to decode part of {eml_file.name}: {e}")
    else:
        # Simple single-part email
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to decode body of {eml_file.name}: {e}")

    # Limit body to 1000 characters to avoid LLM token limits and reduce processing time
    body = body[:1000]

    # Dynamically import ollama library (only after we know it's installed)
    try:
        import ollama
    except ImportError:
        logger.info("Ollama Python package not found, installing")
        # sys.executable ensures we use the same Python interpreter running this script
        # -m pip install runs pip as a module to avoid PATH issues
        run_cmd(
            f"{sys.executable} -m pip install ollama",
            "Installing ollama",
            capture=False,
        )
        import ollama

    # Construct LLM prompt for classification
    # Simple YES/NO question with clear criteria for what makes an email important
    prompt = f"""YES or NO only.

Important? (finances, law, identity, immigration, payments, invoices, reports, work, freelancing)

Subject: {subject}
Body: {body}

Answer:"""

    try:
        logger.debug(f"Sending classification request to LLM for {eml_file.name}")
        # ollama.chat() sends the prompt to the local Ollama server
        # model="llama3.2:3b" specifies which LLM model to use (3 billion parameters)
        response = ollama.chat(
            model="llama3.2:3b", messages=[{"role": "user", "content": prompt}]
        )
        # Check if response contains "YES" (case-insensitive)
        is_important = "YES" in response["message"]["content"].upper()
        classification = "important" if is_important else "unimportant"
        logger.debug(f"Email {eml_file.name} classified as {classification}")
        return classification
    except Exception as e:
        # If LLM fails, default to important to be safe
        logger.warning(
            f"LLM classification failed for {eml_file.name}, defaulting to important: {e}"
        )
        return "important"


def backup_and_classify():
    """
    Main function that orchestrates the complete email backup and classification process.

    This function performs three major steps:
    1. Backs up emails from configured Gmail and/or Outlook accounts
    2. Discovers all .eml files in the backup directories
    3. Classifies and organizes each email into appropriate categories

    The function tracks timing and statistics for all operations and logs comprehensive
    information about what was accomplished.

    Returns:
                    None

    Side effects:
                    - Creates backup directories in EMAIL_OUTPUT_DIR
                    - Creates categorized email directories (important, unimportant, attachments)
                    - Copies email files to categorized directories
                    - Logs detailed timing and statistics
    """
    # time.time() returns current time in seconds since epoch (Unix timestamp)
    # We use this to measure how long each operation takes
    start_time = time.time()
    logger.info("Starting email backup and classification process")

    # Track which directories contain backed up emails
    backup_dirs = []
    # Track statistics for each email account
    backup_stats = {}

    # === GMAIL BACKUP SECTION ===
    if GMAIL_ADDRESS and GMAIL_ADDRESS != "your@gmail.com":
        gmail_start = time.time()
        # Path object represents a file system path (from pathlib)
        # The / operator concatenates paths in a platform-independent way
        gmail_dir = EMAIL_OUTPUT_DIR / "gmail_backup"
        logger.info(f"Starting Gmail backup for {GMAIL_ADDRESS}")
        logger.debug(f"Gmail backup directory: {gmail_dir}")

        # GYB (Got Your Back) is a command-line tool for backing up Gmail
        # --email specifies the Gmail account to backup
        # --local-folder specifies where to store the backup
        success, _ = run_cmd(
            f'gyb --email {GMAIL_ADDRESS} --local-folder "{gmail_dir}"',
            "Gmail backup",
            capture=False,
        )

        gmail_duration = time.time() - gmail_start
        if success:
            # rglob("*.eml") recursively finds all .eml files in directory and subdirectories
            # len(list(...)) counts how many email files were backed up
            gmail_count = (
                len(list(gmail_dir.rglob("*.eml"))) if gmail_dir.exists() else 0
            )
            logger.info(f"Gmail backup completed successfully for {GMAIL_ADDRESS}")
            logger.info(
                f"Gmail backup took {gmail_duration:.1f} seconds, {gmail_count} emails backed up"
            )
            backup_dirs.append(gmail_dir)
            # Store stats in dictionary for later reporting
            backup_stats["gmail"] = {"count": gmail_count, "duration": gmail_duration}
        else:
            logger.warning(
                f"Gmail backup encountered issues for {GMAIL_ADDRESS}, took {gmail_duration:.1f} seconds"
            )
    else:
        logger.debug("Gmail backup skipped: no valid email address configured")

    # === OUTLOOK BACKUP SECTION ===
    if OUTLOOK_ADDRESS and OUTLOOK_ADDRESS != "your@outlook.com":
        outlook_start = time.time()
        outlook_dir = EMAIL_OUTPUT_DIR / "outlook_backup"
        logger.info(f"Starting Outlook backup for {OUTLOOK_ADDRESS}")
        logger.debug(f"Outlook backup directory: {outlook_dir}")

        # Gmvault is a command-line tool for backing up Gmail and IMAP accounts (including Outlook)
        # sync command synchronizes the remote account with local storage
        # -d specifies the local directory for storage
        success, _ = run_cmd(
            f'gmvault sync {OUTLOOK_ADDRESS} -d "{outlook_dir}"',
            "Outlook backup",
            capture=False,
        )

        outlook_duration = time.time() - outlook_start
        if success:
            outlook_count = (
                len(list(outlook_dir.rglob("*.eml"))) if outlook_dir.exists() else 0
            )
            logger.info(f"Outlook backup completed successfully for {OUTLOOK_ADDRESS}")
            logger.info(
                f"Outlook backup took {outlook_duration:.1f} seconds, {outlook_count} emails backed up"
            )
            backup_dirs.append(outlook_dir)
            backup_stats["outlook"] = {
                "count": outlook_count,
                "duration": outlook_duration,
            }
        else:
            logger.warning(
                f"Outlook backup encountered issues for {OUTLOOK_ADDRESS}, took {outlook_duration:.1f} seconds"
            )
    else:
        logger.debug("Outlook backup skipped: no valid email address configured")

    # Exit early if no backups were successful
    if not backup_dirs:
        logger.warning("No emails backed up, check configuration")
        return

    # === EMAIL CLASSIFICATION SECTION ===
    classify_start = time.time()
    logger.info("Starting email classification process")

    # Collect all email files from all backup directories
    all_emails = []
    for backup_dir in backup_dirs:
        if backup_dir.exists():
            # rglob recursively searches all subdirectories for .eml files
            emails_found = list(backup_dir.rglob("*.eml"))
            logger.debug(f"Found {len(emails_found)} emails in {backup_dir}")
            # extend() adds all items from emails_found to all_emails list
            all_emails.extend(emails_found)
        else:
            logger.warning(f"Backup directory does not exist: {backup_dir}")

    if not all_emails:
        logger.warning("No .eml files found in backup directories")
        return

    logger.info(f"Found {len(all_emails)} total emails to classify")

    # Initialize counters for tracking classification results
    stats = {
        "important": 0,
        "unimportant": 0,
        "attachments": 0,
        "skipped": 0,
        "failed": 0,
    }
    total_size = 0  # Track total file size in bytes

    # Process each email file
    # enumerate(list, 1) gives us both the item and a counter starting from 1
    for i, eml_file in enumerate(all_emails, 1):
        # Skip emails that are already classified (already in one of the destination folders)
        # This prevents re-classifying emails if the script is run multiple times
        if any(
            p in eml_file.parents
            for p in [IMPORTANT_EMAILS_DIR, UNIMPORTANT_EMAILS_DIR, EMAIL_ARTIFACTS_DIR]
        ):
            logger.debug(f"Skipping already classified email: {eml_file.name}")
            stats["skipped"] += 1
            continue

        # Classify the email using LLM
        category = classify_email(eml_file)

        # Determine destination directory based on classification
        if category == "attachments":
            dest = EMAIL_ARTIFACTS_DIR / eml_file.name
            stats["attachments"] += 1
        elif category == "important":
            dest = IMPORTANT_EMAILS_DIR / eml_file.name
            stats["important"] += 1
        else:
            dest = UNIMPORTANT_EMAILS_DIR / eml_file.name
            stats["unimportant"] += 1

        # Handle filename conflicts by adding a numeric suffix
        # If email.eml exists, try email_1.eml, email_2.eml, etc.
        counter = 1
        original_dest = dest
        while dest.exists():
            # with_stem() replaces the filename (without extension)
            dest = original_dest.with_stem(f"{original_dest.stem}_{counter}")
            counter += 1

        try:
            # shutil.copy2() copies the file while preserving metadata (timestamps, permissions)
            shutil.copy2(eml_file, dest)
            # Get file size in bytes using .stat().st_size
            file_size = eml_file.stat().st_size
            total_size += file_size
            logger.debug(
                f"Copied {eml_file.name} to {dest.parent.name}/{dest.name} ({file_size} bytes)"
            )
        except Exception as e:
            logger.error(f"Failed to copy {eml_file.name} to {dest}: {e}")
            stats["failed"] += 1
            continue

        # Log progress every 100 emails to show the process is active
        # Modulo operator (%) gives remainder, so i % 100 == 0 every 100 iterations
        if i % 100 == 0:
            logger.info(
                f"Classification progress: {i}/{len(all_emails)} emails processed"
            )

    # === FINAL STATISTICS AND REPORTING ===
    classify_duration = time.time() - classify_start
    total_duration = time.time() - start_time
    # Convert bytes to megabytes for readability (1 MB = 1024 * 1024 bytes)
    total_size_mb = total_size / (1024 * 1024)

    total_classified = stats["important"] + stats["unimportant"] + stats["attachments"]

    logger.info(f"Email classification completed in {classify_duration:.1f} seconds")
    logger.info(
        f"Classification results: {stats['important']} important, {stats['unimportant']} unimportant, {stats['attachments']} with attachments"
    )
    logger.info(
        f"Total emails processed: {total_classified}, skipped: {stats['skipped']}, failed: {stats['failed']}"
    )
    logger.info(f"Total size of classified emails: {total_size_mb:.2f} MB")
    logger.info(
        f"Classification rate: {total_classified/classify_duration:.1f} emails per second"
    )

    # Generate backup summary if we have backup statistics
    if backup_stats:
        # List comprehension to format each account's stats
        # ', '.join() combines the strings with commas
        logger.info(
            f"Backup summary: {', '.join([f'{k}: {v['count']} emails in {v['duration']:.1f}s' for k, v in backup_stats.items()])}"
        )

    logger.info(f"Total archival process completed in {total_duration:.1f} seconds")
    # .absolute() converts relative path to absolute path for clarity
    logger.info(f"Classified emails stored in: {EMAIL_OUTPUT_DIR.absolute()}")
