"""
Email Archiving Workflow Orchestrator

This module provides the main entry point for the email archiving system.
It coordinates the complete workflow:

1. Dependency Verification - Ensures all required tools are installed
   (GYB for Gmail, Gmvault for Outlook, Ollama for email classification)

2. OAuth Configuration - Sets up authentication for email accounts
   (Gmail and/or Outlook accounts as configured)

3. Email Backup and Classification - Executes the backup and classification process
   (downloads emails and organizes them by importance and content type)

This module serves as the high-level orchestrator, delegating specific tasks
to utility modules while tracking timing and providing comprehensive logging.
"""

import logging
import time

from config import GMAIL_ADDRESS , OUTLOOK_ADDRESS
from utilities.dependancy_ensurance import (
	ensure_gmvault ,
	ensure_gyb ,
	ensure_oauth ,
	ensure_ollama ,
)
from utilities.email_archival import (
	backup_and_classify ,
)

# Create module-level logger using __name__ to get the module's name
# This allows log filtering and organization by module
logger = logging.getLogger(__name__)


def archiving_emails(logger_param: logging.Logger = None):
    """
    Execute the complete email archiving workflow with comprehensive logging.

    This is the main entry point that orchestrates the entire email archiving process.
    It performs the following steps in sequence:

    1. Dependency verification - ensures all required command-line tools are installed:
             - gmvault: for Outlook/IMAP email backup
             - GYB (Got Your Back): for Gmail backup
             - Ollama: for LLM-based email classification

    2. OAuth authentication setup - configures access to email accounts:
             - Gmail: authenticates using GYB's OAuth flow
             - Outlook: authenticates using Gmvault's OAuth flow

    3. Backup and classification - downloads and organizes emails:
             - Downloads all emails from configured accounts
             - Classifies each email using LLM
             - Organizes into folders by importance and content type

    Args:
                    logger_param (logging.Logger, optional): Custom logger instance to use.
                                    If provided, replaces the module-level logger. Useful for integration
                                    with existing logging configurations.

    Returns:
                    None

    Raises:
                    Exception: If any critical step fails (dependency installation, OAuth setup,
                                    or backup process), the exception is logged and re-raised to allow
                                    calling code to handle the failure.

    Side effects:
                    - Installs missing dependencies (gmvault, GYB, Ollama)
                    - Creates OAuth tokens for email accounts
                    - Downloads emails to local storage
                    - Creates and populates categorized email directories
                    - Generates comprehensive logs of all operations
    """
    # Record start time for calculating total workflow duration
    workflow_start = time.time()

    # Allow caller to provide custom logger for integration with existing logging setup
    if logger_param:
        # global keyword allows us to modify the module-level logger variable
        global logger
        logger = logger_param

    logger.info("Starting email archiving workflow")

    # === DEPENDENCY VERIFICATION PHASE ===
    logger.info("Ensuring required dependencies are installed")
    dependency_start = time.time()

    try:
        logger.debug("Checking gmvault installation")
        # ensure_gmvault() checks if gmvault is installed, installs if missing
        # gmvault is used for backing up Outlook and other IMAP email accounts
        ensure_gmvault()
        logger.info("gmvault is available")
    except Exception as e:
        logger.error(f"Failed to ensure gmvault installation: {e}")
        # Re-raise exception to allow calling code to handle the failure
        raise

    try:
        logger.debug("Checking GYB installation")
        # ensure_gyb() checks if GYB (Got Your Back) is installed, installs if missing
        # GYB is a specialized tool for backing up Gmail accounts
        ensure_gyb()
        logger.info("GYB is available")
    except Exception as e:
        logger.error(f"Failed to ensure GYB installation: {e}")
        raise

    try:
        logger.debug("Checking Ollama installation")
        # ensure_ollama() checks if Ollama is installed, installs if missing
        # Ollama runs the local LLM used for classifying emails
        ensure_ollama()
        logger.info("Ollama is available")
    except Exception as e:
        logger.error(f"Failed to ensure Ollama installation: {e}")
        raise

    # Calculate how long dependency verification took
    dependency_duration = time.time() - dependency_start
    logger.info(f"All dependencies verified in {dependency_duration:.1f} seconds")

    # === OAUTH AUTHENTICATION PHASE ===
    oauth_start = time.time()
    accounts_configured = 0  # Track how many accounts were successfully configured

    # Configure Gmail OAuth if a valid Gmail address is provided
    if GMAIL_ADDRESS and GMAIL_ADDRESS != "your@gmail.com":
        logger.info(f"Setting up OAuth for Gmail account: {GMAIL_ADDRESS}")
        try:
            # ensure_oauth() handles the OAuth flow for email account authentication
            # It runs a test command to verify authentication works
            # For Gmail: runs GYB with --action estimate to test OAuth token
            ensure_oauth(
                GMAIL_ADDRESS,
                "GYB (Gmail)",
                f"gyb --email {GMAIL_ADDRESS} --action estimate",
            )
            logger.info(f"Gmail OAuth configured successfully for {GMAIL_ADDRESS}")
            accounts_configured += 1
        except Exception as e:
            logger.error(f"Failed to setup Gmail OAuth for {GMAIL_ADDRESS}: {e}")
            raise
    else:
        # Skip Gmail if no valid address is configured
        # This allows users to backup only Outlook if they prefer
        logger.debug("Gmail OAuth setup skipped: no valid email address configured")

    # Configure Outlook OAuth if a valid Outlook address is provided
    if OUTLOOK_ADDRESS and OUTLOOK_ADDRESS != "your@outlook.com":
        logger.info(f"Setting up OAuth for Outlook account: {OUTLOOK_ADDRESS}")
        try:
            # For Outlook: runs Gmvault with -t quick to test OAuth token
            # -t quick does a quick sync test without downloading all emails
            ensure_oauth(
                OUTLOOK_ADDRESS,
                "Gmvault (Outlook)",
                f"gmvault sync {OUTLOOK_ADDRESS} -t quick",
            )
            logger.info(f"Outlook OAuth configured successfully for {OUTLOOK_ADDRESS}")
            accounts_configured += 1
        except Exception as e:
            logger.error(f"Failed to setup Outlook OAuth for {OUTLOOK_ADDRESS}: {e}")
            raise
    else:
        # Skip Outlook if no valid address is configured
        logger.debug("Outlook OAuth setup skipped: no valid email address configured")

    # Calculate OAuth setup duration
    oauth_duration = time.time() - oauth_start
    logger.info(
        f"OAuth configuration completed in {oauth_duration:.1f} seconds for {accounts_configured} account(s)"
    )

    # === BACKUP AND CLASSIFICATION PHASE ===
    logger.info("Beginning email backup and classification")

    try:
        # backup_and_classify() is the main worker function that:
        # 1. Downloads emails from all configured accounts
        # 2. Classifies each email using the LLM
        # 3. Organizes emails into categorized directories
        backup_and_classify()

        # Calculate total workflow duration from start to finish
        workflow_duration = time.time() - workflow_start
        logger.info(
            f"Email archiving workflow completed successfully in {workflow_duration:.1f} seconds total"
        )
    except Exception as e:
        # Log failure with timing information even if the process fails
        workflow_duration = time.time() - workflow_start
        logger.error(
            f"Email backup and classification failed after {workflow_duration:.1f} seconds: {e}"
        )
        raise

    # Return None explicitly to indicate this function is called for side effects
    # (logging, file operations) rather than returning a value
    return None
