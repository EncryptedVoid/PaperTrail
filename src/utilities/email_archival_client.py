"""
Email Archival Module

Functions to archive flagged/starred emails from Gmail and Outlook
with full thread context, attachments, and metadata preservation.
"""

import base64
import email
import logging
import os
from pathlib import Path
from typing import Dict, Set

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from msal import PublicClientApplication


def archive_gmail_starred(target_dir: Path, logger: logging.Logger) -> None:
    """
    Archive all starred emails from Gmail with full threads and attachments.

    Handles OAuth2 authentication, downloads complete conversation threads,
    preserves all metadata, extracts attachments, and attempts to decrypt
    encrypted emails (S/MIME).

    Required .env variables:
        GMAIL_CLIENT_ID: Google OAuth2 client ID
        GMAIL_CLIENT_SECRET: Google OAuth2 client secret
        GMAIL_REFRESH_TOKEN: (optional) Stored refresh token for re-auth
        GMAIL_SMIME_CERT_PATH: (optional) Path to S/MIME certificate for decryption
        GMAIL_SMIME_KEY_PATH: (optional) Path to S/MIME private key for decryption

    Args:
        target_dir: Directory where emails and attachments will be saved
        logger: Logger instance for logging operations and errors

    Returns:
        None

    Raises:
        ValueError: If required environment variables are missing
        Exception: If authentication or API calls fail

    Directory structure created:
        target_dir/
        ├── threads/
        │   ├── thread_<id>_msg_<id>.eml
        │   └── thread_<id>_msg_<id>.eml
        └── attachments/
            └── thread_<id>_<filename>

    Note:
        - First run opens browser for OAuth consent
        - Subsequent runs use stored refresh token
        - Encrypted emails are decrypted if certificates are provided
        - Thread messages are saved as individual EML files with thread ID prefix
    """
    logger.info("Starting Gmail starred email archival process")

    try:
        # Load environment variables
        load_dotenv()
        logger.debug("Environment variables loaded")

        # Validate required environment variables
        required_vars = ["GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Setup directory structure
        logger.info(f"Setting up archive directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        threads_dir = target_dir / "threads"
        attachments_dir = target_dir / "attachments"
        threads_dir.mkdir(exist_ok=True)
        attachments_dir.mkdir(exist_ok=True)
        logger.debug("Archive directories created successfully")

        # Authenticate with Gmail API
        logger.info("Authenticating with Gmail API")
        service = _authenticate_gmail(target_dir, logger)
        logger.info("Gmail authentication successful")

        # Fetch starred messages
        logger.info("Fetching starred emails from Gmail")
        messages = _fetch_starred_messages(service, logger)
        logger.info(f"Found {len(messages)} starred emails")

        if not messages:
            logger.warning("No starred emails found")
            return

        # Track processed threads to avoid duplicates
        processed_threads: Set[str] = set()
        successful_threads = 0
        failed_threads = 0

        # Process each message
        for idx, msg_ref in enumerate(messages, 1):
            try:
                msg_id = msg_ref["id"]
                logger.debug(f"Processing message {idx}/{len(messages)}: {msg_id}")

                # Get thread ID for this message
                message = (
                    service.users()
                    .messages()
                    .get(userId="me", id=msg_id, format="minimal")
                    .execute()
                )

                thread_id = message["threadId"]

                # Skip if we already processed this thread
                if thread_id in processed_threads:
                    logger.debug(f"Thread {thread_id} already processed, skipping")
                    continue

                processed_threads.add(thread_id)
                logger.info(
                    f"Processing thread {thread_id} ({len(processed_threads)}/{len(messages)})"
                )

                # Process the entire thread
                success = _process_gmail_thread(
                    service, thread_id, threads_dir, attachments_dir, logger
                )

                if success:
                    successful_threads += 1
                else:
                    failed_threads += 1

            except Exception as e:
                logger.error(
                    f"Error processing message {msg_ref.get('id', 'unknown')}: {str(e)}",
                    exc_info=True,
                )
                failed_threads += 1
                continue

        # Log summary
        logger.info(f"Archive complete. Processed {len(processed_threads)} threads")
        logger.info(f"Successful: {successful_threads}, Failed: {failed_threads}")
        logger.info(f"Archive saved to: {target_dir}")

    except Exception as e:
        logger.error(f"Fatal error in Gmail archival process: {str(e)}", exc_info=True)
        raise


def archive_outlook_flagged(target_dir: Path, logger: logging.Logger) -> None:
    """
    Archive all flagged emails from Outlook/Microsoft 365 with full threads.

    Handles OAuth2 authentication, downloads complete conversation threads,
    preserves all metadata, extracts attachments, and attempts to decrypt
    encrypted emails (S/MIME).

    Required .env variables:
        OUTLOOK_CLIENT_ID: Azure App Registration client ID
        OUTLOOK_TENANT_ID: Azure tenant ID (or 'common' for personal accounts)
        OUTLOOK_SMIME_CERT_PATH: (optional) Path to S/MIME certificate for decryption
        OUTLOOK_SMIME_KEY_PATH: (optional) Path to S/MIME private key for decryption

    Args:
        target_dir: Directory where emails and attachments will be saved
        logger: Logger instance for logging operations and errors

    Returns:
        None

    Raises:
        ValueError: If required environment variables are missing
        Exception: If authentication or API calls fail

    Directory structure created:
        target_dir/
        ├── conversations/
        │   ├── conv_<id>_msg_<id>.eml
        │   └── conv_<id>_msg_<id>.eml
        └── attachments/
            └── conv_<id>_<filename>

    Note:
        - First run opens browser for OAuth consent
        - Uses Microsoft Graph API for message access
        - Encrypted emails are decrypted if certificates are provided
        - Conversation messages are saved as individual EML files
    """
    logger.info("Starting Outlook flagged email archival process")

    try:
        # Load environment variables
        load_dotenv()
        logger.debug("Environment variables loaded")

        # Validate required environment variables
        required_vars = ["OUTLOOK_CLIENT_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Setup directory structure
        logger.info(f"Setting up archive directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        conversations_dir = target_dir / "conversations"
        attachments_dir = target_dir / "attachments"
        conversations_dir.mkdir(exist_ok=True)
        attachments_dir.mkdir(exist_ok=True)
        logger.debug("Archive directories created successfully")

        # Authenticate with Microsoft Graph API
        logger.info("Authenticating with Microsoft Graph API")
        access_token = _authenticate_outlook(logger)
        logger.info("Outlook authentication successful")

        # Setup API headers
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Fetch flagged messages
        logger.info("Fetching flagged emails from Outlook")
        all_messages = _fetch_flagged_messages(headers, logger)
        logger.info(f"Found {len(all_messages)} flagged emails")

        if not all_messages:
            logger.warning("No flagged emails found")
            return

        # Track processed conversations to avoid duplicates
        processed_conversations: Set[str] = set()
        successful_conversations = 0
        failed_conversations = 0

        # Process each message
        for idx, msg_ref in enumerate(all_messages, 1):
            try:
                conversation_id = msg_ref["conversationId"]
                logger.debug(
                    f"Processing message {idx}/{len(all_messages)}: {msg_ref['id']}"
                )

                # Skip if already processed this conversation
                if conversation_id in processed_conversations:
                    logger.debug(
                        f"Conversation {conversation_id} already processed, skipping"
                    )
                    continue

                processed_conversations.add(conversation_id)
                logger.info(
                    f"Processing conversation {conversation_id} ({len(processed_conversations)}/{len(all_messages)})"
                )

                # Process the entire conversation
                success = _process_outlook_conversation(
                    headers, conversation_id, conversations_dir, attachments_dir, logger
                )

                if success:
                    successful_conversations += 1
                else:
                    failed_conversations += 1

            except Exception as e:
                logger.error(
                    f"Error processing message {msg_ref.get('id', 'unknown')}: {str(e)}",
                    exc_info=True,
                )
                failed_conversations += 1
                continue

        # Log summary
        logger.info(
            f"Archive complete. Processed {len(processed_conversations)} conversations"
        )
        logger.info(
            f"Successful: {successful_conversations}, Failed: {failed_conversations}"
        )
        logger.info(f"Archive saved to: {target_dir}")

    except Exception as e:
        logger.error(
            f"Fatal error in Outlook archival process: {str(e)}", exc_info=True
        )
        raise


# ==============================================================================
# GMAIL HELPER FUNCTIONS
# ==============================================================================


def _authenticate_gmail(target_dir: Path, logger: logging.Logger):
    """
    Authenticate with Gmail API using OAuth2.

    Args:
        target_dir: Directory to store token file
        logger: Logger instance

    Returns:
        Authenticated Gmail service object

    Raises:
        Exception: If authentication fails
    """
    try:
        # Define Gmail API scopes
        SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

        creds = None
        token_path = target_dir / "gmail_token.json"

        # Check for existing token
        if token_path.exists():
            logger.debug(f"Found existing token at {token_path}")
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
                logger.debug("Loaded credentials from token file")
            except Exception as e:
                logger.warning(f"Failed to load existing token: {e}")
                creds = None

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                try:
                    creds.refresh(Request())
                    logger.debug("Credentials refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    creds = None

            if not creds:
                # Create credentials from environment variables
                logger.info("Initiating new OAuth2 flow")
                credentials_config = {
                    "installed": {
                        "client_id": os.getenv("GMAIL_CLIENT_ID"),
                        "client_secret": os.getenv("GMAIL_CLIENT_SECRET"),
                        "redirect_uris": ["http://localhost"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }

                flow = InstalledAppFlow.from_client_config(credentials_config, SCOPES)
                creds = flow.run_local_server(port=0)
                logger.info("OAuth2 flow completed successfully")

            # Save credentials for next run
            try:
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
                logger.debug(f"Credentials saved to {token_path}")
            except Exception as e:
                logger.warning(f"Failed to save credentials: {e}")

        # Build and return Gmail service
        service = build("gmail", "v1", credentials=creds)
        logger.debug("Gmail service object created")
        return service

    except Exception as e:
        logger.error(f"Gmail authentication failed: {str(e)}", exc_info=True)
        raise


def _fetch_starred_messages(service, logger: logging.Logger) -> list:
    """
    Fetch all starred messages from Gmail with pagination support.

    Args:
        service: Authenticated Gmail service object
        logger: Logger instance

    Returns:
        List of message references

    Raises:
        HttpError: If Gmail API call fails
    """
    try:
        messages = []
        page_token = None
        page_count = 0

        # Fetch messages with pagination
        while True:
            page_count += 1
            logger.debug(f"Fetching page {page_count} of starred messages")

            try:
                # Build API request
                request_params = {"userId": "me", "q": "is:starred", "maxResults": 500}

                if page_token:
                    request_params["pageToken"] = page_token

                # Execute request
                results = service.users().messages().list(**request_params).execute()

                # Add messages from this page
                page_messages = results.get("messages", [])
                messages.extend(page_messages)
                logger.debug(f"Page {page_count}: Found {len(page_messages)} messages")

                # Check for next page
                page_token = results.get("nextPageToken")
                if not page_token:
                    logger.debug("No more pages to fetch")
                    break

            except HttpError as e:
                logger.error(
                    f"HTTP error fetching messages (page {page_count}): {e}",
                    exc_info=True,
                )
                raise

        logger.info(
            f"Fetched total of {len(messages)} starred messages across {page_count} pages"
        )
        return messages

    except Exception as e:
        logger.error(f"Error fetching starred messages: {str(e)}", exc_info=True)
        raise


def _process_gmail_thread(
    service,
    thread_id: str,
    threads_dir: Path,
    attachments_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Process a complete Gmail thread including all messages and attachments.

    Args:
        service: Authenticated Gmail service object
        thread_id: ID of the thread to process
        threads_dir: Directory to save thread messages
        attachments_dir: Directory to save attachments
        logger: Logger instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.debug(f"Fetching thread {thread_id}")

        # Get entire thread with full message data
        thread = (
            service.users()
            .threads()
            .get(userId="me", id=thread_id, format="full")
            .execute()
        )

        thread_messages = thread.get("messages", [])
        logger.debug(f"Thread {thread_id} contains {len(thread_messages)} messages")

        # Process each message in the thread
        for msg_idx, thread_msg in enumerate(thread_messages, 1):
            try:
                msg_id = thread_msg["id"]
                logger.debug(
                    f"Processing message {msg_idx}/{len(thread_messages)} in thread {thread_id}: {msg_id}"
                )

                # Download raw message in RFC822 format
                try:
                    raw_msg = (
                        service.users()
                        .messages()
                        .get(userId="me", id=msg_id, format="raw")
                        .execute()
                    )
                except HttpError as e:
                    logger.error(f"Failed to fetch raw message {msg_id}: {e}")
                    continue

                # Decode base64-encoded message
                try:
                    msg_bytes = base64.urlsafe_b64decode(raw_msg["raw"])
                    logger.debug(f"Decoded message {msg_id} ({len(msg_bytes)} bytes)")
                except Exception as e:
                    logger.error(f"Failed to decode message {msg_id}: {e}")
                    continue

                # Parse email structure
                try:
                    parsed_msg = email.message_from_bytes(msg_bytes)
                except Exception as e:
                    logger.error(f"Failed to parse message {msg_id}: {e}")
                    continue

                # Attempt decryption if message is encrypted
                parsed_msg = _decrypt_email_if_needed(parsed_msg, logger)

                # Save message as EML file
                eml_filename = f"thread_{thread_id}_msg_{msg_id}.eml"
                eml_path = threads_dir / eml_filename

                try:
                    with open(eml_path, "wb") as f:
                        f.write(parsed_msg.as_bytes())
                    logger.debug(f"Saved EML file: {eml_filename}")
                except Exception as e:
                    logger.error(f"Failed to save EML file {eml_filename}: {e}")
                    continue

                # Extract and save attachments
                payload = thread_msg.get("payload", {})
                _extract_gmail_attachments(
                    service, msg_id, payload, attachments_dir, thread_id, logger
                )

            except Exception as e:
                logger.error(
                    f"Error processing message {msg_idx} in thread {thread_id}: {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Successfully processed thread {thread_id}")
        return True

    except Exception as e:
        logger.error(f"Error processing thread {thread_id}: {str(e)}", exc_info=True)
        return False


def _extract_gmail_attachments(
    service,
    msg_id: str,
    payload: Dict,
    attachments_dir: Path,
    thread_id: str,
    logger: logging.Logger,
) -> None:
    """
    Extract and save attachments from Gmail message payload.

    Recursively processes message parts to find all attachments.

    Args:
        service: Authenticated Gmail service object
        msg_id: Message ID containing the attachments
        payload: Message payload dictionary
        attachments_dir: Directory to save attachments
        thread_id: Thread ID for filename prefix
        logger: Logger instance

    Returns:
        None
    """
    try:
        # Check if payload has nested parts
        if "parts" in payload:
            for part_idx, part in enumerate(payload["parts"]):
                # Recursively handle nested parts (multipart messages)
                if "parts" in part:
                    logger.debug(
                        f"Processing nested part {part_idx} in message {msg_id}"
                    )
                    _extract_gmail_attachments(
                        service, msg_id, part, attachments_dir, thread_id, logger
                    )

                # Check if this part is an attachment
                filename = part.get("filename")
                if filename:
                    logger.debug(f"Found attachment: {filename}")

                    attachment_id = part.get("body", {}).get("attachmentId")

                    if attachment_id:
                        try:
                            # Download attachment from Gmail API
                            logger.debug(f"Downloading attachment {attachment_id}")
                            attachment = (
                                service.users()
                                .messages()
                                .attachments()
                                .get(userId="me", messageId=msg_id, id=attachment_id)
                                .execute()
                            )

                            # Decode attachment data
                            file_data = base64.urlsafe_b64decode(attachment["data"])
                            logger.debug(
                                f"Decoded attachment {filename} ({len(file_data)} bytes)"
                            )

                            # Save attachment to disk
                            att_path = (
                                attachments_dir / f"thread_{thread_id}_{filename}"
                            )
                            with open(att_path, "wb") as f:
                                f.write(file_data)

                            logger.info(f"Saved attachment: {filename}")

                        except HttpError as e:
                            logger.error(
                                f"HTTP error downloading attachment {filename}: {e}"
                            )
                        except Exception as e:
                            logger.error(f"Error saving attachment {filename}: {e}")
                    else:
                        logger.warning(f"Attachment {filename} has no attachment ID")

    except Exception as e:
        logger.error(
            f"Error extracting attachments from message {msg_id}: {str(e)}",
            exc_info=True,
        )


# ==============================================================================
# OUTLOOK HELPER FUNCTIONS
# ==============================================================================


def _authenticate_outlook(logger: logging.Logger) -> str:
    """
    Authenticate with Microsoft Graph API using OAuth2.

    Args:
        logger: Logger instance

    Returns:
        str: Access token for Microsoft Graph API

    Raises:
        Exception: If authentication fails
    """
    try:
        # Get credentials from environment
        client_id = os.getenv("OUTLOOK_CLIENT_ID")
        tenant_id = os.getenv("OUTLOOK_TENANT_ID", "common")

        logger.debug(f"Authenticating with tenant: {tenant_id}")

        # Create MSAL public client application
        app = PublicClientApplication(
            client_id, authority=f"https://login.microsoftonline.com/{tenant_id}"
        )

        # Define required scopes
        scopes = ["https://graph.microsoft.com/Mail.Read"]

        # Check for cached token
        accounts = app.get_accounts()
        result = None

        if accounts:
            logger.debug(f"Found {len(accounts)} cached accounts")
            # Try silent authentication first
            try:
                result = app.acquire_token_silent(scopes, account=accounts[0])
                if result:
                    logger.info("Used cached authentication token")
            except Exception as e:
                logger.debug(f"Silent authentication failed: {e}")

        # Fall back to interactive authentication if needed
        if not result:
            logger.info("Initiating interactive authentication")
            try:
                result = app.acquire_token_interactive(scopes=scopes)
            except Exception as e:
                logger.error(f"Interactive authentication failed: {e}")
                raise

        # Validate authentication result
        if "access_token" not in result:
            error_msg = result.get("error_description", "Unknown authentication error")
            logger.error(f"Authentication failed: {error_msg}")
            raise Exception(f"Authentication failed: {error_msg}")

        logger.info("Successfully obtained access token")
        return result["access_token"]

    except Exception as e:
        logger.error(f"Outlook authentication failed: {str(e)}", exc_info=True)
        raise


def _fetch_flagged_messages(headers: Dict, logger: logging.Logger) -> list:
    """
    Fetch all flagged messages from Outlook with pagination support.

    Args:
        headers: HTTP headers with authorization token
        logger: Logger instance

    Returns:
        List of message references

    Raises:
        requests.HTTPError: If API call fails
    """
    try:
        all_messages = []
        messages_url = "https://graph.microsoft.com/v1.0/me/messages"

        # Define query parameters
        params = {
            "$filter": "flag/flagStatus eq 'flagged'",
            "$top": 999,  # Maximum messages per page
            "$select": "id,conversationId,internetMessageId",
        }

        page_count = 0

        # Fetch messages with pagination
        while messages_url:
            page_count += 1
            logger.debug(f"Fetching page {page_count} of flagged messages")

            try:
                response = requests.get(
                    messages_url, headers=headers, params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Add messages from this page
                page_messages = data.get("value", [])
                all_messages.extend(page_messages)
                logger.debug(f"Page {page_count}: Found {len(page_messages)} messages")

                # Get next page URL (contains full URL with parameters)
                messages_url = data.get("@odata.nextLink")
                params = None  # NextLink already contains query parameters

            except requests.HTTPError as e:
                logger.error(
                    f"HTTP error fetching messages (page {page_count}): {e}",
                    exc_info=True,
                )
                logger.error(
                    f"Response content: {e.response.text if hasattr(e, 'response') else 'N/A'}"
                )
                raise
            except requests.Timeout:
                logger.error(f"Timeout fetching messages (page {page_count})")
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching messages (page {page_count}): {e}",
                    exc_info=True,
                )
                raise

        logger.info(
            f"Fetched total of {len(all_messages)} flagged messages across {page_count} pages"
        )
        return all_messages

    except Exception as e:
        logger.error(f"Error fetching flagged messages: {str(e)}", exc_info=True)
        raise


def _process_outlook_conversation(
    headers: Dict,
    conversation_id: str,
    conversations_dir: Path,
    attachments_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Process a complete Outlook conversation including all messages and attachments.

    Args:
        headers: HTTP headers with authorization token
        conversation_id: ID of the conversation to process
        conversations_dir: Directory to save conversation messages
        attachments_dir: Directory to save attachments
        logger: Logger instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.debug(f"Fetching conversation {conversation_id}")

        # Get all messages in the conversation
        conv_url = "https://graph.microsoft.com/v1.0/me/messages"
        conv_params = {"$filter": f"conversationId eq '{conversation_id}'", "$top": 999}

        try:
            conv_response = requests.get(
                conv_url, headers=headers, params=conv_params, timeout=30
            )
            conv_response.raise_for_status()
            conv_messages = conv_response.json().get("value", [])
            logger.debug(
                f"Conversation {conversation_id} contains {len(conv_messages)} messages"
            )
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching conversation {conversation_id}: {e}")
            logger.error(
                f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}"
            )
            return False
        except Exception as e:
            logger.error(f"Error fetching conversation {conversation_id}: {e}")
            return False

        # Process each message in the conversation
        for msg_idx, conv_msg in enumerate(conv_messages, 1):
            try:
                msg_id = conv_msg["id"]
                logger.debug(
                    f"Processing message {msg_idx}/{len(conv_messages)} in conversation {conversation_id}: {msg_id}"
                )

                # Download message as EML (MIME format)
                eml_url = (
                    f"https://graph.microsoft.com/v1.0/me/messages/{msg_id}/$value"
                )

                try:
                    eml_response = requests.get(eml_url, headers=headers, timeout=30)
                    eml_response.raise_for_status()
                    logger.debug(
                        f"Downloaded message {msg_id} ({len(eml_response.content)} bytes)"
                    )
                except requests.HTTPError as e:
                    logger.error(f"HTTP error downloading message {msg_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error downloading message {msg_id}: {e}")
                    continue

                # Parse email structure
                try:
                    parsed_msg = email.message_from_bytes(eml_response.content)
                except Exception as e:
                    logger.error(f"Failed to parse message {msg_id}: {e}")
                    continue

                # Attempt decryption if message is encrypted
                parsed_msg = _decrypt_email_if_needed(parsed_msg, logger)

                # Save message as EML file
                eml_filename = f"conv_{conversation_id}_msg_{msg_id}.eml"
                eml_path = conversations_dir / eml_filename

                try:
                    with open(eml_path, "wb") as f:
                        f.write(parsed_msg.as_bytes())
                    logger.debug(f"Saved EML file: {eml_filename}")
                except Exception as e:
                    logger.error(f"Failed to save EML file {eml_filename}: {e}")
                    continue

                # Download attachments via Graph API
                _extract_outlook_attachments(
                    headers, msg_id, attachments_dir, conversation_id, logger
                )

            except Exception as e:
                logger.error(
                    f"Error processing message {msg_idx} in conversation {conversation_id}: {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Successfully processed conversation {conversation_id}")
        return True

    except Exception as e:
        logger.error(
            f"Error processing conversation {conversation_id}: {str(e)}", exc_info=True
        )
        return False


def _extract_outlook_attachments(
    headers: Dict,
    msg_id: str,
    attachments_dir: Path,
    conversation_id: str,
    logger: logging.Logger,
) -> None:
    """
    Extract and save attachments from Outlook message.

    Args:
        headers: HTTP headers with authorization token
        msg_id: Message ID containing the attachments
        attachments_dir: Directory to save attachments
        conversation_id: Conversation ID for filename prefix
        logger: Logger instance

    Returns:
        None
    """
    try:
        # Fetch attachments list from Graph API
        att_url = f"https://graph.microsoft.com/v1.0/me/messages/{msg_id}/attachments"

        try:
            att_response = requests.get(att_url, headers=headers, timeout=30)
            att_response.raise_for_status()
            attachments = att_response.json().get("value", [])

            if attachments:
                logger.debug(
                    f"Found {len(attachments)} attachments in message {msg_id}"
                )
        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching attachments for message {msg_id}: {e}")
            return
        except Exception as e:
            logger.error(f"Error fetching attachments for message {msg_id}: {e}")
            return

        # Process each attachment
        for att_idx, attachment in enumerate(attachments, 1):
            try:
                # Only process file attachments (not item attachments)
                if attachment.get("@odata.type") == "#microsoft.graph.fileAttachment":
                    filename = attachment.get("name", f"attachment_{att_idx}")
                    logger.debug(
                        f"Processing attachment {att_idx}/{len(attachments)}: {filename}"
                    )

                    # Decode base64-encoded attachment content
                    try:
                        content_bytes = base64.b64decode(attachment["contentBytes"])
                        logger.debug(
                            f"Decoded attachment {filename} ({len(content_bytes)} bytes)"
                        )
                    except Exception as e:
                        logger.error(f"Failed to decode attachment {filename}: {e}")
                        continue

                    # Save attachment to disk
                    try:
                        att_path = (
                            attachments_dir / f"conv_{conversation_id}_{filename}"
                        )
                        with open(att_path, "wb") as f:
                            f.write(content_bytes)
                        logger.info(f"Saved attachment: {filename}")
                    except Exception as e:
                        logger.error(f"Failed to save attachment {filename}: {e}")
                        continue
                else:
                    logger.debug(
                        f"Skipping non-file attachment type: {attachment.get('@odata.type')}"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing attachment {att_idx} in message {msg_id}: {e}",
                    exc_info=True,
                )
                continue

    except Exception as e:
        logger.error(
            f"Error extracting attachments from message {msg_id}: {str(e)}",
            exc_info=True,
        )


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================


def _decrypt_email_if_needed(
    parsed_msg: email.message.Message, logger: logging.Logger
) -> email.message.Message:
    """
    Attempt to decrypt S/MIME or PGP encrypted emails.

    Args:
        parsed_msg: Parsed email message object
        logger: Logger instance

    Returns:
        email.message.Message: Decrypted message if successful, otherwise original message

    Note:
        - Requires M2Crypto for S/MIME decryption
        - Requires python-gnupg for PGP decryption
        - Certificate/key paths must be set in environment variables
    """
    try:
        content_type = parsed_msg.get_content_type()
        logger.debug(f"Message content type: {content_type}")

        # Check for S/MIME encryption
        if content_type in ["application/pkcs7-mime", "application/x-pkcs7-mime"]:
            logger.info("Detected S/MIME encrypted message")

            try:
                from M2Crypto import SMIME, BIO

                # Load certificates from environment
                cert_path = os.getenv("GMAIL_SMIME_CERT_PATH") or os.getenv(
                    "OUTLOOK_SMIME_CERT_PATH"
                )
                key_path = os.getenv("GMAIL_SMIME_KEY_PATH") or os.getenv(
                    "OUTLOOK_SMIME_KEY_PATH"
                )

                if not cert_path or not key_path:
                    logger.warning("S/MIME certificates not configured, cannot decrypt")
                    return parsed_msg

                logger.debug(f"Loading S/MIME cert: {cert_path}, key: {key_path}")

                # Initialize S/MIME handler
                s = SMIME.SMIME()
                s.load_key(key_path, cert_path)

                # Get encrypted payload
                encrypted_data = parsed_msg.get_payload(decode=True)
                logger.debug(f"Encrypted data size: {len(encrypted_data)} bytes")

                # Decrypt the message
                p7 = BIO.MemoryBuffer(encrypted_data)
                decrypted = s.decrypt(SMIME.PKCS7(p7))

                # Parse decrypted content
                decrypted_msg = email.message_from_bytes(decrypted)
                logger.info("Successfully decrypted S/MIME message")
                return decrypted_msg

            except ImportError:
                logger.warning("M2Crypto not installed, cannot decrypt S/MIME messages")
                return parsed_msg
            except Exception as e:
                logger.warning(f"Failed to decrypt S/MIME message: {e}")
                return parsed_msg

        # Check for PGP encryption
        elif "BEGIN PGP MESSAGE" in str(parsed_msg):
            logger.info("Detected PGP encrypted message")

            try:
                import gnupg

                # Initialize GPG handler
                gpg = gnupg.GPG()
                encrypted_text = parsed_msg.get_payload()

                # Decrypt the message
                decrypted = gpg.decrypt(encrypted_text)

                if decrypted.ok:
                    logger.info("Successfully decrypted PGP message")
                    return email.message_from_string(str(decrypted))
                else:
                    logger.warning(f"PGP decryption failed: {decrypted.status}")
                    return parsed_msg

            except ImportError:
                logger.warning(
                    "python-gnupg not installed, cannot decrypt PGP messages"
                )
                return parsed_msg
            except Exception as e:
                logger.warning(f"Failed to decrypt PGP message: {e}")
                return parsed_msg

        # No encryption detected
        return parsed_msg

    except Exception as e:
        logger.error(f"Error in decryption check: {str(e)}", exc_info=True)
        return parsed_msg
