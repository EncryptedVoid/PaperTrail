import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
import hashlib
import json
import re
from typing import List, Optional
import icalendar
from ...config import EMAILS_FOR_ARCHIVAL

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class EmailAccount:
    name: str
    email: str
    maildir: Path
    oauth_provider: str
    token_file: Path
    oauth_helper: str


@dataclass
class EmailMessage:
    uuid: str
    account: str
    message_id: str
    subject: str
    sender: str
    recipients: List[str]
    date: datetime
    body_text: str
    body_html: Optional[str]
    headers: dict
    maildir_path: Path
    has_attachments: bool
    attachment_count: int
    is_calendar: bool


@dataclass
class Attachment:
    uuid: str
    email_uuid: str
    original_filename: str
    sanitized_filename: str
    content_type: str
    size_bytes: int
    content: bytes
    metadata: dict


@dataclass
class CalendarEvent:
    email_uuid: str
    title: str
    start_time: datetime
    end_time: datetime
    attendees: List[str]
    location: str
    description: str
    meeting_links: List[str]
    ics_content: bytes


@dataclass
class PhishingAssessment:
    is_phishing: bool
    confidence: float
    score: float
    reasons: List[str]


@dataclass
class PipelineStats:
    total_emails: int = 0
    important_emails: int = 0
    archived_emails: int = 0
    phishing_detected: int = 0
    attachments_extracted: int = 0
    calendars_extracted: int = 0
    errors: int = 0
    processing_time: float = 0.0


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def archive_emails(logger: logging.Logger) -> bool:

    start_time = datetime.now()
    stats = PipelineStats()

    # ========================================================================
    # STEP 1: PRE-FLIGHT CHECKS
    # ========================================================================

    print("\n[1/9] Running pre-flight checks...")

    # Check if OAuth tokens exist
    for account_config in EMAILS_FOR_ARCHIVAL:
        token_file = account_config["token_file"]
        if not token_file.exists():
            print(f"❌ ERROR: No OAuth token found for {account_config['email']}")
            print(f"   Expected: {token_file}")
            print(f"\n   Please authenticate first:")
            if account_config["oauth_provider"] == "google":
                print(
                    f"   {account_config['oauth_helper']} --authorize --provider google"
                )
            else:
                print(
                    f"   oauth2ms --client-id YOUR_ID --authorize --email {account_config['email']}"
                )
            return False
        else:
            print(f"✓ Token found for {account_config['email']}")

    # Check if mbsync is configured
    if not CONFIG["mbsync_config"].exists():
        print(f"❌ ERROR: mbsync config not found: {CONFIG['mbsync_config']}")
        print("   Please create ~/.mbsyncrc (see setup guide)")
        return False
    else:
        print(f"✓ mbsync config found")

    # Check if output directories exist, create if needed
    for dir_name, dir_path in CONFIG["output_dirs"].items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory ready: {dir_name}")

    # Check if LLM is running
    if not check_llm_available():
        print("❌ ERROR: LLM not available")
        print(f"   Start Ollama and pull model: ollama pull {CONFIG['llm']['model']}")
        return False
    else:
        print(f"✓ LLM ready: {CONFIG['llm']['model']}")

    print("✓ All pre-flight checks passed\n")

    # ========================================================================
    # STEP 2: SYNC EMAILS FROM SERVERS
    # ========================================================================

    print("[2/9] Syncing emails from servers...")

    try:
        # Run mbsync to download all new emails
        result = subprocess.run(
            ["mbsync", "-a"],  # -a = all accounts in config
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print("✓ Email sync completed")
            # Parse output to show sync stats
            for line in result.stdout.split("\n"):
                if "C:" in line or "N:" in line or "D:" in line:
                    print(f"  {line}")
        else:
            print(f"⚠️  Warning: mbsync had errors:\n{result.stderr}")
            # Continue anyway - partial sync is okay

    except subprocess.TimeoutExpired:
        print("⚠️  Warning: mbsync timed out (server slow?)")
        print("   Continuing with already-synced emails...")

    except Exception as e:
        print(f"❌ ERROR: Failed to sync emails: {e}")
        return False

    # ========================================================================
    # STEP 3: DISCOVER ALL EMAILS IN MAILDIR
    # ========================================================================

    print("\n[3/9] Discovering emails in local Maildir...")

    all_emails = []

    for account_config in CONFIG["accounts"]:
        maildir = account_config["maildir"]
        account_name = account_config["name"]

        print(f"  Scanning {account_name}...")

        # Maildir structure: Maildir/{FOLDER}/cur/ for synced emails
        # We want emails from cur/ (fully synced) and new/ (just arrived)

        folders = []
        if maildir.exists():
            # Find all folders (INBOX, Sent, Archive, etc.)
            for folder_path in maildir.rglob("cur"):
                folders.append(folder_path)
            for folder_path in maildir.rglob("new"):
                folders.append(folder_path)

        email_count = 0
        for folder in folders:
            for email_file in folder.iterdir():
                if email_file.is_file():
                    all_emails.append({"path": email_file, "account": account_name})
                    email_count += 1

        print(f"    Found {email_count} emails in {account_name}")

    stats.total_emails = len(all_emails)
    print(f"✓ Total emails found: {stats.total_emails}\n")

    if stats.total_emails == 0:
        print("⚠️  No emails to process. Exiting.")
        return True

    # ========================================================================
    # STEP 4: PROCESS EACH EMAIL
    # ========================================================================

    print(f"[4/9] Processing {stats.total_emails} emails...")

    for idx, email_info in enumerate(all_emails, 1):
        email_path = email_info["path"]
        account = email_info["account"]

        if idx % 100 == 0:
            print(f"  Progress: {idx}/{stats.total_emails}")

        try:
            # Parse email file
            with open(email_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)

            # Extract basic metadata
            email_obj = EmailMessage(
                uuid=generate_email_uuid(msg, email_path),
                account=account,
                message_id=msg.get("Message-ID", ""),
                subject=msg.get("Subject", "(No Subject)"),
                sender=msg.get("From", ""),
                recipients=msg.get_all("To", []) + msg.get_all("Cc", []),
                date=parse_email_date(msg.get("Date", "")),
                body_text=extract_text_body(msg),
                body_html=extract_html_body(msg),
                headers=dict(msg.items()),
                maildir_path=email_path,
                has_attachments=False,
                attachment_count=0,
                is_calendar=False,
            )

            # ================================================================
            # STEP 4A: EXTRACT ATTACHMENTS
            # ================================================================

            attachments = []
            calendars = []

            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Check if it's an attachment
                if "attachment" in content_disposition or part.get_filename():

                    filename = part.get_filename()
                    if not filename:
                        # Generate filename from content type
                        ext = get_extension_from_mime(content_type)
                        filename = f"attachment{ext}"

                    attachment = Attachment(
                        uuid=generate_attachment_uuid(
                            part.get_payload(decode=True), email_obj.uuid
                        ),
                        email_uuid=email_obj.uuid,
                        original_filename=filename,
                        sanitized_filename=sanitize_filename(filename),
                        content_type=content_type,
                        size_bytes=len(part.get_payload(decode=True)),
                        content=part.get_payload(decode=True),
                        metadata={
                            "email_subject": email_obj.subject,
                            "sender": email_obj.sender,
                            "date": email_obj.date.isoformat(),
                            "account": account,
                        },
                    )

                    attachments.append(attachment)

                # Check if it's a calendar invite
                elif content_type == "text/calendar":
                    calendar = extract_calendar_from_part(part, email_obj.uuid)
                    if calendar:
                        calendars.append(calendar)
                        email_obj.is_calendar = True

            # Also check for .ics attachments
            for att in attachments:
                if att.original_filename.lower().endswith(".ics"):
                    calendar = extract_calendar_from_attachment(att, email_obj.uuid)
                    if calendar:
                        calendars.append(calendar)
                        email_obj.is_calendar = True

            email_obj.has_attachments = len(attachments) > 0
            email_obj.attachment_count = len(attachments)

            # ================================================================
            # STEP 4B: PHISHING DETECTION
            # ================================================================

            phishing_result = detect_phishing(email_obj, attachments)

            if phishing_result.is_phishing:
                # ROUTE: Suspected phishing -> REVIEW folder
                route_phishing_email(email_obj, attachments, phishing_result)
                stats.phishing_detected += 1
                continue  # Skip rest of pipeline for this email

            # ================================================================
            # STEP 4C: IMPORTANCE ASSESSMENT (LLM)
            # ================================================================

            is_important = assess_email_importance(email_obj, attachments)

            if not is_important:
                # ROUTE: Not important -> Archive-only storage
                route_archive_only(email_obj)
                stats.archived_emails += 1
                continue  # Skip conversion/extraction for unimportant emails

            # If we get here, email is important and not phishing
            stats.important_emails += 1

            # ================================================================
            # STEP 4D: SAVE ATTACHMENTS WITH METADATA
            # ================================================================

            for attachment in attachments:
                save_attachment_with_metadata(attachment)
                stats.attachments_extracted += 1

            # ================================================================
            # STEP 4E: SAVE CALENDAR EVENTS
            # ================================================================

            for calendar in calendars:
                save_calendar_event(calendar)
                stats.calendars_extracted += 1

            # ================================================================
            # STEP 4F: CONVERT EMAIL TO PDF
            # ================================================================

            email_pdf_path = convert_email_to_pdf(email_obj)

            # ================================================================
            # STEP 4G: ROUTE TO SANITIZATION
            # ================================================================

            # Move email PDF to sanitization queue
            final_pdf_path = CONFIG["output_dirs"]["sanitization"] / email_pdf_path.name
            email_pdf_path.rename(final_pdf_path)

            # Keep original email in raw archive
            archive_raw_email(email_obj)

        except Exception as e:
            print(f"  ⚠️  Error processing {email_path.name}: {e}")
            stats.errors += 1
            continue

    # ========================================================================
    # STEP 5: FINAL STATISTICS
    # ========================================================================

    stats.processing_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total emails processed:    {stats.total_emails}")
    print(
        f"Important emails:          {stats.important_emails} ({stats.important_emails/stats.total_emails*100:.1f}%)"
    )
    print(
        f"Archived (not important):  {stats.archived_emails} ({stats.archived_emails/stats.total_emails*100:.1f}%)"
    )
    print(
        f"Phishing detected:         {stats.phishing_detected} ({stats.phishing_detected/stats.total_emails*100:.1f}%)"
    )
    print(f"Attachments extracted:     {stats.attachments_extracted}")
    print(f"Calendar events extracted: {stats.calendars_extracted}")
    print(f"Errors:                    {stats.errors}")
    print(f"Processing time:           {stats.processing_time:.1f} seconds")
    print(
        f"Speed:                     {stats.total_emails/stats.processing_time:.1f} emails/sec"
    )
    print()
    print(
        f"Next step: Run Stage 1 (Sanitization) on {CONFIG['output_dirs']['sanitization']}"
    )
    print("=" * 80)

    return True
