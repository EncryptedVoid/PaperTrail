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


def check_llm_available() -> bool:
    """
    Check if LLM is running and accessible
    """
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            return CONFIG["llm"]["model"] in model_names
        return False
    except:
        return False


def generate_email_uuid(msg, email_path: Path) -> str:
    """
    Generate stable UUID from email Message-ID + timestamp
    """
    message_id = msg.get("Message-ID", "")
    date = msg.get("Date", "")
    subject = msg.get("Subject", "")

    # Create hash from identifiable fields
    hash_input = f"{message_id}|{date}|{subject}|{email_path.stat().st_mtime}"
    return "email-" + hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def generate_attachment_uuid(content: bytes, email_uuid: str) -> str:
    """
    Generate UUID from attachment content + email UUID
    """
    hash_input = content + email_uuid.encode()
    return "att-" + hashlib.sha256(hash_input).hexdigest()[:16]


def parse_email_date(date_str: str) -> datetime:
    """
    Parse email date header (handles various formats)
    """
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(date_str)
    except:
        return datetime.now()


def extract_text_body(msg) -> str:
    """
    Extract plain text body from email
    """
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors="ignore")
    else:
        if msg.get_content_type() == "text/plain":
            body = msg.get_payload(decode=True).decode(errors="ignore")
    return body


def extract_html_body(msg) -> Optional[str]:
    """
    Extract HTML body from email
    """
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        if msg.get_content_type() == "text/html":
            return msg.get_payload(decode=True).decode(errors="ignore")
    return None


def sanitize_filename(filename: str) -> str:
    """
    Remove dangerous characters from filename
    """
    # Remove path traversal attempts
    filename = filename.replace("../", "").replace("..\\", "")
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*]', "_", filename)
    # Limit length
    name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
    if len(name) > 200:
        name = name[:200]
    return f"{name}.{ext}" if ext else name


def get_extension_from_mime(mime_type: str) -> str:
    """
    Get file extension from MIME type
    """
    mime_map = {
        "application/pdf": ".pdf",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "application/zip": ".zip",
        "text/plain": ".txt",
        # ... add more as needed
    }
    return mime_map.get(mime_type, ".bin")


# ============================================================================
# CALENDAR EXTRACTION
# ============================================================================


def extract_calendar_from_part(part, email_uuid: str) -> Optional[CalendarEvent]:
    """
    Extract calendar event from email part
    """
    try:
        ics_content = part.get_payload(decode=True)
        cal = icalendar.Calendar.from_ical(ics_content)

        for component in cal.walk():
            if component.name == "VEVENT":
                summary = str(component.get("summary", ""))
                dtstart = component.get("dtstart").dt
                dtend = component.get("dtend").dt if component.get("dtend") else dtstart
                location = str(component.get("location", ""))
                description = str(component.get("description", ""))

                # Extract attendees
                attendees = []
                for attendee in component.get("attendee", []):
                    if isinstance(attendee, list):
                        attendees.extend([str(a) for a in attendee])
                    else:
                        attendees.append(str(attendee))

                # Extract meeting links (Zoom, Teams, etc.)
                meeting_links = extract_meeting_links(description)

                return CalendarEvent(
                    email_uuid=email_uuid,
                    title=summary,
                    start_time=dtstart,
                    end_time=dtend,
                    attendees=attendees,
                    location=location,
                    description=description,
                    meeting_links=meeting_links,
                    ics_content=ics_content,
                )
    except:
        return None


def extract_calendar_from_attachment(
    attachment: Attachment, email_uuid: str
) -> Optional[CalendarEvent]:
    """
    Extract calendar from .ics attachment
    """
    try:
        cal = icalendar.Calendar.from_ical(attachment.content)
        # Same logic as extract_calendar_from_part
        # ... (omitted for brevity)
        return None  # Placeholder
    except:
        return None


def extract_meeting_links(text: str) -> List[str]:
    """
    Extract Zoom/Teams/Meet links from text
    """
    patterns = [
        r"https://[a-z0-9\-]+\.zoom\.us/j/[0-9]+",
        r"https://teams\.microsoft\.com/l/meetup-join/[^\s]+",
        r"https://meet\.google\.com/[a-z\-]+",
    ]

    links = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        links.extend(matches)

    return links


# ============================================================================
# PHISHING DETECTION
# ============================================================================


def detect_phishing(
    email: EmailMessage, attachments: List[Attachment]
) -> PhishingAssessment:
    """
    Two-stage phishing detection:
    1. Fast rspamd scan
    2. LLM confirmation if score > threshold
    """

    # Stage 1: rspamd quick scan
    rspamd_score = run_rspamd_scan(email)

    if rspamd_score < CONFIG["phishing"]["threshold"]:
        # Low score = likely safe
        return PhishingAssessment(
            is_phishing=False, confidence=0.9, score=rspamd_score, reasons=[]
        )

    # Stage 2: LLM confirmation for suspicious emails
    if CONFIG["phishing"]["use_llm_confirmation"]:
        llm_assessment = llm_confirm_phishing(email, attachments, rspamd_score)
        return llm_assessment
    else:
        # Just use rspamd score
        return PhishingAssessment(
            is_phishing=True,
            confidence=0.7,
            score=rspamd_score,
            reasons=["High spam score"],
        )


def run_rspamd_scan(email: EmailMessage) -> float:
    """
    Run rspamd on email, return spam score
    """
    try:
        # Write email to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".eml") as f:
            # Reconstruct email from parts
            temp_path = f.name
            # ... write email content ...

        # Run rspamd
        result = subprocess.run(
            ["rspamc", temp_path], capture_output=True, text=True, timeout=5
        )

        # Parse score from output
        # Format: "Action: no action; Score: 2.5 / 15.0"
        match = re.search(r"Score: ([\d.]+)", result.stdout)
        if match:
            return float(match.group(1))

        return 0.0

    except:
        return 0.0  # On error, assume safe
    finally:
        Path(temp_path).unlink(missing_ok=True)


def llm_confirm_phishing(
    email: EmailMessage, attachments: List[Attachment], rspamd_score: float
) -> PhishingAssessment:
    """
    Use LLM to confirm phishing detection
    """
    import requests

    # Construct prompt
    attachment_names = [a.original_filename for a in attachments]

    prompt = f"""Analyze this email for phishing indicators.

**Email Details:**
- Subject: {email.subject}
- From: {email.sender}
- Date: {email.date}
- Spam Score: {rspamd_score}/15.0

**Email Body (first 1000 chars):**
{email.body_text[:1000]}

**Attachments:** {', '.join(attachment_names) if attachment_names else 'None'}

**Phishing Indicators to Check:**
- Urgent language ("act now", "verify immediately")
- Mismatched sender/reply-to addresses
- Suspicious links
- Requests for sensitive information
- Impersonation of known brands
- Unusual attachments (.exe, .scr, .zip with executables)

**Response Format:**
PHISHING: [YES/NO]
CONFIDENCE: [0.0-1.0]
REASONS: [Brief list of red flags]

Analyze:"""

    try:
        response = requests.post(
            CONFIG["llm"]["api_url"],
            json={"model": CONFIG["llm"]["model"], "prompt": prompt, "stream": False},
            timeout=CONFIG["llm"]["timeout"],
        )

        llm_response = response.json()["response"]

        # Parse response
        is_phishing = "YES" in llm_response.split("\n")[0].upper()

        # Extract confidence (simple regex)
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", llm_response)
        confidence = float(conf_match.group(1)) if conf_match else 0.7

        # Extract reasons
        reasons_section = (
            llm_response.split("REASONS:")[1] if "REASONS:" in llm_response else ""
        )
        reasons = [
            r.strip()
            for r in reasons_section.split("\n")
            if r.strip() and r.strip() != "-"
        ]

        return PhishingAssessment(
            is_phishing=is_phishing,
            confidence=confidence,
            score=rspamd_score,
            reasons=reasons,
        )

    except Exception as e:
        print(f"  ⚠️  LLM phishing check failed: {e}")
        # Fallback to rspamd score
        return PhishingAssessment(
            is_phishing=True,
            confidence=0.6,
            score=rspamd_score,
            reasons=["High spam score (LLM unavailable)"],
        )


# ============================================================================
# IMPORTANCE ASSESSMENT (LLM)
# ============================================================================


def assess_email_importance(email: EmailMessage, attachments: List[Attachment]) -> bool:
    """
    Use LLM to determine if email is important for archival
    Returns: True = important, False = archive-only
    """
    import requests

    # Build attachment list
    attachment_info = (
        ", ".join([f"{a.original_filename} ({a.content_type})" for a in attachments])
        if attachments
        else "None"
    )

    prompt = f"""Analyze this email and determine if it's important for long-term archival.

**Important emails include:**
- Financial transactions, receipts, invoices
- Legal documents, contracts, agreements
- Immigration/identity documents (passports, visas, IDs)
- Professional certifications, licenses
- Tax documents, W-2s, 1099s
- Medical records, prescriptions, test results
- Property records, leases, deeds
- Insurance documents
- Educational transcripts, diplomas
- Important receipts (over $50)
- Account confirmations for important services

**NOT important (archive-only):**
- Marketing emails, newsletters
- Social media notifications
- Casual personal correspondence
- Spam, promotional emails
- Old conversations with no attachments
- Meeting reminders (past events)
- Transient information

**Email to analyze:**
Subject: {email.subject}
From: {email.sender}
Date: {email.date}
Body (first 2000 chars):
{email.body_text[:2000]}

Attachments: {attachment_info}

**Respond with ONLY:** TRUE or FALSE

Important:"""

    try:
        response = requests.post(
            CONFIG["llm"]["api_url"],
            json={"model": CONFIG["llm"]["model"], "prompt": prompt, "stream": False},
            timeout=CONFIG["llm"]["timeout"],
        )

        llm_response = response.json()["response"].strip().upper()

        # Parse TRUE/FALSE
        if "TRUE" in llm_response:
            return True
        elif "FALSE" in llm_response:
            return False
        else:
            # If unclear, default to important (better safe than sorry)
            print(
                f"  ⚠️  Unclear LLM response for email {email.uuid[:12]}: {llm_response}"
            )
            return True

    except Exception as e:
        print(f"  ⚠️  LLM importance check failed: {e}")
        # On error, assume important (conservative approach)
        return True


# ============================================================================
# EMAIL TO PDF CONVERSION
# ============================================================================


def convert_email_to_pdf(email: EmailMessage) -> Path:
    """
    Convert email to PDF using weasyprint
    Preserves headers, formatting, inline images
    """
    import tempfile
    from weasyprint import HTML, CSS

    # Generate HTML version of email
    html_content = generate_email_html(email)

    # Create temp HTML file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
        f.write(html_content)
        html_path = Path(f.name)

    # Output PDF path
    pdf_filename = f"{email.uuid}.pdf"
    pdf_path = Path(tempfile.gettempdir()) / pdf_filename

    try:
        # Convert to PDF
        HTML(filename=str(html_path)).write_pdf(
            target=str(pdf_path),
            stylesheets=[
                CSS(
                    string="""
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    font-size: 12pt;
                }
                .email-header { 
                    border-bottom: 2px solid #333; 
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }
                .header-field {
                    margin: 5px 0;
                }
                .header-label {
                    font-weight: bold;
                    display: inline-block;
                    width: 100px;
                }
                .email-body {
                    line-height: 1.6;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
            """
                )
            ],
        )

        return pdf_path

    finally:
        html_path.unlink(missing_ok=True)


def generate_email_html(email: EmailMessage) -> str:
    """
    Generate HTML representation of email
    """
    # Escape HTML in text body
    import html

    body_content = (
        email.body_html
        if email.body_html
        else html.escape(email.body_text).replace("\n", "<br>")
    )

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{html.escape(email.subject)}</title>
</head>
<body>
    <div class="email-header">
        <div class="header-field">
            <span class="header-label">From:</span>
            <span>{html.escape(email.sender)}</span>
        </div>
        <div class="header-field">
            <span class="header-label">To:</span>
            <span>{html.escape(', '.join(email.recipients))}</span>
        </div>
        <div class="header-field">
            <span class="header-label">Subject:</span>
            <span>{html.escape(email.subject)}</span>
        </div>
        <div class="header-field">
            <span class="header-label">Date:</span>
            <span>{email.date.strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        {f'<div class="header-field"><span class="header-label">Attachments:</span><span>{email.attachment_count}</span></div>' if email.has_attachments else ''}
    </div>
    <div class="email-body">
        {body_content}
    </div>
</body>
</html>"""

    return html_template


# ============================================================================
# FILE SAVING & ROUTING
# ============================================================================


def save_attachment_with_metadata(attachment: Attachment):
    """
    Save attachment and its metadata JSON to SANITIZATION folder
    """
    # Save attachment file
    output_dir = CONFIG["output_dirs"]["sanitization"]

    # Filename: {email_uuid}_{original_name}_{attachment_uuid}.ext
    name, ext = (
        attachment.sanitized_filename.rsplit(".", 1)
        if "." in attachment.sanitized_filename
        else (attachment.sanitized_filename, "bin")
    )
    final_filename = f"{attachment.email_uuid[:12]}_{name}_{attachment.uuid}{('.' + ext) if ext else ''}"

    file_path = output_dir / final_filename
    file_path.write_bytes(attachment.content)

    # Save metadata JSON
    metadata = {
        "attachment_uuid": attachment.uuid,
        "email_uuid": attachment.email_uuid,
        "original_filename": attachment.original_filename,
        "email_subject": attachment.metadata["email_subject"],
        "sender": attachment.metadata["sender"],
        "date": attachment.metadata["date"],
        "account": attachment.metadata["account"],
        "content_type": attachment.content_type,
        "size_bytes": attachment.size_bytes,
    }

    metadata_path = output_dir / f"{attachment.uuid}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def save_calendar_event(calendar: CalendarEvent):
    """
    Save calendar .ics file to CALENDARS folder
    """
    output_dir = CONFIG["output_dirs"]["calendars"]

    # Filename: {email_uuid}_calendar.ics
    filename = f"{calendar.email_uuid[:12]}_{sanitize_filename(calendar.title)}.ics"
    file_path = output_dir / filename

    file_path.write_bytes(calendar.ics_content)

    # Also save metadata JSON
    metadata = {
        "email_uuid": calendar.email_uuid,
        "title": calendar.title,
        "start_time": calendar.start_time.isoformat(),
        "end_time": calendar.end_time.isoformat(),
        "attendees": calendar.attendees,
        "location": calendar.location,
        "meeting_links": calendar.meeting_links,
        "ics_file": filename,
    }

    metadata_path = output_dir / f"{calendar.email_uuid[:12]}_calendar.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def route_archive_only(email: EmailMessage):
    """
    Move non-important email to archive-only storage
    Just copies the original maildir file
    """
    output_dir = CONFIG["output_dirs"]["archive_only"]

    # Keep original maildir structure
    dest_path = output_dir / f"{email.account}_{email.uuid}.eml"

    # Copy maildir file
    import shutil

    shutil.copy2(email.maildir_path, dest_path)


def route_phishing_email(
    email: EmailMessage, attachments: List[Attachment], assessment: PhishingAssessment
):
    """
    Move suspected phishing to review folder
    """
    output_dir = CONFIG["output_dirs"]["phishing_review"]

    # Save email
    dest_path = output_dir / f"{email.uuid}_PHISHING.eml"
    import shutil

    shutil.copy2(email.maildir_path, dest_path)

    # Save assessment report
    report = {
        "email_uuid": email.uuid,
        "subject": email.subject,
        "sender": email.sender,
        "date": email.date.isoformat(),
        "is_phishing": assessment.is_phishing,
        "confidence": assessment.confidence,
        "score": assessment.score,
        "reasons": assessment.reasons,
        "attachments": [a.original_filename for a in attachments],
    }

    report_path = output_dir / f"{email.uuid}_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2))


def archive_raw_email(email: EmailMessage):
    """
    Keep copy of original email in raw archive
    """
    output_dir = CONFIG["output_dirs"]["raw_emails"]
    dest_path = output_dir / f"{email.account}_{email.uuid}.eml"

    import shutil

    shutil.copy2(email.maildir_path, dest_path)
