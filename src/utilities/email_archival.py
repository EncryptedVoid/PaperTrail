#!/usr/bin/env python3
"""
Complete Email Backup & Classification System
Auto-installs dependencies and backs up Gmail/Outlook
"""

import email
import platform
import shutil
import subprocess
import sys

from config import (
    EMAIL_ARTIFACTS_DIR,
    EMAIL_OUTPUT_DIR,
    GMAIL_ADDRESS,
    IMPORTANT_EMAILS_DIR,
    OUTLOOK_ADDRESS,
    UNIMPORTANT_EMAILS_DIR,
)


def run_cmd(cmd, description, shell=True, check=False, capture=True):
    """Run command with optional output"""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            text=True,
            capture_output=capture,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        return result.returncode == 0, result.stdout if capture else ""
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, str(e)


def check_command_exists(cmd):
    """Check if a command exists"""
    success, _ = run_cmd(
        f"which {cmd}" if platform.system() != "Windows" else f"where {cmd}",
        "",
        capture=True,
    )
    return success


def classify_email(eml_file):
    """Classify email with LLM"""
    try:
        msg = email.message_from_bytes(eml_file.read_bytes())
    except:
        return "important"

    # Check attachments
    has_attachments = any(
        part.get_content_disposition() == "attachment" for part in msg.walk()
    )

    if has_attachments:
        return "attachments"

    # Get content
    subject = msg.get("Subject", "")[:100]
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode(errors="ignore")
                except:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except:
            pass

    body = body[:1000]

    # Import ollama here (after we know it's installed)
    try:
        import ollama
    except ImportError:
        print("   ⚠️  Ollama Python package not found, installing...")
        run_cmd(
            f"{sys.executable} -m pip install ollama",
            "Installing ollama",
            capture=False,
        )
        import ollama

    prompt = f"""YES or NO only.

Important? (finances, law, identity, immigration, payments, invoices, reports, work, freelancing)

Subject: {subject}
Body: {body}

Answer:"""

    try:
        response = ollama.chat(
            model="llama3.2:3b", messages=[{"role": "user", "content": prompt}]
        )
        is_important = "YES" in response["message"]["content"].upper()
        return "important" if is_important else "unimportant"
    except:
        return "important"


def backup_and_classify():
    """Main backup and classification process"""
    print("\n" + "=" * 60)
    print("📥 BACKING UP EMAILS")
    print("=" * 60)

    backup_dirs = []

    # Backup Gmail
    if GMAIL_ADDRESS and GMAIL_ADDRESS != "your@gmail.com":
        gmail_dir = EMAIL_OUTPUT_DIR / "gmail_backup"
        print(f"\n📧 Backing up Gmail: {GMAIL_ADDRESS}")

        success, _ = run_cmd(
            f'gyb --email {GMAIL_ADDRESS} --local-folder "{gmail_dir}"',
            "Gmail backup",
            capture=False,
        )

        if success:
            print("   ✅ Gmail backup complete")
            backup_dirs.append(gmail_dir)
        else:
            print("   ⚠️  Gmail backup had issues")

    # Backup Outlook
    if OUTLOOK_ADDRESS and OUTLOOK_ADDRESS != "your@outlook.com":
        outlook_dir = EMAIL_OUTPUT_DIR / "outlook_backup"
        print(f"\n📧 Backing up Outlook: {OUTLOOK_ADDRESS}")

        success, _ = run_cmd(
            f'gmvault sync {OUTLOOK_ADDRESS} -d "{outlook_dir}"',
            "Outlook backup",
            capture=False,
        )

        if success:
            print("   ✅ Outlook backup complete")
            backup_dirs.append(outlook_dir)
        else:
            print("   ⚠️  Outlook backup had issues")

    if not backup_dirs:
        print("\n⚠️  No emails backed up. Check your configuration.")
        return

    # Classify emails
    print("\n" + "=" * 60)
    print("🤖 CLASSIFYING EMAILS")
    print("=" * 60)

    all_emails = []
    for backup_dir in backup_dirs:
        if backup_dir.exists():
            all_emails.extend(backup_dir.rglob("*.eml"))

    if not all_emails:
        print("⚠️  No .eml files found")
        return

    print(f"\nClassifying {len(all_emails)} emails...\n")

    stats = {"important": 0, "unimportant": 0, "attachments": 0}

    for i, eml_file in enumerate(all_emails, 1):
        if any(
            p in eml_file.parents
            for p in [IMPORTANT_EMAILS_DIR, UNIMPORTANT_EMAILS_DIR, EMAIL_ARTIFACTS_DIR]
        ):
            continue

        print(f"[{i}/{len(all_emails)}] {eml_file.name[:35]}... ", end="")

        category = classify_email(eml_file)

        if category == "attachments":
            dest = EMAIL_ARTIFACTS_DIR / eml_file.name
            print("📎 ARTIFACTS")
            stats["attachments"] += 1
        elif category == "important":
            dest = IMPORTANT_EMAILS_DIR / eml_file.name
            print("⭐ IMPORTANT")
            stats["important"] += 1
        else:
            dest = UNIMPORTANT_EMAILS_DIR / eml_file.name
            print("📧 unimportant")
            stats["unimportant"] += 1

        counter = 1
        original_dest = dest
        while dest.exists():
            dest = original_dest.with_stem(f"{original_dest.stem}_{counter}")
            counter += 1

        shutil.copy2(eml_file, dest)

    print("\n" + "=" * 60)
    print("✅ COMPLETE")
    print("=" * 60)
    print(f"⭐ Important:    {stats['important']}")
    print(f"📧 Unimportant:  {stats['unimportant']}")
    print(f"📎 Attachments:  {stats['attachments']}")
    print(f"\n📁 {EMAIL_OUTPUT_DIR.absolute()}")
