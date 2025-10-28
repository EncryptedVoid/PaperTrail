import logging

from config import GMAIL_ADDRESS, OUTLOOK_ADDRESS
from utilities.dependancy_ensurance import (
    ensure_gmvault,
    ensure_gyb,
    ensure_oauth,
    ensure_ollama,
)
from utilities.email_archival import (
    backup_and_classify,
)


def archiving_emails(logger: logging.logger):
    ensure_gmvault()
    ensure_gyb()
    ensure_ollama()

    if GMAIL_ADDRESS and GMAIL_ADDRESS != "your@gmail.com":
        ensure_oauth(
            GMAIL_ADDRESS,
            "GYB (Gmail)",
            f"gyb --email {GMAIL_ADDRESS} --action estimate",
        )

    if OUTLOOK_ADDRESS and OUTLOOK_ADDRESS != "your@outlook.com":
        ensure_oauth(
            OUTLOOK_ADDRESS,
            "Gmvault (Outlook)",
            f"gmvault sync {OUTLOOK_ADDRESS} -t quick",
        )

    backup_and_classify()
    return None
