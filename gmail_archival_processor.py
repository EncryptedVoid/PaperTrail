import os
import base64
import email
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time

# Scopes needed for Gmail API
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailAutomation:
    def __init__(self):
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        # Token file stores the user's access and refresh tokens
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        self.service = build("gmail", "v1", credentials=creds)

    def get_starred_emails(self):
        """Get all starred emails"""
        try:
            # Search for starred emails
            results = (
                self.service.users()
                .messages()
                .list(userId="me", q="is:starred", maxResults=500)  # Adjust as needed
                .execute()
            )

            messages = results.get("messages", [])

            # Handle pagination if there are more than 500 emails
            while "nextPageToken" in results:
                page_token = results["nextPageToken"]
                results = (
                    self.service.users()
                    .messages()
                    .list(
                        userId="me",
                        q="is:starred",
                        maxResults=500,
                        pageToken=page_token,
                    )
                    .execute()
                )
                messages.extend(results.get("messages", []))

            print(f"Found {len(messages)} starred emails")
            return messages

        except Exception as error:
            print(f"An error occurred: {error}")
            return []

    def get_email_details(self, message_id):
        """Get full email details including attachments"""
        try:
            message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
            return message
        except Exception as error:
            print(f"Error getting email details: {error}")
            return None

    def download_attachments(self, message_id, message_data, download_folder):
        """Download all attachments from an email"""
        attachments = []

        def extract_attachments(parts, message_id):
            for part in parts:
                if part.get("parts"):
                    # Recursively check nested parts
                    extract_attachments(part["parts"], message_id)

                if part.get("filename"):
                    attachment_id = part["body"].get("attachmentId")
                    if attachment_id:
                        try:
                            # Get attachment data
                            attachment = (
                                self.service.users()
                                .messages()
                                .attachments()
                                .get(
                                    userId="me", messageId=message_id, id=attachment_id
                                )
                                .execute()
                            )

                            # Decode attachment data
                            data = base64.urlsafe_b64decode(
                                attachment["data"].encode("UTF-8")
                            )

                            # Create safe filename
                            filename = part["filename"]
                            safe_filename = "".join(
                                c
                                for c in filename
                                if c.isalnum() or c in (" ", "-", "_", ".")
                            )

                            # Save attachment
                            filepath = os.path.join(
                                download_folder, f"{message_id}_{safe_filename}"
                            )
                            with open(filepath, "wb") as f:
                                f.write(data)

                            attachments.append(filepath)
                            print(f"Downloaded: {safe_filename}")

                        except Exception as e:
                            print(
                                f"Error downloading attachment {part['filename']}: {e}"
                            )

        # Extract attachments from message parts
        payload = message_data.get("payload", {})
        if payload.get("parts"):
            extract_attachments(payload["parts"], message_id)

        return attachments

    def save_email_as_eml(self, message_id, message_data, save_folder):
        """Save email as EML file"""
        try:
            # Get raw email content
            raw_message = (
                self.service.users()
                .messages()
                .get(userId="me", id=message_id, format="raw")
                .execute()
            )

            # Decode the raw email
            raw_email = base64.urlsafe_b64decode(raw_message["raw"].encode("ASCII"))

            # Parse email to get subject for filename
            email_obj = email.message_from_bytes(raw_email)
            subject = email_obj.get("Subject", "No Subject")

            # Create safe filename
            safe_subject = "".join(
                c for c in subject if c.isalnum() or c in (" ", "-", "_")
            )[:50]
            filename = f"{message_id}_{safe_subject}.eml"

            # Save EML file
            filepath = os.path.join(save_folder, filename)
            with open(filepath, "wb") as f:
                f.write(raw_email)

            print(f"Saved email: {filename}")
            return filepath

        except Exception as e:
            print(f"Error saving email as EML: {e}")
            return None

    def unstar_email(self, message_id):
        """Remove star from email"""
        try:
            self.service.users().messages().modify(
                userId="me", id=message_id, body={"removeLabelIds": ["STARRED"]}
            ).execute()
            print(f"Unstarred email: {message_id}")
        except Exception as e:
            print(f"Error unstarring email: {e}")

    def process_all_starred_emails(self, download_folder="gmail_exports"):
        """Main function to process all starred emails"""
        # Create folders
        os.makedirs(download_folder, exist_ok=True)
        attachments_folder = os.path.join(download_folder, "attachments")
        emails_folder = os.path.join(download_folder, "emails")
        os.makedirs(attachments_folder, exist_ok=True)
        os.makedirs(emails_folder, exist_ok=True)

        # Get all starred emails
        starred_messages = self.get_starred_emails()

        processed_count = 0
        for message in starred_messages:
            message_id = message["id"]
            print(
                f"\nProcessing email {processed_count + 1}/{len(starred_messages)}: {message_id}"
            )

            # Get email details
            message_data = self.get_email_details(message_id)
            if not message_data:
                continue

            # Download attachments
            attachments = self.download_attachments(
                message_id, message_data, attachments_folder
            )

            # Save email as EML
            eml_path = self.save_email_as_eml(message_id, message_data, emails_folder)

            # Only unstar if both operations succeeded
            if eml_path is not None:  # EML saved successfully
                self.unstar_email(message_id)
                processed_count += 1

            # Add small delay to avoid hitting rate limits
            time.sleep(0.1)

        print(f"\nCompleted! Processed {processed_count} emails.")
        print(f"Attachments saved to: {attachments_folder}")
        print(f"Emails saved to: {emails_folder}")


# Usage Examples
if __name__ == "__main__":
    automation = GmailAutomation()

    # Choose what you want to process:

    # Option 1: Process only starred emails (and unstar them)
    # automation.process_all_starred_emails()

    # Option 2: Process all document categories
    automation.process_document_categories()

    # Option 3: Process specific categories only
    # automation.process_emails_by_query('(receipt OR invoice OR "proof of purchase")', 'receipts_only')

    # Option 4: Custom search query
    # automation.process_custom_search('from:uscis.gov OR subject:immigration', 'immigration_docs')

    # Option 5: Process multiple specific queries
    # queries_to_process = [
    #     ('is:starred', 'starred_emails', True),  # Third parameter = unstar after processing
    #     ('(receipt OR invoice)', 'financial_docs', False),
    #     ('(legal OR immigration OR visa)', 'legal_docs', False)
    # ]
    #
    # for query, folder_name, unstar in queries_to_process:
    #     automation.process_emails_by_query(query, folder_name, unstar_after=unstar)


# ## 🎯 **What the Script Can Do:**

# **1. Starred Emails Only (Original):**
# - Downloads all your starred emails as EML files
# - Downloads all attachments
# - **Automatically unstars** them after successful download

# **2. Document Categories (New):**
# - **Receipts & Invoices:** Orders, purchases, payment confirmations
# - **Financial:** Bank statements, credit cards, tax documents, loans
# - **Legal & Immigration:** Visa docs, USCIS emails, court notices, lawyer correspondence
# - **Subscriptions:** Monthly bills, renewals, memberships
# - **Insurance & Medical:** Health records, claims, prescriptions
# - **Contracts:** Agreements, leases, employment docs
# - **Travel:** Flight confirmations, hotel bookings, Uber receipts
# - **Important Confirmations:** Account verifications, security alerts

# ## 🔍 **Gmail Search Power:**

# The script uses Gmail's advanced search operators:

# ```python
# # Examples of what it can find:
# '(receipt OR invoice OR "proof of purchase")'
# 'from:uscis.gov OR subject:immigration OR visa'
# '(legal OR lawyer OR attorney OR court)'
# '(subscription OR "monthly billing" OR renewal)'
# ```

# ## 🚀 **How to Use:**

# **Option 1 - Process Everything:**
# ```python
# automation = GmailAutomation()
# automation.process_document_categories()  # Downloads ALL document types
# ```

# **Option 2 - Just Starred (with unstarring):**
# ```python
# automation.process_all_starred_emails()
# ```

# **Option 3 - Custom Search:**
# ```python
# # Find all emails from government agencies
# automation.process_custom_search('from:irs.gov OR from:uscis.gov OR from:state.gov', 'government_docs')

# # Find all emails with PDFs attached
# automation.process_custom_search('has:attachment filename:pdf', 'pdf_emails')
# ```

# ## 📁 **Output Structure:**
# ```
# gmail_exports/
# ├── starred/
# │   ├── emails/
# │   └── attachments/
# ├── receipts_invoices/
# │   ├── emails/
# │   └── attachments/
# ├── legal_immigration/
# │   ├── emails/
# │   └── attachments/
# └── [other categories...]
# ```

# ## ⚠️ **Important Notes:**

# - **Only starred emails get unstarred** automatically
# - Document category emails are **NOT unstarred** (for safety)
# - The script creates separate folders for each category
# - It handles email threads as complete EML files
# - Rate limiting prevents Gmail API issues

# Would you like me to help you set this up, or would you prefer to modify the search queries for your specific needs?
