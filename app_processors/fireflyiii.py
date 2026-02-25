import json

import ollama
import requests

FIREFLY_URL = "http://localhost/api/v1"
FIREFLY_TOKEN = "your-token-here"


def extract_and_import(pdf_path, source_account_id):
    # Extract with Qwen2-VL (handles PDF directly!)
    response = ollama.chat(
        model="qwen2-vl:7b",
        messages=[
            {
                "role": "user",
                "content": """You are extracting financial transactions from a bank statement or invoice.

CRITICAL RULES:
1. Extract ONLY transactions from the transaction table
2. Ignore headers, footers, account summaries, and letterhead
3. Date format: YYYY-MM-DD
4. Amounts: negative for withdrawals/debits, positive for deposits/credits
5. Be precise with decimal amounts (always 2 decimal places)
6. Extract the EXACT description text from the statement

Return ONLY valid JSON (no markdown, no extra text):
{
  "transactions": [
    {
      "date": "2024-01-15",
      "description": "GROCERY STORE #123",
      "amount": -45.67
    }
  ]
}

Extract ALL transactions from this page.""",
                "images": [pdf_path],
            }
        ],
    )

    # Parse response
    data = json.loads(response["message"]["content"])

    # Import each transaction
    headers = {
        "Authorization": f"Bearer {FIREFLY_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    for txn in data["transactions"]:
        is_withdrawal = txn["amount"] < 0

        payload = {
            "transactions": [
                {
                    "type": "withdrawal" if is_withdrawal else "deposit",
                    "date": txn["date"],
                    "amount": str(abs(txn["amount"])),
                    "description": txn["description"],
                    "source_id": source_account_id if is_withdrawal else None,
                    "destination_name": txn["description"] if is_withdrawal else None,
                    "source_name": txn["description"] if not is_withdrawal else None,
                    "destination_id": source_account_id if not is_withdrawal else None,
                }
            ]
        }

        # Remove None values
        payload["transactions"][0] = {
            k: v for k, v in payload["transactions"][0].items() if v is not None
        }

        resp = requests.post(
            f"{FIREFLY_URL}/transactions", headers=headers, json=payload
        )

        if resp.status_code == 200:
            print(f"✓ {txn['date']}: {txn['description']} (${abs(txn['amount'])})")
        else:
            print(f"✗ Failed: {resp.json()}")


# Usage
extract_and_import("rbc_statement.pdf", source_account_id=1)
