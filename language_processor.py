#!/usr/bin/env python3
"""
Simple script to communicate with OLLAMA LLM models
Gets response in one batch (non-streaming)
"""

import requests
import json
import sys


def chat_with_ollama(prompt, model="llama2", host="http://localhost:11434"):
    """
    Send a prompt to OLLAMA and get the complete response

    Args:
        prompt (str): The text prompt to send
        model (str): The model name (default: llama2)
        host (str): OLLAMA server URL (default: http://localhost:11434)

    Returns:
        str: The LLM's response text
    """

    url = f"{host}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # This ensures we get the complete response in one batch
    }

    try:
        print(f"Sending prompt to {model}...")
        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to OLLAMA. Make sure it's running on " + host
    except requests.exceptions.Timeout:
        return (
            "Error: Request timed out. The model might be taking too long to respond."
        )
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except json.JSONDecodeError:
        return "Error: Invalid response from OLLAMA"


def main():
    # Example usage
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Enter your prompt: ")

    # You can change the model here - common options: llama2, mistral, codellama, etc.
    model = "llama2"

    print(f"\n🤖 Asking {model}: {prompt}")
    print("-" * 50)

    response = chat_with_ollama(prompt, model)

    print("Response:")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()
