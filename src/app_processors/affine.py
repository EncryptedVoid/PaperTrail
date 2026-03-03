#!/usr/bin/env python3
import glob
import json
import os

import requests


def generate_with_ollama(content, filename):
    """Use local Ollama to generate title and tags"""
    prompt = f"""Read this document and generate:
1. A clear, descriptive title (max 60 chars)
2. 3-5 relevant tags (single words or short phrases)

Document filename: {filename}
Content preview:
{content[:1500]}

Respond ONLY with valid JSON in this exact format:
{{"title": "Your Title Here", "tags": ["tag1", "tag2", "tag3"]}}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Forces JSON output
        },
    )

    result = response.json()["response"]
    return json.loads(result)


def add_frontmatter(filepath):
    """Add YAML frontmatter with AI-generated metadata"""
    with open(filepath, "r") as f:
        content = f.read()

    # Skip if already has frontmatter
    if content.startswith("---"):
        print(f"⏭️  Skipping {filepath} - already has frontmatter")
        return

    # Generate metadata
    filename = os.path.basename(filepath)
    print(f"🤖 Generating metadata for: {filename}")

    try:
        metadata = generate_with_ollama(content, filename)

        # Show user what was generated
        print(f"   Title: {metadata['title']}")
        print(f"   Tags: {', '.join(metadata['tags'])}")

        # Add frontmatter
        new_content = f"""---
title: {metadata['title']}
tags: {', '.join(metadata['tags'])}
---

{content}"""

        with open(filepath, "w") as f:
            f.write(new_content)

        print(f"✅ Added metadata to {filename}\n")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}\n")


def process_directory(notes_dir):
    """Process all markdown files in directory"""
    files = glob.glob(f"{notes_dir}/**/*.md", recursive=True)

    print(f"Found {len(files)} markdown files\n")

    for filepath in files:
        add_frontmatter(filepath)

    print("✨ Done! Files are ready to upload to AFFiNE")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python add_metadata.py /path/to/notes")
        sys.exit(1)

    # Make sure Ollama is running
    try:
        requests.get("http://localhost:11434")
    except:
        print("❌ Ollama is not running! Start it with: ollama serve")
        sys.exit(1)

    process_directory(sys.argv[1])
