def parse_html_bookmarks(file_path):
    """Extract bookmarks from HTML export"""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    bookmarks = []
    for link in soup.find_all("a"):
        bookmarks.append(
            {
                "url": link.get("href"),
                "title": link.text,
                "date_added": link.get("add_date"),
                "tags": link.get("tags", "").split(","),
            }
        )
    return bookmarks


def import_to_linkwarden(bookmark, linkwarden_url, api_key):
    """Send bookmark to Linkwarden API"""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        f"{linkwarden_url}/api/v1/links", json=bookmark, headers=headers
    )
    return response.json()


def generate_tags_ollama(title, url):
    """Use local LLM to generate relevant tags"""
    prompt = f"""Analyze this bookmark and generate 3-5 relevant tags:
    Title: {title}
    URL: {url}

    Return only comma-separated tags."""

    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": prompt}]
    )
    tags = response["message"]["content"].strip().split(",")
    return [tag.strip() for tag in tags]


import time

import ollama
import requests
from bs4 import BeautifulSoup


def process_bookmarks(html_file, linkwarden_url, api_key):
    # 1. Extract bookmarks
    bookmarks = parse_html_bookmarks(html_file)

    # 2. Process each bookmark
    for bookmark in bookmarks:
        # 3. Generate AI tags
        ai_tags = generate_tags_ollama(bookmark["title"], bookmark["url"])

        # 4. Merge with existing tags
        bookmark["tags"] = list(set(bookmark["tags"] + ai_tags))

        # 5. Import to Linkwarden
        result = import_to_linkwarden(bookmark, linkwarden_url, api_key)
        print(f"Imported: {bookmark['title']}")
        time.sleep(0.5)  # Rate limiting


# Usage
# process_bookmarks("bookmarks.html", "http://localhost:3000", "your-api-key")
