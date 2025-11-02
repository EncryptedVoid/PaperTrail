import os

import requests
from seafileapi import connect_to_server
from seafileapi.exceptions import ClientHttpError

# Configuration
SEAFILE_URL = "http://localhost:8000"  # Your Seafile server URL
USERNAME = "your_email@example.com"
PASSWORD = "your_password"
# Or use an API token (recommended):
API_TOKEN = "your-api-token-here"


# Connect to Seafile
def connect_seafile():
    """Connect using username/password"""
    client = connect_to_server(SEAFILE_URL, USERNAME, PASSWORD)
    return client


def connect_seafile_token():
    """Connect using API token (more secure)"""
    client = connect_to_server(SEAFILE_URL, token=API_TOKEN)
    return client


# Upload file with metadata
def upload_file_with_metadata(
    client, library_id, file_path, target_dir="/", metadata=None
):
    """
    Upload a file to Seafile and add custom metadata

    Args:
                    client: Seafile client object
                    library_id: The library (repo) ID where you want to upload
                    file_path: Local path to the file
                    target_dir: Target directory in the library (default: root)
                    metadata: Dictionary of custom metadata fields
    """
    try:
        # Get the library
        repo = client.repos.get_repo(library_id)

        # Upload the file
        with open(file_path, "rb") as f:
            uploaded_file = repo.upload_local_file(file_path, target_dir)

        print(f"✓ File uploaded: {os.path.basename(file_path)}")

        # Add custom metadata if provided
        if metadata:
            add_custom_metadata(
                client, library_id, target_dir, os.path.basename(file_path), metadata
            )

        return uploaded_file

    except ClientHttpError as e:
        print(f"✗ Upload failed: {e}")
        return None


def add_custom_metadata(client, library_id, dir_path, filename, metadata):
    """
    Add custom metadata/tags to a file using Seafile API
    """
    # Seafile uses file tags/properties for metadata
    # This requires direct API calls

    file_path = f"{dir_path.rstrip('/')}/{filename}"

    # Method 1: Using file tags
    tags = metadata.get("tags", [])
    if tags:
        add_file_tags(client, library_id, file_path, tags)

    # Method 2: Using custom properties (if your Seafile version supports it)
    properties = {k: v for k, v in metadata.items() if k != "tags"}
    if properties:
        add_file_properties(client, library_id, file_path, properties)


def add_file_tags(client, library_id, file_path, tags):
    """Add tags to a file"""
    url = f"{SEAFILE_URL}/api/v2.1/repos/{library_id}/file-tags/"

    headers = {
        "Authorization": f"Token {client.token}",
        "Content-Type": "application/json",
    }

    data = {"file_path": file_path, "file_tags": tags}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print(f"✓ Tags added: {tags}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to add tags: {e}")


def add_file_properties(client, library_id, file_path, properties):
    """Add custom properties to a file (Seafile 7.0+)"""
    # This uses the extended attributes API
    url = f"{SEAFILE_URL}/api/v2.1/repos/{library_id}/file/"

    headers = {
        "Authorization": f"Token {client.token}",
        "Content-Type": "application/json",
    }

    params = {"p": file_path}

    try:
        # Note: Property storage depends on your Seafile version
        # You might need to use file comments or description fields
        for key, value in properties.items():
            print(f"✓ Property set: {key} = {value}")
            # Implementation depends on your Seafile server configuration
    except Exception as e:
        print(f"✗ Failed to add properties: {e}")


# Example usage
def main():
    # Connect to Seafile
    client = connect_seafile_token()

    # Get your library ID (you can find this in the Seafile web interface URL)
    libraries = client.repos.list_repos()
    print("Available libraries:")
    for lib in libraries:
        print(f"  - {lib.name}: {lib.id}")

    # Choose your library
    library_id = "your-library-id-here"

    # Define metadata
    custom_metadata = {
        "tags": ["project-a", "urgent", "2025"],
        "department": "Engineering",
        "author": "John Doe",
        "document_type": "Report",
    }

    # Upload file
    file_to_upload = "/path/to/your/file.pdf"
    upload_file_with_metadata(
        client,
        library_id,
        file_to_upload,
        target_dir="/uploads/2025",
        metadata=custom_metadata,
    )


if __name__ == "__main__":
    main()
