from pathlib import Path
from datetime import datetime
import hashlib
import uuid


# Basic UUID generation
def generate_item_id() -> str:
    return str(uuid.uuid4())
