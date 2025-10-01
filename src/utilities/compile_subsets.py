import random
from pathlib import Path

import cv2
import img2pdf
import numpy as np
from pypdf import PdfReader, PdfWriter


def compile_doc_subset(input_pdf: Path, set_size: int, temp_dir: Path) -> Path:
    """Extract subset of PDF pages: first 2, last 2, and random pages in between."""
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    # Calculate how many random middle pages we need
    n_random = max(0, set_size - 4)  # Subtract first 2 and last 2

    # First 2 pages
    for i in range(min(2, total_pages)):
        writer.add_page(reader.pages[i])

    # N random pages from the middle
    if total_pages > 4 and n_random > 0:
        middle_pages = list(range(2, total_pages - 2))
        random_pages = random.sample(middle_pages, min(n_random, len(middle_pages)))
        for page_num in sorted(random_pages):
            writer.add_page(reader.pages[page_num])

    # Last 2 pages
    if total_pages > 2:
        for i in range(max(2, total_pages - 2), total_pages):
            writer.add_page(reader.pages[i])

    # Write to temp directory
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"subset_{Path(input_pdf).stem}.pdf"

    with open(output_path, "wb") as output_file:
        writer.write(output_file)

    return output_path


def compile_video_snapshot_subset(
    video_path: Path, set_size: int, temp_dir: Path
) -> Path:
    """Extract meaningful frames from video and compile into PDF."""

    def is_meaningful_frame(frame, black_thresh=30, white_thresh=225):
        """Check if frame is not mostly black or white"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # Reject if too dark, too bright, or no variation
        if mean_val < black_thresh or mean_val > white_thresh or std_val < 10:
            return False
        return True

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample more candidates than needed to filter out bad frames
    candidate_frames = random.sample(
        range(0, total_frames), min(set_size * 3, total_frames)
    )

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    images = []
    temp_files = []

    for frame_num in candidate_frames:
        if len(images) >= set_size:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret and is_meaningful_frame(frame):
            # Save as temporary image
            temp_file = temp_dir / f"frame_{len(images)}.jpg"
            cv2.imwrite(str(temp_file), frame)
            images.append(str(temp_file))
            temp_files.append(temp_file)

    cap.release()

    # Convert images to PDF
    output_path = temp_dir / f"snapshots_{Path(video_path).stem}.pdf"

    if images:
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(images))

        # Cleanup temp image files
        for temp_file in temp_files:
            temp_file.unlink()

    return output_path
