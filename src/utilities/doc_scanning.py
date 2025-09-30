"""
Interactive Document Scanner
A module for reviewing images and converting them to enhanced PDFs

Requirements:
pip install pillow opencv-python numpy ocrmypdf img2pdf
"""

import shutil
import subprocess
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog , messagebox
from typing import List , Optional

import cv2
import img2pdf
import numpy as np
from PIL import Image , ImageTk


class DocumentScanner:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Interactive Document Scanner")
        self.root.geometry("1200x800")

        # Queue management
        self.image_queue: List[Path] = []
        self.current_index = 0
        self.archive_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None

        # State tracking
        self.current_image_path: Optional[Path] = None
        self.pending_scan: Optional[Path] = (
            None  # Path to scanned PDF awaiting approval
        )
        self.archive_copy: Optional[Path] = None  # Path to archived original

        self.setup_ui()
        self.bind_keys()

    def setup_ui(self):
        """Setup the user interface"""
        # Top frame for controls
        control_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        control_frame.pack(fill=tk.X)

        # Buttons
        tk.Button(
            control_frame,
            text="Load Images",
            command=self.load_images,
            bg="#3498db",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
        ).pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_label = tk.Label(
            control_frame,
            text="No images loaded",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 12),
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Image display frame
        self.image_frame = tk.Frame(self.root, bg="#34495e")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg="#34495e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="← Left Arrow: Skip | → Right Arrow: Scan to PDF | ESC: Quit",
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Arial", 14, "bold"),
            pady=15,
        )
        instructions.pack(fill=tk.X)

    def bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind("<Left>", lambda e: self.handle_left_arrow())
        self.root.bind("<Right>", lambda e: self.handle_right_arrow())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def load_images(self):
        """Load images from a directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if not directory:
            return

        directory = Path(directory)

        # Set up archive and output directories
        self.archive_dir = directory / "archive"
        self.output_dir = directory / "scanned_pdfs"
        self.archive_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Supported image formats
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        # Load all images
        self.image_queue = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not self.image_queue:
            messagebox.showinfo(
                "No Images", "No image files found in the selected directory."
            )
            return

        self.current_index = 0
        self.show_current_image()

    def show_current_image(self):
        """Display the current image"""
        if not self.image_queue or self.current_index >= len(self.image_queue):
            messagebox.showinfo("Complete", "All images processed!")
            self.status_label.config(text="All images processed!")
            return

        self.current_image_path = self.image_queue[self.current_index]

        # Check if this is a pending scan approval
        is_pending = (
            self.pending_scan is not None
            and self.current_image_path == self.pending_scan
        )

        # Update status
        status_text = f"Image {self.current_index + 1}/{len(self.image_queue)}"
        if is_pending:
            status_text += " [REVIEWING SCAN - Right=Accept, Left=Reject]"
        status_text += f" - {self.current_image_path.name}"
        self.status_label.config(text=status_text)

        # Load and display image
        try:
            # Load image with PIL
            img = Image.open(self.current_image_path)

            # Resize to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas not yet sized, use defaults
                canvas_width, canvas_height = 1000, 600

            img.thumbnail(
                (canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS
            )

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)

            # Clear canvas and display
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo,
                anchor=tk.CENTER,
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def handle_left_arrow(self):
        """Skip current image or reject scan"""
        if not self.image_queue:
            return

        # Check if we're reviewing a scanned PDF
        if self.pending_scan and self.current_image_path == self.pending_scan:
            # Reject the scan
            self.reject_scan()
        else:
            # Skip this image
            self.current_index += 1
            self.show_current_image()

    def handle_right_arrow(self):
        """Process image to PDF or accept scan"""
        if not self.image_queue:
            return

        # Check if we're reviewing a scanned PDF
        if self.pending_scan and self.current_image_path == self.pending_scan:
            # Accept the scan
            self.accept_scan()
        else:
            # Start scanning process
            self.scan_to_pdf()

    def scan_to_pdf(self):
        """Scan current image to PDF with enhancement"""
        if not self.current_image_path:
            return

        try:
            # Update status
            self.status_label.config(
                text=f"Processing {self.current_image_path.name}..."
            )
            self.root.update()

            # 1. Make archive copy
            archive_path = self.archive_dir / self.current_image_path.name
            shutil.copy2(self.current_image_path, archive_path)
            self.archive_copy = archive_path

            # 2. Preprocess image with OpenCV
            processed_img = self.preprocess_image(self.current_image_path)

            # 3. Save preprocessed image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_img_path = Path(tmp.name)
                cv2.imwrite(str(temp_img_path), processed_img)

            # 4. Convert to PDF using img2pdf
            pdf_name = self.current_image_path.stem + ".pdf"
            temp_pdf = self.output_dir / f"temp_{pdf_name}"

            with open(temp_pdf, "wb") as f:
                f.write(img2pdf.convert(str(temp_img_path)))

            # 5. Enhance with OCRmyPDF
            final_pdf = self.output_dir / pdf_name
            self.enhance_pdf(temp_pdf, final_pdf)

            # Clean up temp files
            temp_img_path.unlink()
            temp_pdf.unlink(missing_ok=True)

            # 6. Add PDF to queue for review
            self.image_queue.append(final_pdf)
            self.pending_scan = final_pdf

            # 7. Move to next item
            self.current_index += 1
            self.show_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan image: {str(e)}")
            # Restore from archive if it exists
            if self.archive_copy and self.archive_copy.exists():
                shutil.copy2(self.archive_copy, self.current_image_path)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """Preprocess image with OpenCV for better scanning"""
        # Read image
        img = cv2.imread(str(image_path))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply edge detection for document detection (optional - basic preprocessing)
        # For a full document scanner, you'd add perspective correction here

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Increase contrast and brightness
        alpha = 1.5  # Contrast
        beta = 20  # Brightness
        adjusted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)

        # Apply adaptive thresholding for better text visibility
        thresh = cv2.adaptiveThreshold(
            adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
        )

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)

        return sharpened

    def enhance_pdf(self, input_pdf: Path, output_pdf: Path):
        """Enhance PDF using OCRmyPDF"""
        try:
            cmd = [
                "ocrmypdf",
                "--deskew",  # Deskew crooked PDFs
                "--clean",  # Clean up image before OCR
                "--optimize",
                "3",  # Highest optimization
                "--output-type",
                "pdf",
                str(input_pdf),
                str(output_pdf),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                raise Exception(f"OCRmyPDF failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise Exception("OCR processing timed out")
        except FileNotFoundError:
            raise Exception(
                "OCRmyPDF not installed. Install with: pip install ocrmypdf"
            )

    def accept_scan(self):
        """Accept the scanned PDF"""
        if not self.pending_scan:
            return

        # Scan accepted, move to next
        self.pending_scan = None
        self.archive_copy = None
        self.current_index += 1
        self.show_current_image()

    def reject_scan(self):
        """Reject the scan and restore original"""
        if not self.pending_scan or not self.archive_copy:
            return

        try:
            # Delete the rejected PDF
            if self.pending_scan.exists():
                self.pending_scan.unlink()

            # Remove from queue
            self.image_queue.remove(self.pending_scan)

            # Restore original from archive
            original_path = self.archive_copy.parent.parent / self.archive_copy.name
            if self.archive_copy.exists():
                shutil.copy2(self.archive_copy, original_path)

            # Reset state
            self.pending_scan = None
            self.archive_copy = None

            # Don't increment index - show next image
            self.show_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to reject scan: {str(e)}")
