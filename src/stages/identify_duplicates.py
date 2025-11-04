#!/usr/bin/env python3
"""
Simple Duplicate File Reviewer

This module provides a GUI application for reviewing and managing duplicate files.
It displays pairs of duplicate files side-by-side with their metadata and allows
users to interactively choose which files to keep or delete.

Features:
- Load duplicate file pairs from JSON format
- Display file metadata (size, modification time, PDF properties)
- Show image previews for supported formats
- Interactive file deletion with confirmation
- Keyboard shortcuts for quick navigation
- Decision logging to JSON file
- Comprehensive statistics and timing information

Usage:
    reviewer = DuplicateReviewer(review_directory=Path("."), logger=my_logger)
    reviewer.run()
"""
import json
import logging
import os
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog , messagebox , ttk

# pypdf is used for reading PDF metadata and page counts
from pypdf import PdfReader

# Attempt to import customtkinter for modern dark theme UI
# Falls back to standard tkinter if not available
try:
    import customtkinter as ctk
    from customtkinter import CTkScrollableFrame

    # Set dark mode appearance and blue color theme for customtkinter
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    # If customtkinter is not installed, use standard tkinter as fallback
    import tkinter as ctk

    CTkScrollableFrame = tk.Frame

# pikepdf provides advanced PDF manipulation capabilities (optional)
try:
    import pikepdf

    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False

# PIL (Pillow) for image handling and fitz (PyMuPDF) for PDF rendering
try:
    from PIL import Image, ImageTk
    import fitz  # PyMuPDF library for PDF rendering

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OpenCV for video processing (optional)
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# pygame for audio playback (optional)
try:
    import pygame

    # Initialize the mixer module for audio playback
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Determine the best available font for the UI
FONT_FAMILY = "Segoe UI"
try:
    import tkinter.font as tkfont

    # Create temporary root to query available fonts
    root_test = tk.Tk()
    available_fonts = tkfont.families()
    root_test.destroy()
    # Prefer Segoe UI, fallback to Arial or default
    if "Segoe UI" not in available_fonts:
        FONT_FAMILY = "Arial" if "Arial" in available_fonts else "TkDefaultFont"
except:
    FONT_FAMILY = "Arial"


class DuplicateReviewer:
    """
    GUI application for reviewing and managing duplicate file pairs.

    This class creates a tkinter-based interface that allows users to:
    - Load pairs of duplicate files from JSON
    - View metadata and previews side-by-side
    - Make decisions about which files to keep or delete
    - Track all decisions and save them to a log file

    The application logs detailed information about the review process including
    timing, statistics, and user decisions.
    """

    def __init__(self, review_directory: Path, logger: logging.Logger):
        """
        Initialize the Duplicate Reviewer application.

        Args:
                        review_directory: Path object pointing to the directory where review files are located
                        logger: Logger instance for recording application events and statistics
        """
        # Store the logger instance for use throughout the application
        self.logger = logger
        self.logger.info("Initializing Duplicate File Reviewer application")

        # Track start time for calculating total session duration
        self.session_start_time = time.time()

        # Store the review directory path
        self.review_directory = review_directory
        self.logger.info(f"Review directory set to: {review_directory}")

        # Create the main application window using customtkinter or tkinter
        self.root = ctk.CTk()
        self.root.title("Duplicate File Reviewer")
        self.root.geometry("1200x700")
        self.logger.info("Main application window created with size 1200x700")

        # Initialize data structures for managing duplicate pairs and decisions
        self.pairs = []  # List of [file1, file2] pairs loaded from JSON
        self.current_index = 0  # Current position in the pairs list
        self.log = []  # Log of all user decisions (keep/delete)

        # Statistics tracking for session summary
        self.stats = {
            "pairs_loaded": 0,  # Total pairs loaded from JSON
            "pairs_reviewed": 0,  # Pairs that user has made decisions on
            "files_deleted": 0,  # Total files deleted during session
            "files_kept": 0,  # Total files explicitly kept
            "both_kept": 0,  # Pairs where both files were kept
            "left_kept": 0,  # Times left file was chosen
            "right_kept": 0,  # Times right file was chosen
            "file_types": Counter(),  # Count of file extensions encountered
            "total_size_deleted": 0,  # Total bytes deleted
        }
        self.logger.info("Statistics tracking initialized")

        # Build the user interface components
        self.setup_ui()
        self.logger.info("User interface setup complete")

    def setup_ui(self):
        """
        Create and arrange all UI components in the main window.

        This method builds:
        - Top control bar with load button and progress indicator
        - Left and right file comparison panels
        - Action buttons for making decisions
        - Keyboard shortcut bindings
        """
        self.logger.info("Setting up user interface components")

        # Top bar - contains load button and progress display
        top = ttk.Frame(self.root)
        # pack() geometry manager: fill horizontally, add padding
        top.pack(fill=tk.X, padx=10, pady=10)

        # Button to trigger JSON file loading dialog
        ttk.Button(top, text="Load JSON Pairs", command=self.load_json).pack(
            side=tk.LEFT, padx=5
        )
        # Label showing current progress (Pair X/Y)
        self.progress = ttk.Label(top, text="No files loaded")
        self.progress.pack(side=tk.LEFT, padx=20)
        self.logger.info("Top control bar created with load button and progress label")

        # Main comparison area - holds left and right file panels
        compare = ttk.Frame(self.root)
        # Expand to fill available space in both directions
        compare.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left file panel
        left = ttk.LabelFrame(compare, text="Left File")
        # Fill vertically and horizontally, expand to take available space
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Prominent filename label - bold, wrapped text for long names
        self.left_name = ttk.Label(
            left, text="", font=("Arial", 10, "bold"), wraplength=350, justify=tk.CENTER
        )
        self.left_name.pack(pady=5)
        # File size label below filename
        self.left_size = ttk.Label(left, text="", font=("Arial", 9))
        self.left_size.pack()

        # Text widget for displaying detailed metadata (multiline)
        self.left_info = tk.Text(left, height=10, wrap=tk.WORD)
        self.left_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Label for showing image preview thumbnail
        self.left_preview = ttk.Label(left)
        self.left_preview.pack(pady=10)
        # Button to open file in system default application
        ttk.Button(left, text="Open File", command=lambda: self.open_file("left")).pack(
            pady=5
        )
        self.logger.info("Left file panel created with metadata display and preview")

        # Right file panel - mirror of left panel
        right = ttk.LabelFrame(compare, text="Right File")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Same structure as left panel
        self.right_name = ttk.Label(
            right,
            text="",
            font=("Arial", 10, "bold"),
            wraplength=350,
            justify=tk.CENTER,
        )
        self.right_name.pack(pady=5)
        self.right_size = ttk.Label(right, text="", font=("Arial", 9))
        self.right_size.pack()

        self.right_info = tk.Text(right, height=10, wrap=tk.WORD)
        self.right_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.right_preview = ttk.Label(right)
        self.right_preview.pack(pady=10)
        ttk.Button(
            right, text="Open File", command=lambda: self.open_file("right")
        ).pack(pady=5)
        self.logger.info("Right file panel created with metadata display and preview")

        # Action buttons panel - bottom of window
        actions = ttk.Frame(self.root)
        actions.pack(fill=tk.X, padx=10, pady=10)

        # Three main action buttons for user decisions
        ttk.Button(
            actions, text="← Keep Left (Delete Right)", command=self.keep_left, width=30
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            actions,
            text="Keep Right (Delete Left) →",
            command=self.keep_right,
            width=30,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="Keep Both", command=self.keep_both, width=15).pack(
            side=tk.LEFT, padx=5
        )
        self.logger.info(
            "Action buttons created for keep left, keep right, and keep both"
        )

        # Keyboard shortcuts for faster navigation
        # Left arrow = keep left file, Right arrow = keep right file, Up arrow = keep both
        self.root.bind("<Left>", lambda e: self.keep_left())
        self.root.bind("<Right>", lambda e: self.keep_right())
        self.root.bind("<Up>", lambda e: self.keep_both())
        self.logger.info(
            "Keyboard shortcuts bound: Left arrow (keep left), Right arrow (keep right), Up arrow (keep both)"
        )

    def load_json(self):
        """
        Open file dialog to load duplicate pairs from JSON file.

        Expected JSON format: [["file1_path", "file2_path"], ...]
        Each inner array represents a pair of duplicate files.

        After loading, resets the current index and displays the first pair.
        Logs statistics about loaded pairs including file type breakdown.
        """
        self.logger.info("Load JSON button clicked, opening file dialog")
        load_start = time.time()

        # Open file dialog using tkinter's filedialog module
        # Filters to show only JSON files
        path = filedialog.askopenfilename(
            title="Select JSON file with duplicate pairs",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            self.logger.info("File dialog cancelled by user")
            return

        self.logger.info(f"Loading duplicate pairs from: {path}")

        try:
            # Open and parse JSON file using built-in json module
            with open(path) as f:
                self.pairs = json.load(f)

            # Update statistics
            self.stats["pairs_loaded"] = len(self.pairs)
            self.logger.info(
                f"Successfully loaded {len(self.pairs)} duplicate pairs from JSON"
            )

            # Analyze file types in loaded pairs
            for pair in self.pairs:
                for file_path in pair:
                    # Extract file extension and convert to lowercase
                    ext = Path(file_path).suffix.lower()
                    # Increment counter for this file type
                    self.stats["file_types"][ext] += 1

            # Log file type breakdown
            self.logger.info(f"File type breakdown: {dict(self.stats['file_types'])}")

            # Reset to first pair
            self.current_index = 0

            # Display the first pair in the UI
            self.show_pair()

            load_duration = time.time() - load_start
            self.logger.info(
                f"JSON loading and initial display completed in {load_duration:.3f} seconds"
            )

        except json.JSONDecodeError as e:
            # Handle invalid JSON format
            self.logger.error(f"Failed to parse JSON file: {e}")
            messagebox.showerror("JSON Error", f"Invalid JSON format: {e}")
        except Exception as e:
            # Handle other errors (file not found, permissions, etc.)
            self.logger.error(f"Error loading JSON file: {e}")
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def get_metadata(self, filepath):
        """
        Extract metadata from a file including size, modification time, and format-specific info.

        For PDF files, this extracts:
        - PDF metadata fields (Title, Author, Subject, etc.)
        - Page count

        Args:
                        filepath: String path to the file to analyze

        Returns:
                        Dictionary containing metadata key-value pairs
        """
        meta = {}
        metadata_start = time.time()

        try:
            # Use os.stat() to get file system information
            stat = os.stat(filepath)
            # Convert bytes to megabytes with 2 decimal places
            size_bytes = stat.st_size
            meta["Size"] = f"{size_bytes / (1024*1024):.2f} MB"
            # Get modification timestamp from Path.stat()
            meta["Modified"] = Path(filepath).stat().st_mtime

            # Special handling for PDF files
            if filepath.lower().endswith(".pdf"):
                self.logger.debug(f"Extracting PDF metadata from: {filepath}")
                # Use pypdf.PdfReader to open and read PDF
                pdf = PdfReader(filepath)
                # Check if PDF has metadata dictionary
                if pdf.metadata:
                    # Iterate through metadata items
                    for key, val in pdf.metadata.items():
                        if val:
                            # Strip leading "/" from PDF metadata keys
                            meta[key.strip("/")] = str(val)
                # Count pages using len() on pages list
                meta["Pages"] = len(pdf.pages)
                self.logger.debug(f"PDF has {len(pdf.pages)} pages")

        except Exception as e:
            # Log any errors but don't crash - store error in metadata
            self.logger.warning(f"Error extracting metadata from {filepath}: {e}")
            meta["Error"] = str(e)

        metadata_duration = time.time() - metadata_start
        self.logger.debug(
            f"Metadata extraction took {metadata_duration:.4f} seconds for: {filepath}"
        )

        return meta

    def show_pair(self):
        """
        Display the current pair of duplicate files in the UI.

        Updates:
        - File names and sizes in prominent labels
        - Detailed metadata in text widgets
        - Image previews if applicable
        - Progress counter

        If all pairs have been reviewed, shows completion message and logs statistics.
        """
        show_start = time.time()

        # Check if we've reached the end of the pairs list
        if self.current_index >= len(self.pairs):
            self.logger.info("All pairs have been reviewed")
            self.log_session_statistics()
            messagebox.showinfo("Done", "All pairs reviewed!")
            return

        # Get current pair from the list
        pair = self.pairs[self.current_index]
        self.left_file, self.right_file = pair[0], pair[1]

        self.logger.info(f"Displaying pair {self.current_index + 1}/{len(self.pairs)}")
        self.logger.debug(f"Left file: {self.left_file}")
        self.logger.debug(f"Right file: {self.right_file}")

        # Update progress label with current position
        self.progress.config(text=f"Pair {self.current_index + 1}/{len(self.pairs)}")

        # Process both files: left then right
        # Iterate through both files with their associated UI widgets
        for side, path, name_label, size_label, widget in [
            ("left", self.left_file, self.left_name, self.left_size, self.left_info),
            (
                "right",
                self.right_file,
                self.right_name,
                self.right_size,
                self.right_info,
            ),
        ]:
            # Update prominent filename label (just the filename, not full path)
            name_label.config(text=Path(path).name)

            # Extract metadata using get_metadata method
            meta = self.get_metadata(path)
            # Update size label with formatted size
            size_label.config(text=meta.get("Size", "Unknown size"))

            # Clear existing text and insert new metadata
            # Text widget uses "1.0" to mean line 1, character 0 (beginning)
            widget.delete("1.0", tk.END)
            widget.insert("1.0", f"Full Path: {path}\n\n")

            # Insert all metadata key-value pairs
            for k, v in meta.items():
                widget.insert("end", f"{k}: {v}\n")

        # Show image previews if files are images
        self.show_preview(self.left_file, self.left_preview)
        self.show_preview(self.right_file, self.right_preview)

        show_duration = time.time() - show_start
        self.logger.debug(f"Pair display completed in {show_duration:.3f} seconds")

    def show_preview(self, filepath, label):
        """
        Display a thumbnail preview of an image file.

        Supports common image formats: jpg, jpeg, png, gif, bmp, webp
        For non-image files, displays the file extension.

        Args:
                        filepath: Path to the file to preview
                        label: tkinter Label widget to display the preview in
        """
        # Clear any existing image or text
        label.config(image="", text="")

        # Get file extension in lowercase
        ext = Path(filepath).suffix.lower()

        # Check if file is a supported image format
        if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            try:
                self.logger.debug(f"Generating image preview for: {filepath}")
                # Use PIL Image.open() to load image
                img = Image.open(filepath)
                # Resize to fit in 300x300 box while maintaining aspect ratio
                img.thumbnail((300, 300))
                # Convert PIL Image to PhotoImage for tkinter display
                photo = ImageTk.PhotoImage(img)
                # Set image in label and keep reference to prevent garbage collection
                label.config(image=photo)
                label.image = photo  # Keep reference
                self.logger.debug(f"Preview generated successfully for: {filepath}")
            except Exception as e:
                # If preview fails, show error message
                self.logger.warning(f"Failed to generate preview for {filepath}: {e}")
                label.config(text="Preview unavailable")
        else:
            # For non-image files, just show the file type
            label.config(text=f"{ext.upper()} file")

    def open_file(self, side):
        """
        Open a file in the system's default application.

        Uses platform-specific commands:
        - Windows: os.startfile()
        - macOS: open command
        - Linux: xdg-open command

        Args:
                        side: Either "left" or "right" to indicate which file to open
        """
        # Select file based on which side was clicked
        path = self.left_file if side == "left" else self.right_file
        self.logger.info(f"Opening {side} file in system viewer: {path}")

        try:
            # Windows-specific file opening
            if os.name == "nt":
                os.startfile(path)
            # Unix-like systems (macOS and Linux)
            elif os.name == "posix":
                # Check if macOS (Darwin) and use "open", otherwise use "xdg-open"
                os.system(
                    f'open "{path}"'
                    if os.uname().sysname == "Darwin"
                    else f'xdg-open "{path}"'
                )
            self.logger.info(f"File opened successfully: {path}")
        except Exception as e:
            self.logger.error(f"Failed to open file {path}: {e}")
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def keep_left(self):
        """
        User chose to keep the left file and delete the right file.

        Prompts for confirmation, then deletes the right file and logs the decision.
        Updates statistics and moves to the next pair.
        """
        self.logger.info(f"User selected to keep left file: {self.left_file}")
        self.logger.info(f"Will delete right file: {self.right_file}")

        # Show confirmation dialog using messagebox.askyesno()
        if messagebox.askyesno("Confirm", f"Delete:\n{self.right_file}?"):
            delete_start = time.time()
            try:
                # Get file size before deletion for statistics
                file_size = os.path.getsize(self.right_file)

                # Delete file using os.remove()
                os.remove(self.right_file)

                delete_duration = time.time() - delete_start
                self.logger.info(
                    f"Successfully deleted right file in {delete_duration:.3f} seconds: {self.right_file}"
                )

                # Record decision in log
                self.log.append({"kept": self.left_file, "deleted": self.right_file})

                # Update statistics
                self.stats["pairs_reviewed"] += 1
                self.stats["files_deleted"] += 1
                self.stats["files_kept"] += 1
                self.stats["left_kept"] += 1
                self.stats["total_size_deleted"] += file_size
                self.logger.info(
                    f"Deleted {file_size / (1024*1024):.2f} MB, cumulative deleted: {self.stats['total_size_deleted'] / (1024*1024):.2f} MB"
                )

                # Move to next pair
                self.next_pair()
            except Exception as e:
                # Handle deletion errors
                self.logger.error(f"Error deleting file {self.right_file}: {e}")
                messagebox.showerror("Error", str(e))
        else:
            self.logger.info("User cancelled deletion of right file")

    def keep_right(self):
        """
        User chose to keep the right file and delete the left file.

        Prompts for confirmation, then deletes the left file and logs the decision.
        Updates statistics and moves to the next pair.
        """
        self.logger.info(f"User selected to keep right file: {self.right_file}")
        self.logger.info(f"Will delete left file: {self.left_file}")

        # Show confirmation dialog
        if messagebox.askyesno("Confirm", f"Delete:\n{self.left_file}?"):
            delete_start = time.time()
            try:
                # Get file size before deletion for statistics
                file_size = os.path.getsize(self.left_file)

                # Delete file
                os.remove(self.left_file)

                delete_duration = time.time() - delete_start
                self.logger.info(
                    f"Successfully deleted left file in {delete_duration:.3f} seconds: {self.left_file}"
                )

                # Record decision
                self.log.append({"kept": self.right_file, "deleted": self.left_file})

                # Update statistics
                self.stats["pairs_reviewed"] += 1
                self.stats["files_deleted"] += 1
                self.stats["files_kept"] += 1
                self.stats["right_kept"] += 1
                self.stats["total_size_deleted"] += file_size
                self.logger.info(
                    f"Deleted {file_size / (1024*1024):.2f} MB, cumulative deleted: {self.stats['total_size_deleted'] / (1024*1024):.2f} MB"
                )

                # Move to next pair
                self.next_pair()
            except Exception as e:
                # Handle deletion errors
                self.logger.error(f"Error deleting file {self.left_file}: {e}")
                messagebox.showerror("Error", str(e))
        else:
            self.logger.info("User cancelled deletion of left file")

    def keep_both(self):
        """
        User chose to keep both files without deleting either.

        Logs the decision and moves to the next pair.
        """
        self.logger.info(
            f"User selected to keep both files: {self.left_file} and {self.right_file}"
        )

        # Record decision to keep both
        self.log.append({"kept": "both", "files": [self.left_file, self.right_file]})

        # Update statistics
        self.stats["pairs_reviewed"] += 1
        self.stats["both_kept"] += 1
        self.stats["files_kept"] += 2

        # Move to next pair
        self.next_pair()

    def next_pair(self):
        """
        Advance to the next pair of duplicate files.

        If more pairs remain, displays the next pair.
        If all pairs have been reviewed, saves the decision log and shows completion stats.
        """
        # Increment index to move to next pair
        self.current_index += 1
        self.logger.debug(f"Moving to next pair, new index: {self.current_index}")

        # Check if more pairs remain
        if self.current_index < len(self.pairs):
            # Display next pair
            self.show_pair()
        else:
            # All pairs reviewed - save log and show statistics
            self.logger.info("All pairs have been reviewed, saving decision log")

            # Write decisions to JSON file using json.dump()
            log_path = "decisions.json"
            with open(log_path, "w") as f:
                # indent=2 makes the JSON human-readable
                json.dump(self.log, f, indent=2)
            self.logger.info(f"Decision log saved to: {log_path}")

            # Log final session statistics
            self.log_session_statistics()

            # Show completion message
            messagebox.showinfo(
                "Complete", "All pairs reviewed!\nLog saved to decisions.json"
            )

    def log_session_statistics(self):
        """
        Log comprehensive statistics about the review session.

        Includes:
        - Total session duration
        - Pairs loaded and reviewed
        - Files deleted vs kept
        - Total space freed
        - File type breakdown
        - Decision breakdown (left vs right vs both)
        """
        # Calculate total session time
        session_duration = time.time() - self.session_start_time

        self.logger.info("Session Statistics Summary")
        self.logger.info(
            f"Total session duration: {session_duration:.2f} seconds ({session_duration/60:.2f} minutes)"
        )
        self.logger.info(f"Pairs loaded: {self.stats['pairs_loaded']}")
        self.logger.info(f"Pairs reviewed: {self.stats['pairs_reviewed']}")
        self.logger.info(f"Files deleted: {self.stats['files_deleted']}")
        self.logger.info(f"Files kept: {self.stats['files_kept']}")
        self.logger.info(f"Pairs where both kept: {self.stats['both_kept']}")
        self.logger.info(f"Times left file chosen: {self.stats['left_kept']}")
        self.logger.info(f"Times right file chosen: {self.stats['right_kept']}")
        self.logger.info(
            f"Total storage freed: {self.stats['total_size_deleted'] / (1024*1024):.2f} MB"
        )

        # Log file type breakdown
        if self.stats["file_types"]:
            self.logger.info("File types encountered:")
            for ext, count in self.stats["file_types"].most_common():
                self.logger.info(f"  {ext if ext else '(no extension)'}: {count} files")

        # Calculate review rate if any pairs were reviewed
        if self.stats["pairs_reviewed"] > 0:
            avg_time_per_pair = session_duration / self.stats["pairs_reviewed"]
            self.logger.info(f"Average time per pair: {avg_time_per_pair:.2f} seconds")

    def run(self):
        """
        Start the GUI application main event loop.

        This method blocks until the window is closed.
        Uses tkinter's mainloop() to process events and user interactions.
        """
        self.logger.info("Starting application main loop")
        # Enter tkinter main event loop - this blocks until window is closed
        self.root.mainloop()
        self.logger.info("Application closed by user")
