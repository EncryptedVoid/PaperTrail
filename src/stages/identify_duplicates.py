#!/usr/bin/env python3
"""
Simple Duplicate File Reviewer
Reads duplicate pairs, shows metadata side-by-side, lets you choose which to delete
"""
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from pypdf import PdfReader

try:
    import customtkinter as ctk
    from customtkinter import CTkScrollableFrame

    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    import tkinter as ctk

    CTkScrollableFrame = tk.Frame

try:
    import pikepdf

    PIKEPDF_AVAILABLE = True
except ImportError:
    PIKEPDF_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    import fitz  # PyMuPDF

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pygame

    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

FONT_FAMILY = "Segoe UI"
try:
    import tkinter.font as tkfont

    root_test = tk.Tk()
    available_fonts = tkfont.families()
    root_test.destroy()
    if "Segoe UI" not in available_fonts:
        FONT_FAMILY = "Arial" if "Arial" in available_fonts else "TkDefaultFont"
except:
    FONT_FAMILY = "Arial"


class DuplicateReviewer:
    def __init__(self, root):
        self.root = ctk.CTk()
        self.root.title("Duplicate File Reviewer")
        self.root.geometry("1200x700")

        self.pairs = []
        self.current_index = 0
        self.log = []

        self.setup_ui()

    def setup_ui(self):
        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(top, text="Load JSON Pairs", command=self.load_json).pack(
            side=tk.LEFT, padx=5
        )
        self.progress = ttk.Label(top, text="No files loaded")
        self.progress.pack(side=tk.LEFT, padx=20)

        # Main comparison area
        compare = ttk.Frame(self.root)
        compare.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left file
        left = ttk.LabelFrame(compare, text="Left File")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Prominent filename and size labels
        self.left_name = ttk.Label(
            left, text="", font=("Arial", 10, "bold"), wraplength=350, justify=tk.CENTER
        )
        self.left_name.pack(pady=5)
        self.left_size = ttk.Label(left, text="", font=("Arial", 9))
        self.left_size.pack()

        self.left_info = tk.Text(left, height=10, wrap=tk.WORD)
        self.left_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.left_preview = ttk.Label(left)
        self.left_preview.pack(pady=10)
        ttk.Button(left, text="Open File", command=lambda: self.open_file("left")).pack(
            pady=5
        )

        # Right file
        right = ttk.LabelFrame(compare, text="Right File")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Prominent filename and size labels
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

        # Action buttons
        actions = ttk.Frame(self.root)
        actions.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            actions, text="⬅ Keep Left (Delete Right)", command=self.keep_left, width=30
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            actions,
            text="Keep Right (Delete Left) ➡",
            command=self.keep_right,
            width=30,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="Keep Both", command=self.keep_both, width=15).pack(
            side=tk.LEFT, padx=5
        )

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.keep_left())
        self.root.bind("<Right>", lambda e: self.keep_right())
        self.root.bind("<Up>", lambda e: self.keep_both())

    def load_json(self):
        """Load duplicate pairs from JSON: [["file1", "file2"], ...]"""
        path = filedialog.askopenfilename(
            title="Select JSON file with duplicate pairs",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return

        with open(path) as f:
            self.pairs = json.load(f)

        self.current_index = 0
        self.show_pair()

    def get_metadata(self, filepath):
        """Extract metadata - simple and fast"""
        meta = {}
        try:
            stat = os.stat(filepath)
            meta["Size"] = f"{stat.st_size / (1024*1024):.2f} MB"
            meta["Modified"] = Path(filepath).stat().st_mtime

            # PDF metadata with pypdf
            if filepath.lower().endswith(".pdf"):
                pdf = PdfReader(filepath)
                if pdf.metadata:
                    for key, val in pdf.metadata.items():
                        if val:
                            meta[key.strip("/")] = str(val)
                meta["Pages"] = len(pdf.pages)
        except Exception as e:
            meta["Error"] = str(e)

        return meta

    def show_pair(self):
        """Display current pair"""
        if self.current_index >= len(self.pairs):
            messagebox.showinfo("Done", "All pairs reviewed!")
            return

        pair = self.pairs[self.current_index]
        self.left_file, self.right_file = pair[0], pair[1]

        self.progress.config(text=f"Pair {self.current_index + 1}/{len(self.pairs)}")

        # Show metadata
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
            # Update prominent name and size labels
            name_label.config(text=Path(path).name)

            meta = self.get_metadata(path)
            size_label.config(text=meta.get("Size", "Unknown size"))

            # Update text widget with full details
            widget.delete("1.0", tk.END)
            widget.insert("1.0", f"Full Path: {path}\n\n")

            for k, v in meta.items():
                widget.insert("end", f"{k}: {v}\n")

        # Show image previews if applicable
        self.show_preview(self.left_file, self.left_preview)
        self.show_preview(self.right_file, self.right_preview)

    def show_preview(self, filepath, label):
        """Show image preview if it's an image"""
        label.config(image="", text="")

        ext = Path(filepath).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            try:
                img = Image.open(filepath)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo)
                label.image = photo
            except:
                label.config(text="Preview unavailable")
        else:
            label.config(text=f"{ext.upper()} file")

    def open_file(self, side):
        """Open file in system default viewer"""
        path = self.left_file if side == "left" else self.right_file
        if os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            os.system(
                f'open "{path}"'
                if os.uname().sysname == "Darwin"
                else f'xdg-open "{path}"'
            )

    def keep_left(self):
        """Delete right, keep left"""
        if messagebox.askyesno("Confirm", f"Delete:\n{self.right_file}?"):
            try:
                os.remove(self.right_file)
                self.log.append({"kept": self.left_file, "deleted": self.right_file})
                self.next_pair()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def keep_right(self):
        """Delete left, keep right"""
        if messagebox.askyesno("Confirm", f"Delete:\n{self.left_file}?"):
            try:
                os.remove(self.left_file)
                self.log.append({"kept": self.right_file, "deleted": self.left_file})
                self.next_pair()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def keep_both(self):
        """Keep both files"""
        self.log.append({"kept": "both", "files": [self.left_file, self.right_file]})
        self.next_pair()

    def next_pair(self):
        """Move to next pair"""
        self.current_index += 1
        if self.current_index < len(self.pairs):
            self.show_pair()
        else:
            with open("decisions.json", "w") as f:
                json.dump(self.log, f, indent=2)
            messagebox.showinfo(
                "Complete", "All pairs reviewed!\nLog saved to decisions.json"
            )

    def run(self):
        self.root.mainloop()
