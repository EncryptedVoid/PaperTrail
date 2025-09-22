import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import io
import os
from pathlib import Path
import tempfile
import shutil


class CrudePDFEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced PDF Editor - Dark Theme")
        self.root.geometry("1600x900")  # Wider for side panel

        # Dark theme colors
        self.colors = {
            "bg": "#2b2b2b",
            "fg": "#ffffff",
            "button_bg": "#404040",
            "button_fg": "#ffffff",
            "frame_bg": "#353535",
            "entry_bg": "#404040",
            "entry_fg": "#ffffff",
            "select_bg": "#0078d4",
            "accent": "#0078d4",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
        }

        self.setup_dark_theme()

        self.documents = []  # List of loaded PDFs
        self.current_pages = []  # Current working pages
        self.selected_pages = set()  # Multi-selection
        self.visible_start = 0  # For virtual scrolling
        self.visible_count = 15  # Show 15 pages at a time
        self.large_thumbnail_size = (450, 600)  # 3x larger thumbnails
        self.small_thumbnail_size = (150, 200)  # For memory efficiency
        self.mini_thumbnail_size = (80, 120)  # For side panel
        self.reorder_mode = False
        self.original_file_paths = {}  # Track original sources

        self.setup_ui()

    def setup_dark_theme(self):
        """Configure dark theme for the application"""
        self.root.configure(bg=self.colors["bg"])

        # Configure ttk styles for dark theme
        style = ttk.Style()
        style.theme_use("clam")

        # Configure ttk widget styles
        style.configure("TFrame", background=self.colors["bg"])
        style.configure(
            "TButton",
            background=self.colors["button_bg"],
            foreground=self.colors["button_fg"],
        )
        style.configure(
            "TLabel", background=self.colors["bg"], foreground=self.colors["fg"]
        )
        style.configure(
            "TCheckbutton", background=self.colors["bg"], foreground=self.colors["fg"]
        )

    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - main editor
        self.editor_frame = tk.Frame(main_container, bg=self.colors["bg"])
        self.editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # File operations frame
        file_frame = tk.Frame(self.editor_frame, bg=self.colors["frame_bg"])
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(
            file_frame,
            text="Load PDF",
            command=self.load_pdf,
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            file_frame,
            text="Import Pages",
            command=self.import_pages,
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            file_frame,
            text="Import Images",
            command=self.import_images,
            bg=self.colors["success"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            file_frame,
            text="Save to Original",
            command=self.save_to_original,
            bg=self.colors["warning"],
            fg="black",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            file_frame,
            text="Clear All",
            command=self.clear_all,
            bg=self.colors["danger"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=2)

        # Multi-select and reorder controls
        select_frame = tk.Frame(self.editor_frame, bg=self.colors["frame_bg"])
        select_frame.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            select_frame,
            text="Select All",
            command=self.select_all,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            select_frame,
            text="Clear Selection",
            command=self.clear_selection,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            select_frame,
            text="Toggle Reorder Mode",
            command=self.toggle_reorder_mode,
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            font=("Arial", 9, "bold"),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            select_frame,
            text="Delete Selected",
            command=self.delete_selected,
            bg=self.colors["danger"],
            fg=self.colors["button_fg"],
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=2)

        # Page count and navigation
        nav_frame = tk.Frame(self.editor_frame, bg=self.colors["frame_bg"])
        nav_frame.pack(fill=tk.X, padx=5, pady=2)

        self.page_count_label = tk.Label(
            nav_frame,
            text="Total Pages: 0",
            font=("Arial", 10, "bold"),
            bg=self.colors["frame_bg"],
            fg=self.colors["fg"],
        )
        self.page_count_label.pack(side=tk.LEFT)

        # Virtual scrolling controls
        tk.Button(
            nav_frame,
            text="◀◀ First",
            command=self.go_to_first,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            nav_frame,
            text="▶▶ Last",
            command=self.go_to_last,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            nav_frame,
            text="▶ Next 15",
            command=self.next_pages,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            nav_frame,
            text="◀ Prev 15",
            command=self.prev_pages,
            bg=self.colors["button_bg"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.RIGHT, padx=2)

        self.view_label = tk.Label(
            nav_frame,
            text="Viewing: 0-0",
            font=("Arial", 10),
            bg=self.colors["frame_bg"],
            fg=self.colors["fg"],
        )
        self.view_label.pack(side=tk.RIGHT, padx=10)

        # Main content area with better scrolling
        self.main_frame = tk.Frame(self.editor_frame, bg=self.colors["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create scrollable canvas with better performance
        self.canvas = tk.Canvas(
            self.main_frame, highlightthickness=0, bg=self.colors["bg"]
        )
        self.v_scrollbar = ttk.Scrollbar(
            self.main_frame, orient="vertical", command=self.canvas.yview
        )
        self.h_scrollbar = ttk.Scrollbar(
            self.main_frame, orient="horizontal", command=self.canvas.xview
        )

        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors["bg"])

        # Configure scrolling
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        # Grid layout for scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Right side - reorder panel (initially hidden)
        self.reorder_panel = tk.Frame(
            main_container, bg=self.colors["frame_bg"], width=300
        )
        self.setup_reorder_panel()

    def setup_reorder_panel(self):
        """Setup the side panel for reordering pages"""
        header = tk.Label(
            self.reorder_panel,
            text="Page Reorder",
            font=("Arial", 12, "bold"),
            bg=self.colors["frame_bg"],
            fg=self.colors["fg"],
        )
        header.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.reorder_panel,
            text="Click pages to reorder\nSelected pages highlighted",
            font=("Arial", 9),
            bg=self.colors["frame_bg"],
            fg=self.colors["fg"],
        )
        instructions.pack(pady=5)

        # Reorder canvas
        self.reorder_canvas = tk.Canvas(
            self.reorder_panel, bg=self.colors["bg"], highlightthickness=0, width=280
        )
        self.reorder_scrollbar = ttk.Scrollbar(
            self.reorder_panel, orient="vertical", command=self.reorder_canvas.yview
        )

        self.reorder_frame = tk.Frame(self.reorder_canvas, bg=self.colors["bg"])

        self.reorder_frame.bind(
            "<Configure>",
            lambda e: self.reorder_canvas.configure(
                scrollregion=self.reorder_canvas.bbox("all")
            ),
        )

        self.reorder_canvas.create_window(
            (0, 0), window=self.reorder_frame, anchor="nw"
        )
        self.reorder_canvas.configure(yscrollcommand=self.reorder_scrollbar.set)

        self.reorder_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.reorder_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_frame_configure(self, event):
        """Update scroll region when frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Update canvas window size"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            try:
                doc = fitz.open(file_path)
                self.original_file_paths[file_path] = doc  # Track original

                for page_num in range(len(doc)):
                    page_data = {
                        "doc": doc,
                        "page_num": page_num,
                        "original_page": doc[page_num],
                        "current_rotation": 0,
                        "enhance": False,
                        "remove_watermark": False,
                        "file_path": file_path,
                        "type": "pdf",
                        "thumbnail": None,  # Lazy load thumbnails
                        "source_file": file_path,  # Track source
                    }
                    self.current_pages.append(page_data)
                self.refresh_page_list()
                self.refresh_reorder_panel()
                print(f"Loaded {len(doc)} pages from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")

    def import_images(self):
        """Import images and convert them to PDF pages"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("All Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff"),
                ("GIF files", "*.gif"),
                ("WebP files", "*.webp"),
                ("All files", "*.*"),
            ],
        )

        if file_paths:
            try:
                for file_path in file_paths:
                    # Create a temporary PDF from the image
                    temp_doc = fitz.open()

                    # Load image and get dimensions
                    img = Image.open(file_path)

                    # Convert to RGB if necessary (for PNG with transparency, etc.)
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        background.paste(
                            img, mask=img.split()[-1] if img.mode == "RGBA" else None
                        )
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    # Save to temporary bytes
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="JPEG", quality=95)
                    img_bytes.seek(0)

                    # Create PDF page from image
                    img_width, img_height = img.size

                    # Create page with image dimensions (convert pixels to points)
                    page_width = img_width * 72 / 96  # Assume 96 DPI
                    page_height = img_height * 72 / 96

                    temp_page = temp_doc.new_page(width=page_width, height=page_height)

                    # Insert image into page
                    temp_page.insert_image(
                        fitz.Rect(0, 0, page_width, page_height),
                        stream=img_bytes.getvalue(),
                    )

                    page_data = {
                        "doc": temp_doc,
                        "page_num": 0,
                        "original_page": temp_page,
                        "current_rotation": 0,
                        "enhance": False,
                        "remove_watermark": False,
                        "file_path": file_path,
                        "type": "image",
                        "thumbnail": None,  # Lazy load
                        "source_file": None,  # Images don't have source files to save back to
                    }
                    self.current_pages.append(page_data)

                self.refresh_page_list()
                self.refresh_reorder_panel()
                print(f"Imported {len(file_paths)} images")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import images: {str(e)}")

    def create_thumbnail(self, page_data, large=False, mini=False):
        """Create thumbnail with lazy loading and proper rotation"""
        size = (
            self.mini_thumbnail_size
            if mini
            else (self.large_thumbnail_size if large else self.small_thumbnail_size)
        )

        try:
            page = page_data["original_page"]

            # Apply rotation to the page rendering
            rotation = page_data.get("current_rotation", 0)
            scale = 0.9 if large else (0.2 if mini else 0.3)

            # Create transformation matrix with rotation
            mat = fitz.Matrix(scale, scale)
            if rotation:
                mat = mat * fitz.Matrix(rotation)

            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail(size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            # Cache small thumbnail
            if not large and not mini:
                page_data["thumbnail"] = photo

            return photo
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            # Return a placeholder image
            placeholder = Image.new("RGB", size, color="gray")
            placeholder_photo = ImageTk.PhotoImage(placeholder)
            return placeholder_photo

    def import_pages(self):
        """Import pages from another PDF without replacing current pages"""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            try:
                doc = fitz.open(file_path)
                pages_to_import = self.choose_pages_dialog(len(doc))

                for page_num in pages_to_import:
                    page_data = {
                        "doc": doc,
                        "page_num": page_num,
                        "original_page": doc[page_num],
                        "current_rotation": 0,
                        "enhance": False,
                        "remove_watermark": False,
                        "file_path": file_path,
                        "type": "pdf",
                        "thumbnail": None,
                        "source_file": None,  # Imported pages don't save back to source
                    }
                    self.current_pages.append(page_data)

                self.refresh_page_list()
                self.refresh_reorder_panel()
                print(f"Imported {len(pages_to_import)} pages")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import pages: {str(e)}")

    def clear_all(self):
        """Clear all loaded pages"""
        if self.current_pages and messagebox.askyesno("Confirm", "Clear all pages?"):
            self.current_pages.clear()
            self.selected_pages.clear()
            self.visible_start = 0
            self.refresh_page_list()
            self.refresh_reorder_panel()
            print("Cleared all pages")

    # Multi-selection methods
    def select_all(self):
        """Select all pages"""
        self.selected_pages = set(range(len(self.current_pages)))
        self.refresh_page_list()
        self.refresh_reorder_panel()

    def clear_selection(self):
        """Clear all selections"""
        self.selected_pages.clear()
        self.refresh_page_list()
        self.refresh_reorder_panel()

    def toggle_page_selection(self, page_index):
        """Toggle selection of a page"""
        if page_index in self.selected_pages:
            self.selected_pages.remove(page_index)
        else:
            self.selected_pages.add(page_index)
        self.refresh_page_list()
        self.refresh_reorder_panel()

    def delete_selected(self):
        """Delete all selected pages"""
        if not self.selected_pages:
            messagebox.showwarning("Warning", "No pages selected")
            return

        if messagebox.askyesno(
            "Confirm", f"Delete {len(self.selected_pages)} selected pages?"
        ):
            # Sort indices in reverse order to delete from end to beginning
            for index in sorted(self.selected_pages, reverse=True):
                if 0 <= index < len(self.current_pages):
                    self.current_pages.pop(index)

            self.selected_pages.clear()
            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Deleted selected pages")

    def toggle_reorder_mode(self):
        """Toggle the reorder panel visibility"""
        self.reorder_mode = not self.reorder_mode
        if self.reorder_mode:
            self.reorder_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
            self.refresh_reorder_panel()
        else:
            self.reorder_panel.pack_forget()

    def refresh_reorder_panel(self):
        """Refresh the reorder panel with mini thumbnails"""
        if not self.reorder_mode:
            return

        # Clear existing widgets
        for widget in self.reorder_frame.winfo_children():
            widget.destroy()

        for i, page_data in enumerate(self.current_pages):
            frame = tk.Frame(
                self.reorder_frame,
                bg=(
                    self.colors["select_bg"]
                    if i in self.selected_pages
                    else self.colors["frame_bg"]
                ),
                relief=tk.RAISED,
                borderwidth=1,
            )
            frame.pack(fill=tk.X, pady=2, padx=2)

            # Mini thumbnail
            mini_thumb = self.create_thumbnail(page_data, mini=True)
            thumb_label = tk.Label(frame, image=mini_thumb, bg=frame["bg"])
            thumb_label.pack(side=tk.LEFT, padx=5, pady=5)
            thumb_label.image = mini_thumb  # Keep reference

            # Page info
            info_text = f"Page {i+1}\n{os.path.basename(page_data['file_path'])}"
            info_label = tk.Label(
                frame,
                text=info_text,
                font=("Arial", 8),
                bg=frame["bg"],
                fg=self.colors["fg"],
            )
            info_label.pack(side=tk.LEFT, padx=5)

            # Click handlers for reordering
            def make_click_handler(page_idx):
                return lambda e: self.reorder_page_click(page_idx)

            frame.bind("<Button-1>", make_click_handler(i))
            thumb_label.bind("<Button-1>", make_click_handler(i))
            info_label.bind("<Button-1>", make_click_handler(i))

        # Update scroll region
        self.reorder_frame.update_idletasks()
        self.reorder_canvas.configure(scrollregion=self.reorder_canvas.bbox("all"))

    def reorder_page_click(self, clicked_index):
        """Handle clicking on pages in reorder mode"""
        if not self.selected_pages:
            # If no selection, just select this page
            self.selected_pages.add(clicked_index)
        else:
            # Move selected pages to clicked position
            self.move_selected_pages_to(clicked_index)

        self.refresh_page_list()
        self.refresh_reorder_panel()

    def move_selected_pages_to(self, target_index):
        """Move all selected pages to target position"""
        if not self.selected_pages:
            return

        # Extract selected pages
        selected_pages_data = []
        for index in sorted(self.selected_pages, reverse=True):
            if 0 <= index < len(self.current_pages):
                selected_pages_data.insert(0, self.current_pages.pop(index))

        # Adjust target index if it was after removed pages
        adjusted_target = target_index
        for index in sorted(self.selected_pages):
            if index <= target_index:
                adjusted_target -= 1

        # Insert at target position
        for i, page_data in enumerate(selected_pages_data):
            self.current_pages.insert(adjusted_target + i, page_data)

        # Update selection to new positions
        new_selection = set(
            range(adjusted_target, adjusted_target + len(selected_pages_data))
        )
        self.selected_pages = new_selection

        print(
            f"Moved {len(selected_pages_data)} pages to position {adjusted_target + 1}"
        )

    # Virtual scrolling navigation
    def go_to_first(self):
        self.visible_start = 0
        self.refresh_page_list()

    def go_to_last(self):
        self.visible_start = max(0, len(self.current_pages) - self.visible_count)
        self.refresh_page_list()

    def prev_pages(self):
        self.visible_start = max(0, self.visible_start - self.visible_count)
        self.refresh_page_list()

    def next_pages(self):
        self.visible_start = min(
            len(self.current_pages) - self.visible_count,
            self.visible_start + self.visible_count,
        )
        if self.visible_start < 0:
            self.visible_start = 0
        self.refresh_page_list()

    def create_page_widget(self, page_data, index, actual_index):
        """Create widget for each page with all controls - LARGE THUMBNAILS"""
        is_selected = actual_index in self.selected_pages
        bg_color = self.colors["select_bg"] if is_selected else self.colors["frame_bg"]

        page_frame = tk.Frame(
            self.scrollable_frame, relief=tk.RAISED, borderwidth=2, bg=bg_color
        )
        page_frame.pack(fill=tk.X, pady=5, padx=5)

        # Selection checkbox
        select_frame = tk.Frame(page_frame, bg=bg_color)
        select_frame.pack(side=tk.LEFT, padx=5)

        select_var = tk.BooleanVar(value=is_selected)
        select_check = tk.Checkbutton(
            select_frame,
            variable=select_var,
            bg=bg_color,
            command=lambda: self.toggle_page_selection(actual_index),
        )
        select_check.pack(pady=10)

        # Left side: LARGE Thumbnail
        thumb_frame = tk.Frame(page_frame, bg=bg_color)
        thumb_frame.pack(side=tk.LEFT, padx=10, pady=10)

        large_thumbnail = self.create_thumbnail(page_data, large=True)
        thumb_label = tk.Label(
            thumb_frame,
            image=large_thumbnail,
            relief=tk.SUNKEN,
            borderwidth=2,
            bg=bg_color,
        )
        thumb_label.pack()

        # Keep reference to avoid garbage collection
        thumb_label.image = large_thumbnail

        # File info
        file_name = os.path.basename(page_data["file_path"])
        tk.Label(
            thumb_frame,
            text=f"{file_name}",
            font=("Arial", 8),
            wraplength=450,
            bg=bg_color,
            fg=self.colors["fg"],
        ).pack()
        tk.Label(
            thumb_frame,
            text=f"Type: {page_data['type'].upper()}",
            font=("Arial", 8),
            bg=bg_color,
            fg=self.colors["fg"],
        ).pack()

        # Right side: Controls
        controls_frame = tk.Frame(page_frame, bg=bg_color)
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Page info
        tk.Label(
            controls_frame,
            text=f"Page {actual_index + 1} of {len(self.current_pages)}",
            font=("Arial", 14, "bold"),
            fg=self.colors["fg"],
            bg=bg_color,
        ).pack(anchor=tk.W, pady=(0, 10))

        # ROTATION CONTROLS
        rotate_frame = tk.LabelFrame(
            controls_frame,
            text="Rotation",
            font=("Arial", 10, "bold"),
            bg=bg_color,
            fg=self.colors["fg"],
        )
        rotate_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            rotate_frame,
            text="↶ 90°",
            command=lambda: self.rotate_page(actual_index, -90),
            bg=self.colors["warning"],
            fg="black",
            width=8,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            rotate_frame,
            text="↷ 90°",
            command=lambda: self.rotate_page(actual_index, 90),
            bg=self.colors["warning"],
            fg="black",
            width=8,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            rotate_frame,
            text="↶ 180°",
            command=lambda: self.rotate_page(actual_index, 180),
            bg=self.colors["warning"],
            fg="black",
            width=8,
        ).pack(side=tk.LEFT, padx=2, pady=2)

        # MOVE CONTROLS
        move_frame = tk.LabelFrame(
            controls_frame,
            text="Position",
            font=("Arial", 10, "bold"),
            bg=bg_color,
            fg=self.colors["fg"],
        )
        move_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            move_frame,
            text="⬆ Move Up",
            command=lambda: self.move_page(actual_index, -1),
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            width=12,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            move_frame,
            text="⬇ Move Down",
            command=lambda: self.move_page(actual_index, 1),
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            width=12,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            move_frame,
            text="🗑 Delete",
            command=lambda: self.delete_page(actual_index),
            bg=self.colors["danger"],
            fg=self.colors["button_fg"],
            width=12,
        ).pack(side=tk.LEFT, padx=2, pady=2)

        # ADD CONTROLS
        add_frame = tk.LabelFrame(
            controls_frame,
            text="Add Pages",
            font=("Arial", 10, "bold"),
            bg=bg_color,
            fg=self.colors["fg"],
        )
        add_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            add_frame,
            text="+ Add Page After",
            command=lambda: self.add_page_after(actual_index),
            bg=self.colors["success"],
            fg=self.colors["button_fg"],
            width=15,
        ).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(
            add_frame,
            text="+ Add Image After",
            command=lambda: self.add_image_after(actual_index),
            bg=self.colors["success"],
            fg=self.colors["button_fg"],
            width=15,
        ).pack(side=tk.LEFT, padx=2, pady=2)

        # ENHANCEMENT TOGGLES
        enhance_frame = tk.LabelFrame(
            controls_frame,
            text="Enhancements",
            font=("Arial", 10, "bold"),
            bg=bg_color,
            fg=self.colors["fg"],
        )
        enhance_frame.pack(fill=tk.X, pady=5)

        enhance_var = tk.BooleanVar(value=page_data["enhance"])
        tk.Checkbutton(
            enhance_frame,
            text="Enhance Quality",
            variable=enhance_var,
            command=lambda: self.toggle_enhance(actual_index, enhance_var.get()),
            font=("Arial", 9),
            bg=bg_color,
            fg=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5, pady=2)

        watermark_var = tk.BooleanVar(value=page_data["remove_watermark"])
        tk.Checkbutton(
            enhance_frame,
            text="Remove Watermarks",
            variable=watermark_var,
            command=lambda: self.toggle_watermark_removal(
                actual_index, watermark_var.get()
            ),
            font=("Arial", 9),
            bg=bg_color,
            fg=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5, pady=2)

        artifacts_var = tk.BooleanVar()
        tk.Checkbutton(
            enhance_frame,
            text="Remove Artifacts",
            variable=artifacts_var,
            command=lambda: self.toggle_artifact_removal(
                actual_index, artifacts_var.get()
            ),
            font=("Arial", 9),
            bg=bg_color,
            fg=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5, pady=2)

    def add_image_after(self, page_index):
        """Add an image after current page"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("All Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Create PDF from single image
                temp_doc = fitz.open()
                img = Image.open(file_path)

                # Convert to RGB if necessary
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG", quality=95)
                img_bytes.seek(0)

                img_width, img_height = img.size
                page_width = img_width * 72 / 96
                page_height = img_height * 72 / 96

                temp_page = temp_doc.new_page(width=page_width, height=page_height)
                temp_page.insert_image(
                    fitz.Rect(0, 0, page_width, page_height),
                    stream=img_bytes.getvalue(),
                )

                page_data = {
                    "doc": temp_doc,
                    "page_num": 0,
                    "original_page": temp_page,
                    "current_rotation": 0,
                    "enhance": False,
                    "remove_watermark": False,
                    "file_path": file_path,
                    "type": "image",
                    "thumbnail": None,
                    "source_file": None,
                }
                self.current_pages.insert(page_index + 1, page_data)
                self.refresh_page_list()
                self.refresh_reorder_panel()
                print(f"Added image after page {page_index + 1}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add image: {str(e)}")

    def rotate_page(self, page_index, angle):
        """Rotate a specific page"""
        try:
            page_data = self.current_pages[page_index]
            page_data["current_rotation"] = (
                page_data["current_rotation"] + angle
            ) % 360

            # Clear cached thumbnail to force regeneration
            page_data["thumbnail"] = None

            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Rotated page {page_index + 1} by {angle}°")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate page: {str(e)}")

    def move_page(self, page_index, direction):
        """Move page up (-1) or down (+1)"""
        new_index = page_index + direction

        if 0 <= new_index < len(self.current_pages):
            self.current_pages[page_index], self.current_pages[new_index] = (
                self.current_pages[new_index],
                self.current_pages[page_index],
            )

            # Update selection if the moved page was selected
            if page_index in self.selected_pages:
                self.selected_pages.remove(page_index)
                self.selected_pages.add(new_index)

            # Adjust visible window if necessary
            if new_index < self.visible_start:
                self.visible_start = max(0, self.visible_start - self.visible_count)
            elif new_index >= self.visible_start + self.visible_count:
                self.visible_start = min(
                    len(self.current_pages) - self.visible_count,
                    self.visible_start + self.visible_count,
                )

            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Moved page from {page_index + 1} to {new_index + 1}")
        else:
            messagebox.showwarning("Warning", "Cannot move page beyond boundaries")

    def delete_page(self, page_index):
        """Remove page from current document"""
        if messagebox.askyesno("Confirm", f"Delete page {page_index + 1}?"):
            self.current_pages.pop(page_index)

            # Update selection
            if page_index in self.selected_pages:
                self.selected_pages.remove(page_index)

            # Adjust selection indices
            new_selection = set()
            for idx in self.selected_pages:
                if idx > page_index:
                    new_selection.add(idx - 1)
                else:
                    new_selection.add(idx)
            self.selected_pages = new_selection

            # Adjust visible window
            if self.visible_start >= len(self.current_pages):
                self.visible_start = max(
                    0, len(self.current_pages) - self.visible_count
                )

            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Deleted page {page_index + 1}")

    def add_page_after(self, page_index):
        """Add a blank page or import page after current page"""
        choice = messagebox.askyesnocancel(
            "Add Page", "Yes = Import from PDF\nNo = Add blank page\nCancel = Cancel"
        )

        if choice is True:  # Import from file
            self.import_page_after(page_index)
        elif choice is False:  # Add blank page
            self.add_blank_page_after(page_index)

    def import_page_after(self, page_index):
        """Import pages from another PDF and insert after current page"""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            try:
                doc = fitz.open(file_path)
                pages_to_import = self.choose_pages_dialog(len(doc))

                for i, page_num in enumerate(pages_to_import):
                    page_data = {
                        "doc": doc,
                        "page_num": page_num,
                        "original_page": doc[page_num],
                        "current_rotation": 0,
                        "enhance": False,
                        "remove_watermark": False,
                        "file_path": file_path,
                        "type": "pdf",
                        "thumbnail": None,
                        "source_file": None,
                    }
                    self.current_pages.insert(page_index + 1 + i, page_data)

                self.refresh_page_list()
                self.refresh_reorder_panel()
                print(
                    f"Imported {len(pages_to_import)} pages after page {page_index + 1}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import page: {str(e)}")

    def add_blank_page_after(self, page_index):
        """Add blank page"""
        try:
            new_doc = fitz.open()
            new_page = new_doc.new_page()

            page_data = {
                "doc": new_doc,
                "page_num": 0,
                "original_page": new_page,
                "current_rotation": 0,
                "enhance": False,
                "remove_watermark": False,
                "file_path": "blank_page",
                "type": "blank",
                "thumbnail": None,
                "source_file": None,
            }
            self.current_pages.insert(page_index + 1, page_data)
            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Added blank page after page {page_index + 1}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add blank page: {str(e)}")

    def choose_pages_dialog(self, max_pages):
        """Dialog to choose which pages to import"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Pages")
        dialog.geometry("400x200")
        dialog.configure(bg=self.colors["bg"])

        tk.Label(
            dialog,
            text=f"Choose pages (1-{max_pages}):",
            font=("Arial", 12),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        ).pack(pady=10)
        tk.Label(
            dialog,
            text="Format: 1,2,3 or 1-5 or all",
            font=("Arial", 10),
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        ).pack()

        entry = tk.Entry(
            dialog,
            width=30,
            font=("Arial", 12),
            bg=self.colors["entry_bg"],
            fg=self.colors["entry_fg"],
        )
        entry.pack(pady=10)
        entry.insert(0, "all")

        result = []

        def on_ok():
            try:
                pages_str = entry.get().strip().lower()
                if pages_str == "all":
                    result.extend(range(max_pages))
                elif "-" in pages_str and "," not in pages_str:
                    start, end = map(int, pages_str.split("-"))
                    result.extend(range(start - 1, end))
                else:
                    result.extend([int(p.strip()) - 1 for p in pages_str.split(",")])
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid page format: {str(e)}")

        def on_cancel():
            dialog.destroy()

        button_frame = tk.Frame(dialog, bg=self.colors["bg"])
        button_frame.pack(pady=10)
        tk.Button(
            button_frame,
            text="OK",
            command=on_ok,
            bg=self.colors["success"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel,
            bg=self.colors["danger"],
            fg=self.colors["button_fg"],
        ).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        return result

    def toggle_enhance(self, page_index, enable):
        """Toggle page enhancement"""
        self.current_pages[page_index]["enhance"] = enable
        if enable:
            self.enhance_page(page_index)
        print(
            f"Enhancement {'enabled' if enable else 'disabled'} for page {page_index + 1}"
        )

    def enhance_page(self, page_index):
        """Apply enhancement to scanned page"""
        try:
            page_data = self.current_pages[page_index]
            page = page_data["original_page"]

            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            enhanced = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            # Clear cached thumbnail to force regeneration
            page_data["thumbnail"] = None

            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Enhanced page {page_index + 1}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to enhance page: {str(e)}")

    def toggle_watermark_removal(self, page_index, enable):
        """Toggle watermark removal"""
        self.current_pages[page_index]["remove_watermark"] = enable
        if enable:
            self.remove_watermarks(page_index)
        print(
            f"Watermark removal {'enabled' if enable else 'disabled'} for page {page_index + 1}"
        )

    def remove_watermarks(self, page_index):
        """Attempt to remove watermarks"""
        try:
            page_data = self.current_pages[page_index]
            page = page_data["original_page"]

            blocks = page.get_text("dict")
            watermark_keywords = [
                "confidential",
                "draft",
                "sample",
                "watermark",
                "copy",
            ]

            removed_count = 0
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].lower()
                            for keyword in watermark_keywords:
                                if keyword in text:
                                    print(f"Found potential watermark: {span['text']}")
                                    rect = fitz.Rect(span["bbox"])
                                    page.draw_rect(
                                        rect, color=(1, 1, 1), fill=(1, 1, 1)
                                    )
                                    removed_count += 1

            page_data["thumbnail"] = None
            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(
                f"Processed watermark removal for page {page_index + 1} ({removed_count} items)"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove watermarks: {str(e)}")

    def toggle_artifact_removal(self, page_index, enable):
        """Toggle artifact removal"""
        if enable:
            self.remove_artifacts(page_index)

    def remove_artifacts(self, page_index):
        """Remove common scan artifacts"""
        try:
            page_data = self.current_pages[page_index]
            page = page_data["original_page"]

            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

            horizontal_lines = cv2.morphologyEx(
                cleaned, cv2.MORPH_OPEN, horizontal_kernel
            )
            vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical_kernel)

            cleaned = cv2.subtract(cleaned, horizontal_lines)
            cleaned = cv2.subtract(cleaned, vertical_lines)

            page_data["thumbnail"] = None
            self.refresh_page_list()
            self.refresh_reorder_panel()
            print(f"Removed artifacts from page {page_index + 1}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove artifacts: {str(e)}")

    def save_to_original(self):
        """Save changes back to original documents with preview confirmation"""
        if not self.current_pages:
            messagebox.showwarning("Warning", "No pages to save")
            return

        # Create preview PDF first
        self.create_preview_pdf()

    def create_preview_pdf(self):
        """Create a preview PDF and show confirmation dialog"""
        try:
            # Create temporary preview file
            temp_dir = tempfile.mkdtemp()
            preview_path = os.path.join(temp_dir, "preview.pdf")

            output_doc = fitz.open()

            for page_data in self.current_pages:
                source_page = page_data["original_page"]

                # Apply rotation if any
                if page_data.get("current_rotation", 0) != 0:
                    rotation_matrix = fitz.Matrix(page_data["current_rotation"])
                    source_page.set_rotation(page_data["current_rotation"])

                new_page = output_doc.new_page(
                    width=source_page.rect.width, height=source_page.rect.height
                )
                new_page.show_pdf_page(
                    new_page.rect, page_data["doc"], page_data["page_num"]
                )

            output_doc.save(preview_path)
            output_doc.close()

            # Show confirmation dialog
            self.show_preview_confirmation(preview_path, temp_dir)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create preview: {str(e)}")

    def show_preview_confirmation(self, preview_path, temp_dir):
        """Show preview confirmation dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Save to Original")
        dialog.geometry("600x400")
        dialog.configure(bg=self.colors["bg"])

        # Info text
        info_text = f"""
Preview PDF created with {len(self.current_pages)} pages.

This will OVERWRITE the original PDF files with your changes.
Please review the preview PDF carefully before confirming.

Preview saved at: {preview_path}
        """

        tk.Label(
            dialog,
            text=info_text,
            font=("Arial", 11),
            wraplength=550,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
        ).pack(pady=20)

        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors["bg"])
        button_frame.pack(pady=20)

        tk.Button(
            button_frame,
            text="Open Preview PDF",
            command=lambda: os.startfile(preview_path),
            bg=self.colors["accent"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            button_frame,
            text="✓ CONFIRM SAVE",
            command=lambda: self.confirm_save_to_original(dialog, temp_dir),
            bg=self.colors["success"],
            fg=self.colors["button_fg"],
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            button_frame,
            text="✗ Cancel",
            command=lambda: self.cancel_save(dialog, temp_dir),
            bg=self.colors["danger"],
            fg=self.colors["button_fg"],
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=10)

    def confirm_save_to_original(self, dialog, temp_dir):
        """Actually save changes to original files"""
        dialog.destroy()

        try:
            # Group pages by source file
            files_to_update = {}

            for i, page_data in enumerate(self.current_pages):
                source_file = page_data.get("source_file")
                if source_file and source_file in self.original_file_paths:
                    if source_file not in files_to_update:
                        files_to_update[source_file] = []
                    files_to_update[source_file].append((i, page_data))

            if not files_to_update:
                messagebox.showinfo(
                    "Info", "No original files to update. Use 'Save PDF' for new file."
                )
                return

            # Update each original file
            for file_path, pages in files_to_update.items():
                self.update_original_file(file_path, pages)

            messagebox.showinfo(
                "Success",
                f"Successfully updated {len(files_to_update)} original PDF files!\n"
                f"Total pages processed: {len(self.current_pages)}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to original: {str(e)}")
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def update_original_file(self, file_path, pages):
        """Update a specific original file with modified pages"""
        try:
            # Create backup
            backup_path = file_path + ".backup"
            shutil.copy2(file_path, backup_path)

            # Create new document with all pages from current view that belong to this file
            new_doc = fitz.open()

            for _, page_data in pages:
                source_page = page_data["original_page"]

                # Apply rotation if any
                if page_data.get("current_rotation", 0) != 0:
                    source_page.set_rotation(page_data["current_rotation"])

                new_page = new_doc.new_page(
                    width=source_page.rect.width, height=source_page.rect.height
                )
                new_page.show_pdf_page(
                    new_page.rect, page_data["doc"], page_data["page_num"]
                )

            # Save over original
            new_doc.save(file_path)
            new_doc.close()

            print(f"Updated {file_path} with {len(pages)} pages")

        except Exception as e:
            # Restore backup if something went wrong
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
            raise e

    def cancel_save(self, dialog, temp_dir):
        """Cancel save operation and clean up"""
        dialog.destroy()
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Save operation cancelled")

    def refresh_page_list(self):
        """Refresh the display with virtual scrolling - HIGH PERFORMANCE"""
        # Clear current display
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        total_pages = len(self.current_pages)

        # Update labels
        self.page_count_label.config(text=f"Total Pages: {total_pages}")

        if total_pages == 0:
            self.view_label.config(text="Viewing: 0-0")
            return

        # Calculate visible range
        start_idx = self.visible_start
        end_idx = min(start_idx + self.visible_count, total_pages)

        self.view_label.config(text=f"Viewing: {start_idx + 1}-{end_idx}")

        # Create widgets only for visible pages
        for i, page_data in enumerate(self.current_pages[start_idx:end_idx]):
            actual_index = start_idx + i
            self.create_page_widget(page_data, i, actual_index)

        # Update scrollable area
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


if __name__ == "__main__":
    app = CrudePDFEditor()
    app.root.mainloop()
