import io
import os
import shutil
import subprocess
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox
from zipfile import ZipFile

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


class FileReviewUI:
    def __init__(self, folder_path: Path):
        self.root = ctk.CTk()
        self.root.title("File Review Queue - Simple Edition")
        self.root.geometry("1800x1000")

        self.colors = {
            "bg": "#1a1a1a",
            "fg": "#ffffff",
            "sidebar_bg": "#2a2a2a",
            "preview_bg": "#1e1e1e",
            "accent1": "#FF6B6B",
            "accent2": "#4ECDC4",
            "accent3": "#4F86F7",
            "accent4": "#FFC144",
            "accent5": "#9966CC",
            "selected": "#4F86F7",
            "flagged": "#FF0000",
            "card_bg": "#2d2d2d",
        }

        self.folder_path = Path(folder_path)
        self.queue = deque()
        self.history = []
        self.current_index = -1
        self.current_file = None
        self.pending_actions = {}
        self.flagged_files = set()

        # PDF state - SIMPLE
        self.pdf_doc = None
        self.pdf_page_order = []  # Just track the order
        self.drag_source = None

        self.combination_files = []
        self.selected_for_combine = []

        self.start_time = time.time()
        self.file_start_time = None
        self.file_times = []

        # Video playback
        self.video_cap = None
        self.video_after_id = None
        self.video_label = None
        self.is_playing = False

        self.zoom_scale = 1.0

        self._setup_folders()
        self._load_files()
        self._setup_ui()
        self._bind_keys()
        self._update_timing()
        self._update_combination_sidebar()

        if self.queue:
            self.show_next()

    def _setup_folders(self):
        self.folders = {
            "ARCHIVE": self.folder_path / "ARCHIVE",
            "COMBINATION": self.folder_path / "COMBINATION",
            "IMMICH": self.folder_path / "IMMICH",
            "RESOURCESPACE": self.folder_path / "RESOURCESPACE",
            "AFFINE": self.folder_path / "AFFINE",
            "OTHER": self.folder_path / "OTHER",
        }
        for folder in self.folders.values():
            folder.mkdir(exist_ok=True)

    def _load_files(self):
        files = [
            f
            for f in self.folder_path.iterdir()
            if f.is_file() and f.parent == self.folder_path
        ]
        self.queue.extend(files)
        print(f"✓ Loaded {len(self.queue)} files")

    def _lighten_color(self, color):
        color = color.lstrip("#")
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r, g, b = (
            min(255, int(r * 1.15)),
            min(255, int(g * 1.15)),
            min(255, int(b * 1.15)),
        )
        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_timing(self):
        elapsed = int(time.time() - self.start_time)
        avg_time = sum(self.file_times) / len(self.file_times) if self.file_times else 0
        remaining_min = int((len(self.queue) * avg_time) / 60) if avg_time > 0 else 0
        self.timing_label.configure(
            text=f"⏱ {elapsed}s | Avg: {avg_time:.1f}s | ~{remaining_min}m left"
        )
        self.root.after(1000, self._update_timing)

    def _update_labels(self):
        if self.current_file:
            total = len(self.history) + len(self.queue)
            flag = " 🚩" if self.current_file in self.flagged_files else ""
            self.file_label.configure(
                text=f"📄 {self.current_file.name}{flag} ({self.current_index + 1}/{total})"
            )
        self.queue_label.configure(
            text=f"Queue: {len(self.queue)} | Flagged: {len(self.flagged_files)}"
        )

    def _update_action_display(self):
        action = self.pending_actions.get(self.current_file)
        if action:
            action_type, target = action
            if action_type == "pdf_edit":
                self.current_action_label.configure(
                    text="⚡ PDF EDITED", text_color=self.colors["accent4"]
                )
            else:
                self.current_action_label.configure(
                    text=f"→ {target}", text_color=self.colors["accent3"]
                )
        else:
            self.current_action_label.configure(
                text="No action", text_color=self.colors["fg"]
            )

    def _setup_ui(self):
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill=tk.BOTH, expand=True)

        # LEFT SIDEBAR
        left = ctk.CTkFrame(main, width=320, fg_color=self.colors["sidebar_bg"])
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        left.pack_propagate(False)
        ctk.CTkLabel(
            left,
            text="📚 HISTORY",
            font=(FONT_FAMILY, 16, "bold"),
            text_color=self.colors["accent2"],
        ).pack(pady=(15, 10))
        self.history_scroll_frame = CTkScrollableFrame(
            left, fg_color=self.colors["sidebar_bg"]
        )
        self.history_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 10))

        # CENTER
        center = ctk.CTkFrame(main, fg_color="transparent")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)

        info = ctk.CTkFrame(center, fg_color="transparent", height=50)
        info.pack(fill=tk.X, pady=(5, 0))
        self.file_label = ctk.CTkLabel(
            info,
            text="No file",
            font=(FONT_FAMILY, 14, "bold"),
            text_color=self.colors["fg"],
        )
        self.file_label.pack(side=tk.LEFT, padx=10)
        self.current_action_label = ctk.CTkLabel(
            info,
            text="No action",
            font=(FONT_FAMILY, 12, "bold"),
            text_color=self.colors["fg"],
        )
        self.current_action_label.pack(side=tk.LEFT, padx=10)
        self.queue_label = ctk.CTkLabel(
            info, text="Queue: 0", font=(FONT_FAMILY, 14), text_color=self.colors["fg"]
        )
        self.queue_label.pack(side=tk.RIGHT, padx=10)

        preview = ctk.CTkFrame(center, fg_color=self.colors["preview_bg"])
        preview.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        toolbar = ctk.CTkFrame(preview, fg_color="transparent", height=40)
        toolbar.pack(fill=tk.X, pady=(5, 0), padx=5)

        left_tools = ctk.CTkFrame(toolbar, fg_color="transparent")
        left_tools.pack(side=tk.LEFT)
        ctk.CTkButton(
            left_tools,
            text="✓ SAVE PDF",
            command=self._apply_pdf_changes,
            fg_color=self.colors["accent3"],
            width=130,
            height=35,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(
            left_tools,
            text="🚩 FLAG",
            command=self._toggle_flag,
            fg_color=self.colors["flagged"],
            width=100,
            height=35,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)

        right_tools = ctk.CTkFrame(toolbar, fg_color="transparent")
        right_tools.pack(side=tk.RIGHT)
        ctk.CTkButton(
            right_tools,
            text="🔗 OPEN",
            command=self._open_external,
            fg_color=self.colors["accent5"],
            width=120,
            height=35,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)

        self.zoom_label = ctk.CTkLabel(
            preview,
            text=f"🔍 100%",
            font=(FONT_FAMILY, 10),
            text_color=self.colors["accent4"],
        )
        self.zoom_label.pack(side=tk.TOP, pady=2)

        self.preview_scroll_frame = CTkScrollableFrame(
            preview, fg_color=self.colors["preview_bg"]
        )
        self.preview_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.preview_scroll_frame.bind("<Control-MouseWheel>", self._handle_zoom)
        self.root.bind("<Control-MouseWheel>", self._handle_zoom)

        btn_frame = ctk.CTkFrame(center, fg_color="transparent")
        btn_frame.pack(pady=10)
        for i in range(5):
            btn_frame.grid_columnconfigure(i, weight=1, uniform="button")

        actions = [
            ("1 - Archive", "1", self.colors["accent1"]),
            ("2 - Handwriting", "2", self.colors["accent2"]),
            ("3 - Document", "3", self.colors["accent3"]),
            ("4 - Combine", "4", self.colors["accent4"]),
            ("5 - Video→Audio", "5", self.colors["accent5"]),
            ("I - Immich", "i", "#66BB6A"),
            ("R - ResourceSpace", "r", "#17A2B8"),
            ("A - AFFiNE", "a", "#F48024"),
            ("O - Other", "o", "#DC3545"),
        ]
        for i, (label, key, color) in enumerate(actions):
            ctk.CTkButton(
                btn_frame,
                text=label,
                command=lambda k=key: self._handle_action(k),
                fg_color=color,
                width=200,
                height=38,
                text_color=self.colors["fg"],
            ).grid(row=i // 5, column=i % 5, padx=5, pady=5)

        self.status_label = ctk.CTkLabel(
            center, text="", text_color=self.colors["accent3"], font=(FONT_FAMILY, 10)
        )
        self.status_label.pack(pady=2)

        footer = ctk.CTkFrame(center, fg_color="transparent", height=60)
        footer.pack(fill=tk.X, pady=5)
        self.timing_label = ctk.CTkLabel(
            footer,
            text="⏱ 0s",
            text_color=self.colors["accent4"],
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.timing_label.pack(side=tk.LEFT, padx=10)

        nav = ctk.CTkFrame(footer, fg_color="transparent")
        nav.pack(side=tk.RIGHT, padx=10)
        ctk.CTkButton(
            nav,
            text="← Prev",
            command=self.show_previous,
            fg_color=self.colors["accent2"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(
            nav,
            text="Next →",
            command=self.show_next,
            fg_color=self.colors["accent2"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(
            nav,
            text="🚩 FLAG",
            command=self._jump_to_next_flagged,
            fg_color=self.colors["flagged"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(
            nav,
            text="✓ DONE",
            command=self._finish_review,
            fg_color="#66bb66",
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=10)

        # RIGHT SIDEBAR
        right = ctk.CTkFrame(main, width=350, fg_color=self.colors["sidebar_bg"])
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)
        right.pack_propagate(False)
        ctk.CTkLabel(
            right,
            text="🔀 COMBINE",
            font=(FONT_FAMILY, 14, "bold"),
            text_color=self.colors["accent4"],
        ).pack(pady=(10, 5))
        self.combo_scroll_frame = CTkScrollableFrame(
            right, fg_color=self.colors["sidebar_bg"]
        )
        self.combo_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        ctk.CTkButton(
            right,
            text="🔗 MERGE PDFs",
            command=self._combine_selected,
            fg_color=self.colors["accent3"],
            width=250,
            height=40,
            text_color=self.colors["fg"],
        ).pack(pady=5)
        self.combo_status = ctk.CTkLabel(
            right, text="0 files", text_color=self.colors["fg"], font=(FONT_FAMILY, 10)
        )
        self.combo_status.pack(pady=(2, 10))

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.show_previous())
        self.root.bind("<Right>", lambda e: self.show_next())
        self.root.bind("f", lambda e: self._toggle_flag())
        self.root.bind("F", lambda e: self._toggle_flag())
        for key in ["1", "2", "3", "4", "5", "i", "I", "r", "R", "a", "A", "o", "O"]:
            self.root.bind(key, lambda e, k=key: self._handle_action(k.lower()))

    def _handle_zoom(self, event):
        self.zoom_scale = max(
            0.5, min(3.0, self.zoom_scale + (0.1 if event.delta > 0 else -0.1))
        )
        self.zoom_label.configure(text=f"🔍 {int(self.zoom_scale * 100)}%")
        self._display_file()

    def _toggle_flag(self):
        if not self.current_file:
            return
        if self.current_file in self.flagged_files:
            self.flagged_files.remove(self.current_file)
            self.status_label.configure(
                text="✓ Flag removed", text_color=self.colors["accent3"]
            )
        else:
            self.flagged_files.add(self.current_file)
            self.status_label.configure(
                text="🚩 FLAGGED", text_color=self.colors["flagged"]
            )
        self._update_labels()
        self._update_history_listbox()

    def _jump_to_next_flagged(self):
        if not self.flagged_files:
            messagebox.showinfo("No Flags", "No flagged files.")
            return
        for i in range(self.current_index + 1, len(self.history)):
            if self.history[i][0] in self.flagged_files:
                self._stop_video()
                self.current_index = i
                self.current_file = self.history[i][0]
                self._display_file()
                self._update_labels()
                self._update_action_display()
                self._update_history_listbox()
                return
        messagebox.showinfo("Done", "No more flagged files ahead.")

    def _open_external(self):
        if not self.current_file or not self.current_file.exists():
            return
        try:
            if os.name == "nt":
                os.startfile(self.current_file)
            else:
                subprocess.run(["xdg-open", str(self.current_file)])
            self.status_label.configure(
                text=f"✓ Opened {self.current_file.name}",
                text_color=self.colors["accent3"],
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    def _handle_action(self, key):
        if not self.current_file:
            return
        if key == "4":
            try:
                target = self.folders["COMBINATION"] / self.current_file.name
                counter = 1
                while target.exists():
                    target = (
                        self.folders["COMBINATION"]
                        / f"{self.current_file.stem}_{counter}{self.current_file.suffix}"
                    )
                    counter += 1
                shutil.move(str(self.current_file), str(target))
                self.status_label.configure(
                    text="✓ Moved to Combination", text_color=self.colors["accent4"]
                )
                self._update_combination_sidebar()
                self.show_next()
                return
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {e}")
                return

        action_map = {
            "1": ("move", "ARCHIVE"),
            "2": ("process_2", "AFFINE"),
            "3": ("process_3", "RESOURCESPACE"),
            "5": ("process_5", "IMMICH"),
            "i": ("move", "IMMICH"),
            "r": ("move", "RESOURCESPACE"),
            "a": ("move", "AFFINE"),
            "o": ("move", "OTHER"),
        }
        if key in action_map:
            self.pending_actions[self.current_file] = action_map[key]
            self._update_action_display()
            self._update_history_listbox()
            self.show_next()

    def show_next(self):
        if self.file_start_time:
            self.file_times.append(time.time() - self.file_start_time)
        self._stop_video()

        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.current_file = self.history[self.current_index][0]
        elif self.queue:
            if self.current_file and self.current_file not in [
                h[0] for h in self.history
            ]:
                self.history.append((self.current_file, "manual", 0))
            self.current_file = self.queue.popleft()
            self.current_index = len(self.history)
            self.file_start_time = time.time()
            self._reset_pdf_state()
        else:
            self.current_file = None
            self._clear_preview()
            self._update_labels()
            return

        self._display_file()
        self._update_labels()
        self._update_action_display()
        self._update_history_listbox()

    def show_previous(self):
        self._stop_video()
        if self.current_index > 0:
            self.current_index -= 1
            self.current_file = self.history[self.current_index][0]
            self._display_file()
            self._update_labels()
            self._update_action_display()
            self._update_history_listbox()

    def _stop_video(self):
        self.is_playing = False
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.video_cap:
            try:
                self.video_cap.release()
            except:
                pass
            self.video_cap = None

    def _reset_pdf_state(self):
        self.pdf_page_order = []
        if self.pdf_doc:
            try:
                self.pdf_doc.close()
            except:
                pass
            self.pdf_doc = None

    def _on_history_select(self, idx):
        if idx < len(self.history):
            self._stop_video()
            self.current_index = idx
            self.current_file = self.history[idx][0]
            self._display_file()
            self._update_labels()
            self._update_action_display()
            self._update_history_listbox()

    def _update_history_listbox(self):
        for w in self.history_scroll_frame.winfo_children():
            w.destroy()

        items = self.history[:]
        if self.current_file and self.current_index == len(self.history):
            items.append((self.current_file, "manual", 0))

        for idx, (file, _, _) in enumerate(items):
            is_curr = idx == self.current_index
            is_flag = file in self.flagged_files
            card_col = (
                self.colors["flagged"]
                if is_flag
                else (self.colors["selected"] if is_curr else self.colors["card_bg"])
            )

            frame = ctk.CTkFrame(
                self.history_scroll_frame,
                fg_color=card_col,
                corner_radius=10,
                border_width=2 if is_curr else 0,
                border_color=self.colors["accent3"] if is_curr else card_col,
            )
            frame.pack(pady=6, padx=6, fill=tk.X)

            thumb = self._create_thumbnail(file, size=(280, 160))
            if thumb:
                lbl = tk.Label(frame, image=thumb, bg=card_col, cursor="hand2")
                lbl.image = thumb
                lbl.pack(pady=8, padx=8)
                lbl.bind("<Button-1>", lambda e, i=idx: self._on_history_select(i))

            name = file.name
            if len(name) > 35:
                name = name[:32] + "..."
            if is_flag:
                name = "🚩 " + name

            ctk.CTkLabel(
                frame,
                text=name,
                text_color=self.colors["fg"],
                font=(FONT_FAMILY, 11, "bold" if is_curr else "normal"),
            ).pack(fill=tk.X, padx=10, pady=(0, 10))

            frame.bind("<Button-1>", lambda e, i=idx: self._on_history_select(i))

    def _update_combination_sidebar(self):
        for w in self.combo_scroll_frame.winfo_children():
            w.destroy()

        combo_folder = self.folders["COMBINATION"]
        self.combination_files = [f for f in combo_folder.iterdir() if f.is_file()]

        if not self.combination_files:
            ctk.CTkLabel(
                self.combo_scroll_frame, text="No files", text_color=self.colors["fg"]
            ).pack(pady=20)
            self.combo_status.configure(text="0 files")
            return

        self.combo_status.configure(text=f"{len(self.combination_files)} files")

        for file in sorted(self.combination_files, key=lambda x: x.name):
            frame = ctk.CTkFrame(
                self.combo_scroll_frame,
                fg_color=self.colors["card_bg"],
                corner_radius=8,
            )
            frame.pack(fill=tk.X, padx=5, pady=4)

            thumb = self._create_thumbnail(file, size=(300, 100))
            if thumb:
                lbl = tk.Label(frame, image=thumb, bg=self.colors["card_bg"])
                lbl.image = thumb
                lbl.pack(pady=5, padx=5)

            var = tk.BooleanVar(value=file in self.selected_for_combine)

            def toggle(p, v):
                if v.get():
                    if p not in self.selected_for_combine:
                        self.selected_for_combine.append(p)
                else:
                    if p in self.selected_for_combine:
                        self.selected_for_combine.remove(p)
                self.combo_status.configure(
                    text=f"{len(self.selected_for_combine)} selected"
                )

            name = file.name
            if len(name) > 35:
                name = name[:32] + "..."
            ctk.CTkCheckBox(
                frame,
                text=name,
                variable=var,
                command=lambda p=file, v=var: toggle(p, v),
                text_color=self.colors["fg"],
                font=(FONT_FAMILY, 10),
            ).pack(padx=12, pady=8)

    def _combine_selected(self):
        if len(self.selected_for_combine) < 2:
            messagebox.showwarning("Error", "Select 2+ PDFs to merge.")
            return

        output = self.folder_path / f"MERGED_{int(time.time())}.pdf"
        if PIKEPDF_AVAILABLE:
            try:
                pdf = pikepdf.new()
                for file in sorted(self.selected_for_combine, key=lambda x: x.name):
                    if file.suffix.lower() == ".pdf":
                        with pikepdf.open(file) as src:
                            pdf.pages.extend(src.pages)
                        os.remove(file)
                if len(pdf.pages) > 0:
                    pdf.save(output)
                    self.status_label.configure(
                        text=f"✓ Merged into {output.name}",
                        text_color=self.colors["accent3"],
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Merge failed: {e}")
        else:
            messagebox.showinfo("Error", "Install pikepdf: pip install pikepdf")
        self.selected_for_combine.clear()
        self._update_combination_sidebar()

    def _apply_pdf_changes(self):
        """Save reordered PDF pages"""
        if (
            not self.current_file
            or self.current_file.suffix.lower() != ".pdf"
            or not self.pdf_page_order
        ):
            return

        try:
            doc = fitz.open(self.current_file)
            doc.select(self.pdf_page_order)  # Reorder pages
            temp = self.current_file.parent / f"temp_{self.current_file.name}"
            doc.save(temp)
            doc.close()
            os.remove(self.current_file)
            os.rename(temp, self.current_file)
            self.status_label.configure(
                text="✓ PDF saved!", text_color=self.colors["accent3"]
            )
            self._reset_pdf_state()
            self._display_file()
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def _clear_preview(self):
        for w in self.preview_scroll_frame.winfo_children():
            w.destroy()

    def _display_file(self):
        self._clear_preview()
        if not self.current_file or not self.current_file.exists():
            return

        ext = self.current_file.suffix.lower()
        if ext == ".pdf":
            self._display_pdf()
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]:
            self._display_image()
        elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
            self._display_video()
        elif ext in [".mp3", ".wav", ".flac", ".ogg"]:
            self._display_audio()
        elif ext in [".txt", ".log", ".json", ".csv", ".xml"]:
            self._display_text()

    def _display_pdf(self):
        """PDF with simple drag-drop reorder"""
        try:
            if not self.pdf_doc or self.pdf_doc.name != str(self.current_file):
                if self.pdf_doc:
                    self.pdf_doc.close()
                self.pdf_doc = fitz.open(self.current_file)
                self.pdf_page_order = list(range(len(self.pdf_doc)))

            doc = self.pdf_doc
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"📄 PDF ({len(doc)} pages) - Click & drag to reorder",
                text_color=self.colors["accent4"],
                font=(FONT_FAMILY, 13, "bold"),
            ).pack(pady=10)

            grid = ctk.CTkFrame(self.preview_scroll_frame, fg_color="transparent")
            grid.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)

            cols = min(3, len(doc))
            for c in range(cols):
                grid.grid_columnconfigure(c, weight=1)

            for idx, page_num in enumerate(self.pdf_page_order):
                row, col = idx // cols, idx % cols

                frame = ctk.CTkFrame(
                    grid, fg_color=self.colors["sidebar_bg"], corner_radius=12
                )
                frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                # Drag-drop handlers
                def start_drag(e, i=idx):
                    self.drag_source = i
                    e.widget.configure(
                        border_width=3, border_color=self.colors["accent4"]
                    )

                def end_drag(e, i=idx):
                    if self.drag_source is not None and self.drag_source != i:
                        # Swap pages
                        (
                            self.pdf_page_order[self.drag_source],
                            self.pdf_page_order[i],
                        ) = (
                            self.pdf_page_order[i],
                            self.pdf_page_order[self.drag_source],
                        )
                        self.drag_source = None
                        self._display_pdf()  # Refresh
                    else:
                        e.widget.configure(border_width=0)
                        self.drag_source = None

                frame.bind("<Button-1>", start_drag)
                frame.bind("<ButtonRelease-1>", end_drag)

                page = doc[page_num]
                scale = 0.3 * self.zoom_scale
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                photo = ImageTk.PhotoImage(img)

                lbl = tk.Label(
                    frame, image=photo, bg=frame.cget("fg_color"), cursor="hand2"
                )
                lbl.image = photo
                lbl.pack(pady=5)
                lbl.bind("<Button-1>", start_drag)
                lbl.bind("<ButtonRelease-1>", end_drag)

                ctk.CTkLabel(
                    frame,
                    text=f"Page {page_num + 1}",
                    text_color=self.colors["accent4"],
                    font=(FONT_FAMILY, 11, "bold"),
                ).pack(pady=5)

        except Exception as e:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"❌ Error: {e}",
                text_color=self.colors["accent1"],
            ).pack(pady=50)

    def _display_image(self):
        try:
            img = Image.open(self.current_file)
            w, h = int(img.width * self.zoom_scale), int(img.height * self.zoom_scale)
            if w > 1200:
                h = int(h * (1200 / w))
                w = 1200
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(
                self.preview_scroll_frame,
                image=photo,
                bg=self.preview_scroll_frame.cget("fg_color"),
            )
            lbl.image = photo
            lbl.pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"❌ {e}",
                text_color=self.colors["accent1"],
            ).pack(pady=50)

    def _display_video(self):
        """Video with OpenCV (NO AUDIO - use Open button for audio)"""
        if not CV2_AVAILABLE:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text="Install: pip install opencv-python",
                text_color=self.colors["accent1"],
            ).pack(pady=50)
            return

        try:
            self.video_cap = cv2.VideoCapture(str(self.current_file))

            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"🎬 {self.current_file.name}",
                text_color=self.colors["accent4"],
                font=(FONT_FAMILY, 14, "bold"),
            ).pack(pady=10)

            ctk.CTkLabel(
                self.preview_scroll_frame,
                text="⚠️ No audio in preview - Use 'OPEN' button to play with audio",
                text_color=self.colors["accent1"],
                font=(FONT_FAMILY, 10),
            ).pack(pady=5)

            self.video_label = tk.Label(self.preview_scroll_frame, bg="#000000")
            self.video_label.pack(pady=10)

            controls = ctk.CTkFrame(self.preview_scroll_frame, fg_color="transparent")
            controls.pack(pady=5)

            def toggle_play():
                self.is_playing = not self.is_playing
                play_btn.configure(text="⏸ Pause" if self.is_playing else "▶ Play")
                if self.is_playing:
                    update_frame()

            play_btn = ctk.CTkButton(
                controls,
                text="▶ Play",
                command=toggle_play,
                fg_color=self.colors["accent3"],
                width=100,
                height=35,
                text_color=self.colors["fg"],
            )
            play_btn.pack(side=tk.LEFT, padx=5)

            def update_frame():
                if not self.is_playing or not self.video_cap:
                    return
                ret, frame = self.video_cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    nw, nh = int(w * self.zoom_scale * 0.6), int(
                        h * self.zoom_scale * 0.6
                    )
                    frame = cv2.resize(frame, (nw, nh))
                    img = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(img)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                    self.video_after_id = self.root.after(33, update_frame)
                else:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.is_playing = False
                    play_btn.configure(text="▶ Play")

            # Auto-start
            self.is_playing = True
            play_btn.configure(text="⏸ Pause")
            update_frame()

        except Exception as e:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"❌ {e}",
                text_color=self.colors["accent1"],
            ).pack(pady=50)

    def _display_audio(self):
        """Audio with pygame"""
        if not PYGAME_AVAILABLE:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text="Install: pip install pygame",
                text_color=self.colors["accent1"],
            ).pack(pady=50)
            return

        ctk.CTkLabel(
            self.preview_scroll_frame,
            text=f"🎵 {self.current_file.name}",
            text_color=self.colors["accent4"],
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(pady=20)

        controls = ctk.CTkFrame(self.preview_scroll_frame, fg_color="transparent")
        controls.pack(pady=10)

        def play():
            try:
                pygame.mixer.music.load(str(self.current_file))
                pygame.mixer.music.play()
                self.status_label.configure(
                    text="▶ Playing...", text_color=self.colors["accent3"]
                )
            except Exception as e:
                messagebox.showerror("Error", f"Cannot play: {e}")

        def pause():
            pygame.mixer.music.pause()
            self.status_label.configure(
                text="⏸ Paused", text_color=self.colors["accent4"]
            )

        def stop():
            pygame.mixer.music.stop()
            self.status_label.configure(text="⏹ Stopped", text_color=self.colors["fg"])

        ctk.CTkButton(
            controls,
            text="▶ Play",
            command=play,
            fg_color=self.colors["accent3"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(
            controls,
            text="⏸ Pause",
            command=pause,
            fg_color=self.colors["accent4"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(
            controls,
            text="⏹ Stop",
            command=stop,
            fg_color=self.colors["accent1"],
            width=100,
            height=40,
            text_color=self.colors["fg"],
        ).pack(side=tk.LEFT, padx=5)

        # Auto-play
        self.root.after(300, play)

    def _display_text(self):
        try:
            with open(self.current_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            box = ctk.CTkTextbox(
                self.preview_scroll_frame,
                wrap=tk.WORD,
                fg_color=self.colors["sidebar_bg"],
                text_color=self.colors["fg"],
                font=(FONT_FAMILY, 10),
            )
            box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            box.insert("1.0", content)
            box.configure(state="disabled")
        except Exception as e:
            ctk.CTkLabel(
                self.preview_scroll_frame,
                text=f"❌ {e}",
                text_color=self.colors["accent1"],
            ).pack(pady=50)

    def _finish_review(self):
        if messagebox.askyesno(
            "Finish", f"Execute {len(self.pending_actions)} pending actions?"
        ):
            for file, (action_type, target) in self.pending_actions.items():
                if action_type == "move" and file.exists():
                    try:
                        dest = self.folders[target] / file.name
                        if dest.exists():
                            dest = (
                                self.folders[target]
                                / f"{file.stem}_{int(time.time())}{file.suffix}"
                            )
                        shutil.move(str(file), str(dest))
                        print(f"✓ Moved {file.name} → {target}")
                    except Exception as e:
                        print(f"❌ Failed {file.name}: {e}")
            messagebox.showinfo("Done", "✓ All actions executed!")
            self.root.destroy()

    def _create_thumbnail(self, file, size=(280, 160)):
        """Simple thumbnails with office doc support"""
        if not file.exists():
            return None

        ext = file.suffix.lower()
        try:
            if ext == ".pdf" and PIL_AVAILABLE:
                doc = fitz.open(file)
                if len(doc) > 0:
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(0.4, 0.4))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    doc.close()
                    return ImageTk.PhotoImage(img)

            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                img = Image.open(file)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)

            elif ext in [".mp4", ".avi", ".mkv", ".mov"] and CV2_AVAILABLE:
                cap = cv2.VideoCapture(str(file))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    cap.release()
                    return ImageTk.PhotoImage(img)

            elif ext in [".docx", ".pptx", ".xlsx"]:
                # Office docs are ZIP files - extract embedded thumbnail
                try:
                    with ZipFile(file, "r") as z:
                        thumbs = [n for n in z.namelist() if "thumbnail" in n.lower()]
                        if thumbs:
                            data = z.read(thumbs[0])
                            if thumbs[0].lower().endswith((".jpeg", ".jpg", ".png")):
                                img = Image.open(io.BytesIO(data))
                                img.thumbnail(size, Image.Resampling.LANCZOS)
                                return ImageTk.PhotoImage(img)
                except:
                    pass

        except Exception as e:
            print(f"⚠️ Thumb error {file.name}: {e}")

        return None

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FileReviewUI(r"C:\Users\UserX\Desktop\PaperTrail-Load")
    app.run()
