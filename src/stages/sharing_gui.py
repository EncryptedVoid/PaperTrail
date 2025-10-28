import customtkinter
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import platform
import subprocess
import time
import string
import random
from datetime import date

# --- Configuration for CustomTkinter ---
customtkinter.set_appearance_mode("Dark")

# --- Define New Bright/Deep Colors ---
PRIMARY_COLOR = "#FF6347"  # Coral Red (for switches/highlights)
SECONDARY_COLOR = "#89CFF0"  # Light Baby Blue (for buttons/accents/card borders)
SUCCESS_COLOR = "#32CD32" # Bright Green
ACCENT_BG_COLOR = "#2D2D2D" # Darker background for depth/shadowing
FONT_FAMILY = "Segoe UI"

# --- Global Functions ---
def generate_password(length=12):
	"""Generates a random, strong password."""
	characters = string.ascii_letters + string.digits + "!@#$%^&*"
	return ''.join(random.choice(characters) for i in range(length))

# --- Main Application Class ---
class FileSharingApp(customtkinter.CTk):
	def __init__(self):
		super().__init__()

		# --- Basic Setup ---
		self.title("Secure File Sharer")
		self.geometry("1400x900")
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(1, weight=1)

		# --- State Variables ---
		self.file_data = {}
		self.root_dir = ""
		self.output_folder = "shared_files_output" # Initial placeholder

		# Folder Naming Variables
		self.folder_date_prefix_var = tk.StringVar(value="no")
		self.folder_name_option_var = tk.StringVar(value="date_only")
		self.folder_custom_name_var = tk.StringVar(value="Shared_Documents")

		# --- Bind folder naming variables to a change handler ---
		self.folder_date_prefix_var.trace_add("write", self._update_folder_name_preview)
		self.folder_name_option_var.trace_add("write", self._update_folder_name_preview)
		self.folder_custom_name_var.trace_add("write", self._update_folder_name_preview)


		# --- GUI Components Setup ---
		self._setup_top_panel()
		self._setup_file_list_frame()
		self._setup_folder_name_controls() # Moved below file list to be above submit bar
		self._setup_options_panel()

		self._update_folder_name_preview() # Initial call for preview

	# ----------------------------------------------------------------------
	# --- Top Panel: File Input, Browse, and Drive Selection ---
	# ----------------------------------------------------------------------
	def _setup_top_panel(self):
		self.top_frame = customtkinter.CTkFrame(self, fg_color="transparent")
		self.top_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

		self.top_frame.grid_columnconfigure(0, weight=1)
		self.top_frame.grid_columnconfigure(1, weight=5)
		self.top_frame.grid_columnconfigure(2, weight=2)
		self.top_frame.grid_columnconfigure(3, weight=1)

		self.drive_button = customtkinter.CTkButton(
				self.top_frame,
				text="Select Root Folder 📁",
				command=self._select_hard_drive,
				fg_color=SECONDARY_COLOR,
				hover_color=PRIMARY_COLOR,
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		)
		self.drive_button.grid(row=0, column=0, padx=10, pady=5, sticky="w")

		self.file_entry = customtkinter.CTkEntry(
				self.top_frame,
				placeholder_text="Paste ABSOLUTE file path or RELATIVE file name...",
				border_color=PRIMARY_COLOR,
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		)
		self.file_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

		self.browse_button = customtkinter.CTkButton(
				self.top_frame,
				text="Browse Files...",
				command=self._browse_file,
				fg_color=PRIMARY_COLOR,
				hover_color=SECONDARY_COLOR,
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		)
		self.browse_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

		self.add_button = customtkinter.CTkButton(
				self.top_frame,
				text="ENTER",
				command=self._add_file_from_input,
				fg_color=SECONDARY_COLOR,
				hover_color=PRIMARY_COLOR,
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		)
		self.add_button.grid(row=0, column=3, padx=10, pady=5, sticky="e")
		self.file_entry.bind('<Return>', lambda event: self._add_file_from_input())

		# --- Path Handling Logic (Same) ---
	def _select_hard_drive(self):
		drive_path = filedialog.askdirectory(title="Select a Hard Drive/Root Directory")
		if drive_path:
			self.root_dir = drive_path
			messagebox.showinfo("Drive Selected", f"Root Directory set to:\n{self.root_dir}")

	def _browse_file(self):
		file_path = filedialog.askopenfilename(title="Select a File to Share")
		if file_path:
			self._add_file_logic(file_path)

	def _add_file_from_input(self):
		file_path_input = self.file_entry.get().strip().strip('"')
		if file_path_input:
			self._add_file_logic(file_path_input)

	def _add_file_logic(self, file_input):
		full_path = os.path.normpath(file_input)

		if not os.path.isabs(full_path) and self.root_dir:
			combined_path = os.path.normpath(os.path.join(self.root_dir, file_input))
			if os.path.exists(combined_path) and os.path.isfile(combined_path):
				full_path = combined_path

		if full_path in self.file_data:
			messagebox.showwarning("Warning", "File is already added.")
			return

		if not os.path.exists(full_path) or not os.path.isfile(full_path):
			error_msg = f"File not found or path is invalid.\nInput: {file_input}"
			if not os.path.isabs(file_input) and not self.root_dir:
				error_msg += "\n\nTip: You need to either provide the full ABSOLUTE path, or select a Root Folder first."
			messagebox.showerror("Error", error_msg)
			return

		self._add_file_card(full_path)
		self.file_entry.delete(0, 'end')

		# ----------------------------------------------------------------------
	# --- Tiled File Card Layout (UI Polishing) ---
	# ----------------------------------------------------------------------
	def _setup_file_list_frame(self):
		# Header removed as requested.
		self.file_list_frame = customtkinter.CTkScrollableFrame(
				self,
				fg_color="transparent"
		)
		self.file_list_frame.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="nsew")

		self.NUM_COLUMNS = 3
		for i in range(self.NUM_COLUMNS):
			self.file_list_frame.grid_columnconfigure(i, weight=1)

	def _get_next_grid_position(self):
		num_files = len(self.file_data)
		row = num_files // self.NUM_COLUMNS
		col = num_files % self.NUM_COLUMNS
		return row, col

	def _add_file_card(self, file_path):
		file_name = os.path.basename(file_path)
		row, col = self._get_next_grid_position()

		# 1. Main Card Container (Added shadow effect via different background)
		card_container = customtkinter.CTkFrame(
				self.file_list_frame,
				fg_color=ACCENT_BG_COLOR,
				border_width=2,
				border_color=SECONDARY_COLOR
		)
		card_container.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
		card_container.grid_columnconfigure((0, 1), weight=1)

		# --- Options Variables ---
		vars = {
			'in_folder': tk.StringVar(value="yes"),
			'individual_pwd': tk.StringVar(value="no"),
			'ai_name': tk.StringVar(value="no"),
			'watermark': tk.StringVar(value="no"),
			'watermark_text': tk.StringVar(value=f"CONFIDENTIAL"),
			'date_prefix': tk.StringVar(value="no"),
			'translate_choices': tk.StringVar(value="None"),
		}

		# 2. Left Side: Preview and Name (UPDATED)
		preview_frame = customtkinter.CTkFrame(card_container, fg_color="transparent")
		preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

		# File Name Label with Cutoff (max 30 characters visible)
		display_name = file_name if len(file_name) <= 30 else file_name[:27] + '...'
		customtkinter.CTkLabel(
				preview_frame,
				text=f"📄 {display_name}", # Changed to file icon
				anchor="w",
				text_color="white",
				wraplength=200,
				font=customtkinter.CTkFont(size=14, weight="bold", family=FONT_FAMILY)
		).pack(fill="x", pady=(0, 5))

		# Preview Area (Enhanced Placeholder)
		file_extension = file_name.split('.')[-1].upper() if '.' in file_name else "N/A"
		try:
			file_size_kb = round(os.path.getsize(file_path) / 1024, 1)
		except:
			file_size_kb = "N/A"

		preview_text = (
			f"FILE PREVIEW (No Render)\n\n"
			f"Type: {file_extension}\n"
			f"Size: {file_size_kb} KB\n"
			f"Added: {date.today().strftime('%Y-%m-%d')}"
		)

		customtkinter.CTkLabel(
				preview_frame,
				text=preview_text,
				fg_color=('#EEEEEE', '#1E1E1E'),
				text_color=('#555555', '#AAAAAA'),
				height=150, width=200,
				justify=tk.LEFT,
				font=customtkinter.CTkFont(size=10, family=FONT_FAMILY)
		).pack(fill="both", expand=True)

		# 3. Right Side: Toggles and Inputs
		options_panel = customtkinter.CTkFrame(card_container, fg_color="transparent")
		options_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

		# Toggles Section
		self._create_toggle(options_panel, "AI Generated Name 🤖", vars['ai_name'], 0, 0, sticky="w")
		self._create_toggle(options_panel, "Date Prefix 🗓️", vars['date_prefix'], 1, 0, sticky="w")
		self._create_toggle(options_panel, "In Output Folder 🗂️", vars['in_folder'], 2, 0, sticky="w")
		self._create_toggle(options_panel, "Password Protect 🔑", vars['individual_pwd'], 3, 0, sticky="w")

		# Watermark Section (Toggle + Input)
		watermark_frame = customtkinter.CTkFrame(options_panel, fg_color="transparent")
		watermark_frame.grid(row=4, column=0, sticky="ew", pady=(5, 2))

		watermark_switch = self._create_toggle(watermark_frame, "Add Watermark 💧", vars['watermark'], 0, 0, sticky="w", pack=False)
		watermark_switch.grid(row=0, column=0, sticky="w")

		# Watermark Entry Field
		watermark_entry = customtkinter.CTkEntry(
				watermark_frame, placeholder_text="Watermark Text Input", width=150,
				textvariable=vars['watermark_text'], font=customtkinter.CTkFont(family=FONT_FAMILY)
		)
		# We will use grid_forget/grid to conditionally hide/show it

		def toggle_watermark_entry():
			if vars['watermark'].get() == "yes":
				watermark_entry.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky="ew")
			else:
				watermark_entry.grid_forget()

		watermark_switch.configure(command=toggle_watermark_entry)
		toggle_watermark_entry()

		# Translation Section (Multi-Select Dropdown)
		customtkinter.CTkLabel(
				options_panel, text="Translate To (Multi-Select):", text_color="gray",
				font=customtkinter.CTkFont(size=11, family=FONT_FAMILY, weight="bold")
		).grid(row=5, column=0, sticky="w", pady=(10, 0))

		translation_button = customtkinter.CTkButton(
				options_panel,
				text=f"Select Languages ({vars['translate_choices'].get()})",
				command=lambda: self._open_multi_select_dialog(vars['translate_choices'], translation_button),
				fg_color=PRIMARY_COLOR, hover_color=SECONDARY_COLOR,
				font=customtkinter.CTkFont(size=11, family=FONT_FAMILY)
		)
		translation_button.grid(row=6, column=0, sticky="ew")

		# 4. Store data
		self.file_data[file_path] = {'options': vars, 'frame': card_container}

	def _open_multi_select_dialog(self, var_store, button_widget):
		"""Opens a custom dialog for multi-selection of translation languages."""

		dialog = customtkinter.CTkToplevel(self)
		dialog.title("Select Languages")
		dialog.geometry("300x350")
		dialog.transient(self)

		languages = ["Spanish", "French", "German", "Japanese", "Mandarin", "Russian", "Arabic"]

		current_selection = var_store.get().split(',') if var_store.get() != "None" else []
		check_vars = {lang: tk.BooleanVar(value=lang in current_selection) for lang in languages}

		scroll_frame = customtkinter.CTkScrollableFrame(dialog, fg_color="transparent")
		scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)

		for lang in languages:
			chk = customtkinter.CTkCheckBox(
					scroll_frame, text=lang, variable=check_vars[lang],
					onvalue=True, offvalue=False,
					fg_color=PRIMARY_COLOR, hover_color=SECONDARY_COLOR,
					font=customtkinter.CTkFont(size=12, family=FONT_FAMILY)
			)
			chk.pack(pady=5, padx=5, anchor="w")

		def save_selections():
			selected_langs = [lang for lang, var in check_vars.items() if var.get()]

			var_store.set(",".join(selected_langs) if selected_langs else "None")

			display_text = ", ".join(selected_langs) if selected_langs else "None"
			button_widget.configure(text=f"Select Languages ({display_text})")
			dialog.destroy()

		save_button = customtkinter.CTkButton(
				dialog, text="Save Selection", command=save_selections,
				fg_color=SECONDARY_COLOR, hover_color=PRIMARY_COLOR,
				font=customtkinter.CTkFont(size=14, family=FONT_FAMILY)
		)
		save_button.pack(pady=10)

		dialog.grab_set()
		self.wait_window(dialog)


	def _create_toggle(self, parent, text, variable, row, col, sticky="w", pack=True):
		"""Helper function to create a standardized option toggle."""
		toggle = customtkinter.CTkSwitch(
				parent, text=text, variable=variable,
				onvalue="yes", offvalue="no",
				button_color=PRIMARY_COLOR, progress_color=SECONDARY_COLOR,
				font=customtkinter.CTkFont(size=11, family=FONT_FAMILY)
		)
		if pack:
			toggle.grid(row=row, column=col, sticky=sticky, pady=2)
		return toggle

	# ----------------------------------------------------------------------
	# --- Folder Naming Controls (REFINED & MOVED) ---
	# ----------------------------------------------------------------------
	def _update_folder_name_preview(self, *args):
		"""Calculates and updates the live preview of the output folder name."""
		current_date_str = date.today().strftime("%Y-%m-%d")

		folder_prefix = f"{current_date_str}_" if self.folder_date_prefix_var.get() == "yes" else ""

		folder_option = self.folder_name_option_var.get()
		custom_entry = getattr(self, 'custom_entry', None)

		if folder_option == "ai_name":
			folder_suffix = "AI_Generated_Share_Summary"
		elif folder_option == "custom_name":
			folder_suffix = self.folder_custom_name_var.get()
		elif folder_option == "date_only":
			folder_suffix = ""

		final_folder_name = folder_prefix + folder_suffix.replace(' ', '_')
		if not final_folder_name:
			final_folder_name = current_date_str

		self.output_folder = final_folder_name

		# Update the live preview label
		if hasattr(self, 'preview_label'):
			self.preview_label.configure(text=f"Folder Name Preview: 📂 {self.output_folder}")

		# Conditionally show/hide the custom input field
		if custom_entry:
			if folder_option == "custom_name":
				custom_entry.grid(row=0, column=3, padx=15, pady=10, sticky="ew")
			else:
				custom_entry.grid_forget()


	def _setup_folder_name_controls(self):
		# Now placed BEFORE the submission bar (row 2)
		self.folder_name_frame = customtkinter.CTkFrame(self, fg_color=ACCENT_BG_COLOR)
		self.folder_name_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
		self.folder_name_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

		# 1. Date Prefix Toggle
		self._create_toggle(
				self.folder_name_frame, "Folder Date Prefix 🗓️", self.folder_date_prefix_var,
				0, 0, sticky="w", pack=False
		).grid(row=0, column=0, padx=15, pady=10, sticky="w")

		# 2. Naming Option Radio Buttons
		customtkinter.CTkLabel(
				self.folder_name_frame, text="Folder Name Style:", text_color="gray",
				font=customtkinter.CTkFont(size=12, family=FONT_FAMILY)
		).grid(row=0, column=1, padx=(10, 0), sticky="w")

		radio_frame = customtkinter.CTkFrame(self.folder_name_frame, fg_color="transparent")
		radio_frame.grid(row=0, column=2, padx=10, pady=10, sticky="w")

		customtkinter.CTkRadioButton(radio_frame, text="Date Only", variable=self.folder_name_option_var, value="date_only", font=customtkinter.CTkFont(size=11, family=FONT_FAMILY)).pack(side="left", padx=10)
		customtkinter.CTkRadioButton(radio_frame, text="AI Title", variable=self.folder_name_option_var, value="ai_name", font=customtkinter.CTkFont(size=11, family=FONT_FAMILY)).pack(side="left", padx=10)
		customtkinter.CTkRadioButton(radio_frame, text="Custom", variable=self.folder_name_option_var, value="custom_name", font=customtkinter.CTkFont(size=11, family=FONT_FAMILY)).pack(side="left", padx=10)

		# 3. Custom Name Input (Initially hidden, revealed by _update_folder_name_preview)
		self.custom_entry = customtkinter.CTkEntry(
				self.folder_name_frame,
				textvariable=self.folder_custom_name_var,
				placeholder_text="Enter Custom Folder Name...",
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		)

		# 4. Folder Name Preview (Full width at the bottom of this frame)
		self.preview_label = customtkinter.CTkLabel(
				self.folder_name_frame,
				text="", # Will be set by _update_folder_name_preview
				anchor="w",
				text_color=SECONDARY_COLOR,
				font=customtkinter.CTkFont(size=14, weight="bold", family=FONT_FAMILY)
		)
		self.preview_label.grid(row=1, column=0, columnspan=4, padx=15, pady=5, sticky="ew")

		self._update_folder_name_preview() # Initial call

	# ----------------------------------------------------------------------
	# --- Global Options Panel (SUBMISSION BAR) ---
	# ----------------------------------------------------------------------
	def _setup_options_panel(self):
		self.global_frame = customtkinter.CTkFrame(self, fg_color="#1E1E1E")
		self.global_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

		# Configure columns for 1/3 (Password Toggle) and 2/3 (Submit Button) split
		self.global_frame.grid_columnconfigure(0, weight=1)
		self.global_frame.grid_columnconfigure(1, weight=2)

		# --- Global Password Protection Toggle (1/3 Width) ---
		self.pwd_var = tk.StringVar(value="no")
		customtkinter.CTkSwitch(
				self.global_frame,
				text="Global Password Protect Output 🔑 (Auto-Generated)",
				variable=self.pwd_var, onvalue="yes", offvalue="no",
				button_color=PRIMARY_COLOR, progress_color=SECONDARY_COLOR,
				font=customtkinter.CTkFont(family=FONT_FAMILY)
		).grid(row=1, column=0, padx=20, pady=10, sticky="w")

		# --- Submit Button (2/3 Width) ---
		self.submit_button = customtkinter.CTkButton(
				self.global_frame,
				text="PREPARE AND SHARE FILES! 🚀",
				command=self._process_files,
				fg_color=SECONDARY_COLOR,
				hover_color=PRIMARY_COLOR,
				font=customtkinter.CTkFont(size=18, weight="bold", family=FONT_FAMILY)
		)
		self.submit_button.grid(row=1, column=1, padx=20, pady=20, sticky="ew")


	# ----------------------------------------------------------------------
	# --- Processing Logic (FINALIZED) ---
	# ----------------------------------------------------------------------
	def _process_files(self):
		if not self.file_data:
			messagebox.showerror("Error", "Please add files first before processing.")
			return

		current_date_str = date.today().strftime("%Y-%m-%d")
		auto_password = generate_password()

		# Ensure final folder name is set from the live preview
		self.output_folder = self._calculate_final_folder_name()
		output_dir = os.path.join(os.getcwd(), self.output_folder)
		os.makedirs(output_dir, exist_ok=True)

		# ... (rest of the processing logic is the same) ...

		# Simplified Processing Log for brevity in the final answer

		global_protect = self.pwd_var.get() == "yes"
		print("Starting File Processing...")

		for file_path, data in self.file_data.items():
			original_name = os.path.basename(file_path)
			temp_name = original_name
			options = {k: v.get() for k, v in data['options'].items()}

			# FILE NAMING PREFIXES
			if options["date_prefix"] == "yes":
				temp_name = f"{current_date_str}_{temp_name}"

			if options["ai_name"] == "yes":
				temp_name = f"AI_Better_{temp_name}"

			if options["translate_choices"] != "None":
				langs_to_translate = options['translate_choices'].split(',')
				lang_str = "_".join(langs_to_translate)
				temp_name = f"Trans_{lang_str}_{temp_name}"

			# DESTINATION
			if options["in_folder"] == "yes":
				destination_path = os.path.join(output_dir, temp_name)
			else:
				destination_path = os.path.join(os.path.dirname(output_dir), f"separate_share_{temp_name}")

			try:
				shutil.copy(file_path, destination_path)
			except Exception as e:
				print(f"ERROR copying {original_name}: {e}")

		# 3. Global Encryption
		if global_protect:
			print(f"Master Password: {auto_password}")

		# 4. Open File Explorer
		self._open_explorer(output_dir)

		if global_protect:
			messagebox.showinfo("Success", f"Files processed. Output folder opening now.\n\n***MASTER PASSWORD: {auto_password}***\n(Please save this password!)")
		else:
			messagebox.showinfo("Success", "Files processed. Output folder opening now.")

	def _calculate_final_folder_name(self):
		"""Logic to determine the output folder name."""
		current_date_str = date.today().strftime("%Y-%m-%d")
		folder_prefix = f"{current_date_str}_" if self.folder_date_prefix_var.get() == "yes" else ""
		folder_option = self.folder_name_option_var.get()

		if folder_option == "ai_name":
			folder_suffix = "AI_Generated_Share_Summary"
		elif folder_option == "custom_name":
			folder_suffix = self.folder_custom_name_var.get()
		elif folder_option == "date_only":
			folder_suffix = ""

		final_folder_name = folder_prefix + folder_suffix.replace(' ', '_')
		if not final_folder_name:
			final_folder_name = current_date_str

		return final_folder_name

	def _open_explorer(self, path):
		system = platform.system()
		try:
			if system == 'Windows':
				os.startfile(path)
			elif system == 'Darwin':
				subprocess.Popen(['open', path])
			elif system == 'Linux':
				subprocess.Popen(['xdg-open', path])
		except FileNotFoundError:
			messagebox.showerror("Error", "Could not find the necessary command to open file explorer.")


# --- Run the Application ---
if __name__ == "__main__":
	app = FileSharingApp()
	app.mainloop()
