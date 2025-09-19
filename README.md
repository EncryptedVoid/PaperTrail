# 📄✨ PaperTrail: The Document Processing Beast 🚀

> *"Drop your documents in, get a magical spreadsheet out!"* ✨

## 🤔 What Even Is This Thing?

PaperTrail is an **absolutely bonkers** document processing pipeline that takes your messy pile of files and transforms them into a beautifully organized spreadsheet database! 🗂️➡️📊

Think of it as a digital filing assistant that:
- 🔍 **Sniffs out duplicates** (using fancy SHA3-512 checksums)
- 🏷️ **Labels everything** with unique UUIDs
- 🧠 **Reads your documents** with AI vision models
- 📝 **Extracts structured data** using local LLM models
- 📈 **Spits out spreadsheets** ready for your enjoyment

## 🎪 The 8-Stage Circus Performance

Your documents go through this wild ride:

### 🎭 Stage 1: The Bouncer
- Kicks out duplicates and zero-byte files
- Moves unsupported files to the "review" folder
- Maintains a permanent checksum history (no duplicates allowed!)

### 🏷️ Stage 2: The Name Tag Party
- Gives every document a shiny new UUID name
- Creates a JSON profile for each file
- Preserves original filenames for posterity

### 📊 Stage 3: The Data Detective
- Extracts technical metadata from files
- Handles images, PDFs, Word docs, Excel files, and more
- Gets file size, creation dates, EXIF data - the works!

### 👁️ Stage 4: The Vision Wizard
- Uses Qwen2-VL models to "see" your documents
- Extracts all text via OCR magic
- Describes visual elements in detail
- Auto-detects your hardware and picks the best model

### 🧠 Stage 5: The Smart Librarian
- Uses local OLLAMA models to extract structured fields
- Pulls out titles, document types, dates, people names
- Creates searchable tags and metadata
- Hardware-aware model selection (because efficiency matters!)

### ✅ Stage 6: The Completion Ceremony
- Marks documents as fully processed
- Creates processing summaries
- Moves everything to the "completed" folder

### 📈 Stage 7: The Spreadsheet Factory
- Generates beautiful Excel and CSV files
- Organizes all extracted data into columns
- Creates a searchable database of your documents

### 🔐 Stage 8: The Vault (Coming Soon™)
- *Encryption capabilities exist but aren't wired up yet*

## 🎯 What You Get Out of It

**Input:** A folder full of random documents 📁
**Output:** Professional spreadsheet database with extracted metadata 📊

The final spreadsheet includes columns like:
- 📋 Title, Document Type, Language
- 👥 Issuer, Translator, Official Authority
- 📅 Creation Date, Issue Date, Expiry Date
- 🏷️ Tags, Notes, Confidentiality Level
- 🔧 Technical metadata (file size, checksums, etc.)

## 🛠️ What You Need to Run This Beast

- 🐍 Python 3.8+
- 🤖 OLLAMA running locally
- 🖥️ Decent hardware (it auto-detects and optimizes)
- 📦 A bunch of Python packages (transformers, torch, PIL, etc.)

## 📁 The Folder Structure Dance

The pipeline creates this beautiful folder hierarchy:
```
📂 test_run/
├── 📁 unprocessed_artifacts/     (Drop files here!)
├── 📁 identified_artifacts/
├── 📁 metadata_extracted/
├── 📁 visually_processed/
├── 📁 processed_artifacts/       (Final processed files)
├── 📁 artifact_profiles/         (JSON profiles)
├── 📁 session_logs/              (All the juicy logs)
├── 📁 review_required/           (Problem files)
└── 📊 PaperTrail_Artifact_Registry_TIMESTAMP.xlsx (THE PRIZE!)
```

## 🎮 How to Use It

1. 📥 Drop your documents in `unprocessed_artifacts/`
2. 🏃‍♂️ Run `python papertrail.py`
3. ☕ Grab coffee and watch the magic happen
4. 🎉 Find your shiny new spreadsheet in the base folder!

## 🚨 Current State: "It Actually Works!"

This is a **functional prototype** that:
- ✅ Successfully processes PDFs, images, and Office docs
- ✅ Uses real AI models for text extraction and field parsing
- ✅ Creates actual usable spreadsheets
- ✅ Has comprehensive logging and error handling
- ✅ Automatically optimizes for your hardware

## 🎪 Fun Features That Actually Exist

- 🧮 **Hardware Auto-Detection:** Figures out your GPU/RAM and picks models accordingly
- 🔄 **Model Refreshing:** Prevents AI brain fog by refreshing models periodically
- 📊 **Progress Tracking:** Real-time stats on processing speed and success rates
- 🔍 **Duplicate Prevention:** Uses cryptographic checksums (because we're fancy)
- 📝 **Session Logging:** Detailed logs of everything that happened
- 🎯 **Quality Metrics:** Tracks how well the AI extraction is working

## 🎊 The Bottom Line

PaperTrail is a surprisingly capable document processing pipeline that actually works! It's like having a very patient digital assistant that never gets tired of organizing your paperwork. Drop in your documents, wait for the AI magic to happen, and get back a professional database spreadsheet.

*Perfect for anyone drowning in documents and wanting to bring some order to the chaos!* 🌪️➡️📋✨