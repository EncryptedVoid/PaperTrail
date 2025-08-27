from ocr_extractor import OCRExtractor

# Initialize
extractor = OCRExtractor()

# Extract text from a single image
result = extractor.extract_with_easyocr("document.jpg")
print(result["text"])

# Compare all three methods
all_results = extractor.extract_all_methods("document.jpg")
