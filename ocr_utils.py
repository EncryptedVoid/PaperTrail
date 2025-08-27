import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# OCR imports
import pytesseract
import easyocr
from paddleocr import PaddleOCR

# Additional imports for different file formats
try:
    from pdf2image import convert_from_path

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    try:
        import pyheif

        HEIC_SUPPORT = True
    except ImportError:
        HEIC_SUPPORT = False


class OCREngine(Enum):
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    PYTESSERACT = "pytesseract"


class FileFormat(Enum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    TIFF = ".tiff"
    TIF = ".tif"
    BMP = ".bmp"
    WEBP = ".webp"
    PDF = ".pdf"
    HEIC = ".heic"
    HEIF = ".heif"
    GIF = ".gif"
    SVG = ".svg"
    ICO = ".ico"
    TGA = ".tga"
    PCX = ".pcx"


@dataclass
class FileMetadata:
    filename: str
    full_path: str
    file_size_bytes: int
    file_size_mb: float
    created: str
    modified: str
    file_hash: str
    image_width: Union[int, str]
    image_height: Union[int, str]
    color_mode: Union[str, None]
    image_format: Union[str, None]
    file_extension: str
    page_number: Optional[int] = None


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[Union[int, float]]
    length: int


@dataclass
class EngineResult:
    full_text: str
    word_count: int
    character_count: int
    average_confidence: float
    total_detections: int
    detailed_results: List[OCRResult]
    error: Optional[str] = None
    language_info: Optional[str] = None


@dataclass
class ProcessingResult:
    timestamp: str
    file_info: FileMetadata
    ocr_results: Dict[str, Union[EngineResult, Dict[str, Any]]]
    analysis: Dict[str, Any]
    processing_summary: List[str]
    total_pages: Optional[int] = None
    individual_pages: Optional[List["ProcessingResult"]] = None


class ImagePreprocessor:
    @staticmethod
    def enhance_image_quality(image_path: str, target_dpi: int = 300) -> np.ndarray:
        """Enhance image quality for better OCR accuracy"""
        img = cv2.imread(image_path)

        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Resize if image is too small or too large
        height, width = img.shape[:2]

        # Target dimensions for optimal OCR (around 2000px width for good quality)
        if width < 800:
            scale_factor = 800 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        elif width > 4000:
            scale_factor = 4000 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    @staticmethod
    def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for better OCR results"""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned


class FormatConverter:
    @staticmethod
    def convert_pdf_to_images(pdf_path: str, temp_dir: str) -> List[str]:
        """Convert PDF pages to images"""
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support not available. Install: pip install pdf2image"
            )

        try:
            images = convert_from_path(pdf_path, dpi=300, fmt="PNG")
            temp_files: List[str] = []

            for i, image in enumerate(images):
                temp_file = os.path.join(temp_dir, f"page_{i+1:03d}.png")
                image.save(temp_file, "PNG")
                temp_files.append(temp_file)

            print(f"Converted PDF to {len(temp_files)} page images")
            return temp_files

        except Exception as e:
            raise RuntimeError(f"PDF conversion failed: {e}")

    @staticmethod
    def convert_heic_to_image(heic_path: str, temp_dir: str) -> List[str]:
        """Convert HEIC file to PNG"""
        if not HEIC_SUPPORT:
            raise ImportError(
                "HEIC support not available. Install: pip install pillow-heif"
            )

        try:
            try:
                # Method 1: pillow-heif
                image = Image.open(heic_path)
            except Exception:
                # Method 2: pyheif fallback
                import pyheif

                heif_file = pyheif.read(heic_path)
                image = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )

            temp_file = os.path.join(temp_dir, "converted.png")
            image.save(temp_file, "PNG")

            print("Converted HEIC to PNG")
            return [temp_file]

        except Exception as e:
            raise RuntimeError(f"HEIC conversion failed: {e}")

    @staticmethod
    def convert_unsupported_to_png(file_path: str, temp_dir: str) -> List[str]:
        """Convert unsupported formats to PNG using PIL"""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")

                temp_file = os.path.join(temp_dir, "converted.png")
                img.save(temp_file, "PNG", quality=95)

            print(f"Converted {os.path.splitext(file_path)[1]} to PNG")
            return [temp_file]

        except Exception as e:
            raise RuntimeError(f"Format conversion failed: {e}")


class FileProcessor:
    def __init__(self) -> None:
        self.temp_dirs: List[str] = []
        self.preprocessor = ImagePreprocessor()
        self.converter = FormatConverter()

    def cleanup_temp_dirs(self) -> None:
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not delete temp directory {temp_dir}: {e}")
        self.temp_dirs.clear()

    def prepare_file_for_ocr(self, file_path: str) -> List[str]:
        """Prepare file for OCR by converting if necessary"""
        file_ext = os.path.splitext(file_path)[1].lower()

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="ocr_temp_")
        self.temp_dirs.append(temp_dir)

        try:
            if file_ext == FileFormat.PDF.value:
                return self.converter.convert_pdf_to_images(file_path, temp_dir)
            elif file_ext in [FileFormat.HEIC.value, FileFormat.HEIF.value]:
                return self.converter.convert_heic_to_image(file_path, temp_dir)
            elif file_ext in [
                fmt.value
                for fmt in [
                    FileFormat.PNG,
                    FileFormat.JPG,
                    FileFormat.JPEG,
                    FileFormat.TIFF,
                    FileFormat.TIF,
                    FileFormat.BMP,
                    FileFormat.WEBP,
                ]
            ]:
                # Standard formats - enhance and return
                enhanced_file = os.path.join(temp_dir, f"enhanced{file_ext}")
                enhanced_img = self.preprocessor.enhance_image_quality(file_path)
                cv2.imwrite(enhanced_file, enhanced_img)
                return [enhanced_file]
            else:
                # Try to convert unsupported formats
                print(f"Attempting to convert unsupported format: {file_ext}")
                return self.converter.convert_unsupported_to_png(file_path, temp_dir)

        except Exception as e:
            print(f"File preparation failed: {e}")
            return []


class OCREngineManager:
    def __init__(self) -> None:
        self.engines: Dict[str, Any] = {}
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize all OCR engines"""
        try:
            self.engines["easyocr"] = easyocr.Reader(
                ["en", "fr", "es", "de"], gpu=False
            )
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            self.engines["easyocr"] = None

        try:
            self.engines["paddleocr"] = {
                "en": PaddleOCR(use_angle_cls=True, lang="en", show_log=False),
                "fr": PaddleOCR(use_angle_cls=True, lang="fr", show_log=False),
                "es": PaddleOCR(use_angle_cls=True, lang="es", show_log=False),
                "de": PaddleOCR(use_angle_cls=True, lang="de", show_log=False),
            }
        except Exception as e:
            print(f"PaddleOCR initialization failed: {e}")
            self.engines["paddleocr"] = None

    def run_easyocr(self, image_path: str) -> Dict[str, Any]:
        """Run EasyOCR with multilingual support"""
        reader = self.engines.get("easyocr")
        if reader is None:
            return {"error": "EasyOCR not initialized"}

        try:
            results = reader.readtext(image_path, detail=1, paragraph=False)

            extracted_data: List[OCRResult] = []
            full_text_pieces: List[str] = []
            total_confidence = 0.0

            for bbox, text, confidence in results:
                ocr_result = OCRResult(
                    text=text, confidence=float(confidence), bbox=bbox, length=len(text)
                )
                extracted_data.append(ocr_result)
                full_text_pieces.append(text)
                total_confidence += confidence

            avg_confidence = total_confidence / len(results) if results else 0.0

            return {
                "full_text": " ".join(full_text_pieces),
                "word_count": len(" ".join(full_text_pieces).split()),
                "character_count": len("".join(full_text_pieces)),
                "average_confidence": avg_confidence,
                "total_detections": len(results),
                "detailed_results": [
                    {
                        "text": r.text,
                        "confidence": r.confidence,
                        "bbox": r.bbox,
                        "length": r.length,
                    }
                    for r in extracted_data
                ],
                "languages_detected": "Multiple (en/fr/es/de supported)",
            }

        except Exception as e:
            return {"error": f"EasyOCR processing failed: {str(e)}"}

    def run_paddleocr(self, image_path: str) -> Dict[str, Any]:
        """Run PaddleOCR with multiple language attempts"""
        ocr_engines = self.engines.get("paddleocr")
        if ocr_engines is None:
            return {"error": "PaddleOCR not initialized"}

        results_by_language: Dict[str, Dict[str, Any]] = {}
        best_result: Optional[Dict[str, Any]] = None
        best_confidence = 0.0

        for lang_code, ocr_engine in ocr_engines.items():
            try:
                results = ocr_engine.ocr(image_path, cls=True)

                if results and results[0]:
                    text_pieces: List[str] = []
                    confidences: List[float] = []
                    detailed_data: List[Dict[str, Any]] = []

                    for line in results[0]:
                        bbox, (text, confidence) = line
                        text_pieces.append(text)
                        confidences.append(float(confidence))
                        detailed_data.append(
                            {
                                "text": text,
                                "confidence": float(confidence),
                                "bbox": (
                                    bbox.tolist() if hasattr(bbox, "tolist") else bbox
                                ),
                            }
                        )

                    full_text = " ".join(text_pieces)
                    avg_confidence = (
                        sum(confidences) / len(confidences) if confidences else 0.0
                    )

                    lang_result = {
                        "language": lang_code,
                        "full_text": full_text,
                        "word_count": len(full_text.split()),
                        "character_count": len(full_text.replace(" ", "")),
                        "average_confidence": avg_confidence,
                        "total_detections": len(detailed_data),
                        "detailed_results": detailed_data,
                    }

                    results_by_language[lang_code] = lang_result

                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_result = lang_result
                else:
                    results_by_language[lang_code] = {"error": "No text detected"}

            except Exception as e:
                results_by_language[lang_code] = {"error": str(e)}

        return {
            "best_result": best_result,
            "all_languages": results_by_language,
            "best_language": best_result["language"] if best_result else "none",
        }

    def run_pytesseract(
        self, image_path: str, processed_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Run PyTesseract with multiple language configurations"""
        try:
            if processed_image is not None:
                img = Image.fromarray(processed_image)
            else:
                img = Image.open(image_path)

            lang_configs = ["eng+fra+spa+deu", "eng", "fra", "spa", "deu"]

            results_by_config: Dict[str, Dict[str, Any]] = {}
            best_result: Optional[Dict[str, Any]] = None
            best_word_count = 0

            for lang in lang_configs:
                try:
                    text = pytesseract.image_to_string(img, lang=lang).strip()
                    data = pytesseract.image_to_data(
                        img, lang=lang, output_type=pytesseract.Output.DICT
                    )

                    confident_words: List[Dict[str, Any]] = []
                    total_conf = 0.0
                    conf_count = 0

                    for i in range(len(data["text"])):
                        confidence = int(data["conf"][i])
                        if confidence > 30:
                            word_info = {
                                "text": data["text"][i],
                                "confidence": confidence,
                                "bbox": [
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                ],
                            }
                            confident_words.append(word_info)
                            total_conf += confidence
                            conf_count += 1

                    avg_confidence = total_conf / conf_count if conf_count > 0 else 0.0
                    word_count = len(text.split())

                    lang_result = {
                        "language_config": lang,
                        "full_text": text,
                        "word_count": word_count,
                        "character_count": len(text.replace(" ", "")),
                        "average_confidence": avg_confidence,
                        "confident_words": confident_words,
                        "total_detections": len(confident_words),
                    }

                    results_by_config[lang] = lang_result

                    if word_count > best_word_count:
                        best_word_count = word_count
                        best_result = lang_result

                except Exception as e:
                    results_by_config[lang] = {"error": str(e)}

            return {
                "best_result": best_result,
                "all_configs": results_by_config,
                "best_language": (
                    best_result["language_config"] if best_result else "none"
                ),
            }

        except Exception as e:
            return {"error": f"PyTesseract processing failed: {str(e)}"}


class ResultAnalyzer:
    @staticmethod
    def analyze_ocr_results(
        ocr_results: Dict[str, Union[Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """Analyze and compare results from all OCR engines"""
        analysis: Dict[str, Any] = {
            "engines_succeeded": [],
            "engines_failed": [],
            "text_comparison": {},
            "confidence_comparison": {},
            "recommendations": [],
        }

        for engine, result in ocr_results.items():
            if isinstance(result, dict) and "error" in result:
                analysis["engines_failed"].append(engine)
            else:
                analysis["engines_succeeded"].append(engine)

        texts: Dict[str, str] = {}
        confidences: Dict[str, float] = {}

        for engine in analysis["engines_succeeded"]:
            result = ocr_results[engine]
            if isinstance(result, dict):
                if engine == "easyocr":
                    texts[engine] = result.get("full_text", "")
                    confidences[engine] = result.get("average_confidence", 0.0)
                elif engine in ["paddleocr", "pytesseract"]:
                    best = result.get("best_result", {})
                    texts[engine] = best.get("full_text", "") if best else ""
                    confidences[engine] = (
                        best.get("average_confidence", 0.0) if best else 0.0
                    )

        analysis["text_comparison"] = texts
        analysis["confidence_comparison"] = confidences

        if confidences:
            best_engine = max(confidences, key=confidences.get)
            best_conf = confidences[best_engine]
            analysis["recommendations"].append(
                f"Highest confidence: {best_engine} ({best_conf:.2f})"
            )

            if len(texts) > 1:
                lengths = {k: len(v.split()) for k, v in texts.items()}
                longest_text = max(lengths, key=lengths.get)
                analysis["recommendations"].append(
                    f"Most text detected: {longest_text} ({lengths[longest_text]} words)"
                )

        return analysis

    @staticmethod
    def create_processing_summary(
        ocr_results: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[str]:
        """Create a human-readable summary"""
        summary: List[str] = []

        succeeded = len(analysis["engines_succeeded"])
        failed = len(analysis["engines_failed"])
        summary.append(
            f"OCR Processing: {succeeded} engines succeeded, {failed} failed"
        )

        if analysis["confidence_comparison"]:
            best_engine = max(
                analysis["confidence_comparison"],
                key=analysis["confidence_comparison"].get,
            )
            best_conf = analysis["confidence_comparison"][best_engine]
            summary.append(f"Best confidence: {best_engine} ({best_conf:.1%})")

        texts = analysis["text_comparison"]
        if texts:
            total_chars = sum(len(text.replace(" ", "")) for text in texts.values())
            avg_chars = total_chars // len(texts) if texts else 0
            summary.append(f"Average characters detected: {avg_chars}")

        return summary


class MultilingualOCRProcessor:
    def __init__(self) -> None:
        self.file_processor = FileProcessor()
        self.ocr_manager = OCREngineManager()
        self.analyzer = ResultAnalyzer()
        self.preprocessor = ImagePreprocessor()

    def get_file_metadata(
        self, file_path: str, page_number: Optional[int] = None
    ) -> FileMetadata:
        """Extract detailed file metadata"""
        stat = os.stat(file_path)

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_info = img.format
        except Exception:
            width = height = "Unknown"
            mode = format_info = None

        return FileMetadata(
            filename=os.path.basename(file_path),
            full_path=os.path.abspath(file_path),
            file_size_bytes=stat.st_size,
            file_size_mb=round(stat.st_size / (1024 * 1024), 2),
            created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            file_hash=file_hash,
            image_width=width,
            image_height=height,
            color_mode=mode,
            image_format=format_info,
            file_extension=os.path.splitext(file_path)[1].lower(),
            page_number=page_number,
        )

    def process_multilingual_file(
        self, file_path: str, output_dir: str = "ocr_results"
    ) -> Dict[str, Any]:
        """Main processing function"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            processed_files = self.file_processor.prepare_file_for_ocr(file_path)

            if not processed_files:
                return {"error": "File preparation failed"}

            all_results: List[Dict[str, Any]] = []

            for i, image_path in enumerate(processed_files):
                print(f"Processing page/image {i+1}/{len(processed_files)}")

                file_info = self.get_file_metadata(
                    file_path, page_number=i + 1 if len(processed_files) > 1 else None
                )

                processed_image = self.preprocessor.preprocess_for_ocr(
                    self.preprocessor.enhance_image_quality(image_path)
                )

                ocr_results: Dict[str, Any] = {}

                print("  Running OCR engines...")

                try:
                    print("    - EasyOCR...")
                    ocr_results["easyocr"] = self.ocr_manager.run_easyocr(image_path)
                except Exception as e:
                    ocr_results["easyocr"] = {"error": str(e)}

                try:
                    print("    - PaddleOCR...")
                    ocr_results["paddleocr"] = self.ocr_manager.run_paddleocr(
                        image_path
                    )
                except Exception as e:
                    ocr_results["paddleocr"] = {"error": str(e)}

                try:
                    print("    - PyTesseract...")
                    ocr_results["pytesseract"] = self.ocr_manager.run_pytesseract(
                        image_path, processed_image
                    )
                except Exception as e:
                    ocr_results["pytesseract"] = {"error": str(e)}

                analysis = self.analyzer.analyze_ocr_results(ocr_results)

                page_results = {
                    "page_number": i + 1,
                    "timestamp": datetime.now().isoformat(),
                    "file_info": file_info.__dict__,
                    "ocr_results": ocr_results,
                    "analysis": analysis,
                    "processing_summary": self.analyzer.create_processing_summary(
                        ocr_results, analysis
                    ),
                }

                all_results.append(page_results)

            final_results = self._combine_multi_page_results(all_results, file_path)
            output_file = self._save_results_to_file(
                final_results, file_path, output_dir
            )

            print(f"Results saved to: {output_file}")
            return final_results

        finally:
            self.file_processor.cleanup_temp_dirs()

    def _combine_multi_page_results(
        self, all_results: List[Dict[str, Any]], original_file: str
    ) -> Dict[str, Any]:
        """Combine results from multiple pages"""
        if len(all_results) == 1:
            return all_results[0]

        combined_text: Dict[str, List[str]] = {}
        total_confidence: Dict[str, List[float]] = {}
        engines = ["easyocr", "paddleocr", "pytesseract"]

        for engine in engines:
            combined_text[engine] = []
            total_confidence[engine] = []

        for page_result in all_results:
            for engine in engines:
                ocr_result = page_result["ocr_results"].get(engine, {})

                if "error" not in ocr_result:
                    if engine == "easyocr":
                        text = ocr_result.get("full_text", "")
                        conf = ocr_result.get("average_confidence", 0.0)
                    else:
                        best_result = ocr_result.get("best_result", {})
                        text = best_result.get("full_text", "") if best_result else ""
                        conf = (
                            best_result.get("average_confidence", 0.0)
                            if best_result
                            else 0.0
                        )

                    if text.strip():
                        combined_text[engine].append(
                            f"[Page {page_result['page_number']}] {text}"
                        )
                        total_confidence[engine].append(conf)

        combined_ocr_results: Dict[str, Any] = {}
        for engine in engines:
            if combined_text[engine]:
                avg_conf = sum(total_confidence[engine]) / len(total_confidence[engine])
                full_text = "\n\n".join(combined_text[engine])

                combined_ocr_results[engine] = {
                    "full_text": full_text,
                    "average_confidence": avg_conf,
                    "total_pages": len(all_results),
                    "pages_with_text": len(combined_text[engine]),
                }
            else:
                combined_ocr_results[engine] = {"error": "No text found on any page"}

        analysis = self.analyzer.analyze_ocr_results(combined_ocr_results)

        return {
            "timestamp": datetime.now().isoformat(),
            "file_info": self.get_file_metadata(original_file).__dict__,
            "total_pages": len(all_results),
            "individual_pages": all_results,
            "combined_results": {
                "ocr_results": combined_ocr_results,
                "analysis": analysis,
                "processing_summary": self.analyzer.create_processing_summary(
                    combined_ocr_results, analysis
                ),
            },
        }

    def _save_results_to_file(
        self, results: Dict[str, Any], original_file_path: str, output_dir: str
    ) -> str:
        """Save results to timestamped text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.splitext(os.path.basename(original_file_path))[0]
        output_filename = f"ocr_results_{original_name}_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MULTILINGUAL OCR PROCESSING RESULTS\n")
            f.write("=" * 80 + "\n\n")

            is_multipage = "total_pages" in results

            if is_multipage:
                self._write_multipage_results(f, results)
            else:
                self._write_single_page_results(f, results)

            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")

        return output_path

    def _write_multipage_results(self, f, results: Dict[str, Any]) -> None:
        """Write multipage results to file"""
        file_info = results["file_info"]
        f.write(f"Processing Time: {results['timestamp']}\n")
        f.write(f"Original File: {file_info['filename']}\n")
        f.write(f"File Type: {file_info['file_extension'].upper()}\n")
        f.write(f"File Size: {file_info['file_size_mb']} MB\n")
        f.write(f"Total Pages: {results['total_pages']}\n")
        f.write(f"File Hash: {file_info['file_hash']}\n\n")

        combined_results = results["combined_results"]

        f.write("OVERALL PROCESSING SUMMARY\n")
        f.write("-" * 40 + "\n")
        for item in combined_results["processing_summary"]:
            f.write(f"• {item}\n")
        f.write("\n")

        f.write("COMBINED OCR RESULTS (ALL PAGES)\n")
        f.write("=" * 50 + "\n\n")

        for engine, result in combined_results["ocr_results"].items():
            f.write(f"{engine.upper()}\n")
            f.write("-" * 20 + "\n")

            if "error" in result:
                f.write(f"Error: {result['error']}\n\n")
                continue

            f.write(
                f"Pages with text: {result.get('pages_with_text', 0)}/{result.get('total_pages', 0)}\n"
            )
            f.write(f"Average Confidence: {result.get('average_confidence', 0):.1%}\n")
            f.write(f"Combined Text:\n{result.get('full_text', 'No text')}\n\n")

    def _write_single_page_results(self, f, results: Dict[str, Any]) -> None:
        """Write single page results to file"""
        file_info = results["file_info"]
        f.write(f"Processing Time: {results['timestamp']}\n")
        f.write(f"Original File: {file_info['filename']}\n")
        f.write(f"File Size: {file_info['file_size_mb']} MB\n")
        f.write(
            f"Image Dimensions: {file_info['image_width']} x {file_info['image_height']}\n"
        )
        f.write(f"File Hash: {file_info['file_hash']}\n\n")

        f.write("PROCESSING SUMMARY\n")
        f.write("-" * 40 + "\n")
        for item in results["processing_summary"]:
            f.write(f"• {item}\n")
        f.write("\n")

        f.write("OCR RESULTS BY ENGINE\n")
        f.write("=" * 40 + "\n\n")

        for engine, result in results["ocr_results"].items():
            f.write(f"{engine.upper()}\n")
            f.write("-" * 20 + "\n")

            if "error" in result:
                f.write(f"Error: {result['error']}\n\n")
                continue

            if engine == "easyocr":
                f.write(f"Text: {result.get('full_text', 'No text')}\n")
                f.write(f"Confidence: {result.get('average_confidence', 0):.1%}\n")
                f.write(f"Word Count: {result.get('word_count', 0)}\n\n")
            elif engine in ["paddleocr", "pytesseract"]:
                best = result.get("best_result", {})
                lang_key = "best_language" if engine == "paddleocr" else "best_language"
                f.write(f"Best Config: {result.get(lang_key, 'unknown')}\n")
                f.write(
                    f"Text: {best.get('full_text', 'No text') if best else 'No text'}\n"
                )
                f.write(f"Confidence: {best.get('average_confidence', 0):.1%}\n")
                f.write(f"Word Count: {best.get('word_count', 0)}\n\n")


def process_multilingual_file(
    file_path: str, output_dir: str = "ocr_results"
) -> Dict[str, Any]:
    """Main entry point function"""
    processor = MultilingualOCRProcessor()
    return processor.process_multilingual_file(file_path, output_dir)


def batch_process_files(
    file_paths: List[str], output_dir: str = "batch_ocr_results"
) -> List[Dict[str, Any]]:
    """Process multiple files in batch"""
    results: List[Dict[str, Any]] = []

    print(f"Batch processing {len(file_paths)} files...")
    print(f"PDF support: {'✓' if PDF_SUPPORT else '✗ (install pdf2image)'}")
    print(f"HEIC support: {'✓' if HEIC_SUPPORT else '✗ (install pillow-heif)'}")
    print("-" * 60)

    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
        print("=" * 60)

        try:
            result = process_multilingual_file(file_path, output_dir)
            results.append({"file": file_path, "status": "success", "result": result})
            print("✓ Successfully processed")
        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")
            results.append({"file": file_path, "status": "failed", "error": str(e)})

    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(results)} | Success: {successful} | Failed: {failed}")

    return results


if __name__ == "__main__":
    # Example usage
    example_files = [
        # r"samples\ACFrOgAAXYXVopawF1uC8XvhakYkE2U54MUEakYRpVeP0qNTo9Wm-OsbOVPp2haoRTV9gjj-Y61Yh_zB279emeOzlTW-J9UX302T7YFPqxuVK-ZhpohgJtYSk_RzOdh-s99ARXxbFiq5J0FGQRgOFMqqetTgzfN2_VS6C7NuWg==.pdf",
        r"samples\Adobe Scan Aug 23, 2025 (3).pdf",
        r"samples\Adobe Scan Aug 23, 2025 (5).pdf",
        # r"samples\2024 WORK PERMIT.JPG",
        # r"receipt.jpg",
    ]

    existing_files = [f for f in example_files if os.path.exists(f)]
    if existing_files:
        batch_results = batch_process_files(existing_files)
    else:
        print("No example files found. Update file paths in the __main__ section.")
