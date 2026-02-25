import logging
import random
from pathlib import Path
from typing import List , Optional

import bitsandbytes  # noqa: F401
import pypdf
import torch
from PIL import Image
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor , BitsAndBytesConfig , Qwen2_5_VLForConditionalGeneration

from config import HUGGING_FACE_TOKEN , QWEN_VL_CPU_FALLBACK , QWEN_VL_MODEL_TIERS
from utilities.dependancy_ensurance import ensure_bitsandbytes , ensure_poppler


def _build_quantization_config( mode: Optional[ str ] ) -> Optional[ "BitsAndBytesConfig" ] :
	"""
	Build a BitsAndBytesConfig for the requested quantization mode.

	Returns None (full precision) if bitsandbytes is not installed rather than
	crashing — the tier selection logic will have already avoided quantized tiers
	in that case, so this is purely a safety net.

	Args:
			mode: None for full precision, "int8" for 8-bit, "nf4" for 4-bit NormalFloat.

	Returns:
			BitsAndBytesConfig instance, or None for full precision.
	"""
	if mode is None :
		return None

	if mode == "int8" :
		return BitsAndBytesConfig( load_in_8bit=True )
	if mode == "nf4" :
		return BitsAndBytesConfig(
				load_in_4bit=True ,
				bnb_4bit_quant_type="nf4" ,
				bnb_4bit_use_double_quant=True ,
				bnb_4bit_compute_dtype=torch.bfloat16 ,
		)
	raise ValueError( f"Unknown quantization mode: {mode!r}" )


def _select_tier_for_vram(
		available_gb: float , logger: logging.Logger ,
) -> dict :
	"""
	Return the best-fitting tier dict for available VRAM.

	Quantized tiers are skipped automatically if bitsandbytes is not installed,
	falling through to the next non-quantized tier instead.

	Args:
			available_gb: Free VRAM in gigabytes.
			logger:       Logger instance.

	Returns:
			Tier dict with keys: model_id, min_vram_gb, quantization.
	"""
	logger.warning(
			"bitsandbytes is not installed — quantized tiers (int8/nf4) will be "
			"skipped. Run `pip install bitsandbytes --prefer-binary` to enable them." ,
	)

	for tier in QWEN_VL_MODEL_TIERS :
		if tier[ "quantization" ] is not None :
			continue  # skip quantized tiers silently; warning already emitted above
		if available_gb >= tier[ "min_vram_gb" ] :
			q_label = tier[ "quantization" ] or "bf16"
			logger.info(
					f"VRAM {available_gb:.1f} GB — selecting "
					f"{tier[ 'model_id' ]} ({q_label}, requires ~{tier[ 'min_vram_gb' ]} GB)" ,
			)
			return tier

	logger.warning(
			f"VRAM {available_gb:.1f} GB is below the minimum GPU tier "
			f"({QWEN_VL_MODEL_TIERS[ -1 ][ 'min_vram_gb' ]} GB). "
			f"Falling back to CPU with {QWEN_VL_CPU_FALLBACK}." ,
	)
	return { "model_id" : QWEN_VL_CPU_FALLBACK , "min_vram_gb" : 0 , "quantization" : None }


def _get_available_vram_gb( ) -> float :
	"""
	Return the available (free) VRAM in GB across all visible CUDA devices.
	Uses the device with the most free memory when multiple GPUs are present.
	Returns 0.0 if no CUDA device is available.
	"""
	if not torch.cuda.is_available( ) :
		return 0.0

	max_free = 0
	for i in range( torch.cuda.device_count( ) ) :
		free , _ = torch.cuda.mem_get_info( i )
		max_free = max( max_free , free )

	return max_free / (1024 ** 3)  # bytes → GB


class VisualProcessor :
	"""
	Qwen2.5-VL processor for PDF and image text extraction and description.

	Automatically selects the largest Qwen2.5-VL model that fits within
	the available GPU VRAM, falling back to CPU if no GPU is detected or
	if VRAM is insufficient for even the smallest tier.

	Processes documents with intelligent page sampling for large PDFs.
	Sends multiple images in a single model call for better context.
	"""

	def __init__(
			self ,
			logger: logging.Logger ,
			model_id: Optional[ str ] = None ,  # None → auto-select from VRAM
			max_pages: int = 6 ,
			max_tokens: int = 512 ,
	) -> None :
		"""
		Initialize processor.

		Args:
				logger:     Logger instance.
				model_id:   HuggingFace model identifier. When None (default) the
										constructor detects available VRAM and picks the best
										fitting model automatically.
				max_pages:  Max pages to process (first, last, + random middle pages).
				max_tokens: Max tokens to generate per inference.
		"""
		ensure_poppler( )
		ensure_bitsandbytes( )

		if not isinstance( logger , logging.Logger ) :
			raise TypeError( f"Expected logging.Logger, got {type( logger )}" )

		self.logger = logger
		self.max_pages = max_pages
		self.max_tokens = max_tokens

		# ── Device & VRAM detection ────────────────────────────────────────
		self.device = "cuda" if torch.cuda.is_available( ) else "cpu"
		self.logger.info( f"Using device: {self.device}" )

		if self.device == "cuda" :
			available_gb = _get_available_vram_gb( )
			self.logger.info( f"Free VRAM detected: {available_gb:.1f} GB" )
		else :
			available_gb = 0.0
			self.logger.info( "No CUDA device found — running on CPU." )

		# ── Model selection ────────────────────────────────────────────────
		if model_id is not None :
			# Caller supplied an explicit model; use full precision, no quantization.
			resolved_model = model_id
			quant_config = None
			if self.device == "cpu" :
				self.logger.warning(
						"Explicit model_id provided but no GPU is available. "
						"Inference will be slow on CPU." ,
				)
		else :
			tier = _select_tier_for_vram( available_gb , self.logger )
			resolved_model = tier[ "model_id" ]
			quant_config = _build_quantization_config( tier[ "quantization" ] )

			# If nothing fit in VRAM, switch fully to CPU
			if available_gb > 0 and tier[ "min_vram_gb" ] == 0 :
				self.device = "cpu"

		# ── Load model ────────────────────────────────────────────────────
		# Quantized models manage their own dtype internally; only set
		# torch_dtype for full-precision loads to avoid conflicting configs.
		self.logger.info( f"Loading {resolved_model}" )
		load_kwargs = dict(
				device_map="auto" if self.device == "cuda" else None ,
				token=HUGGING_FACE_TOKEN ,
		)
		if quant_config is not None :
			load_kwargs[ "quantization_config" ] = quant_config
		else :
			load_kwargs[ "torch_dtype" ] = (
				torch.bfloat16 if self.device == "cuda" else torch.float32
			)

		self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
				resolved_model , **load_kwargs ,
		)
		self.processor = AutoProcessor.from_pretrained(
				resolved_model , token=HUGGING_FACE_TOKEN ,
		)
		self.logger.info( "Model loaded successfully." )

	# ── Private helpers ────────────────────────────────────────────────────

	def _load_images(
			self , file_path: Path , subset_size: Optional[ int ] = None ,
	) -> List[ Image.Image ] :
		"""
		Load images from PDF or image file with page sampling.

		For PDFs exceeding max_pages:
		- Always includes first and last page
		- Randomly samples remaining pages from middle

		Args:
				file_path:   Path to file.
				subset_size: Override max_pages for this file.

		Returns:
				List of PIL Images.
		"""
		ext = file_path.suffix.lower( )

		if ext == ".pdf" :
			self.logger.info( f"Converting PDF: {file_path.name}" )
			images = convert_from_path( str( file_path ) )
			total = len( images )
			max_allowed = subset_size if subset_size else self.max_pages

			if total > max_allowed :
				indices = [ 0 , total - 1 ]
				middle_count = max_allowed - 2
				if middle_count > 0 and total > 2 :
					middle = list( range( 1 , total - 1 ) )
					indices.extend(
							random.sample( middle , min( middle_count , len( middle ) ) ) ,
					)
				indices.sort( )
				images = [ images[ i ] for i in indices ]
				self.logger.info( f"Sampled {len( images )} of {total} pages: {indices}" )
			else :
				self.logger.info( f"Processing all {total} pages" )

			return images

		elif ext in [ ".jpg" , ".jpeg" , ".png" , ".bmp" , ".tiff" , ".webp" ] :
			return [ Image.open( file_path ) ]

		else :
			raise ValueError( f"Unsupported file type: {ext}" )

	def _generate( self , images: List[ Image.Image ] , prompt: str ) -> str :
		"""
		Generate text from images using the loaded Qwen2.5-VL model.

		Args:
				images: List of PIL Images.
				prompt: Text prompt.

		Returns:
				Generated text.
		"""
		content = [ { "type" : "image" , "image" : img } for img in images ]
		content.append( { "type" : "text" , "text" : prompt } )
		messages = [ { "role" : "user" , "content" : content } ]

		text = self.processor.apply_chat_template(
				messages , tokenize=False , add_generation_prompt=True ,
		)
		image_inputs , video_inputs = process_vision_info( messages )

		inputs = self.processor(
				text=[ text ] ,
				images=image_inputs ,
				videos=video_inputs ,
				padding=True ,
				return_tensors="pt" ,
		).to( self.device )

		with torch.no_grad( ) :
			generated_ids = self.model.generate(
					**inputs , max_new_tokens=self.max_tokens ,
			)

		generated_ids_trimmed = [
			out_ids[ len( in_ids ) : ]
			for in_ids , out_ids in zip( inputs.input_ids , generated_ids )
		]

		output = self.processor.batch_decode(
				generated_ids_trimmed ,
				skip_special_tokens=True ,
				clean_up_tokenization_spaces=False ,
		)[ 0 ]

		return output.strip( )

	# ── Public API ─────────────────────────────────────────────────────────

	def extract_text( self , file_path: Path , subset_size: Optional[ int ] = None ) -> str :
		"""
		Extract text from document.

		Args:
				file_path:   Path to PDF or image.
				subset_size: Override max_pages.

		Returns:
				Extracted text.
		"""
		self.logger.info( f"Extracting text: {file_path.name}" )
		images = self._load_images( file_path , subset_size )
		prompt = (
			"Extract all text from these images. "
			"If multiple pages, extract text from each page separately. "
			"Return only the text content with clear page separations."
		)
		return self._generate( images , prompt )

	def extract_visual_description(
			self , file_path: Path , subset_size: Optional[ int ] = None ,
	) -> str :
		"""
		Generate visual descriptions of document.

		Args:
				file_path:   Path to PDF or image.
				subset_size: Override max_pages.

		Returns:
				Visual descriptions.
		"""
		self.logger.info( f"Describing: {file_path.name}" )
		images = self._load_images( file_path , subset_size )
		prompt = (
			"Describe the visual elements in these images. "
			"Include layout, colors, text formatting, charts, diagrams. "
			"If multiple pages, describe each separately."
		)
		return self._generate( images , prompt )


def compile_doc_subset(
		input_pdf: Path ,
		subset_size: int ,
		temp_dir: Path ,
) -> Path :
	reader = pypdf.PdfReader( str( input_pdf ) )
	total_pages = len( reader.pages )

	if total_pages <= subset_size :
		return input_pdf

	first , last = 0 , total_pages - 1
	middle_indices = list( range( 1 , last ) )  # everything except first and last
	sampled = random.sample( middle_indices , min( subset_size - 2 , len( middle_indices ) ) )
	selected = sorted( [ first ] + sampled + [ last ] )

	writer = pypdf.PdfWriter( )
	for i in selected :
		writer.add_page( reader.pages[ i ] )

	output_path = Path( temp_dir ) / f"SUBSET-{input_pdf.stem}.pdf"
	with open( output_path , "wb" ) as f :
		writer.write( f )

	return output_path
