				def has_audio(file_path, silence_thresh=-50, min_silence_len=1000):
					audio = AudioSegment.from_file(file_path)
					nonsilent = detect_nonsilent(audio,
																			 min_silence_len=min_silence_len,
																			 silence_thresh=silence_thresh)
					# If less than 1% of audio is non-silent, consider it silent
					return len(nonsilent) > 0 and sum(end-start for start, end in nonsilent) > len(audio) * 0.01

Best Transcription Options:
1. faster-whisper (Recommended)

Most accurate AND faster than original Whisper
Uses CTranslate2 optimization
With your 12GB VRAM, run the large-v3 model for maximum accuracy
Supports 99+ languages

bashpip install faster-whisper
pythonfrom faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

import librosa
import numpy as np

def has_meaningful_audio(file_path, threshold=-40):
    y, sr = librosa.load(file_path, sr=None)
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms)
    # Check if significant portion is above threshold
    return np.mean(db) > threshold

from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def transcribe_if_has_audio(file_path):
	# Check for audio first
	audio = AudioSegment.from_file(file_path)
	nonsilent = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-50)

	if not nonsilent or sum(end-start for start, end in nonsilent) < len(audio) * 0.01:
		print("No meaningful audio detected, skipping transcription")
		return None

	# Transcribe with large-v3 for accuracy
	model = WhisperModel("large-v3", device="cuda", compute_type="float16")
	segments, info = model.transcribe(file_path, beam_size=5)

	return list(segments)
