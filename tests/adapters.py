from __future__ import annotations

import os
from typing import Any
import fasttext
from data_parsing import extract_text, identify_language, mask_email, mask_phone_numbers, mask_ip_addresses, nsfw, toxic, gopher, deduplicate_lines, minhash_deduplication
MODEL_PATH = "retrain_quality_classifier.bin"
_model = None



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)

def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return toxic(text)


def _load_model():
    global _model
    if _model is None:
        _model = fasttext.load_model(MODEL_PATH)
    return _model

def run_classify_quality(text: str) -> tuple[Any, float]:
    model = _load_model()
    
    # Preprocess text to remove newlines (fastText requirement)
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    
    prediction = model.predict(cleaned_text)
    label_raw = prediction[0][0]
    confidence = prediction[1][0]
    
    # Convert fasttext label to readable format
    if label_raw == '__label__positive':
        quality_label = 'wiki'
    else:
        quality_label = 'cc'
    
    return quality_label, float(confidence)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return deduplicate_lines(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
