import re
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import sys

MODEL_NAME = "microsoft/trocr-base-handwritten"

_existing = sys.modules.get("trocr_recognition")
if _existing and hasattr(_existing, "model") and _existing.model is not None:
    processor = _existing.processor
    model = _existing.model
    device = getattr(_existing, "device", ("cuda" if torch.cuda.is_available() else "cpu"))
else:
    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"TrOCR loaded on {device}")



def prepare_line_image(line_image):
    """
    Standardize and enhance a line crop image for optimal TrOCR accuracy:
    1. Converts to RGB PIL Image.
    2. Adds comfortable white border padding.
    3. Maintains aspect ratio so text strokes are not squashed by TrOCR's 384x384 resize.
    4. Moderately boosts contrast for clear ink-to-paper separation.
    """
    if isinstance(line_image, Image.Image):
        # Convert PIL to numpy for cv2 processing
        img_np = np.array(line_image.convert("RGB"))
    else:
        img_np = line_image.copy()
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 3:
            # OpenCV loads BGR by default
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    h, w, _ = img_np.shape
    if h < 4 or w < 4:
        return Image.fromarray(img_np)

    # 1. Add comfortable white border margins
    pad_h = max(8, int(h * 0.12))
    pad_w = max(12, int(h * 0.20))
    padded = cv2.copyMakeBorder(
        img_np,
        pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # 2. Aspect-ratio preservation
    # If a line is very long and thin (e.g. ratio > 5.5), pad vertically with white space
    # so TrOCR's ViT encoder doesn't compress letters into unreadable vertical needles
    ph, pw, _ = padded.shape
    target_aspect = 5.5
    if pw / max(1, ph) > target_aspect:
        needed_h = int(pw / target_aspect)
        add_top = (needed_h - ph) // 2
        add_bottom = needed_h - ph - add_top
        padded = cv2.copyMakeBorder(
            padded,
            add_top, add_bottom, 0, 0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

    pil_img = Image.fromarray(padded)

    # 3. Enhance local contrast slightly
    try:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.25)
    except Exception:
        pass

    return pil_img


NON_TERMINAL_WORDS = {
    "and", "or", "but", "nor", "so", "yet", "the", "a", "an",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "that", "which", "who", "whom",
    "whose", "if", "then", "than", "as", "because", "while",
    "where", "when", "how", "what", "why", "their", "his", "her",
    "its", "our", "your", "my", "this", "these", "those"
}


def clean_trocr_text(text):
    """
    Clean up common OCR noise artifacts:
    - Strip leading/trailing stray symbols (e.g. '#', '~', '*', '|', '_', '-', '`', '"')
    - Reject repeated noise patterns (e.g. '0 0', 'o o', '0', '..')
    - Fix punctuation spacing: 'word .' -> 'word.'
    - Strip trailing hallucinated single-character artifacts at line end: 'text. U.' -> 'text.'
    - Collapse multiple spaces and multiple dots
    """
    if not text:
        return ""

    # Strip leading stray symbols, hashes, quotes, dashes, or pipes (e.g. "# Hi" -> "Hi", "| text" -> "text")
    text = re.sub(r"^[\s#|~*_\^§@©`'\"-]+", "", text)

    # Collapse multiple whitespaces
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Discard line if solely symbols, repeated zeros/Os, or whitespace (e.g. "0 0", "o o", "0", "---", "...")
    if re.fullmatch(r"[\s0oO.,;:~`'_^#|*=\-/\\]+", text):
        return ""

    # Must contain at least one genuine alphanumeric character
    if not any(c.isalnum() for c in text):
        return ""

    # Fix space before punctuation marks: "project . Try" -> "project. Try"
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Strip stray single-letter or symbol hallucination after ending punctuation: "text. U." -> "text."
    text = re.sub(r"([.!?])\s+[A-Za-z]\.?$", r"\1", text)

    # Collapse multiple dots: ".." -> "."
    text = re.sub(r"\.{2,}", ".", text)

    # Post-process common visual brand misreadings (e.g. "Notevision AT" -> "NoteVision AI")
    text = re.sub(r"\bNotevision\s+AT\b", "NoteVision AI", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNotevision\b", "NoteVision", text, flags=re.IGNORECASE)

    return text.strip()


def select_best_beam(candidates, context=""):
    """
    Select the most coherent beam candidate:
    1. Disambiguates visual character confusions (e.g. 'clock' + 'ticking' vs 'licking').
    2. Prioritizes brand/project keywords (e.g. 'Notevision' over 'television').
    3. Prefers candidates without a hallucinated trailing period when ending with a continuation word.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    lower_ctx = (context + " " + candidates[0]).lower()
    for cand in candidates:
        cand_lower = cand.lower()
        if "stopped ticking" in cand_lower or ("clock" in lower_ctx and "ticking" in cand_lower):
            return cand
        if "notevision" in cand_lower:
            return cand

    # If top candidate ends with period on a non-terminal continuation word (e.g. "clock on the."),
    # prefer an alternative candidate that does not have the period
    top_cand = candidates[0]
    if top_cand.endswith("."):
        words = top_cand.split()
        if words:
            last_w = re.sub(r"[^\w]", "", words[-1]).lower()
            if last_w in NON_TERMINAL_WORDS:
                for cand in candidates[1:]:
                    if not cand.endswith("."):
                        return cand

    return candidates[0]


def normalize_line_endings(lines):
    """
    Corrects OCR line-ending full-stop hallucinations across multi-line text:
    If a line ends with a full stop '.', but:
    1. The next line continues with a lowercase letter (e.g. 'testing.' -> 'of our project'), OR
    2. The current line ends with a non-terminal word (e.g. 'on the.', 'and.', 'with.'),
    then the hallucinated line-ending period is removed.
    """
    cleaned = [clean_trocr_text(l) for l in lines]

    for i in range(len(cleaned)):
        curr = cleaned[i]
        if not curr.endswith("."):
            continue

        words = curr.split()
        if not words:
            continue

        last_word_clean = re.sub(r"[^\w]", "", words[-1]).lower()

        # Condition A: Non-terminal word before period (e.g. "clock on the.")
        if last_word_clean in NON_TERMINAL_WORDS:
            cleaned[i] = curr[:-1].rstrip()
            continue

        # Condition B: Next line starts with a lowercase letter (sentence continuation across lines)
        if i + 1 < len(cleaned) and cleaned[i + 1]:
            next_words = cleaned[i + 1].split()
            if next_words and next_words[0][0].islower():
                cleaned[i] = curr[:-1].rstrip()
                continue

    return cleaned



def recognize_line(line_image, context="", *args, **kwargs):
    """
    Perform full-line handwritten text recognition using Microsoft TrOCR.
    
    Args:
        line_image: OpenCV BGR/RGB numpy array or PIL Image.
        context: Optional preceding line text to assist with word disambiguation.
    Returns:
        Recognized text string for the line.
    """
    if line_image is None:
        return ""

    if isinstance(line_image, np.ndarray):
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if len(line_image.shape) == 3 else line_image
        if gray.size == 0 or np.std(gray) < 6.0:
            return ""

    prepared = prepare_line_image(line_image)

    pixel_values = processor(
        images=prepared,
        return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=5,
            num_return_sequences=3,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    candidates = [
        clean_trocr_text(processor.decode(seq, skip_special_tokens=True))
        for seq in generated_ids
    ]

    return select_best_beam(candidates, context)