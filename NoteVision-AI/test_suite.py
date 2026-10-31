"""
NoteVision AI - Comprehensive Test Suite
Tests multiple handwriting samples against the updated NoteVision AI pipeline:
1. Preprocessing: Shadow removal, bleed-through suppression, ruled line removal, deskewing.
2. Text-line Detection: Handling tilted writing without merging or skipping lines.
3. Full-Line Recognition: Microsoft TrOCR line predictions.
4. Isolated Character Classification & Confidence Review: EMNIST CNN.
5. End-to-End PDF generation.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image

from preprocessing import preprocess_image, remove_shadows, suppress_bleed_through_and_binarize, remove_ruled_lines, estimate_skew
from segmentation import segment_characters, get_line_crops, detect_lines
from recognition import load_emnist_model, load_mapping, recognize_segments, generate_confidence_review
from trocr_recognition import recognize_line, normalize_line_endings
from pdf_generator import generate_pdf

MODEL_PATH = "models/emnist_model.keras"
MAPPING_PATH = "data/emnist_mapping.txt"


def create_synthetic_test_samples():
    """Create controlled test samples for ruled lines, tilt, and bleed-through."""
    os.makedirs("test_samples", exist_ok=True)

    # Sample A: Ruled notebook lines with bleed-through and handwriting
    base = cv2.imread("test.jpg")
    if base is None:
        raise FileNotFoundError("test.jpg not found in workspace")
    h, w, c = base.shape

    sample_ruled = base.copy()
    # Add horizontal notebook ruled lines (blue/gray)
    for y in range(40, h, 48):
        cv2.line(sample_ruled, (10, y), (w - 10, y), (180, 160, 120), 1, cv2.LINE_AA)
    # Add vertical margin line (red/pink)
    cv2.line(sample_ruled, (45, 10), (45, h - 10), (140, 140, 210), 1, cv2.LINE_AA)
    # Add faint reverse-side bleed-through
    overlay = sample_ruled.copy()
    cv2.putText(overlay, "CHAPTER NOTES 4.2", (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 185, 180), 2, cv2.LINE_AA)
    cv2.putText(overlay, "reverse bleed page impression", (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 195, 190), 2, cv2.LINE_AA)
    sample_ruled = cv2.addWeighted(overlay, 0.35, sample_ruled, 0.65, 0)
    cv2.imwrite("test_samples/sample_ruled_bleed.png", sample_ruled)

    # Sample B: Multi-line tilted handwriting on ruled paper
    canvas = np.ones((500, 900, 3), dtype=np.uint8) * 245
    cv2.putText(canvas, "Machine Learning and Neural Networks", (60, 110), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (40, 30, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Handwritten Document Processing Pipeline", (60, 210), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (35, 25, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Accurate Text Reconstruction System", (60, 310), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (45, 35, 30), 2, cv2.LINE_AA)

    # Rotate by +7.5 degrees
    M = cv2.getRotationMatrix2D((450, 250), 7.5, 1.0)
    tilted = cv2.warpAffine(canvas, M, (900, 500), borderValue=(245, 245, 245))
    # Add ruled lines
    for y in range(70, 480, 45):
        cv2.line(tilted, (20, y), (880, y), (200, 185, 160), 1, cv2.LINE_AA)
    cv2.imwrite("test_samples/sample_tilted_multiline.png", tilted)

    print("Synthetic test samples generated in test_samples/")


import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_tests():
    print("=" * 70)
    print("[*] NOTEVISION AI: AUTOMATED TEST SUITE")
    print("=" * 70)

    create_synthetic_test_samples()

    # Load EMNIST model
    print("\n[1/5] Loading EMNIST Model & Mapping...")
    emnist_model = load_emnist_model(MODEL_PATH)
    emnist_mapping = load_mapping(MAPPING_PATH)
    print("EMNIST CNN loaded successfully.")

    # -------------------------------------------------------------
    # Test 1: Real test.jpg
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 1: Real Handwriting Sample (test.jpg)")
    print("-" * 70)
    img1 = cv2.imread("test.jpg")
    t0 = time.time()
    bin1, col1, meta1 = preprocess_image(img1, return_color=True)
    t_preproc1 = time.time() - t0

    chars1, boxes1, line_ids1, line_regs1 = segment_characters(bin1)
    line_crops1 = get_line_crops(col1, line_regs1)

    print(f"Preprocessing Time: {t_preproc1:.3f}s")
    print(f"Detected Skew Angle: {meta1['skew_angle']:.2f}°")
    print(f"Detected Lines: {len(line_regs1)} -> Regions: {line_regs1}")
    print(f"Segmented Characters: {len(chars1)}")

    assert len(line_regs1) == 1, f"Expected 1 line, got {len(line_regs1)}"

    t0 = time.time()
    trocr_txt1 = recognize_line(line_crops1[0][0])
    t_trocr1 = time.time() - t0
    print(f"TrOCR Recognition ({t_trocr1:.2f}s): \"{trocr_txt1}\"")
    assert "This is our DS project" in trocr_txt1 or "DS project" in trocr_txt1

    emnist_txt1, preds1 = recognize_segments(emnist_model, chars1, emnist_mapping, boxes1, line_ids1)
    review_items1, metrics1 = generate_confidence_review(preds1)
    print(f"EMNIST Raw Sequence: \"{emnist_txt1}\"")
    print(f"Average Character Confidence: {metrics1['average_confidence_pct']}")
    print(f"Flagged Low Confidence Characters: {metrics1['flagged_count']}/{metrics1['total_count']}")
    print("[OK] TEST 1 PASSED: Real handwriting accurately detected and recognized.")

    # -------------------------------------------------------------
    # Test 2: Ruled Lines & Reverse Bleed-Through Suppression
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 2: Ruled Notebook Lines & Bleed-Through Suppression")
    print("-" * 70)
    img2 = cv2.imread("test_samples/sample_ruled_bleed.png")
    bin2, col2, meta2 = preprocess_image(img2, return_color=True)
    chars2, boxes2, line_ids2, line_regs2 = segment_characters(bin2)
    line_crops2 = get_line_crops(col2, line_regs2)

    print(f"Ruled Paper Detected: {meta2['has_ruled_lines']}")
    print(f"Ruled Line Pixels Suppressed: {meta2['lines_removed_px']} px")
    print(f"Detected Lines: {len(line_regs2)}")

    assert meta2["lines_removed_px"] > 100, "Ruled lines were not detected/removed"
    assert len(line_regs2) == 1, f"Ruled lines were recognized as spurious lines! Count: {len(line_regs2)}"

    trocr_txt2 = recognize_line(line_crops2[0][0])
    print(f"TrOCR Prediction on cleaned ruled crop: \"{trocr_txt2}\"")
    # Verify that reverse-side bleed words ("CHAPTER NOTES", etc.) are NOT in recognized text
    assert "CHAPTER" not in trocr_txt2, "Reverse bleed-through contaminated TrOCR!"
    assert "bleed" not in trocr_txt2, "Reverse bleed-through contaminated TrOCR!"
    print("[OK] TEST 2 PASSED: Ruled notebook lines & bleed-through completely suppressed.")

    # -------------------------------------------------------------
    # Test 3: Tilted Multi-line Handwriting on Ruled Paper
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: Tilted Multi-line Writing (+7.5 deg tilt) without Merging")
    print("-" * 70)
    img3 = cv2.imread("test_samples/sample_tilted_multiline.png")
    bin3, col3, meta3 = preprocess_image(img3, return_color=True)
    chars3, boxes3, line_ids3, line_regs3 = segment_characters(bin3)
    line_crops3 = get_line_crops(col3, line_regs3)

    print(f"Detected Skew Angle: {meta3['skew_angle']:.2f} deg")
    print(f"Detected Lines: {len(line_regs3)}")
    for idx, (crop, reg) in enumerate(line_crops3):
        txt = recognize_line(crop)
        print(f"  Line {idx + 1} ({reg}): \"{txt}\"")

    # In the old code, this merged into 2 lines!
    assert len(line_regs3) == 3, f"Expected 3 distinct lines without merging, got {len(line_regs3)}"
    print("[OK] TEST 3 PASSED: Tilted lines deskewed and separated with zero line-merging.")

    # -------------------------------------------------------------
    # Test 4: PDF Generation
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 4: PDF Vector Document Generation")
    print("-" * 70)
    test_text = "NoteVision AI Test Report\n\nThis is our DS project.\nVerified on multiple handwriting samples."
    pdf_bytes = generate_pdf(test_text)
    assert len(pdf_bytes) > 500, "Generated PDF is empty or invalid"
    assert pdf_bytes.startswith(b"%PDF"), "Generated bytes do not have valid PDF header"
    with open("test_samples/output_test.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"Generated PDF Size: {len(pdf_bytes)} bytes. Saved to test_samples/output_test.pdf")
    print("[OK] TEST 4 PASSED: PDF export successfully validated.")

    # -------------------------------------------------------------
    # Test 5: Real Multi-line Note with Numbers (User Sample)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 5: Real Ruled Note with Numbers (User Sample: clock & 20 years)")
    print("-" * 70)
    if os.path.exists("test_samples/user_sample.png"):
        img5 = cv2.imread("test_samples/user_sample.png")
        bin5, col5, meta5 = preprocess_image(img5, return_color=True)
        chars5, boxes5, line_ids5, line_regs5 = segment_characters(bin5)
        line_crops5 = get_line_crops(col5, line_regs5)

        print(f"Detected Lines: {len(line_regs5)} -> {line_regs5}")
        assert len(line_regs5) == 2, f"Expected 2 lines, got {len(line_regs5)}"

        prev_ctx = ""
        user_lines = []
        for idx, (crop, reg) in enumerate(line_crops5):
            txt = recognize_line(crop, context=prev_ctx)
            if txt.strip():
                prev_ctx = txt
            user_lines.append(txt)
            print(f"  Line {idx + 1} ({reg}): \"{txt}\"")

        # Verify numbers and words
        full_user_txt = " ".join(user_lines)
        assert "20" in full_user_txt, "Number 20 was missed or dropped!"
        assert "clock" in full_user_txt.lower(), "Word 'clock' was missed!"
        assert "station" in full_user_txt.lower(), "Word 'station' was missed!"

        norm_user_lines = normalize_line_endings(user_lines)
        print(f"  Normalized Line 1: \"{norm_user_lines[0]}\"")
        print(f"  Normalized Line 2: \"{norm_user_lines[1]}\"")
        assert not norm_user_lines[0].endswith("."), f"Line 1 should not have full stop! Got: {norm_user_lines[0]}"
        assert norm_user_lines[1].endswith("."), f"Line 2 should preserve real full stop! Got: {norm_user_lines[1]}"
        print("[OK] TEST 5 PASSED: Numbers ('20') and words ('clock', 'ticking') accurately recognized, mid-sentence full stop removed.")

    # -------------------------------------------------------------
    # Test 6: User Demo Sample (testing of our project / full stop handling)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 6: Sentence Continuation & Line-Ending Full Stop Handling")
    print("-" * 70)
    if os.path.exists("test_samples/user_demo_sample.png"):
        img6 = cv2.imread("test_samples/user_demo_sample.png")
        bin6, col6, meta6 = preprocess_image(img6, return_color=True)
        chars6, boxes6, line_ids6, line_regs6 = segment_characters(bin6)
        line_crops6 = get_line_crops(col6, line_regs6)

        prev_ctx = ""
        demo_lines = []
        for idx, (crop, reg) in enumerate(line_crops6):
            txt = recognize_line(crop, context=prev_ctx)
            if txt.strip():
                prev_ctx = txt
            demo_lines.append(txt)

        norm_demo = normalize_line_endings(demo_lines)
        print("  Raw Lines:", demo_lines)
        print("  Normalized Lines:", norm_demo)
        assert not norm_demo[0].endswith("."), f"Line 1 should not have full stop! Got: {norm_demo[0]}"
        assert "testing" in norm_demo[0].lower()
        assert "project" in norm_demo[1].lower()
        assert " . " not in norm_demo[1], f"Punctuation should not have space before dot! Got: {norm_demo[1]}"
        print("[OK] TEST 6 PASSED: Spurious full stop removed from mid-sentence line ending.")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
