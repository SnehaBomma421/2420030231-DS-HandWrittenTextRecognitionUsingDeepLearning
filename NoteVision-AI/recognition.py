import cv2
import numpy as np
import tensorflow as tf


def load_emnist_model(model_path="models/emnist_model.keras"):
    """Load the trained EMNIST CNN model."""
    return tf.keras.models.load_model(model_path)


def load_mapping(mapping_path="data/emnist_mapping.txt"):
    """
    Load EMNIST class index -> character mapping.
    """
    mapping = {}
    with open(mapping_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_index = int(parts[0])
                ascii_code = int(parts[1])
                mapping[class_index] = chr(ascii_code)
    return mapping


def predict_character(model, char_image, mapping):
    """
    Predict one handwritten character using the EMNIST CNN.
    Returns:
        best_char: str
        confidence: float
        top_predictions: list of {"character": char, "confidence": conf}
    """
    image = np.array(char_image, dtype=np.float32)

    # Ensure shape is (1, 28, 28, 1)
    if image.ndim == 2:
        image = image.reshape(1, 28, 28, 1)
    elif image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    top_predictions = []
    for index in top_indices:
        index = int(index)
        top_predictions.append({
            "character": mapping.get(index, "?"),
            "confidence": float(predictions[index])
        })

    best_index = int(top_indices[0])
    return (
        mapping.get(best_index, "?"),
        float(predictions[best_index]),
        top_predictions
    )


def recognize_segments(model, characters, mapping, boxes=None, line_ids=None):
    """
    Recognize all segmented characters using the EMNIST CNN.

    Returns:
        text: reconstructed string
        predictions: list of detailed prediction dicts
    """
    if not characters:
        return "", []

    text = ""
    predictions = []

    avg_width = 15
    if boxes and len(boxes) > 0:
        avg_width = np.mean([b[2] for b in boxes])
    space_threshold = avg_width * 1.4

    prev_line = -1
    prev_x_end = -1

    for i, char_image in enumerate(characters):
        if boxes is not None and line_ids is not None:
            curr_line = line_ids[i]
            x, _, w, _ = boxes[i]

            if prev_line != -1:
                if curr_line > prev_line:
                    text += "\n"
                    predictions.append({"type": "newline"})
                elif (x - prev_x_end) > space_threshold:
                    text += " "
                    predictions.append({"type": "space"})

            prev_line = curr_line
            prev_x_end = x + w

        character, confidence, alternatives = predict_character(
            model,
            char_image,
            mapping
        )

        text += character
        predictions.append({
            "type": "character",
            "character": character,
            "confidence": confidence,
            "alternatives": alternatives,
            "box": boxes[i] if boxes is not None else None,
            "line_id": line_ids[i] if line_ids is not None else None,
            "char_index": i
        })

    return text, predictions


def generate_confidence_review(predictions, threshold=0.75):
    """
    Generate an AI Confidence Review from EMNIST character predictions.
    Identifies low-confidence character classifications and flags them with
    top alternative predictions.
    
    Returns:
        review_items: list of flagged character dicts
        metrics: summary dict with average_confidence, flagged_count, total_count
    """
    char_preds = [p for p in predictions if p.get("type") == "character"]
    if not char_preds:
        return [], {"average_confidence": 0.0, "flagged_count": 0, "total_count": 0}

    confidences = [p["confidence"] for p in char_preds]
    avg_conf = float(np.mean(confidences))

    review_items = []
    for item in char_preds:
        conf = item["confidence"]
        if conf < threshold:
            review_items.append({
                "char_index": item.get("char_index", 0),
                "predicted": item["character"],
                "confidence": conf,
                "confidence_pct": f"{conf * 100:.1f}%",
                "alternatives": [
                    f"'{alt['character']}' ({alt['confidence']*100:.1f}%)"
                    for alt in item.get("alternatives", [])[1:]
                ],
                "box": item.get("box"),
                "line_id": item.get("line_id", 0)
            })

    metrics = {
        "average_confidence": avg_conf,
        "average_confidence_pct": f"{avg_conf * 100:.1f}%",
        "flagged_count": len(review_items),
        "total_count": len(char_preds)
    }

    return review_items, metrics


def get_ocr_confidence_corrections(characters, mapping, boxes, line_ids, model):
    """
    Use EMNIST CNN to suggest confidence reviews for character segments.
    """
    _, predictions = recognize_segments(model, characters, mapping, boxes, line_ids)
    char_preds = [p for p in predictions if p.get("type") == "character"]
    corrections = []
    for item in char_preds:
        corrections.append({
            "emnist_char": item["character"],
            "confidence": item["confidence"],
            "alternatives": item["alternatives"],
            "char_index": item.get("char_index", 0)
        })
    return corrections


def ocr_preprocess_for_easyocr(original_image):
    """
    Preprocess image for OCR readers to maximize accuracy.
    """
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(resized, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def hybrid_recognize_segments(model, characters, mapping, boxes, line_ids, ocr_reader, original_image):
    """
    Hybrid recognition pipeline compatibility function.
    """
    ocr_results = ocr_reader.readtext(original_image)
    ocr_text = " ".join([res[1] for res in ocr_results])
    _, predictions = recognize_segments(model, characters, mapping, boxes, line_ids)
    ocr_confidences = [{"text": res[1], "confidence": res[2]} for res in ocr_results]
    return ocr_text, predictions, ocr_confidences
