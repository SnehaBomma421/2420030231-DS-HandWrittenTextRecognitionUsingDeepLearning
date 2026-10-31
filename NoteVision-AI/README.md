# NoteVision AI

A Data Science project that converts handwritten note images into editable digital text and downloadable PDFs.

## Workflow

Upload Images → OpenCV Preprocessing → Character Segmentation → Hybrid OCR (EasyOCR + EMNIST CNN) → Text Reconstruction → Editable Text → Confidence Review → PDF

### Hybrid OCR Approach

The system uses a hybrid recognition pipeline:

1. **EasyOCR** — Primary text extraction engine. Handles full-line context and provides high accuracy for handwritten text.
2. **EMNIST CNN** — Character-level confidence review. Provides confidence scores and alternative predictions for each character.
3. **OCR-Optimized Preprocessing** — Grayscale conversion, 2x upscaling, denoising, and CLAHE contrast enhancement to maximize OCR accuracy.
4. **Confidence Review** — Both EasyOCR and EMNIST confidence scores are displayed so low-confidence characters can be flagged for manual review.

## Setup

### 1. Create environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Add EMNIST ByClass files

Place these files inside `data/`:

- `emnist_train.csv`
- `emnist_test.csv`
- `emnist_mapping.txt`

The CSV format expected is:

```text
label,pixel0,pixel1,...pixel783
```

If your downloaded EMNIST files are IDX/GZIP instead of CSV, convert them to this format or update the loader.

### 4. Train

```bash
python train_model.py
```

### 5. Evaluate

```bash
python evaluate_model.py
```

### 6. Run the app

```bash
streamlit run app.py
```

## Important

This project uses a real TensorFlow/Keras CNN. Predictions, confidence values, accuracy, loss, classification report, and confusion matrix are generated from actual model execution.

Character segmentation is intentionally modular in `segmentation.py` so it can be improved later.
