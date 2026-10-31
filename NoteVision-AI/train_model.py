"""
Train NoteVision AI on EMNIST ByClass CSV files.

Expected CSV format:
label,pixel0,pixel1,...pixel783

Run:
python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TRAIN_PATH = os.path.join(DATA_DIR, "emnist_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "emnist_test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "emnist_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path, header=None)

    # Handle optional header row
    first = str(df.iloc[0, 0]).lower()
    if not first.replace(".", "", 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)

    y = df.iloc[:, 0].astype(np.int32).to_numpy()
    X = df.iloc[:, 1:785].astype(np.float32).to_numpy()
    return X, y

def correct_orientation(X):
    # CSV EMNIST images are commonly stored transposed/rotated.
    imgs = X.reshape(-1, 28, 28)
    imgs = np.rot90(imgs, k=-1, axes=(1, 2))
    imgs = np.fliplr(imgs)
    return imgs

def prepare(X):
    X = correct_orientation(X)
    X = X.astype("float32") / 255.0
    return X[..., np.newaxis]

def show_samples(X, y):
    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    for ax, img, label in zip(axes.ravel(), X[:15], y[:15]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Class {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_samples.png"), dpi=150)
    plt.close()

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Dropout(0.30),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading EMNIST training data...")
    X_train_raw, y_train = load_csv(TRAIN_PATH)
    X_test_raw, y_test = load_csv(TEST_PATH)

    X_train = prepare(X_train_raw)
    X_test = prepare(X_test_raw)

    show_samples(X_train, y_train)

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = build_model(num_classes)
    model.summary()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    model.save(MODEL_PATH)

    history_data = history.history
    history_data["test_accuracy"] = [float(test_acc)]
    history_data["test_loss"] = [float(test_loss)]

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_data.items()}, f)

    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()
