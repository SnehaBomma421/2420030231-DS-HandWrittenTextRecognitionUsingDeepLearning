import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/emnist_model.keras"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path, header=None)
    first = str(df.iloc[0, 0]).lower()
    if not first.replace(".", "", 1).isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    return (
        df.iloc[:, 1:785].astype(np.float32).to_numpy(),
        df.iloc[:, 0].astype(np.int32).to_numpy()
    )

def prepare(X):
    X = X.reshape(-1, 28, 28)
    X = np.rot90(X, k=-1, axes=(1, 2))
    X = np.fliplr(X)
    return (X.astype("float32") / 255.0)[..., np.newaxis]

def main():
    X_raw, y_true = load_csv(os.path.join(DATA_DIR, "emnist_test.csv"))
    X = prepare(X_raw)

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(X, batch_size=512, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    loss, accuracy = model.evaluate(X, y_true, verbose=0)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Full matrix can be large; save numeric data and readable plot.
    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)

    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.title("EMNIST ByClass Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    metrics = {"test_accuracy": float(accuracy), "test_loss": float(loss)}
    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)

if __name__ == "__main__":
    main()
