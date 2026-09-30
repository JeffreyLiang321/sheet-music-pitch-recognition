"""Classifier architectures and a common loader for Keras / ONNX weights.

Three small CNNs, all taking a single-channel binary window:
  notehead  - filled vs hollow, and half vs whole for the hollow ones
              (the two binary heads from the notebook, unchanged in spirit)
  clef      - G / F / C clef
  accidental- sharp / flat / natural / something else
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .crops import NOTE_WINDOW, CLEF_WINDOW, CLEF_SHRINK, ACC_WINDOW, ACC_SHRINK

MODEL_DIR = Path("models")

NOTE_CLASSES = ("noteheadBlack", "noteheadHalf", "noteheadWhole")
CLEF_CLASSES = ("gClef", "fClef", "cClef")
ACC_CLASSES = ("accidentalSharp", "accidentalFlat", "accidentalNatural", "other")


def _cnn(input_shape, n_out, widths=(32, 64, 128), dense=64, dropout=(0.25, 0.35, 0.45, 0.5),
         weight_decay=1e-4):
    from tensorflow.keras import layers, regularizers, Sequential

    reg = regularizers.l2(weight_decay)
    # The clef and half/whole sets are small (a few dozen steps per epoch), so
    # BatchNorm's running statistics need a low momentum to settle in time.
    bn = dict(momentum=0.9)
    m = Sequential()
    m.add(layers.Input(input_shape))
    for w, d in zip(widths, dropout):
        m.add(layers.Conv2D(w, 3, padding="same", activation="relu", kernel_regularizer=reg))
        m.add(layers.BatchNormalization(**bn))
        m.add(layers.MaxPooling2D(2))
        m.add(layers.Dropout(d))
    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(dense, activation="relu", kernel_regularizer=reg))
    m.add(layers.BatchNormalization(**bn))
    m.add(layers.Dropout(dropout[-1]))
    if n_out == 1:
        m.add(layers.Dense(1, activation="sigmoid", kernel_regularizer=reg))
    else:
        m.add(layers.Dense(n_out, activation="softmax", kernel_regularizer=reg))
    return m


def build_filled_model():
    return _cnn((*NOTE_WINDOW, 1), 1)


def build_hollow_model():
    return _cnn((*NOTE_WINDOW, 1), 1)


def build_clef_model():
    return _cnn((CLEF_WINDOW[0] // CLEF_SHRINK, CLEF_WINDOW[1] // CLEF_SHRINK, 1), len(CLEF_CLASSES))


def build_accidental_model():
    return _cnn((ACC_WINDOW[0] // ACC_SHRINK, ACC_WINDOW[1] // ACC_SHRINK, 1), len(ACC_CLASSES))


class OnnxModel:
    """Minimal stand-in for a Keras model so the pipeline can run without TF."""

    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x: np.ndarray, verbose=0) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


def load_model(name: str, model_dir: Path = MODEL_DIR):
    """Load models/<name>.onnx if present, otherwise models/<name>.keras."""
    onnx_path = model_dir / f"{name}.onnx"
    if onnx_path.exists():
        return OnnxModel(onnx_path)
    from tensorflow import keras
    return keras.models.load_model(model_dir / f"{name}.keras")
