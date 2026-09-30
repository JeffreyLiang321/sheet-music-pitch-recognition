"""Convert the Keras models in models/ to ONNX so the server can run on
onnxruntime alone (no TensorFlow in the container).

    python scripts/export_onnx.py

Each export is checked against the Keras model on random input.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras

from omr.models import MODEL_DIR

NAMES = ["detector", "note_filled", "note_hollow", "clef", "accidental"]


def export(name: str) -> None:
    import tf2onnx

    model = keras.models.load_model(MODEL_DIR / f"{name}.keras", compile=False)
    shape = [None] + list(model.input_shape[1:])
    spec = (tf.TensorSpec(shape, tf.float32, name="input"),)

    # tf2onnx wants a tf.function; wrap the Keras 3 model's call
    @tf.function(input_signature=spec)
    def call(x):
        return model(x, training=False)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_model, _ = tf2onnx.convert.from_function(call, input_signature=spec, opset=17, output_path=None)
    out = MODEL_DIR / f"{name}.onnx"
    out.write_bytes(onnx_model.SerializeToString())

    x = np.random.rand(2, *[d or 64 for d in model.input_shape[1:]]).astype(np.float32)
    ref = model.predict(x, verbose=0)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    err = float(np.abs(ref - got).max())
    print(f"{name:12s} -> {out} ({out.stat().st_size / 1e6:.2f} MB), max abs diff vs keras {err:.2e}")
    if err > 1e-3:
        sys.exit(f"{name}: ONNX output does not match Keras")


if __name__ == "__main__":
    names = sys.argv[1:] or NAMES
    for n in names:
        if (MODEL_DIR / f"{n}.keras").exists():
            export(n)
        else:
            print(f"{n}: no keras model yet, skipping")
