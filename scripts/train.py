"""Train one of the models on the crops from build_dataset.py.

    python scripts/train.py detector
    python scripts/train.py noteheads     # filled/hollow and half/whole
    python scripts/train.py clef
    python scripts/train.py accidental

Weights land in models/<name>.keras. Validation is on held-out songs, not
held-out pages, so the numbers printed here are not inflated by seeing
other pages of the same piece.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from omr import doremi, models
from omr.crops import to_batch
from omr.detector import build_detector, heatmap_target

CACHE = doremi.CACHE_DIR
MODEL_DIR = models.MODEL_DIR


def augment_layer():
    return keras.Sequential([
        keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        keras.layers.RandomRotation(0.02, fill_mode="constant"),
        keras.layers.RandomZoom(0.08, fill_mode="constant"),
    ])


def load_crops(kind, split):
    d = np.load(CACHE / f"{kind}_{split}.npz")
    return to_batch(list(d["x"])), d["y"].astype(np.int32)


def fit_classifier(model, x, y, xv, yv, name, epochs, class_names=None, batch=64):
    aug = augment_layer()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=0).batch(batch)
    ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE).prefetch(4)
    val = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(256)

    counts = np.bincount(y)
    weights = {i: len(y) / (len(counts) * c) for i, c in enumerate(counts)}
    binary = model.output_shape[-1] == 1
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy" if binary else "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # validation is meaningless until BatchNorm's running stats have settled,
    # so early stopping does not get a say for the first few epochs
    model.fit(ds, validation_data=val, epochs=epochs, class_weight=weights, verbose=2, callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, start_from_epoch=8,
                                      restore_best_weights=True),
    ])
    p = model.predict(xv, verbose=0)
    pred = (p[:, 0] > 0.5).astype(int) if binary else p.argmax(1)
    print(f"\n{name}: val accuracy {np.mean(pred == yv):.4f}  (n={len(yv)}, class counts {np.bincount(yv).tolist()})")
    print("confusion (rows = true):")
    print(confusion_matrix(yv, pred))
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / f"{name}.keras")


def train_noteheads(epochs):
    x, y = load_crops("noteheads", "train")
    xv, yv = load_crops("noteheads", "val")
    # 0 black, 1 half, 2 whole  ->  filled model predicts "is hollow"
    fit_classifier(models.build_filled_model(), x, (y > 0).astype(int), xv, (yv > 0).astype(int),
                   "note_filled", min(epochs, 10))
    # only ~1500 hollow heads to learn from, so this one needs many more
    # passes over the data to see the same number of steps
    hollow, hollow_v = y > 0, yv > 0
    fit_classifier(models.build_hollow_model(), x[hollow], (y[hollow] == 2).astype(int),
                   xv[hollow_v], (yv[hollow_v] == 2).astype(int), "note_hollow", epochs * 3)


def train_clef(epochs):
    x, y = load_crops("clefs", "train")
    xv, yv = load_crops("clefs", "val")
    fit_classifier(models.build_clef_model(), x, y, xv, yv, "clef", epochs)


def train_accidental(epochs):
    x, y = load_crops("accidentals", "train")
    xv, yv = load_crops("accidentals", "val")
    fit_classifier(models.build_accidental_model(), x, y, xv, yv, "accidental", epochs)


# ---- detector -------------------------------------------------------------

CROP = 256


class BandSampler:
    """Random CROPxCROP windows out of the half-res page bands."""

    def __init__(self, split, seed=0):
        with open(CACHE / "bands.json") as f:
            index = [b for b in json.load(f) if b["split"] == split]
        self.rng = random.Random(seed)
        self.bands = []
        for b in index:
            img = cv2.imread(str(CACHE / "bands" / f"{b['name']}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            self.bands.append((img, np.array([[c[0], c[1]] for c in b["centers"]], dtype=np.float32)))
        # synthetic exercise pages outnumber real music; sample real pages
        # more often so the network sees chords and hollow heads
        self.weights = [0.3 if "beam" in b["name"] or "accidental" in b["name"] else 1.0 for b in index]

    def sample(self):
        img, centers = self.rng.choices(self.bands, weights=self.weights)[0]
        h, w = img.shape
        y0 = self.rng.randint(0, max(0, h - CROP))
        x0 = self.rng.randint(0, max(0, w - CROP))
        crop = np.zeros((CROP, CROP), dtype=np.uint8)
        sub = img[y0:y0 + CROP, x0:x0 + CROP]
        crop[:sub.shape[0], :sub.shape[1]] = sub
        local = centers - [x0, y0] if len(centers) else centers
        target = heatmap_target((CROP, CROP), local)
        return (crop > 127).astype(np.float32)[..., None], target[..., None]

    def generator(self):
        while True:
            yield self.sample()


def weighted_bce(y_true, y_pred):
    # most of a page is background; up-weight pixels near a notehead
    w = 1.0 + 4.0 * y_true
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(w[..., 0] * bce)


def train_detector(epochs, steps=300, batch=16):
    sig = (tf.TensorSpec((CROP, CROP, 1), tf.float32), tf.TensorSpec((CROP, CROP, 1), tf.float32))
    train = tf.data.Dataset.from_generator(BandSampler("train").generator, output_signature=sig).batch(batch).prefetch(4)
    val_sampler = BandSampler("val", seed=1)
    xv, yv = zip(*[val_sampler.sample() for _ in range(160)])
    xv, yv = np.stack(xv), np.stack(yv)

    model = build_detector()
    model.summary(line_length=90)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=weighted_bce)
    model.fit(train, steps_per_epoch=steps, epochs=epochs, validation_data=(xv, yv), verbose=2, callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=2e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "detector.keras"), save_best_only=True),
    ])
    model.save(MODEL_DIR / "detector.keras")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["detector", "noteheads", "clef", "accidental"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    keras.utils.set_random_seed(0)
    MODEL_DIR.mkdir(exist_ok=True)
    if args.model == "detector":
        train_detector(args.epochs or 20)
    elif args.model == "noteheads":
        train_noteheads(args.epochs or 30)
    elif args.model == "clef":
        train_clef(args.epochs or 40)
    else:
        train_accidental(args.epochs or 20)


if __name__ == "__main__":
    main()
