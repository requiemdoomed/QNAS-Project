"""Datasets + classical feature extraction (pipeline stage 1).

Raw data  ->  classical feature extractor  ->  ``n_features`` angles in [0, pi]  ->  quantum circuit

Supported datasets
------------------
``digits``         8x8 handwritten digits bundled with scikit-learn (MNIST-like, works offline).
                   Pick the binary task with ``classes=(a, b)`` - e.g. (0, 1) for the easy
                   "MNIST 0 vs 1" calibration task, (3, 5) for a harder one.
``breast_cancer``  569 tabular tumour-measurement samples bundled with scikit-learn (works offline).
``pneumoniamnist`` / ``breastmnist`` / any other 2-class MedMNIST flag. Needs the ``.npz`` file
                   (``<root>/<flag>.npz``); if it is missing we try ``medmnist``'s downloader and,
                   if there is no network, raise a clear error with manual instructions.

Feature extractors
------------------
``pca``  standardise + PCA to ``n_features`` components (fast, deterministic).
``cnn``  small convolutional *autoencoder* trained on the training images only (no labels), whose
         bottleneck gives the ``n_features`` latent features (the "pre-trained CNN backbone").

The extractor is fitted on the training split only - validation/test never leak into it.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/data.py
# Purpose: loading the datasets and turning raw images or tables into a few
#          angle features in [0, pi] that the quantum circuit can encode.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# Listing the datasets that ship with scikit-learn and need no download.
OFFLINE_DATASETS = ("digits", "breast_cancer")


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset file is missing and cannot be downloaded."""


# ------------------------------------------------------------------------------
# Holding the data (containers)
# ------------------------------------------------------------------------------
@dataclass
class RawSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    image_shape: Optional[Tuple[int, int]]  # storing (H, W) for images, None for tabular data
    info: dict = field(default_factory=dict)


@dataclass
class TaskData:
    """Quantum-ready features (angles in [0, pi]) + binary labels for the three splits."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    info: dict = field(default_factory=dict)

    def sizes(self) -> dict:
        # Reporting how many samples each split holds.
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


# ------------------------------------------------------------------------------
# Loading the raw datasets
# ------------------------------------------------------------------------------
def _stratified_split(x, y, seed, fractions=(0.6, 0.2, 0.2)):
    from sklearn.model_selection import train_test_split

    # Splitting 60 / 20 / 20 in two stratified steps so every split keeps the class ratio.
    x_tr, x_rest, y_tr, y_rest = train_test_split(
        x, y, train_size=fractions[0], stratify=y, random_state=seed
    )
    rel = fractions[1] / (fractions[1] + fractions[2])
    x_va, x_te, y_va, y_te = train_test_split(
        x_rest, y_rest, train_size=rel, stratify=y_rest, random_state=seed
    )
    return x_tr, y_tr, x_va, y_va, x_te, y_te


def _balance(x, y, rng: np.random.Generator, per_class: Optional[int] = None):
    """Undersample every class to the size of the smallest (or to ``per_class``)."""
    # Sampling every class down to the size of the rarest one.
    classes = np.unique(y)
    n = min(int((y == c).sum()) for c in classes)
    if per_class is not None:
        n = min(n, per_class)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=n, replace=False) for c in classes])
    rng.shuffle(idx)
    return x[idx], y[idx]


def _as_groups(classes) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(3, 5)`` -> ((3,), (5,));  ``((0, 2, 4, 6, 8), (1, 3, 5, 7, 9))`` -> two digit groups."""
    # Accepting either single digits or groups of digits for each class.
    a, b = classes
    ga = tuple(a) if isinstance(a, (list, tuple)) else (int(a),)
    gb = tuple(b) if isinstance(b, (list, tuple)) else (int(b),)
    if set(ga) & set(gb):
        raise ValueError("the two class groups must not overlap")
    return tuple(int(i) for i in ga), tuple(int(i) for i in gb)


def load_digits_raw(classes=(3, 5), seed: int = 0) -> RawSplits:
    """Binary digits task: label 1 for digits in the second group, 0 for the first.

    ``classes=(3, 5)`` is "3 vs 5"; ``classes=((0,2,4,6,8), (1,3,5,7,9))`` is "even vs odd"
    (uses all 1797 samples -> much larger validation / test sets).
    """
    from sklearn.datasets import load_digits

    # Keeping only the requested digits and scaling pixels from 0..16 to 0..1.
    ga, gb = _as_groups(classes)
    d = load_digits()
    mask = np.isin(d.target, ga + gb)
    x = (d.images[mask] / 16.0).astype(np.float32)  # keeping (N, 8, 8) images in [0, 1]
    y = np.isin(d.target[mask], gb).astype(np.int64)
    parts = _stratified_split(x, y, seed)
    return RawSplits(*parts, image_shape=(8, 8),
                     info={"dataset": "digits", "classes": [list(ga), list(gb)]})


def load_breast_cancer_raw(seed: int = 0) -> RawSplits:
    from sklearn.datasets import load_breast_cancer

    # Using the 30 tabular measurements as they are (the extractor reduces them).
    d = load_breast_cancer()
    x = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    parts = _stratified_split(x, y, seed)
    return RawSplits(*parts, image_shape=None, info={"dataset": "breast_cancer", "classes": ["malignant", "benign"]})


def load_medmnist_raw(flag: str, root: str, size: int = 28, seed: int = 0, max_per_class: Optional[int] = None) -> RawSplits:
    """Load a 2-class MedMNIST dataset from ``<root>/<flag>[_<size>].npz`` (balanced by undersampling)."""
    root_p = Path(root)
    fname = f"{flag}.npz" if size == 28 else f"{flag}_{size}.npz"
    path = root_p / fname
    # Trying the official downloader when the file is missing, and explaining the
    # manual route clearly when the network is not available.
    if not path.exists():
        try:  # using the official downloader if the network allows it
            import medmnist
            from medmnist import INFO

            root_p.mkdir(parents=True, exist_ok=True)
            cls = getattr(medmnist, INFO[flag]["python_class"])
            cls(split="train", download=True, root=str(root_p), size=size)
        except Exception as exc:  # noqa: BLE001 - any failure -> friendly message
            raise DatasetUnavailable(
                f"MedMNIST file '{path}' not found and automatic download failed ({type(exc).__name__}).\n"
                f"  1. Download '{fname}' from https://zenodo.org/records/10519652 (or https://medmnist.com)\n"
                f"  2. Put it in '{root_p}/' and run again.\n"
                f"Meanwhile you can use the bundled offline datasets: --dataset digits | breast_cancer"
            ) from exc
    with np.load(path) as z:
        parts = {k: z[k] for k in z.files}
    # Reading each official split, converting to grey [0, 1] images and balancing it.
    rng = np.random.default_rng(seed)
    out = []
    for split in ("train", "val", "test"):
        x = parts[f"{split}_images"].astype(np.float32) / 255.0
        y = parts[f"{split}_labels"].reshape(-1).astype(np.int64)
        if x.ndim == 4:  # converting RGB to grey
            x = x.mean(axis=-1)
        if len(np.unique(y)) != 2:
            raise ValueError(f"{flag} is not a binary task ({len(np.unique(y))} classes)")
        x, y = _balance(x, y, rng, max_per_class)
        out += [x, y]
    h, w = out[0].shape[1:3]
    return RawSplits(*out, image_shape=(h, w), info={"dataset": flag, "size": size, "balanced": True})


def load_raw(dataset: str, classes=(3, 5), seed: int = 0, root: str = "data", size: int = 28, max_per_class: Optional[int] = None) -> RawSplits:
    # Dispatching on the dataset name.
    if dataset == "digits":
        return load_digits_raw(classes, seed)
    if dataset == "breast_cancer":
        return load_breast_cancer_raw(seed)
    return load_medmnist_raw(dataset, root, size=size, seed=seed, max_per_class=max_per_class)


# ------------------------------------------------------------------------------
# Extracting classical features
# ------------------------------------------------------------------------------
class PCAExtractor:
    kind = "pca"

    def __init__(self, n_features: int, seed: int = 0):
        self.n_features, self.seed = n_features, seed

    def fit(self, x_train: np.ndarray, image_shape=None):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standardising pixels / columns, then keeping the top principal components.
        flat = x_train.reshape(len(x_train), -1)
        self.scaler = StandardScaler().fit(flat)
        self.pca = PCA(n_components=self.n_features, random_state=self.seed).fit(self.scaler.transform(flat))
        self.info = {"explained_variance": float(self.pca.explained_variance_ratio_.sum())}
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        # Applying the scaler and projection learned on the training split.
        flat = x.reshape(len(x), -1)
        return self.pca.transform(self.scaler.transform(flat)).astype(np.float64)


class CNNAutoencoderExtractor:
    """Convolutional autoencoder; the ``n_features``-dimensional bottleneck is the feature vector."""

    kind = "cnn"

    def __init__(self, n_features: int, seed: int = 0, epochs: int = 60, lr: float = 3e-3, batch_size: int = 64):
        self.n_features, self.seed = n_features, seed
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size

    @staticmethod
    def _build(n_features: int, h: int, w: int):
        from torch import nn

        # Encoding 1 -> 8 -> 16 -> 16 channels, pooling to 2x2 and squeezing to n_features.
        enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(64, n_features),
        )
        # Decoding the bottleneck straight back to a full image for the reconstruction loss.
        dec = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, h * w), nn.Sigmoid())
        return enc, dec

    def fit(self, x_train: np.ndarray, image_shape=None):
        import torch
        from torch import nn

        if image_shape is None:
            raise ValueError("the 'cnn' extractor needs image data; use extractor='pca' for tabular data")
        # Training the autoencoder without labels on the training images only.
        h, w = image_shape
        torch.manual_seed(self.seed)
        self.enc, self.dec = self._build(self.n_features, h, w)
        x = torch.tensor(x_train.reshape(-1, 1, h, w), dtype=torch.float32)
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        last = float("nan")
        for _ in range(self.epochs):
            perm = torch.randperm(len(x), generator=g)
            total = 0.0
            for i in range(0, len(x), self.batch_size):
                xb = x[perm[i : i + self.batch_size]]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.dec(self.enc(xb)), xb.reshape(len(xb), -1))
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(xb)
            last = total / len(x)
        # Switching to inference mode and remembering the final reconstruction error.
        self.enc.eval()
        self.info = {"autoencoder_mse": last}
        self._hw = (h, w)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        import torch

        # Running only the encoder to get the latent features.
        h, w = self._hw
        with torch.no_grad():
            z = self.enc(torch.tensor(x.reshape(-1, 1, h, w), dtype=torch.float32))
        return z.numpy().astype(np.float64)


def make_extractor(kind: str, n_features: int, seed: int = 0, ae_epochs: int = 60):
    # Building the requested feature extractor.
    if kind == "pca":
        return PCAExtractor(n_features, seed)
    if kind == "cnn":
        return CNNAutoencoderExtractor(n_features, seed, epochs=ae_epochs)
    raise ValueError(f"unknown extractor {kind!r} (use 'pca' or 'cnn')")


class AngleScaler:
    """z-score (fit on train) then squash linearly into [0, pi]: ``pi * (clip(z, -3, 3) + 3) / 6``."""

    def fit(self, f: np.ndarray):
        # Remembering the training mean and spread (the small constant avoids division by zero).
        self.mu, self.sd = f.mean(axis=0), f.std(axis=0) + 1e-9
        return self

    def transform(self, f: np.ndarray) -> np.ndarray:
        # Clipping outliers at 3 standard deviations, then mapping linearly onto [0, pi].
        z = np.clip((f - self.mu) / self.sd, -3.0, 3.0)
        return math.pi * (z + 3.0) / 6.0


# ------------------------------------------------------------------------------
# Preparing a task (public entry point)
# ------------------------------------------------------------------------------
def _cache_key(**kw) -> str:
    # Hashing the settings so each distinct task gets its own cache file.
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prepare_task(
    dataset: str = "digits",
    classes: Sequence[int] = (3, 5),
    n_features: int = 4,
    extractor: str = "pca",
    seed: int = 0,
    root: str = "data",
    size: int = 28,
    max_per_class: Optional[int] = None,
    ae_epochs: int = 60,
    cache_dir: Optional[str] = None,
) -> TaskData:
    """Load the dataset, fit the classical extractor on the training split and return angle features."""
    key = _cache_key(dataset=dataset, classes=list(classes), n_features=n_features, extractor=extractor,
                     seed=seed, size=size, max_per_class=max_per_class, ae_epochs=ae_epochs)
    # Reusing a cached task when the exact same settings were prepared before.
    cache_file = Path(cache_dir) / f"task_{key}.npz" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as z:
            info = json.loads(str(z["info"]))
            return TaskData(z["x_train"], z["y_train"], z["x_val"], z["y_val"], z["x_test"], z["y_test"], int(z["n_features"]), info)

    # Loading raw data, fitting the extractor and angle scaler on the training split only.
    raw = load_raw(dataset, classes, seed, root, size, max_per_class)
    ext = make_extractor(extractor, n_features, seed, ae_epochs).fit(raw.x_train, raw.image_shape)
    f_train = ext.transform(raw.x_train)
    scaler = AngleScaler().fit(f_train)
    task = TaskData(
        x_train=scaler.transform(f_train), y_train=raw.y_train,
        x_val=scaler.transform(ext.transform(raw.x_val)), y_val=raw.y_val,
        x_test=scaler.transform(ext.transform(raw.x_test)), y_test=raw.y_test,
        n_features=n_features,
        info={**raw.info, "extractor": extractor, "n_features": n_features, "seed": seed, **getattr(ext, "info", {})},
    )
    # Recording sizes and class balance, then caching the result for the next stage.
    task.info["sizes"] = task.sizes()
    task.info["class_balance_train"] = float(task.y_train.mean())
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, x_train=task.x_train, y_train=task.y_train, x_val=task.x_val, y_val=task.y_val,
                 x_test=task.x_test, y_test=task.y_test, n_features=n_features, info=json.dumps(task.info, default=str))
    return task
