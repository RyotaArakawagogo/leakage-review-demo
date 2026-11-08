"""Synthetic dataset: several images per patient, one label per patient.

Each feature value is the sum of three parts:
  patient shift  - fixed per patient and feature, unrelated to the label
  label signal   - small shift shared by all positive patients (the real signal)
  image noise    - independent per image

The patient shift lets a model recognise a patient from any of their images.
When images of one patient are in both train and test, the model can return the
remembered label of that patient instead of learning the weak real signal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "intensity_mean",
    "intensity_std",
    "intensity_skew",
    "texture_contrast",
    "texture_homogeneity",
    "texture_entropy",
    "edge_density",
    "region_area",
    "region_eccentricity",
    "gradient_mean",
]

DEFAULT_SEED = 20261108


@dataclass(frozen=True)
class DataConfig:
    n_patients: int = 40
    images_min: int = 8
    images_max: int = 20
    patient_shift_sd: float = 1.0
    signal: float = 0.4
    noise_sd: float = 0.5
    seed: int = DEFAULT_SEED


def without_patient_shift(config: DataConfig) -> DataConfig:
    """Same data (labels, image counts, noise) with the patient shift set to zero."""
    return replace(config, patient_shift_sd=0.0)


def generate_dataset(config: DataConfig = DataConfig()) -> pd.DataFrame:
    if config.n_patients % 2:
        raise ValueError("n_patients must be even (half positive, half negative)")
    rng = np.random.default_rng(config.seed)
    n_features = len(FEATURE_NAMES)

    # Draw every random number in a fixed order and always draw the patient shift,
    # so that patient_shift_sd=0 changes only that term and nothing else.
    labels = rng.permutation(np.repeat([0, 1], config.n_patients // 2))
    n_images = rng.integers(config.images_min, config.images_max + 1, size=config.n_patients)
    direction = rng.choice([-1.0, 1.0], size=n_features)
    shift = rng.normal(0.0, 1.0, size=(config.n_patients, n_features)) * config.patient_shift_sd
    noise = rng.normal(0.0, config.noise_sd, size=(int(n_images.sum()), n_features))

    patient_index = np.repeat(np.arange(config.n_patients), n_images)
    image_labels = labels[patient_index]
    values = shift[patient_index] + image_labels[:, None] * config.signal * direction + noise

    df = pd.DataFrame(np.round(values, 6), columns=FEATURE_NAMES)
    df.insert(0, "label", image_labels.astype(int))
    df.insert(0, "patient_id", [f"P{p + 1:03d}" for p in patient_index])
    df.insert(0, "image_id", [f"IMG{i + 1:04d}" for i in range(len(df))])
    return df


def to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, float_format="%.6f", lineterminator="\n")
