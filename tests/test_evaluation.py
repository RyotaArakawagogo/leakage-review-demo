"""The headline result: image-level splitting inflates AUC, and the patient shift is the cause."""
import pytest

from leakage_demo.data import DataConfig, generate_dataset, without_patient_shift
from leakage_demo.evaluate import evaluate

MIN_GAP = 0.2  # observed gap with the default config: 0.333
MAX_CONTROL_GAP = 0.05  # observed gap without patient shift: 0.001


@pytest.fixture(scope="module")
def results():
    normal = generate_dataset(DataConfig())
    control = generate_dataset(without_patient_shift(DataConfig()))
    return {
        ("normal", s): evaluate(normal, s) for s in ("image", "patient")
    } | {("control", s): evaluate(control, s) for s in ("image", "patient")}


def test_image_split_auc_is_clearly_higher_than_patient_split(results):
    gap = results["normal", "image"]["auc_mean"] - results["normal", "patient"]["auc_mean"]
    assert gap >= MIN_GAP


def test_gap_disappears_without_patient_shift(results):
    gap = results["control", "image"]["auc_mean"] - results["control", "patient"]["auc_mean"]
    assert abs(gap) <= MAX_CONTROL_GAP


def test_reported_shared_patients(results):
    for data in ("normal", "control"):
        assert all(f["shared_patients"] == 0 for f in results[data, "patient"]["folds"])
        assert all(f["shared_patients"] > 0 for f in results[data, "image"]["folds"])
