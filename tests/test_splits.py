import pytest

from leakage_demo.data import DataConfig, generate_dataset
from leakage_demo.evaluate import split_indices


@pytest.fixture(scope="module")
def df():
    return generate_dataset(DataConfig())


def patients(df, idx):
    return set(df["patient_id"].iloc[idx])


def test_patient_split_never_shares_a_patient(df):
    folds = list(split_indices(df, "patient"))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert patients(df, train_idx).isdisjoint(patients(df, test_idx))
    # Every image is tested exactly once.
    tested = sorted(i for _, test_idx in folds for i in test_idx)
    assert tested == list(range(len(df)))


def test_image_split_shares_patients(df):
    for train_idx, test_idx in split_indices(df, "image"):
        assert len(patients(df, train_idx) & patients(df, test_idx)) > 0
