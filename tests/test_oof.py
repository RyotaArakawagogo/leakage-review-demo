from collections import Counter

import numpy as np
import pytest

from leakage_demo.data import DataConfig, generate_dataset
from leakage_demo.evaluate import evaluate, split_indices


@pytest.fixture(scope="module")
def df():
    return generate_dataset(DataConfig())


@pytest.mark.parametrize("scheme", ["image", "patient"])
def test_every_image_is_predicted_exactly_once(df, scheme):
    # Count test appearances from the splits themselves, independent of evaluate().
    counts = Counter(int(i) for _, test_idx in split_indices(df, scheme) for i in test_idx)
    assert sorted(counts) == list(range(len(df)))
    assert set(counts.values()) == {1}

    r = evaluate(df, scheme)
    assert not np.isnan(r["oof_proba"]).any()
    assert ((r["oof_proba"] >= 0) & (r["oof_proba"] <= 1)).all()
    # The fold recorded for each image is the fold where it was in test.
    for fold, (_, test_idx) in enumerate(split_indices(df, scheme)):
        assert (r["oof_fold"][test_idx] == fold).all()
