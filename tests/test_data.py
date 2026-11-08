from dataclasses import replace

from leakage_demo.data import DataConfig, generate_dataset, to_csv_text, without_patient_shift


def test_same_seed_gives_identical_csv():
    assert to_csv_text(generate_dataset(DataConfig())) == to_csv_text(generate_dataset(DataConfig()))


def test_different_seed_gives_different_csv():
    # Guards the test above: it must be able to fail.
    other = replace(DataConfig(), seed=DataConfig().seed + 1)
    assert to_csv_text(generate_dataset(DataConfig())) != to_csv_text(generate_dataset(other))


def test_label_is_per_patient_and_shape_matches_config():
    cfg = DataConfig()
    df = generate_dataset(cfg)
    assert df.groupby("patient_id")["label"].nunique().max() == 1
    assert df["patient_id"].nunique() == cfg.n_patients
    counts = df.groupby("patient_id").size()
    assert counts.min() >= cfg.images_min and counts.max() <= cfg.images_max
    assert df.groupby("patient_id")["label"].first().sum() == cfg.n_patients // 2


def test_without_patient_shift_keeps_everything_else():
    a = generate_dataset(DataConfig())
    b = generate_dataset(without_patient_shift(DataConfig()))
    assert a[["image_id", "patient_id", "label"]].equals(b[["image_id", "patient_id", "label"]])
    # Within one patient, the shift is a constant per feature, so image-to-image
    # differences are the same in both datasets (up to the 6-decimal rounding).
    features = [c for c in a.columns if c not in ("image_id", "patient_id", "label")]
    diff = (a[features] - b[features]).groupby(a["patient_id"]).agg(lambda s: s.max() - s.min())
    assert diff.to_numpy().max() < 1e-5
