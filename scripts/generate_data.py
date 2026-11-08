"""Write the synthetic datasets to data/ (normal and without patient shift)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_demo.data import DataConfig, generate_dataset, to_csv_text, without_patient_shift  # noqa: E402


def main() -> None:
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    config = DataConfig()
    for name, cfg in [("images.csv", config), ("images_no_patient_shift.csv", without_patient_shift(config))]:
        df = generate_dataset(cfg)
        (out / name).write_text(to_csv_text(df), encoding="utf-8", newline="\n")
        positives = int(df.groupby("patient_id")["label"].first().sum())
        print(f"{name}: {len(df)} images, {df['patient_id'].nunique()} patients, {positives} positive patients")


if __name__ == "__main__":
    main()
