"""Print the image-level vs patient-level comparison for both datasets."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_demo.data import DataConfig, generate_dataset, without_patient_shift  # noqa: E402
from leakage_demo.evaluate import evaluate  # noqa: E402


def main() -> None:
    config = DataConfig()
    print(f"config: {config}")
    print("| data | split | ROC AUC | accuracy | shared / test patients per fold | fold AUCs |")
    print("|---|---|---|---|---|---|")
    for data_name, cfg in [("normal", config), ("no patient shift", without_patient_shift(config))]:
        df = generate_dataset(cfg)
        for scheme in ("image", "patient"):
            r = evaluate(df, scheme)
            shared = ", ".join(f"{f['shared_patients']}/{f['test_patients']}" for f in r["folds"])
            aucs = ", ".join(f"{f['auc']:.3f}" for f in r["folds"])
            print(f"| {data_name} | {scheme} | {r['auc_mean']:.3f} ± {r['auc_std']:.3f} | "
                  f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | {shared} | {aucs} |")


if __name__ == "__main__":
    main()
