"""Streamlit demo: the same model scored with an image-level split and a patient-level split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from leakage_demo.data import FEATURE_NAMES, DataConfig, generate_dataset, without_patient_shift  # noqa: E402
from leakage_demo.evaluate import N_SPLITS, evaluate  # noqa: E402

SCHEMES = {"image": "Split by image", "patient": "Split by patient"}
LABEL_NAMES = {0: "No finding", 1: "Finding"}
LABEL_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # validated pair (CVD-safe)


@st.cache_data(show_spinner="Generating data and running 5-fold cross-validation...")
def load() -> dict:
    config = DataConfig()
    normal = generate_dataset(config)
    control = generate_dataset(without_patient_shift(config))
    return {
        "config": config,
        "df": normal,
        "normal": {s: evaluate(normal, s) for s in SCHEMES},
        "control": {s: evaluate(control, s) for s in SCHEMES},
    }


def pm(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def fold_auc_chart(result: dict, title: str) -> go.Figure:
    aucs = [f["auc"] for f in result["folds"]]
    fig = go.Figure(
        go.Scatter(
            x=[f"Fold {i + 1}" for i in range(len(aucs))],
            y=aucs,
            mode="markers",
            marker={"size": 11, "color": "#2a78d6"},
            hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title=title)
    fig.update_yaxes(range=[0.4, 1.02], title="ROC AUC")
    return fig


def per_patient_summary(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """One row per patient: true label, mean out-of-fold probability, and whether it is on the correct side of 0.5."""
    summary = (
        df[["patient_id", "label"]]
        .assign(proba=result["oof_proba"])
        .groupby("patient_id")
        .agg(label=("label", "first"), mean_proba=("proba", "mean"), images=("proba", "size"))
        .reset_index()
        .sort_values(["label", "patient_id"], kind="stable")
    )
    summary["correct"] = (summary["mean_proba"] >= 0.5) == (summary["label"] == 1)
    return summary


def threshold_line(fig: go.Figure) -> None:
    fig.add_hline(y=0.5, line={"color": "#8a8985", "width": 1, "dash": "dash"})


def per_patient_chart(summary: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for label in (0, 1):
        for correct in (True, False):
            part = summary[(summary["label"] == label) & (summary["correct"] == correct)]
            # An empty group still gets a legend entry (Plotly hides traces with no data),
            # so both charts list the same four entries in the same order.
            empty = part.empty
            fig.add_trace(
                go.Scatter(
                    x=[None] if empty else part["patient_id"],
                    y=[None] if empty else part["mean_proba"],
                    mode="markers",
                    name=f"{LABEL_NAMES[label]}, {'correct side' if correct else 'wrong side'}",
                    marker={
                        "size": 9 if correct else 11,
                        "color": LABEL_COLORS[label],
                        "symbol": "circle" if correct else "x",
                    },
                    customdata=None if empty else part[["images"]],
                    hovertemplate="%{x} (%{customdata[0]} images)<br>Mean P(finding) %{y:.2f}<extra></extra>",
                )
            )
    threshold_line(fig)
    fig.update_layout(
        title=title,
        height=400,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "y": -0.22},
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=summary["patient_id"].tolist(),
        title="Patient (sorted by true label)",
        tickangle=-90,
        tickfont={"size": 9},
    )
    fig.update_yaxes(range=[0, 1], title="Mean predicted probability of finding")
    return fig


def image_chart(df_patient: pd.DataFrame, proba: np.ndarray, label: int, title: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df_patient["image_id"],
            y=proba,
            mode="markers",
            marker={"size": 9, "color": LABEL_COLORS[label]},
            hovertemplate="%{x}<br>P(finding) %{y:.2f}<extra></extra>",
        )
    )
    threshold_line(fig)
    fig.update_layout(title=title, height=280, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    fig.update_xaxes(title="Image", tickangle=-90, tickfont={"size": 9})
    fig.update_yaxes(range=[0, 1], title="P(finding)")
    return fig


st.set_page_config(page_title="Patient-level data leakage in an imaging model", layout="wide")

data = load()
df: pd.DataFrame = data["df"]
config: DataConfig = data["config"]
normal = data["normal"]
control = data["control"]

st.title("Patient-level data leakage in an imaging model")
st.write(
    f"**Same data, same model. Split by image: AUC {normal['image']['auc_mean']:.2f}. "
    f"Split by patient: {normal['patient']['auc_mean']:.2f}.**"
)
st.caption(
    f"Each patient has {config.images_min}–{config.images_max} images. Splitting by image puts images of "
    "the same patient in both training and test, so the model can recognise the patient instead of the finding."
)

st.subheader("Scores")
for col, scheme in zip(st.columns(2), SCHEMES):
    r = normal[scheme]
    with col:
        st.markdown(f"#### {SCHEMES[scheme]}")
        a, b = st.columns(2)
        a.metric(f"ROC AUC ({N_SPLITS}-fold mean ± SD)", pm(r["auc_mean"], r["auc_std"]))
        b.metric(f"Accuracy ({N_SPLITS}-fold mean ± SD)", pm(r["accuracy_mean"], r["accuracy_std"]))
        shared = ", ".join(f"{f['shared_patients']}/{f['test_patients']}" for f in r["folds"])
        st.write(f"Test patients also in training (folds 1–5): {shared}")

st.subheader("Prediction for each patient")
st.caption(
    "Each dot is one patient: the mean out-of-fold prediction over their images. "
    "With a split by image, the model has already seen every test patient in training, "
    "so it gets them right by recognising the patient, not the finding."
)
for col, scheme in zip(st.columns(2), SCHEMES):
    summary = per_patient_summary(df, normal[scheme])
    title = f"{SCHEMES[scheme]}: {int(summary['correct'].sum())} of {len(summary)} patients on the correct side"
    with col:
        st.plotly_chart(per_patient_chart(summary, title), width="stretch", key=f"patients-{scheme}")

st.subheader("Review one patient")
patient_labels = df.groupby("patient_id")["label"].first()
patient = st.selectbox(
    "Patient",
    patient_labels.index.tolist(),
    format_func=lambda p: f"{p} ({LABEL_NAMES[int(patient_labels[p])]}, {int((df['patient_id'] == p).sum())} images)",
)
rows = (df["patient_id"] == patient).to_numpy()
for col, scheme in zip(st.columns(2), SCHEMES):
    proba = normal[scheme]["oof_proba"][rows]
    title = f"{SCHEMES[scheme]}: mean {proba.mean():.2f}"
    with col:
        st.plotly_chart(
            image_chart(df[rows], proba, int(patient_labels[patient]), title), width="stretch", key=f"images-{scheme}"
        )
review = pd.DataFrame(
    {
        "Image": df.loc[rows, "image_id"],
        "True label": df.loc[rows, "label"].map(LABEL_NAMES),
        "Split by image: fold": normal["image"]["oof_fold"][rows] + 1,
        "Split by image: P(finding)": normal["image"]["oof_proba"][rows],
        "Split by patient: fold": normal["patient"]["oof_fold"][rows] + 1,
        "Split by patient: P(finding)": normal["patient"]["oof_proba"][rows],
    }
)
st.dataframe(
    review,
    hide_index=True,
    column_config={
        "Split by image: P(finding)": st.column_config.NumberColumn(format="%.2f"),
        "Split by patient: P(finding)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption("P(finding) is the predicted probability of a finding; accuracy counts 0.5 or above as a finding.")

st.subheader("Notes")
with st.expander("Why does this happen?"):
    st.write(
        "Every feature value is the sum of three parts: a fixed offset for each patient, a weak signal shared by "
        "all patients with a finding, and noise for each image. The patient offset makes a patient's images look "
        "alike. When the same patient is in training and test, the model can return that patient's remembered "
        "label without learning the weak signal."
    )
    st.write(
        "To check that the offset is the cause, the same data was generated again with the patient offset set to "
        "zero (same labels, image counts and noise). The gap between the two splits disappears:"
    )
    rows_ = []
    for data_name, res in (("Normal data", normal), ("Patient offset set to zero", control)):
        rows_.append(
            {
                "Data": data_name,
                "Split by image: ROC AUC": pm(res["image"]["auc_mean"], res["image"]["auc_std"]),
                "Split by patient: ROC AUC": pm(res["patient"]["auc_mean"], res["patient"]["auc_std"]),
                "Gap": f"{res['image']['auc_mean'] - res['patient']['auc_mean']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(rows_), hide_index=True)
    st.write(
        "Compare only the two splits within the same data. Without the offset, patients no longer differ from "
        "each other, so the task itself is easier and all scores are higher. Its scores say nothing about how "
        "well the model does on the normal data."
    )

with st.expander("Why do patient-level scores swing?"):
    pr = normal["patient"]
    aucs = [f["auc"] for f in pr["folds"]]
    tests = sorted({f["test_patients"] for f in pr["folds"]})
    n_test = ", ".join(str(t) for t in tests)
    st.write(
        f"With a patient-level split, each fold tests only {n_test} patients. Images of one patient move together, "
        "so one or two patients ranked the wrong way change the fold's AUC a lot. "
        f"Here the fold AUC ranged from {min(aucs):.3f} to {max(aucs):.3f}."
    )
    st.write(
        "With an image-level split, every fold contains images of almost every patient, and the model has seen "
        "those patients in training, so the folds look stable. That stability is part of the leak, not a sign of "
        "a reliable model."
    )
    for col, scheme in zip(st.columns(2), SCHEMES):
        with col:
            st.plotly_chart(
                fold_auc_chart(normal[scheme], f"{SCHEMES[scheme]}: ROC AUC in each fold"),
                width="stretch",
                key=f"folds-{scheme}",
            )

with st.expander("About this data"):
    counts = df.groupby("patient_id").size()
    n_pos = int(patient_labels.sum())
    st.write(
        f"All data is randomly generated. There are no real patients or images.\n\n"
        f"- {df['patient_id'].nunique()} patients: {n_pos} with a finding, {len(patient_labels) - n_pos} without. "
        "The label is set per patient, so all images of a patient share it.\n"
        f"- {len(df)} images, {counts.min()}–{counts.max()} per patient.\n"
        f"- {len(FEATURE_NAMES)} numeric features per image: {', '.join(FEATURE_NAMES)}.\n"
        f"- Random seed: {config.seed}. The data is generated when the app starts; no files are read or uploaded.\n"
        f"- Model: random forest with 300 trees. {N_SPLITS}-fold cross-validation with StratifiedKFold "
        "(split by image) and StratifiedGroupKFold (split by patient)."
    )
