from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_app_renders_to_the_end_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    headline = at.markdown[0].value
    assert "Split by image: AUC 0.98" in headline and "Split by patient: 0.65" in headline
    assert [e.label for e in at.expander] == [
        "Why does this happen?",
        "Why do patient-level scores swing?",
        "About this data",
    ]

    # Picking another patient re-renders the review table without errors.
    at.selectbox[0].select("P040").run()
    assert not at.exception
    assert len(at.dataframe) >= 1
