"""phase3_eda.py writes figures 01-10 and the probe section into a run folder from the fixture data."""
import importlib.util
from pathlib import Path


def test_eda_writes_probe_figures(dataset_path, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("phase3_eda", root / "notebooks" / "phase3_eda.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(root / "notebooks"))
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path / "figures")
    monkeypatch.setattr(mod, "WRITEUP", tmp_path / "W.md")
    monkeypatch.setattr(mod, "MARKDOWN_ONLY", True)   # no pandoc PDF build in tests
    mod.main(["--data", str(dataset_path)])
    for name in ("08_roc_length_probe.png", "09_auroc_ci_forest.png", "10_spearman_ci.png"):
        assert (tmp_path / "figures" / name).exists()
    assert "Length shortcut with confidence intervals" in (tmp_path / "W.md").read_text()
