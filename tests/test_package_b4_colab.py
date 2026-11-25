import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_b4_colab", ROOT / "scripts/package_b4_colab.py")
package_b4_colab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_b4_colab)


def notebook_source():
    cells = json.loads((ROOT / "notebooks/b4_colab.ipynb").read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_archive_lands_where_the_notebook_reads_it(tmp_path):
    folder = package_b4_colab.package(ROOT, tmp_path / "dist", study="my-study")
    assert folder == tmp_path / "dist/cs7267-b4/my-study"
    with ZipFile(folder / "b4-colab-source.zip") as archive:
        assert "src/models/b4_experiments.py" in archive.namelist()
    # The notebook derives the same folder from DRIVE_PROJECT and STUDY.
    assert "'dist/cs7267-b4' / STUDY" in notebook_source().replace('"', "'")


def test_inputs_use_the_notebook_file_names(tmp_path, monkeypatch):
    def fake_build(data, out):
        out.write_bytes(b"ctx")
        out.with_suffix(".coverage.csv").write_text("source\n", encoding="utf-8")
    monkeypatch.setattr(package_b4_colab, "build_contexts", fake_build)
    data = tmp_path / "dataset.parquet"
    data.write_bytes(b"data")
    inputs = package_b4_colab.package(ROOT, tmp_path / "dist", data=data) / "inputs"
    assert sorted(p.name for p in inputs.iterdir()) == [
        "contexts.coverage.csv", "contexts.parquet", "dataset.parquet"]
    assert (inputs / "dataset.parquet").read_bytes() == b"data"


@pytest.mark.parametrize("study", ["", "../escape", "a b"])
def test_rejects_study_names_the_notebook_rejects(tmp_path, study):
    with pytest.raises(ValueError):
        package_b4_colab.package(ROOT, tmp_path / "dist", study=study)
