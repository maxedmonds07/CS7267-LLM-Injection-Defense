"""Pack the experiment source for Colab, including local edits on unpublished branches.

Writes the layout the notebook reads when DRIVE_PROJECT is set:
    dist/cs7267-b4/<study>/b4-colab-source.zip
    dist/cs7267-b4/<study>/inputs/{dataset.parquet, contexts.parquet, contexts.coverage.csv}
Upload the whole dist directory into the Drive folder named by DRIVE_PROJECT. The inputs
are written only with --inputs; without them, use DATA_MODE = "dvc".
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_STUDY = "b4-relevance-v1"


def source_files(root):
    files = sorted((root / "src/models").glob("b4*.py"))
    files += [root / "src" / name for name in (
        "data/b4_data.py", "data/dataset.py", "eval/harness.py", "eval/metrics.py",
        "models/baselines.py", "models/deberta.py", "models/deberta_run.py", "models/perplexity.py",
        "models/precomputed.py", "models/tfidf.py", "models/train.py")]
    files += [root / name for name in ("configs/b4.yaml", "config.yaml", "pyproject.toml", "uv.lock",
                                        "dvc.yaml", "dvc.lock", ".dvcignore", ".dvc/config", "README.md")]
    return files


def build_contexts(data, out):
    """Build the paired-context sidecar from local raw data (same code the notebook runs)."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src/data"))
    from b4_data import main
    main(["--data", str(data), "--raw", str(root / "data/raw"),
          "--generated", str(root / "data/generated"), "--out", str(out)])


def package(root, dist, *, study=DEFAULT_STUDY, data=None):
    """Write the study folder under dist; with data, also write the upload-mode inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", study):
        raise ValueError("Use the same simple study name as the notebook's STUDY field")
    root = Path(root)
    folder = Path(dist) / "cs7267-b4" / study
    folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(folder / "b4-colab-source.zip", "w", ZIP_DEFLATED) as archive:
        for path in source_files(root):
            archive.write(path, path.relative_to(root).as_posix())
    if data is not None:
        inputs = folder / "inputs"
        inputs.mkdir(exist_ok=True)
        shutil.copyfile(data, inputs / "dataset.parquet")
        # Fingerprint the copy the notebook will read, not the local original.
        build_contexts(inputs / "dataset.parquet", inputs / "contexts.parquet")
    return folder


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", default=DEFAULT_STUDY, help="must match the notebook's STUDY")
    parser.add_argument("--inputs", action="store_true",
                        help="also copy dataset.parquet and build contexts for DATA_MODE = 'upload'")
    parser.add_argument("--data", default=str(root / "data/processed/dataset.parquet"))
    args = parser.parse_args()
    folder = package(root, root / "dist", study=args.study, data=args.data if args.inputs else None)
    print(f"Wrote {folder}. Upload {root / 'dist'} into your Drive project folder.")
