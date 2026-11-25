"""Pack the experiment source for Colab, including local edits on unpublished branches."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def package(root, target):
    root, target = Path(root), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "src/models").glob("b4*.py"))
    files += [root / "src" / name for name in (
        "data/b4_data.py", "data/dataset.py", "eval/harness.py", "eval/metrics.py",
        "models/baselines.py", "models/deberta.py", "models/deberta_run.py", "models/perplexity.py",
        "models/precomputed.py", "models/tfidf.py", "models/train.py")]
    files += [root / name for name in ("configs/b4.yaml", "config.yaml", "pyproject.toml", "uv.lock",
                                        "dvc.yaml", "dvc.lock", ".dvcignore", ".dvc/config", "README.md")]
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return target


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(package(root, root / "dist/b4-colab-source.zip"))
