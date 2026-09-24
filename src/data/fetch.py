"""Fetch the raw datasets listed in config.yaml into data/raw/.

Each dataset is pulled from a GitHub tarball at a pinned commit, keeping only the
configured paths. For PoisonedRAG, the BEIR corpora are reduced to the passages
the attack targets (top-k retrieved + gold) so the full corpora never reach R2.

Usage: python src/data/fetch.py [--config config.yaml] [--only NAME ...]
"""

import argparse
import csv
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/{commit}"


def wanted(rel_path: str, patterns: list[str]) -> bool:
    return any(
        rel_path.startswith(p) if p.endswith("/") else rel_path == p for p in patterns
    )


def fetch_repo_paths(spec: dict, dest: Path) -> list[str]:
    """Stream the repo tarball at the pinned commit and extract only the configured paths."""
    url = TARBALL_URL.format(repo=spec["repo"], commit=spec["commit"])
    written = []
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # Tarball entries are prefixed with "<repo>-<commit>/".
            rel = member.name.split("/", 1)[1] if "/" in member.name else ""
            if not rel or not wanted(rel, spec["paths"]):
                continue
            target = (dest / rel).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"refusing to write outside {dest}: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(rel)

    missing = [p for p in spec["paths"] if not any(wanted(w, [p]) for w in written)]
    if missing:
        raise FileNotFoundError(f"{spec['repo']}@{spec['commit'][:7]} has no {missing}")
    return written


def download(url: str, target: Path) -> Path:
    """Download to target unless a complete copy is already cached."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        expected = int(resp.headers.get("Content-Length", 0))
        shutil.copyfileobj(resp, out, length=1 << 20)
    if expected and part.stat().st_size != expected:
        part.unlink()
        raise IOError(f"truncated download: {url}")
    part.rename(target)
    return target


def subset_beir(name: str, split: str, beir: dict, dataset_dir: Path, cache_dir: Path) -> dict:
    """Write corpus/queries/qrels for PoisonedRAG's target queries only."""
    adv = json.loads((dataset_dir / f"results/adv_targeted_results/{name}.json").read_text())
    retrieved = json.loads(
        (dataset_dir / f"results/beir_results/{name}-contriever.json").read_text()
    )
    target_qids = {case["id"] for case in adv.values()}

    zip_path = download(beir["url"].format(name=name), cache_dir / f"{name}.zip")
    out = dataset_dir / "beir" / name
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{name}/qrels/{split}.tsv") as f:
            rows = csv.DictReader((line.decode() for line in f), delimiter="\t")
            qrels = [r for r in rows if r["query-id"] in target_qids]

        doc_ids = {r["corpus-id"] for r in qrels}
        for qid in target_qids:
            doc_ids.update(list(retrieved[qid])[: beir["top_k"]])

        found = set()
        with zf.open(f"{name}/corpus.jsonl") as f, open(out / "corpus.jsonl", "w") as dst:
            for line in f:
                doc = json.loads(line)
                if doc["_id"] in doc_ids:
                    dst.write(json.dumps(doc) + "\n")
                    found.add(doc["_id"])

        with zf.open(f"{name}/queries.jsonl") as f, open(out / "queries.jsonl", "w") as dst:
            for line in f:
                if json.loads(line)["_id"] in target_qids:
                    dst.write(line.decode())

    with open(out / "qrels.tsv", "w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
        writer.writeheader()
        writer.writerows(qrels)

    if missing := doc_ids - found:
        raise ValueError(f"{name}: {len(missing)} targeted passages not in BEIR corpus")
    return {"split": split, "queries": len(target_qids), "passages": len(found), "qrels": len(qrels)}


def fetch_dataset(name: str, spec: dict, raw_dir: Path, cache_dir: Path) -> None:
    dest = raw_dir / name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)

    print(f"{name}: {spec['repo']}@{spec['commit'][:7]}", file=sys.stderr)
    files = fetch_repo_paths(spec, dest)
    source = {k: spec[k] for k in ("repo", "commit", "license", "surface", "role")}
    source["files"] = sorted(files)

    if beir := spec.get("beir"):
        source["beir"] = {
            sub: subset_beir(sub, split, beir, dest, cache_dir)
            for sub, split in beir["subsets"].items()
        }

    (dest / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"  {len(files)} files -> {dest}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--only", nargs="+", help="fetch just these datasets")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    raw_dir = Path(config["paths"]["raw_dir"])
    cache_dir = Path(config["paths"]["cache_dir"])
    datasets = config["datasets"]

    for name in args.only or datasets:
        fetch_dataset(name, datasets[name], raw_dir, cache_dir)


if __name__ == "__main__":
    main()
