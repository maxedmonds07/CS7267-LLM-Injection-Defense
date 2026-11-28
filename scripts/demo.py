"""Live demo: B1-Optuna learned a voice, not a behavior.

    make demo                      # four scenes, Enter between them
    make demo ARGS="--no-pause"    # run straight through
    make demo ARGS='--text "Create a new GitHub repository in your account"'
    make demo ARGS="--interactive" # score whatever the audience suggests
    make demo ARGS="--prompt-guard"  # also score with Prompt Guard 2 (gated: HF login first)

Every scene is a real held-out row from data/processed/dataset.parquet, scored by B1-Optuna (our
recommended detector, refit from its committed settings) at its deployed threshold (1% FPR on
in-distribution validation, report/phase5/thresholds.json). "Pushed up by" lists the word
n-grams that add most to the score: TF-IDF weight times the logistic coefficient, in log-odds.
The fitted model is cached (default ~/.cache/cs7267-demo) so later runs start at once.
"""

import argparse
import hashlib
import json
import pickle
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/data", "src/models", "src/eval", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

from dataset import DEFAULT_PATH  # noqa: E402
from metrics import threshold_at_fpr  # noqa: E402
from mitigation_normalize import normalize  # noqa: E402
from red_team import mutate  # noqa: E402

THRESHOLDS = ROOT / "report" / "phase5" / "thresholds.json"
CACHE = Path.home() / ".cache" / "cs7267-demo"
WIDTH = 92


@dataclass(frozen=True)
class Scene:
    title: str
    source: str
    row_id: str
    point: str
    mutation: str | None = None  # a red-team mutation, applied exactly as in the red-team run


SCENES = [
    Scene("The voice it learned", "BIPIA email, held-out test row",
          "bipia:email:test:40ea2c1085ef8dfe:Emoji Substitution-2:middle",
          "An attack phrased the way BIPIA writes them: a request about the model's response."),
    Scene("The same kind of goal, written as tool procedure", "MCPTox tool description (never trained on)",
          "mcptox:mcp-simple-arxiv_3",
          "It hijacks another tool's argument, but in the voice of documentation, so it passes."),
    Scene("Benign documentation in the wrong voice", "MCPTox tool description, benign",
          "mcptox:Github:clean:create_repository",
          "Harmless, but 'your account' sounds like the attacks it was trained on."),
    Scene("A flagged attack, lightly disguised", "BIPIA email, held-out test row, red-team leetspeak",
          "bipia:email:test:4bb2e9ac704c323f:Reverse Text-3:end",
          "Swapping letters for digits defeats it, and normalizing cannot undo that. In the red-team\n"
          "    run, leetspeak cut B1-Optuna's test recall from 0.725 to 0.213 (5% FPR threshold).",
          mutation="leetspeak"),
]


# ---- scoring -------------------------------------------------------------------------------

def verdict(score: float, threshold: float) -> str:
    """The harness flags strictly above the threshold."""
    return "FLAGGED" if score > threshold else "passes"


def pushed_up_by(detector, text: str, k: int = 4) -> list[tuple[str, float]]:
    """The k word n-grams adding most to the log-odds of `text`: weight x coefficient, positive only."""
    union, model = detector.pipeline.steps[0][1], detector.pipeline.steps[1][1]
    row = union.transform([text]).tocsr()
    names = union.get_feature_names_out()
    contrib = row.data * model.coef_[0][row.indices]
    pairs = [(names[i], c) for i, c in zip(row.indices, contrib) if names[i].startswith("word__") and c > 0]
    pairs.sort(key=lambda p: -p[1])
    return [(name.removeprefix("word__"), float(c)) for name, c in pairs[:k]]


# ---- model ---------------------------------------------------------------------------------

def _fingerprint(data_path: Path) -> str:
    from tfidf_optuna import BEST_PARAMS_PATH
    h = hashlib.sha256(Path(BEST_PARAMS_PATH).read_bytes())
    stat = Path(data_path).stat()
    h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def load_detector(data_path: Path, cache_dir: Path | None = CACHE):
    """B1-Optuna refit from its committed settings, cached per (settings, dataset) pair. Returns
    the detector and the validation threshold it reproduces."""
    path = cache_dir / f"b1_optuna-{_fingerprint(data_path)}.pkl" if cache_dir else None
    if path and path.exists():
        # Only ever a file this function wrote to the user's own cache directory; --no-cache skips it.
        with path.open("rb") as f:
            return pickle.load(f)
    from harness import load_data
    from train import fit_detector
    print("Fitting B1-Optuna from its committed settings (about 10 s, once)...", flush=True)
    data = load_data(str(data_path))
    detector = fit_detector("tfidf-optuna", data)[0]
    val_benign = data.val.text[data.val.label == 0]
    result = (detector, float(threshold_at_fpr(detector.predict_proba(val_benign), 0.01)))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(result, f)
    return result


def deployed_threshold(reproduced: float) -> float:
    saved = json.loads(THRESHOLDS.read_text())["b1_optuna"]
    if not np.isclose(saved, reproduced, atol=1e-6):
        print(f"warning: refit threshold {reproduced:.4f} differs from the committed {saved:.4f}", file=sys.stderr)
    return saved


def rows_by_id(ids, data_path: Path) -> dict[str, str]:
    table = pq.read_table(data_path, columns=["id", "text"], filters=[("id", "in", list(set(ids)))])
    found = dict(zip(table.column("id").to_pylist(), table.column("text").to_pylist()))
    missing = set(ids) - set(found)
    if missing:
        raise SystemExit(f"rows not in {data_path}: {sorted(missing)} (run `dvc pull` first?)")
    return found


# ---- display -------------------------------------------------------------------------------

def shown(text: str, max_lines: int = 8) -> str:
    """The text as the audience should read it: escaped newlines unescaped, wrapped, indented."""
    lines = []
    for para in text.replace("\\n", "\n").strip().splitlines():
        lines += textwrap.wrap(para, WIDTH - 4) or [""]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join("  │ " + line for line in lines)


def score_line(name: str, score: float, threshold: float) -> str:
    return f"  {name:<22} score {score:.3f}   threshold {threshold:.3f}   →  {verdict(score, threshold)}"


def report(detector, threshold: float, text: str, pg=None) -> str:
    score = float(detector.predict_proba([text])[0])
    out = [score_line("B1-Optuna", score, threshold)]
    words = pushed_up_by(detector, text)
    if words:
        out.append("  pushed up by:  " + ",  ".join(f'"{w}" +{c:.2f}' for w, c in words))
    if pg is not None:
        guard = float(pg([text])[0])
        out.append(f"  {'Prompt Guard 2':<22} score {guard:.4f}   its own cut 0.500   →  {verdict(guard, 0.5)}")
    return "\n".join(out)


def play(scene: Scene, n: int, total: int, text: str, detector, threshold: float, pg=None) -> str:
    head = f"[{n}/{total}] {scene.title}"
    out = ["", "═" * WIDTH, head, f"  {scene.source}", "─" * WIDTH]
    if scene.mutation:
        mutated = mutate(text, scene.row_id, scene.mutation)
        out += [shown(text), "", report(detector, threshold, text, pg),
                "", f"  after the red-team {scene.mutation} mutation:", shown(mutated), "",
                report(detector, threshold, mutated, pg),
                "", "  after normalizing (NFKC, zero-width removed):",
                report(detector, threshold, normalize(mutated), pg)]
    else:
        out += [shown(text), "", report(detector, threshold, text, pg)]
    out += ["", f"  ▸ {scene.point}"]
    return "\n".join(out)


def prompt_guard():
    from prompt_guard_colab import hf_scorer
    print("Loading Prompt Guard 2 (gated model; uses your Hugging Face login)...", flush=True)
    return hf_scorer()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter between scenes")
    ap.add_argument("--text", action="append", help="score this text instead of the scenes (repeatable)")
    ap.add_argument("--interactive", "-i", action="store_true", help="type texts to score; empty line quits")
    ap.add_argument("--prompt-guard", action="store_true", help="also score with Prompt Guard 2")
    ap.add_argument("--no-cache", action="store_true", help="refit instead of using the cached model")
    args = ap.parse_args(argv)
    data_path = Path(args.data)

    started = time.perf_counter()
    detector, reproduced = load_detector(data_path, None if args.no_cache else CACHE)
    threshold = deployed_threshold(reproduced)
    pg = prompt_guard() if args.prompt_guard else None
    print(f"B1-Optuna ready in {time.perf_counter() - started:.1f} s. Deployed threshold {threshold:.3f} "
          f"(1% false positives on in-distribution validation).")

    if args.text or args.interactive:
        for text in args.text or []:
            print("\n" + shown(text) + "\n\n" + report(detector, threshold, text, pg))
        while args.interactive:
            try:
                text = input("\ntext> ").strip()
            except EOFError:
                break
            if not text:
                break
            print(report(detector, threshold, text, pg))
        return

    texts = rows_by_id([s.row_id for s in SCENES], data_path)
    for n, scene in enumerate(SCENES, 1):
        print(play(scene, n, len(SCENES), texts[scene.row_id], detector, threshold, pg))
        if not args.no_pause and n < len(SCENES):
            input("\n  [Enter for the next scene]")
    print("\n" + "═" * WIDTH + "\nIt learned a voice, not a behavior.\n")


if __name__ == "__main__":
    main()
