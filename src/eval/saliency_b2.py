"""Integrated-gradients saliency for B2 (DeBERTa row 2, seed 7267).

    PYTHONPATH=src/data:src/eval:src/models uv run --group deberta python src/eval/saliency_b2.py

For each row, take the window that sets its score (the max-logit chunk, as in predict_proba) and
attribute that window's logit to its tokens with Captum's LayerIntegratedGradients on the
embedding layer: 300 steps, baseline = the same window with every non-special token replaced by
[PAD]. Raw attention is not used: it is not a faithful explanation (Jain and Wallace, 2019).

Rows: a fixed-seed sample of N_PER_GROUP rows from each of MCPTox benign, MCPTox adversarial,
test benign and test adversarial. Writes to report/phase5/saliency/:

  rows.csv      per row: group, score, convergence delta, share of positive attribution on
                instruction words, top tokens (no full text)
  summary.csv   the instruction-word share per group (mean and bootstrap 95% interval)
  heatmaps.png  token heatmaps for six MCPTox rows (shortened)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from harness import load_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "phase5" / "saliency"
MODEL = ROOT / "models" / "deberta" / "row2-s7267"
N_PER_GROUP = 30
STEPS = 300  # 50 steps left completeness errors near 20% of the logit on confident rows
SEED = 7267
# Words that address the reader as an instruction: modals, imperatives common in tool
# descriptions and injections, and second person.
INSTRUCTION_WORDS = frozenset("""must should always never only ensure make sure do not don't use call
run invoke execute first before after important required need you your please instead""".split())
WORD = re.compile(r"[a-z']+")


def is_instruction(token: str) -> bool:
    word = token.replace("▁", "").lower()
    return bool(WORD.fullmatch(word)) and word in INSTRUCTION_WORDS


def instruction_share(tokens, attributions) -> float:
    """Share of positive (toward adversarial) attribution that falls on instruction words."""
    attributions = np.asarray(attributions, dtype=float)
    positive = np.clip(attributions, 0, None)
    total = positive.sum()
    if total <= 0:
        return float("nan")
    mask = np.array([is_instruction(t) for t in tokens])
    return float(positive[mask].sum() / total)


def best_window(detector, text):
    """Token ids of the window that sets the text's score, with [CLS] and [SEP]."""
    import torch

    windows, _ = detector._windows(detector._encode([text]))
    with torch.no_grad():
        logits = detector._chunk_logits(windows)
    return windows[int(torch.argmax(logits))]


def attribute(detector, window, steps=STEPS):
    """(tokens, per-token attribution, logit, convergence delta) for one window."""
    import torch
    from captum.attr import LayerIntegratedGradients

    model = detector.model
    device = next(model.parameters()).device
    tok = detector.tokenizer
    ids = torch.tensor([window], device=device)
    special = {tok.cls_token_id, tok.sep_token_id}
    baseline = torch.tensor([[i if i in special else tok.pad_token_id for i in window]], device=device)
    mask = torch.ones_like(ids)

    def forward(input_ids):
        return model(input_ids=input_ids, attention_mask=mask.expand(input_ids.shape[0], -1)).logits.squeeze(-1)

    lig = LayerIntegratedGradients(forward, model.base_model.embeddings)
    attributions, delta = lig.attribute(ids, baselines=baseline, n_steps=steps, return_convergence_delta=True,
                                        internal_batch_size=25)
    per_token = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    with torch.no_grad():
        logit = float(forward(ids)[0])
    return tok.convert_ids_to_tokens(window), per_token, logit, float(delta.item())


def sample_rows(data, n=N_PER_GROUP, seed=SEED) -> pd.DataFrame:
    parts = []
    for eval_set in ("mcptox", "test"):
        frame = data.evals[eval_set]
        for label in (0, 1):
            rows = frame[frame.label == label]
            parts.append(rows.sample(min(n, len(rows)), random_state=seed).assign(eval_set=eval_set))
    return pd.concat(parts, ignore_index=True)


def bootstrap_mean(values, n_boot=2000, seed=SEED):
    values = np.asarray([v for v in values if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def heatmaps(examples, path, max_tokens=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 1.25 * len(examples)))
    for ax, (title, tokens, attributions) in zip(np.atleast_1d(axes), examples):
        tokens, attributions = tokens[1:-1][:max_tokens], np.asarray(attributions[1:-1][:max_tokens])
        scale = np.abs(attributions).max() or 1.0
        norm = colors.TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8, loc="left")
        x, y = 0.0, 0.75
        for token, value in zip(tokens, attributions):
            word = token.replace("▁", " ")
            width = 0.0105 * max(len(word), 1)
            if x + width > 1.0:
                x, y = 0.0, y - 0.38
            ax.text(x, y, word, fontsize=6.5, family="monospace", transform=ax.transAxes, va="center",
                    bbox={"facecolor": plt.cm.RdBu_r(norm(value)), "edgecolor": "none", "pad": 0.6})
            x += width
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--n", type=int, default=N_PER_GROUP)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    import torch
    from deberta import DebertaDetector

    torch.manual_seed(SEED)
    detector = DebertaDetector.load(args.model, device=args.device)
    detector.model.eval()
    rows = sample_rows(load_data(), args.n)
    records, examples = [], []
    for i, row in enumerate(rows.itertuples()):
        window = best_window(detector, row.text)
        tokens, attributions, logit, delta = attribute(detector, window)
        order = np.argsort(-attributions)[:5]
        records.append({
            "eval_set": row.eval_set, "label": row.label, "source": row.source, "logit": logit,
            "score": float(1 / (1 + np.exp(-logit))), "convergence_delta": delta, "n_tokens": len(tokens),
            "instruction_share": instruction_share(tokens, attributions),
            "instruction_token_share": float(np.mean([is_instruction(t) for t in tokens])),
            "top_tokens": " ".join(tokens[j].replace("▁", "") for j in order),
        })
        print(f"{i + 1}/{len(rows)} {row.eval_set} label={row.label} logit={logit:.2f} delta={delta:.3f}", flush=True)
        if row.eval_set == "mcptox" and sum(e[0].startswith(f"MCPTox {'attack' if row.label else 'benign'}")
                                            for e in examples) < 3:
            examples.append((f"MCPTox {'attack' if row.label else 'benign'} (score {1 / (1 + np.exp(-logit)):.3f})",
                             tokens, attributions))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "rows.csv", index=False)
    summary = []
    for (eval_set, label), part in table.groupby(["eval_set", "label"]):
        mean, low, high = bootstrap_mean(part.instruction_share)
        summary.append({"eval_set": eval_set, "label": label, "n": len(part), "instruction_share": mean,
                        "lo": low, "hi": high, "instruction_token_share": part.instruction_token_share.mean(),
                        "mean_score": part.score.mean(), "median_abs_delta": part.convergence_delta.abs().median()})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    heatmaps(examples, out / "heatmaps.png")
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
