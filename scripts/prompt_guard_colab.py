"""Score val, test, MCPTox and MSB zero-shot with Meta's Prompt Guard 2 (an off-the-shelf
injection detector; follow-up analysis outside the pre-registered protocol).

The model is gated: accept the Llama 4 Community License for
meta-llama/Llama-Prompt-Guard-2-86M on Hugging Face first, then run on a Colab T4:

    !git clone <repo> && cd CS7267-LLM-Injection-Defense && pip install uv
    !uv sync --group deberta && make data          # needs the team's DVC credentials
    import os; os.environ["HF_TOKEN"] = "<your token>"
    !PYTHONPATH=src/data:src/eval:src/models uv run python scripts/prompt_guard_colab.py

and download both files in report/offshelf_scores/ into the same path locally:
- prompt-guard-2.parquet: the B2 score-file layout, so the harness reads it with
  `--detector deberta --scores <file>`; it reports as offshelf_prompt_guard_2, not as a DeBERTa run.
  It lives outside report/deberta_scores so the B2 verdict never picks it up.
- prompt-guard-2_native.csv: recall and FPR at Prompt Guard's own 0.5 cut, the way someone would
  deploy it off the shelf (the harness applies our 1% val-FPR threshold instead), with the number of
  chunked texts and the training-data exposure of each eval set.

Long texts are scored like B2: overlapping 512-token windows, text score = highest window, so an
injection late in a long BIPIA row is not truncated away.

Framing for the paper: Prompt Guard 2's malicious label covers text that explicitly tries to
override earlier instructions; MCPTox attacks are implicit tool procedure, so a low score there is
partly by design. Write "a public detector not built for this surface also misses it". Its training
mix includes open-source and curated injection data, so BIPIA overlap cannot be ruled out: the
in-distribution sets are marked exposure "unknown".
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataset import DEFAULT_PATH
from deberta import chunk_spans
from deberta_run import scores_table
from harness import load_data
from metrics import rates_at_threshold

MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
NAME = "offshelf_prompt_guard_2"
OUT = Path(__file__).resolve().parents[1] / "report" / "offshelf_scores" / "prompt-guard-2.parquet"
CHUNK_TOKENS = 510  # 512 minus [CLS] and [SEP]
CHUNK_STRIDE = 384
WINDOW_BATCH = 64
NATIVE_THRESHOLD = 0.5
# Benchmarks released after Prompt Guard 2; the in-distribution sets may overlap its training data.
HELD_OUT = {"mcptox", "msb"}


def windows(token_lists, chunk_tokens=CHUNK_TOKENS, stride=CHUNK_STRIDE):
    """(windows, owners): every overlapping window of every text's content tokens, and the index
    of the text it came from."""
    out, owners = [], []
    for i, ids in enumerate(token_lists):
        for start, end in chunk_spans(len(ids), chunk_tokens, stride):
            out.append(list(ids[start:end]))
            owners.append(i)
    return out, owners


def max_by_owner(values, owners, n) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, np.asarray(owners), np.asarray(values, dtype=float))
    return out


def check_two_class(config) -> None:
    """Prompt Guard 2 is benign vs malicious. Prompt Guard 1 had three classes, where the last
    index meant jailbreak only, so taking the last column would silently score the wrong thing."""
    if config.num_labels != 2:
        raise ValueError(f"expected 2 classes (benign, malicious), got {config.num_labels}: {getattr(config, 'id2label', None)}")


def hf_scorer(model_name=MODEL, device=None):
    """Batch scorer: probability of the malicious class, max over each text's windows. Counts the
    texts that needed more than one window in the `chunked` attribute."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.model_max_length = 10**9  # we chunk ourselves; silence the length warning
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token).eval()
    check_two_class(model.config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def score(texts):
        ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        chunks, owners = windows(ids)
        score.chunked += sum(np.bincount(owners, minlength=len(ids)) > 1)
        probs = []
        for i in range(0, len(chunks), WINDOW_BATCH):
            inputs = [tokenizer.build_inputs_with_special_tokens(c) for c in chunks[i:i + WINDOW_BATCH]]
            batch = tokenizer.pad({"input_ids": inputs}, return_tensors="pt").to(device)
            with torch.no_grad():
                probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].float().cpu().numpy())
        return max_by_owner(np.concatenate(probs), owners, len(ids))

    score.chunked = 0
    return score


class PromptGuardDetector:
    name = NAME

    def __init__(self, scorer=None, batch_size=32):
        self.scorer = scorer if scorer is not None else hf_scorer()
        self.batch_size = batch_size

    def predict_proba(self, texts):
        texts = list(texts)
        batches = math.ceil(len(texts) / self.batch_size)
        return np.concatenate([self.scorer(texts[i * self.batch_size:(i + 1) * self.batch_size])
                               for i in range(batches)]) if texts else np.array([])


def native_summary(frames: dict, scores: dict, chunked: dict) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        recall, fpr = rates_at_threshold(scores[name], frame.label.to_numpy(), NATIVE_THRESHOLD)
        rows.append({"eval_set": name, "n": len(frame), "chunked": chunked[name],
                     f"recall_at_{NATIVE_THRESHOLD}": recall, f"fpr_at_{NATIVE_THRESHOLD}": fpr,
                     "exposure": "unlikely" if name in HELD_OUT else "unknown"})
    return pd.DataFrame(rows)


def score_file(detector, data, path) -> pd.DataFrame:
    """Score val and every eval set; write the score file and the native-threshold summary."""
    frames = {"val": data.val, **data.evals}
    scores, chunked = {}, {}
    for name, frame in frames.items():
        before = int(getattr(detector.scorer, "chunked", 0))
        scores[name] = detector.predict_proba(frame.text)
        chunked[name] = int(getattr(detector.scorer, "chunked", 0)) - before
    run_info = {"row": "prompt-guard-2", "name": NAME, "seed": None, "model": MODEL,
                "chunk_tokens": CHUNK_TOKENS, "chunk_stride": CHUNK_STRIDE, "chunked": chunked}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(scores_table(frames, scores, run_info, val_only=False), path)
    summary = native_summary(frames, scores, chunked)
    summary.to_csv(path.with_name(f"{path.stem}_native.csv"), index=False, float_format="%.3f")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    summary = score_file(PromptGuardDetector(batch_size=args.batch_size), load_data(args.data), args.out)
    print(f"wrote {args.out}\n{summary.to_string(index=False, float_format='%.3f')}")


if __name__ == "__main__":
    main()
