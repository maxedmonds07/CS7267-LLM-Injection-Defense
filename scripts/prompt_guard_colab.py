"""Score val, test, MCPTox and MSB zero-shot with Meta's Prompt Guard 2 (an off-the-shelf
injection detector; follow-up analysis outside the pre-registered protocol).

The model is gated: accept the Llama 4 Community License for
meta-llama/Llama-Prompt-Guard-2-86M on Hugging Face first, then run
notebooks/prompt_guard_colab.ipynb on a Colab T4. It reads HF_TOKEN and the R2 keys from Colab
Secrets, checks the model on three texts, and runs both passes (windowed, then --truncate).
Save the files it writes in report/offshelf_scores/:
- prompt-guard-2.parquet: the B2 score-file layout, so the harness reads it with
  `--detector deberta --scores <file>`; it reports as offshelf_prompt_guard_2, not as a DeBERTa run.
  It lives outside report/deberta_scores so the B2 verdict never picks it up.
- prompt-guard-2_native.csv: recall and FPR at Prompt Guard's own 0.5 cut, the way someone would
  deploy it off the shelf (the harness applies our 1% val-FPR threshold instead), with the number of
  chunked texts and the training-data exposure of each eval set.

Long texts are scored like B2: overlapping 512-token windows, text score = highest window, so an
injection late in a long BIPIA row is not truncated away. `--truncate` scores only the first window
and writes prompt-guard-2-truncated.parquet (named offshelf_prompt_guard_2_truncated), so a second
run measures what truncation would have cost.

Framing for the paper: Prompt Guard 2's malicious label covers text that explicitly tries to
override earlier instructions; MCPTox attacks are implicit tool procedure, so a low score there is
partly by design. In our run it still ranked MCPTox well within length bins (0.879) while its own cut
flagged 1.4% of attacks; the report's follow-up appendix gives the reading. Its training mix includes
open-source and curated injection data, so BIPIA overlap cannot be ruled out: the in-distribution
sets are marked exposure "unknown".
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


def windows(token_lists, wrap, chunk_tokens=CHUNK_TOKENS, stride=CHUNK_STRIDE, truncate=False):
    """(windows, owners, multi): every overlapping window of every text's content tokens, wrapped
    in special tokens by `wrap`; the index of the text each came from; and how many texts needed
    more than one window. With `truncate`, only each text's first window is kept."""
    out, owners, multi = [], [], 0
    for i, ids in enumerate(token_lists):
        spans = chunk_spans(len(ids), chunk_tokens, stride)
        multi += len(spans) > 1
        for start, end in spans[:1] if truncate else spans:
            out.append(wrap(list(ids[start:end])))
            owners.append(i)
    return out, owners, multi


def max_by_owner(values, owners, n) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, np.asarray(owners), np.asarray(values, dtype=float))
    return out


def max_pool_scores(token_lists, window_scorer, wrap, chunk_tokens=CHUNK_TOKENS, stride=CHUNK_STRIDE,
                    truncate=False, window_batch=WINDOW_BATCH):
    """(scores, multi): each text's highest window score. `window_scorer` maps a batch of wrapped
    windows to one malicious probability each; it is the only part that needs the model."""
    batch, owners, multi = windows(token_lists, wrap, chunk_tokens, stride, truncate)
    probs = [np.asarray(window_scorer(batch[i:i + window_batch]), dtype=float)
             for i in range(0, len(batch), window_batch)]
    return max_by_owner(np.concatenate(probs) if probs else [], owners, len(token_lists)), multi


def malicious_index(id2label: dict) -> int:
    """Column of the malicious class. Prompt Guard 2 is benign vs malicious; Prompt Guard 1 had
    three classes, where the last index meant jailbreak only, so refuse anything but two. A label
    named 'malicious' wins; otherwise column 1, as on the model card."""
    if len(id2label) != 2:
        raise ValueError(f"expected 2 classes (benign, malicious), got {id2label}")
    named = [int(i) for i, label in id2label.items() if str(label).lower() == "malicious"]
    return named[0] if named else 1


def default_device(cuda: bool, mps: bool) -> str:
    """NVIDIA GPU (Colab), then Apple GPU (a Mac), then CPU."""
    return "cuda" if cuda else "mps" if mps else "cpu"


def hf_scorer(model_name=MODEL, device=None, truncate=False):
    """Batch scorer: probability of the malicious class, max over each text's windows. Counts the
    texts that needed more than one window in the `chunked` attribute."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.model_max_length = 10**9  # we chunk ourselves; silence the length warning
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token).eval()
    positive = malicious_index(model.config.id2label)
    device = device or default_device(torch.cuda.is_available(), torch.backends.mps.is_available())
    model.to(device)
    pad = tokenizer.pad_token_id

    def window_scorer(batch):
        width = max(len(w) for w in batch)
        ids = torch.tensor([w + [pad] * (width - len(w)) for w in batch], device=device)
        mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w in batch], device=device)
        with torch.no_grad():
            probs = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)
        return probs[:, positive].float().cpu().numpy()

    def score(texts):
        ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        scores, multi = max_pool_scores(ids, window_scorer, tokenizer.build_inputs_with_special_tokens,
                                        truncate=truncate)
        score.chunked += multi
        return scores

    score.chunked = 0
    score.truncate = truncate
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


def native_summary(frames: dict, scores: dict, chunked: dict, truncate: bool = False) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        recall, fpr = rates_at_threshold(scores[name], frame.label.to_numpy(), NATIVE_THRESHOLD)
        rows.append({"eval_set": name, "n": len(frame), "chunked": chunked[name],
                     f"recall_at_{NATIVE_THRESHOLD}": recall, f"fpr_at_{NATIVE_THRESHOLD}": fpr,
                     "exposure": "unlikely" if name in HELD_OUT else "unknown", "truncate": truncate})
    return pd.DataFrame(rows)


def score_file(detector, data, path) -> pd.DataFrame:
    """Score val and every eval set; write the score file and the native-threshold summary."""
    frames = {"val": data.val, **data.evals}
    scores, chunked = {}, {}
    for name, frame in frames.items():
        before = int(getattr(detector.scorer, "chunked", 0))
        scores[name] = detector.predict_proba(frame.text)
        chunked[name] = int(getattr(detector.scorer, "chunked", 0)) - before
    truncate = bool(getattr(detector.scorer, "truncate", False))
    run_info = {"row": "prompt-guard-2", "name": f"{NAME}_truncated" if truncate else NAME, "seed": None,
                "model": MODEL, "chunk_tokens": CHUNK_TOKENS, "chunk_stride": CHUNK_STRIDE,
                "truncate": truncate, "chunked": chunked}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(scores_table(frames, scores, run_info, val_only=False), path)
    summary = native_summary(frames, scores, chunked, truncate)
    summary.to_csv(path.with_name(f"{path.stem}_native.csv"), index=False, float_format="%.3f")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=None, help=f"default {OUT.name}, or its -truncated twin")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the fastest available)")
    ap.add_argument("--truncate", action="store_true",
                    help="score only each text's first 512-token window, to measure what truncation costs")
    args = ap.parse_args(argv)
    out = args.out or str(OUT.with_stem(f"{OUT.stem}-truncated") if args.truncate else OUT)
    detector = PromptGuardDetector(scorer=hf_scorer(device=args.device, truncate=args.truncate), batch_size=args.batch_size)
    summary = score_file(detector, load_data(args.data), out)
    print(f"wrote {out}\n{summary.to_string(index=False, float_format='%.3f')}")


if __name__ == "__main__":
    main()
