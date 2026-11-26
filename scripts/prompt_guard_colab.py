"""Score val, test, MCPTox and MSB zero-shot with Meta's Prompt Guard 2 (an off-the-shelf
injection detector; follow-up analysis outside the pre-registered protocol).

The model is gated: accept the Llama 4 Community License for
meta-llama/Llama-Prompt-Guard-2-86M on Hugging Face first, then run on a Colab T4:

    !git clone <repo> && cd CS7267-LLM-Injection-Defense && pip install uv
    !uv sync --group deberta && make data          # needs the team's DVC credentials
    import os; os.environ["HF_TOKEN"] = "<your token>"
    !PYTHONPATH=src/data:src/eval:src/models uv run python scripts/prompt_guard_colab.py

and download report/deberta_scores/prompt-guard-2.parquet into the same path locally. The file has
the B2 score-file layout, so the harness reads it with `--detector deberta --scores <file>`.
Texts longer than 512 tokens are truncated (Prompt Guard's context); the count is recorded in the
file's metadata.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset import DEFAULT_PATH
from deberta_run import scores_table
from harness import load_data

MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
OUT = Path(__file__).resolve().parents[1] / "report" / "deberta_scores" / "prompt-guard-2.parquet"
MAX_LENGTH = 512


def hf_scorer(model_name=MODEL, max_length=MAX_LENGTH, device=None):
    """Batch scorer: probability of the malicious class. Also returns how many texts it truncated
    through the `truncated` attribute."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token).eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def score(texts):
        lengths = [len(ids) for ids in tokenizer(list(texts))["input_ids"]]
        score.truncated += sum(n > max_length for n in lengths)
        batch = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1)
        return probs[:, -1].float().cpu().numpy()

    score.truncated = 0
    return score


class PromptGuardDetector:
    name = "prompt-guard-2"

    def __init__(self, scorer=None, batch_size=32):
        self.scorer = scorer if scorer is not None else hf_scorer()
        self.batch_size = batch_size

    def predict_proba(self, texts):
        texts = list(texts)
        batches = math.ceil(len(texts) / self.batch_size)
        return np.concatenate([self.scorer(texts[i * self.batch_size:(i + 1) * self.batch_size])
                               for i in range(batches)]) if texts else np.array([])


def score_file(detector, data, path) -> None:
    """Score val and every eval set, then write them with the truncation count seen while scoring."""
    frames = {"val": data.val, **data.evals}
    scores = {name: detector.predict_proba(frame.text) for name, frame in frames.items()}
    run_info = {"row": "prompt-guard-2", "seed": None, "model": MODEL, "max_length": MAX_LENGTH,
                "truncated": int(getattr(detector.scorer, "truncated", 0))}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(scores_table(frames, scores, run_info, val_only=False), path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    detector = PromptGuardDetector(batch_size=args.batch_size)
    score_file(detector, load_data(args.data), args.out)
    print(f"wrote {args.out}; truncated {detector.scorer.truncated} texts at {MAX_LENGTH} tokens")


if __name__ == "__main__":
    main()
