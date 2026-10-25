"""B0: frozen language-model perplexity as a prompt-injection anomaly score.

High perplexity is treated as more suspicious. The language model is never trained or
fine-tuned on the attack labels. The harness calibrates its decision threshold using
only benign validation examples, exactly as it does for the other detectors.

Install the optional dependencies with ``uv sync --group perplexity`` before running.
"""

import math
from tqdm.auto import tqdm
import numpy as np


def _windows(total_tokens: int, max_context: int, stride: int):
    """Yield (start, end, first_target) for disjoint scored spans with overlapping context.

    Token zero is a BOS/EOS prefix. Every subsequent token is scored exactly once,
    while each window can reuse earlier tokens as conditioning context.
    """
    if max_context < 2 or not 1 <= stride < max_context:
        raise ValueError("max_context must be >= 2 and stride must be in [1, max_context)")

    for first_target in range(1, total_tokens, stride):
        end = min(first_target + stride, total_tokens)
        start = max(0, end - max_context)
        yield start, end, first_target


class PerplexityDetector:
    """Score text using a frozen causal LM: larger perplexity means more suspicious.

    ``predict_proba`` returns a *monotonic anomaly score* in (0, 1), not a calibrated
    attack probability. This matches the harness's detector interface; the ordering of
    examples (and hence AUROC) is identical to using raw perplexity.
    """

    name = "perplexity_b0"

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        max_context: int = 512,
        stride: int = 256,
        device: str | None = None,
    ):
        if not 1 <= stride < max_context:
            raise ValueError("stride must be positive and smaller than max_context")
        self.model_name = model_name
        self.max_context = max_context
        self.stride = stride
        self.device = device
        self._model = None
        self._tokenizer = None

    def fit(self, texts, labels):
        """No-op: B0 uses no labels and does not train the pretrained language model."""
        return self

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PerplexityDetector needs torch and transformers. "
                "Run: uv sync --group perplexity"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        model_context = getattr(model.config, "max_position_embeddings", None)
        if model_context is None:
            model_context = getattr(model.config, "n_positions", None)
        if model_context is not None and self.max_context > model_context:
            raise ValueError(
                f"max_context={self.max_context} exceeds model limit {model_context}"
            )
        if tokenizer.bos_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("The language-model tokenizer needs a BOS or EOS token")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._tokenizer = tokenizer
        self._model = model.to(device).eval()

    def log_perplexity(self, text: str) -> float:
        """Mean token negative log-likelihood, i.e. log(perplexity), across the text.

        The sliding window scores *all* tokens, including those beyond the model's
        context limit. Earlier tokens in overlapping windows contribute context but
        do not contribute twice to the loss.
        """
        self._ensure_model()
        import torch

        tokenizer = self._tokenizer
        prefix = tokenizer.bos_token_id
        if prefix is None:
            prefix = tokenizer.eos_token_id
        ids = tokenizer.encode(text if text else " ", add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(" ", add_special_tokens=False)
        if not ids:
            raise ValueError("The tokenizer produced no tokens for the input")
        ids = [prefix, *ids]

        nll_sum = 0.0
        token_count = 0
        for start, end, first_target in _windows(len(ids), self.max_context, self.stride):
            input_ids = torch.tensor([ids[start:end]], dtype=torch.long, device=self.device)
            labels = input_ids.clone()
            labels[:, : first_target - start] = -100
            with torch.inference_mode():
                loss = self._model(input_ids=input_ids, labels=labels).loss
            n_targets = end - first_target
            nll_sum += float(loss.item()) * n_targets
            token_count += n_targets

        result = nll_sum / token_count
        if not math.isfinite(result):
            raise ValueError("Language model returned a non-finite perplexity score")
        return result

    def perplexities(self, texts) -> np.ndarray:
        """Return conventional raw perplexities for inspection or reporting."""
        return np.exp([self.log_perplexity(str(text)) for text in texts])

    def predict_proba(self, texts) -> np.ndarray:
        """Return monotonically scaled perplexity scores (not calibrated probabilities)."""
        log_ppl = np.array([self.log_perplexity(str(text)) 
                            for text in tqdm(texts, desc="Calculating Perplexity", unit="text", ascii=True, leave=False)], dtype=float)
        
        return 1.0 / (1.0 + np.exp(-log_ppl))
