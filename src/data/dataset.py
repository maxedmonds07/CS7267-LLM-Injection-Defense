"""Read data/processed/dataset.parquet, refusing held-out sources when fitting.

  load("fit", roles=["train_pool"], splits=["train"])  # training data
  load("eval", sources=["mcptox"])                     # LOSO evaluation: every MCPTox row
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data/processed/dataset.parquet"
HOLDOUT_ROLES = frozenset({"loso_holdout", "transfer_holdout"})
PURPOSES = ("fit", "eval")


class HoldoutError(ValueError):
    """Fitting data would include a held-out source."""


def load(purpose, *, splits=None, roles=None, sources=None, allow_holdout=False, path=DEFAULT_PATH) -> pa.Table:
    """Rows matching every given filter. For purpose="fit", held-out roles raise HoldoutError
    unless allow_holdout=True; purpose="eval" reads anything."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, not {purpose!r}")
    filters = [
        (column, "in", list(values))
        for column, values in (("split", splits), ("role", roles), ("source", sources))
        if values
    ]
    table = pq.read_table(path, filters=filters or None)
    if purpose == "fit" and not allow_holdout:
        leaked = sorted(HOLDOUT_ROLES & set(table["role"].to_pylist()))
        if leaked:
            raise HoldoutError(
                f"fitting data includes held-out roles {leaked}; filter with roles=['train_pool'] "
                "or pass allow_holdout=True deliberately"
            )
    return table
