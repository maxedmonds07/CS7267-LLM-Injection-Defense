"""The holdout guard: fitting may never see LOSO or transfer holdouts by accident."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset import HoldoutError, load


@pytest.fixture
def path(tmp_path):
    rows = [
        {"id": "p1", "source": "poisonedrag", "split": "train", "role": "train_pool"},
        {"id": "p2", "source": "poisonedrag", "split": "test", "role": "train_pool"},
        {"id": "m1", "source": "mcptox", "split": "train", "role": "loso_holdout"},
        {"id": "m2", "source": "mcptox", "split": "val", "role": "loso_holdout"},
        {"id": "m3", "source": "mcptox", "split": "test", "role": "loso_holdout"},
    ]
    target = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target)
    return target


def test_fit_refuses_holdout_rows(path):
    with pytest.raises(HoldoutError, match="loso_holdout"):
        load("fit", path=path)


def test_fit_on_train_pool(path):
    table = load("fit", roles=["train_pool"], splits=["train"], path=path)
    assert table["id"].to_pylist() == ["p1"]


def test_fit_escape_hatch(path):
    assert load("fit", allow_holdout=True, path=path).num_rows == 5


def test_eval_reads_every_split_of_a_holdout(path):
    assert load("eval", sources=["mcptox"], path=path)["id"].to_pylist() == ["m1", "m2", "m3"]


def test_unknown_purpose(path):
    with pytest.raises(ValueError, match="purpose"):
        load("train", path=path)
