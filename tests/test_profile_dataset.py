"""Dataset profile: shape, nulls and class balance for the Phase 2 data card."""

import pandas as pd

from profile_dataset import profile, to_markdown


def row(id, label, text="some text", source="bipia", **overrides):
    attack = {"attack_kind": "instruction", "attack_objective": "task_hijack", "attack_family": "x"}
    base = {
        "id": id, "text": text, "label": label,
        **(attack if label else dict.fromkeys(attack)),
        "source": source, "subset": "email", "attack_surface": "tool_output",
        "group": id, "split": "train", "role": "train_pool",
    }
    return base | overrides


def frame(*rows):
    return pd.DataFrame(rows).astype({"label": "int8"})


def test_shape():
    result = profile(frame(row("a", 1), row("b", 0)))
    assert result["shape"] == {"rows": 2, "columns": 12}


def test_benign_attack_fields_are_expected_nulls():
    result = profile(frame(row("a", 1), row("b", 0)))
    assert result["nulls"]["attack_kind"] == {"total": 1, "benign": 1, "adversarial": 0}
    assert result["unexpected_nulls"] == {}


def test_unexpected_nulls_are_reported_per_column():
    result = profile(frame(row("a", 1, attack_objective=None), row("b", 0, text=None)))
    assert result["unexpected_nulls"] == {"attack_objective": 1, "text": 1}


def test_empty_text_counts_as_missing():
    assert profile(frame(row("a", 1, text="   "), row("b", 0)))["unexpected_nulls"] == {"text": 1}


def test_class_balance_overall_and_by_source():
    result = profile(frame(row("a", 1), row("b", 0), row("c", 0), row("d", 1, source="msb")))
    assert result["class_balance"]["overall"] == {"adversarial": 2, "benign": 2, "adversarial_share": 0.5}
    assert result["class_balance"]["by_source"]["bipia"]["adversarial_share"] == 0.333
    assert result["class_balance"]["by_source"]["msb"] == {"adversarial": 1, "benign": 0, "adversarial_share": 1.0}


def test_duplicates():
    result = profile(frame(row("a", 1, text="same"), row("a", 1, text="same"), row("b", 0)))
    assert result["duplicates"] == {"ids": 1, "texts": 1}


def test_text_length_by_source_and_label():
    result = profile(frame(row("a", 1, text="abcd"), row("b", 0, text="ab"), row("c", 0, text="abcdef")))
    assert result["text_length"]["bipia"]["benign"] == {"min": 2, "median": 4.0, "max": 6}
    assert result["text_length"]["bipia"]["adversarial"] == {"min": 4, "median": 4.0, "max": 4}


def test_markdown_report_states_the_findings():
    report = to_markdown(profile(frame(row("a", 1), row("b", 0, text=None))))
    assert "2 rows × 12 columns" in report
    assert "| text | 1 |" in report
    assert "Unexpected nulls: text (1)" in report
