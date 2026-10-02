"""Length separability: how much of a label text length alone could explain, in either direction."""

import pytest

from metrics import length_separability


def test_longer_adversarial_is_fully_separable():
    assert length_separability(["a", "bb", "cccc", "ddddd"], [0, 0, 1, 1]) == 1.0


def test_shorter_adversarial_counts_the_same():
    # PoisonedRAG's direction: poisoned passages are shorter than retrieved ones.
    assert length_separability(["a", "bb", "cccc", "ddddd"], [1, 1, 0, 0]) == 1.0


def test_equal_lengths_carry_no_signal():
    assert length_separability(["ab", "cd", "ef", "gh"], [0, 1, 0, 1]) == 0.5


def test_single_class_is_undefined():
    # MSB tool_output has adversarial rows only.
    with pytest.raises(ValueError, match="both labels"):
        length_separability(["a", "bb"], [1, 1])
