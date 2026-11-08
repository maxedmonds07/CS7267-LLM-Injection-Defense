"""The Colab notebook: committed clean, keys only from Colab Secrets, logic only in src/."""

import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks" / "deberta_colab.ipynb"


def _cells():
    return json.loads(NB.read_text())["cells"]


def _source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _all_source():
    return "\n".join(_source(c) for c in _cells())


def test_notebook_is_committed_without_outputs():
    cells = _cells()
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in cells if c["cell_type"] == "code")


def test_notebook_reads_keys_only_from_colab_secrets():
    source = _all_source()
    assert "userdata.get('R2_ACCESS_KEY_ID')" in source
    assert "userdata.get('R2_SECRET_ACCESS_KEY')" in source
    assert not re.search(r"\b[0-9a-f]{32,}\b", source)          # no pasted key
    assert "config.local" not in source
    assert not re.search(r"print\([^)]*(KEY|SECRET)", source)


def test_keys_never_reach_a_shell_command_line():
    # `!cmd {VAR}` interpolates Python values into the command line; keys must only travel
    # through the environment.
    for line in _all_source().splitlines():
        if line.lstrip().startswith(("!", "%%bash", "%sh")):
            assert not re.search(r"KEY|SECRET|userdata", line), line


NOT_DEBERTA_RUN = {"--depth", "--branch", "--oneline", "--frozen", "--group"}   # git and uv flags


def _cli_flags():
    """Every --flag the deberta_run parsers define, read from the parser itself."""
    import argparse

    import deberta_run

    flags = set()
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        stack = [self]
        while stack:
            parser = stack.pop()
            for action in parser._actions:
                flags.update(o for o in action.option_strings if o.startswith("--"))
                if isinstance(action, argparse._SubParsersAction):
                    stack.extend(action.choices.values())
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            deberta_run.main([])
    finally:
        argparse.ArgumentParser.parse_args = original
    return flags


def test_cli_flags_the_notebook_uses_exist():
    used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", _all_source())) - NOT_DEBERTA_RUN
    assert used, "found no deberta_run flags in the notebook"
    assert used <= _cli_flags(), used - _cli_flags()


def test_checkpoints_and_outputs_go_to_drive():
    source = _all_source()
    assert "/content/drive/MyDrive/cs7267-deberta/" in source
    assert "--checkpoint-dir" in source and "--model-out" in source
