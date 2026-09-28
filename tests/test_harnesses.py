"""Checks that the evaluation harnesses in config.yaml are installed at the pinned versions."""

from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HARNESSES = yaml.safe_load((ROOT / "config.yaml").read_text())["harnesses"]


@pytest.mark.parametrize("name", HARNESSES)
def test_installed_version_matches_config(name):
    spec = HARNESSES[name]
    assert version(spec["package"]) == spec["version"], "run `make install`"


def test_agentdojo_suites_load():
    from agentdojo.task_suite.load_suites import get_suites

    suites = get_suites("v1")
    assert {"workspace", "travel", "banking", "slack"} <= suites.keys()
    assert all(suite.user_tasks and suite.injection_tasks for suite in suites.values())
