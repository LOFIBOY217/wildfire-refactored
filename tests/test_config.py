"""
Unit tests for src/config.py — YAML config loading, env-var expansion,
deep merge, and path resolution.

Run: python -m pytest tests/test_config.py -v
"""
import os

import pytest

import src.config as cfgmod
from src.config import (
    _expand_env_vars,
    _expand_recursive,
    _deep_merge,
    load_config,
    get_path,
    PROJECT_ROOT,
)


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """load_config caches on a module global; reset around every test."""
    cfgmod._cached_config = None
    cfgmod._cached_path = None
    yield
    cfgmod._cached_config = None
    cfgmod._cached_path = None


# ---------- _expand_env_vars ---------- #

def test_expand_env_var_set(monkeypatch):
    monkeypatch.setenv("WF_TEST_VAR", "hello")
    assert _expand_env_vars("${WF_TEST_VAR}") == "hello"


def test_expand_env_var_unset(monkeypatch):
    monkeypatch.delenv("WF_MISSING_VAR", raising=False)
    assert _expand_env_vars("${WF_MISSING_VAR}") == ""


def test_expand_env_var_embedded(monkeypatch):
    monkeypatch.setenv("WF_KEY", "abc")
    assert _expand_env_vars("prefix-${WF_KEY}-suffix") == "prefix-abc-suffix"


def test_expand_env_var_non_string_passthrough():
    assert _expand_env_vars(42) == 42
    assert _expand_env_vars(None) is None


def test_expand_no_placeholder_unchanged():
    assert _expand_env_vars("plain string") == "plain string"


# ---------- _expand_recursive ---------- #

def test_expand_recursive_nested(monkeypatch):
    monkeypatch.setenv("WF_X", "v")
    obj = {"a": "${WF_X}", "b": {"c": "${WF_X}", "d": 1}, "e": ["${WF_X}", 2]}
    assert _expand_recursive(obj) == {
        "a": "v", "b": {"c": "v", "d": 1}, "e": ["v", 2],
    }


# ---------- _deep_merge ---------- #

def test_deep_merge_flat():
    assert _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"paths": {"x": 1, "y": 2}, "k": 0}
    override = {"paths": {"y": 20, "z": 30}}
    assert _deep_merge(base, override) == {
        "paths": {"x": 1, "y": 20, "z": 30}, "k": 0,
    }


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    _deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_deep_merge_non_dict_override_replaces():
    assert _deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}


# ---------- load_config / get_path ---------- #

def test_load_config_default_has_paths():
    cfg = load_config()
    assert "paths" in cfg
    assert "fwi_dir" in cfg["paths"]


def test_get_path_relative_resolves_under_root():
    cfg = load_config()
    p = get_path(cfg, "fwi_dir")
    assert os.path.isabs(p)
    assert p.startswith(str(PROJECT_ROOT))


def test_get_path_absolute_unchanged():
    cfg = {"paths": {"abs": "/tmp/somewhere"}}
    assert get_path(cfg, "abs") == "/tmp/somewhere"


def test_load_config_override_merges_and_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("WF_OVERRIDE_TEST", "seekrit")
    override = tmp_path / "over.yaml"
    override.write_text(
        "paths:\n"
        "  fwi_dir: custom/fwi\n"
        "credentials:\n"
        "  token: ${WF_OVERRIDE_TEST}\n"
    )
    cfg = load_config(str(override))
    # Override wins for the key it sets ...
    assert cfg["paths"]["fwi_dir"] == "custom/fwi"
    # ... env var is expanded ...
    assert cfg["credentials"]["token"] == "seekrit"
    # ... and unrelated default keys survive the deep merge.
    assert "checkpoint_dir" in cfg["paths"]
