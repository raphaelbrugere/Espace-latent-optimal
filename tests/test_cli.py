"""Commande lo (latentopt/cli) : aiguillage vers les modules, aide, erreurs."""

import importlib
import sys

import pytest

from latentopt import cli


@pytest.mark.parametrize("words", list(cli.COMMANDS))
def test_every_command_points_to_an_existing_function(words):
    module, func, family, _ = cli.COMMANDS[words]
    assert callable(getattr(importlib.import_module(module), func))
    assert family in cli.FAMILIES


def test_resolve_prefers_the_longest_command():
    assert cli.resolve(["hub", "push", "x", "--dry-run"]) == (("hub", "push"), ["x", "--dry-run"])
    assert cli.resolve(["train", "task=pets_seg"]) == (("train",), ["task=pets_seg"])
    with pytest.raises(KeyError):
        cli.resolve(["hub"])


def test_help_lists_every_command_by_family(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(f"lo {' '.join(w)}" in out for w in cli.COMMANDS)
    assert out.index("Lancer des runs") < out.index("lo pretrain") < out.index("Tracabilite")


def test_unknown_command_suggests_the_group(capsys):
    assert cli.main(["hub", "nope"]) == 2
    assert "lo hub push, lo hub pull" in capsys.readouterr().err


def test_argparse_commands_receive_their_arguments(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "fake_cmd", type(sys)("fake_cmd"))
    sys.modules["fake_cmd"].main = lambda argv: seen.setdefault("argv", argv) and 3
    monkeypatch.setitem(cli.COMMANDS, ("fake",), ("fake_cmd", "main", "store", ""))
    assert cli.main(["fake", "a", "--b"]) == 3 and seen["argv"] == ["a", "--b"]


def test_hydra_commands_read_their_overrides_from_sys_argv(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "fake_run", type(sys)("fake_run"))
    sys.modules["fake_run"].main = lambda: seen.setdefault("argv", list(sys.argv))
    monkeypatch.setitem(cli.COMMANDS, ("fakerun",), ("fake_run", "main", "run", ""))
    monkeypatch.setattr(sys, "argv", ["lo"])
    assert cli.main(["fakerun", "task=x", "-m"]) == 0
    assert seen["argv"] == ["lo fakerun", "task=x", "-m"]
