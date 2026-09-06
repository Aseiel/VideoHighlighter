"""The AMD simulator must not leak into anything it ran beside.

`tools/simulate_directml.py` fakes a machine by patching global state: it puts
a stand-in package in `sys.modules`, replaces `torch.Tensor.to`, and switches
off this box's real accelerators. All of that is fine inside its own process
and catastrophic if it survives the block — a leaked `Tensor.to` would silently
redirect devices for every test that ran afterwards, and the failures would
appear nowhere near the cause.

So the contract tested here is narrow and total: after `simulate()` returns,
the process is exactly as it was.
"""

from __future__ import annotations

import sys

import pytest

from modules import directml_device as dml

simulate_directml = pytest.importorskip("tools.simulate_directml")


@pytest.fixture(autouse=True)
def clean_module_state():
    dml.set_mode(None)
    yield
    # set_mode(None) nulls the probe cache. Deliberately NOT probe(refresh=True):
    # fixture teardown runs before monkeypatch unwinds, so re-probing here would
    # cache the *fake* AMD box and hand it to every later test in the session.
    dml.set_mode(None)


def _scenario(key):
    return simulate_directml.BY_KEY[key]


def test_every_scenario_is_reachable_by_name():
    """The argparse `choices` and the table are built from the same dict, so
    this only fails if a scenario is added without a key."""
    assert simulate_directml.BY_KEY
    for key, scenario in simulate_directml.BY_KEY.items():
        assert scenario.key == key
        assert scenario.summary


def test_the_fake_package_is_removed_afterwards():
    assert "torch_directml" not in sys.modules
    with simulate_directml.simulate(_scenario("rx570"), hide_real=False,
                                    real_tensors=False) as probe:
        assert probe.available
        assert sys.modules["torch_directml"].is_available() is True
    assert "torch_directml" not in sys.modules


def test_torch_is_left_exactly_as_it_was():
    """The one that would poison the rest of the suite."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    before = (torch.Tensor.to, nn.Module.to, torch.empty, torch.randn)
    with simulate_directml.simulate(_scenario("rx570"), hide_real=True,
                                    real_tensors=True):
        assert torch.Tensor.to is not before[0], "the shim did not install"
    assert (torch.Tensor.to, nn.Module.to, torch.empty, torch.randn) == before


def test_the_real_accelerators_come_back():
    torch = pytest.importorskip("torch")
    before = torch.cuda.is_available
    with simulate_directml.simulate(_scenario("rx570"), hide_real=True,
                                    real_tensors=False):
        assert torch.cuda.is_available() is False
    assert torch.cuda.is_available is before


def test_the_mode_environment_variable_is_restored(monkeypatch):
    monkeypatch.setenv(dml.MODE_ENV, "off")
    with simulate_directml.simulate(_scenario("nvidia-forced"), hide_real=False,
                                    real_tensors=False):
        assert dml.mode() == dml.MODE_FORCE
    import os
    assert os.environ[dml.MODE_ENV] == "off"


def test_the_probe_cache_does_not_survive_a_scenario():
    """Two scenarios in one process must not see each other's answer — the
    failure that would make `--all` report the first machine nine times."""
    with simulate_directml.simulate(_scenario("rx570"), hide_real=False,
                                    real_tensors=False) as probe:
        assert probe.available is True
    with simulate_directml.simulate(_scenario("missing"), hide_real=False,
                                    real_tensors=False) as probe:
        assert probe.available is False


def test_a_software_only_machine_is_refused_in_simulation():
    """Mirrors the module-level rule, through the tool that found it."""
    with simulate_directml.simulate(_scenario("software-only"), hide_real=False,
                                    real_tensors=False) as probe:
        assert probe.available is False
        assert "software renderer" in probe.reason
