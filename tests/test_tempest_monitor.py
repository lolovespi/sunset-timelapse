"""Tests for Tempest monitor armed-flag gating."""

from datetime import datetime

import pytest

from tempest_monitor import TempestMonitor, StormConditions


def test_armed_flag_defaults_to_false():
    monitor = TempestMonitor()
    assert monitor.armed is False


def test_arm_and_disarm():
    monitor = TempestMonitor()
    monitor.arm()
    assert monitor.armed is True
    monitor.disarm()
    assert monitor.armed is False


def test_callback_not_fired_when_disarmed(monkeypatch):
    monitor = TempestMonitor()
    fired = []
    monitor.register_storm_callback(lambda c: fired.append(c))
    monitor.disarm()

    # Synthesize storm conditions and try to fire
    conditions = StormConditions(
        storm_detected=True, confidence=0.5,
        trigger_reasons=['test'], lightning_active=True,
    )
    monitor._fire_storm_callbacks(conditions)
    assert fired == []


def test_callback_fired_when_armed():
    monitor = TempestMonitor()
    fired = []
    monitor.register_storm_callback(lambda c: fired.append(c))
    monitor.arm()

    conditions = StormConditions(
        storm_detected=True, confidence=0.5,
        trigger_reasons=['test'], lightning_active=True,
    )
    monitor._fire_storm_callbacks(conditions)
    assert len(fired) == 1
