"""Tests for drawdown.py."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from evt_ppo import drawdown as dd


def test_underwater_curve_basic():
    values = np.array([100.0, 110.0, 120.0, 90.0, 100.0, 130.0, 115.0])
    uw = dd.underwater_curve(values)
    expected = np.array([0.0, 0.0, 0.0, 30/120, 20/120, 0.0, 15/130])
    print(f"  underwater={uw}")
    assert np.allclose(uw, expected, atol=1e-9)


def test_maximum_drawdown_known_value():
    values = np.array([100.0, 110.0, 120.0, 90.0, 100.0, 130.0, 115.0])
    mdd = dd.maximum_drawdown(values)
    print(f"  MDD={mdd:.4f} (expected {30/120:.4f})")
    assert abs(mdd - 30/120) < 1e-9


def test_monotonic_increasing_no_drawdown():
    values = np.linspace(100, 200, 50)
    assert dd.maximum_drawdown(values) == 0.0
    assert dd.current_drawdown(values) == 0.0


def test_cdar_in_range():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.012, size=500)
    values = 100.0 * np.cumprod(1.0 + returns)
    values = np.concatenate([[100.0], values])
    cdar = dd.conditional_drawdown_at_risk(values, alpha=0.95)
    mdd = dd.maximum_drawdown(values)
    print(f"  MDD={mdd:.4f}, CDaR_95={cdar:.4f}")
    assert 0.0 <= cdar <= mdd + 1e-9


def test_calmar_zero_when_no_drawdown():
    values = np.linspace(100, 110, 252)
    assert dd.calmar_ratio(values) == 0.0


def test_calmar_positive_for_winning_strategy():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0008, 0.010, size=252)
    values = 100.0 * np.cumprod(1.0 + returns)
    values = np.concatenate([[100.0], values])
    cr = dd.calmar_ratio(values)
    print(f"  Calmar={cr:.4f}")
    assert np.isfinite(cr)


def test_drawdown_increment_non_negative():
    assert dd.drawdown_increment(0.05, 0.10) == 0.05
    assert dd.drawdown_increment(0.10, 0.05) == 0.0
    assert dd.drawdown_increment(0.0, 0.0) == 0.0


if __name__ == "__main__":
    tests = [
        ("Underwater curve", test_underwater_curve_basic),
        ("MDD known value", test_maximum_drawdown_known_value),
        ("Monotonic no drawdown", test_monotonic_increasing_no_drawdown),
        ("CDaR in range", test_cdar_in_range),
        ("Calmar zero no DD", test_calmar_zero_when_no_drawdown),
        ("Calmar finite", test_calmar_positive_for_winning_strategy),
        ("Drawdown increment", test_drawdown_increment_non_negative),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            print("  PASSED")
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERRORED: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{'='*60}")
    print(f"Result: {len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)