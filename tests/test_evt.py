"""Sanity tests for the EVT module."""
import sys
from pathlib import Path

# Make `src/` importable regardless of where this file lives.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from scipy import stats
from evt_ppo import evt


def test_gpd_recovers_known_shape():
    """If we sample from a GPD with known xi, MLE should recover xi roughly."""
    rng = np.random.default_rng(0)
    true_shape = 0.3
    true_scale = 1.0
    n = 5000
    excesses = stats.genpareto.rvs(true_shape, loc=0, scale=true_scale,
                                    size=n, random_state=rng)
    threshold = 0.0
    losses = excesses + threshold
    fit = evt.fit_gpd(losses, threshold=threshold)
    print(f"  true xi={true_shape:.3f}, est xi={fit.shape:.3f}")
    print(f"  true sigma={true_scale:.3f}, est sigma={fit.scale:.3f}")
    assert abs(fit.shape - true_shape) < 0.1, "xi not recovered"
    assert abs(fit.scale - true_scale) < 0.2, "sigma not recovered"
    assert fit.converged


def test_gpd_var_cvar_ordering():
    """For a heavy-tailed fit, CVaR_alpha must exceed VaR_alpha for alpha large."""
    rng = np.random.default_rng(1)
    losses = stats.genpareto.rvs(0.2, loc=0, scale=1.0, size=2000,
                                  random_state=rng)
    fit = evt.fit_gpd(losses, quantile=0.90)
    var99 = evt.gpd_var(fit, 0.99)
    cvar99 = evt.gpd_cvar(fit, 0.99)
    var95 = evt.gpd_var(fit, 0.95)
    print(f"  VaR_95={var95:.3f}, VaR_99={var99:.3f}, CVaR_99={cvar99:.3f}")
    assert var99 > var95, "VaR must be monotone in alpha"
    assert cvar99 > var99, "CVaR must exceed VaR"
    assert cvar99 < float('inf')


def test_auto_threshold_picks_reasonable_value():
    """Automated threshold selection should return a finite, in-range threshold."""
    rng = np.random.default_rng(2)
    bulk = rng.normal(0.0, 0.01, size=900)
    tail = stats.genpareto.rvs(0.3, loc=0.02, scale=0.005, size=100,
                                random_state=rng)
    losses = np.concatenate([bulk, tail])
    threshold, fit = evt.select_threshold_auto(losses)
    print(f"  selected threshold={threshold:.4f}, xi={fit.shape:.3f}, "
          f"n_excesses={fit.n_excesses}")
    assert 0.0 < threshold < 0.05
    assert fit.n_excesses >= 30
    assert fit.converged


def test_evt_state_features_shape():
    """The 5-dim EVT state vector must be finite and well-shaped."""
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0005, 0.012, size=250)
    feats = evt.evt_state_features(returns)
    print(f"  feats={feats}")
    assert feats.shape == (5,)
    assert np.all(np.isfinite(feats))
    assert feats[1] > 0   # scale > 0
    assert feats[3] >= feats[2]  # CVaR >= VaR


def test_block_maxima_gev():
    """GEV fit on block maxima of heavy-tailed data should converge."""
    rng = np.random.default_rng(4)
    losses = stats.t.rvs(df=4, size=1000, random_state=rng)
    losses = np.abs(losses)
    fit = evt.fit_gev_block_maxima(losses, block_size=5)
    print(f"  GEV: loc={fit.location:.3f}, scale={fit.scale:.3f}, "
          f"shape={fit.shape:.3f}, n_blocks={fit.n_blocks}")
    assert fit.converged
    assert fit.scale > 0
    assert fit.n_blocks == 200


def test_degenerate_inputs_handled():
    """Tiny samples and constant data should not crash."""
    fit = evt.fit_gpd(np.array([0.01, 0.01, 0.01]))
    print(f"  degenerate fit: xi={fit.shape}, scale={fit.scale}, "
          f"converged={fit.converged}")
    assert not fit.converged
    assert np.isfinite(fit.scale)
    assert fit.scale > 0

    feats = evt.evt_state_features(np.array([0.0] * 10))
    print(f"  degenerate feats={feats}")
    assert np.all(np.isfinite(feats))


if __name__ == "__main__":
    tests = [
        ("GPD recovers known shape", test_gpd_recovers_known_shape),
        ("GPD VaR/CVaR ordering", test_gpd_var_cvar_ordering),
        ("Automated threshold selection", test_auto_threshold_picks_reasonable_value),
        ("EVT state feature vector", test_evt_state_features_shape),
        ("Block maxima GEV", test_block_maxima_gev),
        ("Degenerate inputs", test_degenerate_inputs_handled),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            print(f"  PASSED")
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERRORED: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{'='*60}")
    print(f"Result: {len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)