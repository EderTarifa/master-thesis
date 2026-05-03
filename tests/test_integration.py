"""Integration tests for metrics, benchmarks, statistics."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from evt_ppo import metrics as M
from evt_ppo import benchmarks as B
from evt_ppo import statistics as S
from evt_ppo import data as D


# ---- metrics ----

def test_metrics_on_known_series():
    values = np.linspace(100, 200, 253)
    m = M.all_metrics(values)
    print(f"  CAGR={m['cagr']:.4f}  MDD={m['mdd']:.6f}  Sharpe={m['sharpe']:.2f}")
    assert abs(m["cagr"] - 1.0) < 0.05
    assert m["mdd"] < 1e-6
    assert m["sharpe"] > 0


def test_metrics_handle_constant():
    values = np.full(100, 100.0)
    m = M.all_metrics(values)
    assert m["cagr"] == 0.0
    assert m["mdd"] == 0.0
    assert m["sharpe"] == 0.0


def test_metrics_drawdown_consistent_with_underwater():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.012, 500)
    vals = 100 * np.cumprod(1.0 + rets)
    m = M.all_metrics(vals)
    assert 0.0 <= m["mdd"] <= 1.0
    assert 0.0 <= m["cdar_95"] <= m["mdd"] + 1e-9
    assert 0.0 <= m["dd_q95"] <= m["mdd"] + 1e-9


# ---- benchmarks ----

def test_benchmarks_min_variance_lowers_vol():
    rng = np.random.default_rng(0)
    train = np.column_stack([
        rng.normal(0.0005, 0.02, 1000),
        rng.normal(0.0005, 0.005, 1000),
    ])
    cov = np.cov(train.T) * 252
    w_minvar = B.minimum_variance(cov)
    print(f"  min-var weights: {w_minvar}")
    assert w_minvar[1] > w_minvar[0]
    assert abs(w_minvar.sum() - 1.0) < 1e-6
    assert (w_minvar >= 0).all()


def test_run_benchmarks_returns_all_strategies():
    df = D.synthetic_market(n_assets=8, n_days=1500, seed=1)
    log_ret = np.log(df / df.shift(1)).dropna().values
    train, test = log_ret[:1000], log_ret[1000:]
    res = B.run_benchmarks(train, test, rebalance_freq=21, transaction_cost=0.001)
    assert set(res.keys()) == {"equal_weight", "min_variance", "mean_variance",
                                "buy_and_hold_eq"}
    for name, br in res.items():
        m = M.all_metrics(br.values, br.weights_history)
        print(f"  {name:<20}  CAGR={m['cagr']:+.3f}  MDD={m['mdd']:.3f}  Sharpe={m['sharpe']:+.2f}")
        assert np.isfinite(m["cagr"])
        assert np.isfinite(m["mdd"])
        assert br.weights_history.shape[0] == br.values.size


# ---- statistics ----

def test_paired_t_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.10, 0.30, size=60)
    evt = base - rng.uniform(0.01, 0.03, size=60)
    res = S.paired_one_sided_t(base, evt, alpha=0.05)
    print(f"  t-stat={res.statistic:.2f}  p={res.pvalue:.6f}  reject={res.reject_h0}")
    assert res.reject_h0
    assert res.mean_diff < 0


def test_paired_t_no_effect_does_not_reject():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.10, 0.30, size=60)
    evt = base + rng.normal(0, 0.001, size=60)
    res = S.paired_one_sided_t(base, evt, alpha=0.05)
    print(f"  t-stat={res.statistic:.2f}  p={res.pvalue:.4f}  reject={res.reject_h0}")
    assert not res.reject_h0


def test_full_comparison_runs():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.10, 0.30, size=50)
    evt = base - rng.uniform(0.005, 0.025, size=50)
    res = S.full_comparison(base, evt, alpha=0.05, n_bootstrap=500)
    print(f"  mean_diff={res['mean_diff']:+.4f}  CI=[{res['bootstrap_ci_lower']:+.4f},{res['bootstrap_ci_upper']:+.4f}]")
    print(f"  t p={res['paired_t_pvalue']:.4f}  W p={res['wilcoxon_pvalue']:.4f}")
    assert res["mean_diff"] < 0
    assert res["bootstrap_ci_upper"] < 0


if __name__ == "__main__":
    tests = [
        ("Metrics on known monotone series", test_metrics_on_known_series),
        ("Metrics on constant series", test_metrics_handle_constant),
        ("Metrics drawdown ordering", test_metrics_drawdown_consistent_with_underwater),
        ("Min variance favours stable asset", test_benchmarks_min_variance_lowers_vol),
        ("All benchmarks run end-to-end", test_run_benchmarks_returns_all_strategies),
        ("Paired t detects effect", test_paired_t_detects_consistent_improvement),
        ("Paired t no false reject", test_paired_t_no_effect_does_not_reject),
        ("Full comparison wraps cleanly", test_full_comparison_runs),
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