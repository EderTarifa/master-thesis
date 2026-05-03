"""
Extreme Value Theory estimators.

Implements:
- Peaks-Over-Threshold (POT) with Generalized Pareto Distribution (GPD).
- Block Maxima with Generalized Extreme Value (GEV) distribution.
- Automated threshold selection via ordered Anderson-Darling tests
  with FDR correction (Bader, Yan & Zhang, 2018).
- Closed-form CVaR for GPD tail.

References
----------
Bader, B., Yan, J., & Zhang, X. (2018). Automated threshold selection
    for extreme value analysis via ordered goodness-of-fit tests with
    adjustment for false discovery rate. Annals of Applied Statistics.
McNeil & Frey (2000). Estimation of tail-related risk measures...
Coles (2001). An Introduction to Statistical Modeling of Extreme Values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPDFit:
    """Result of fitting a Generalized Pareto Distribution to threshold excesses.

    Parameters
    ----------
    threshold : float
        Threshold u above which excesses (x - u) were modelled.
    shape : float
        Shape parameter xi. xi>0 heavy tail, xi=0 exponential, xi<0 bounded.
    scale : float
        Scale parameter sigma > 0.
    n_excesses : int
        Number of observations exceeding the threshold.
    n_total : int
        Total number of observations in the sample.
    converged : bool
        Whether the MLE converged successfully.
    """

    threshold: float
    shape: float
    scale: float
    n_excesses: int
    n_total: int
    converged: bool

    @property
    def exceedance_prob(self) -> float:
        """Empirical probability of exceeding the threshold."""
        return self.n_excesses / max(self.n_total, 1)


@dataclass
class GEVFit:
    """Result of fitting a Generalized Extreme Value distribution to block maxima."""

    location: float
    scale: float
    shape: float
    n_blocks: int
    converged: bool


# ---------------------------------------------------------------------------
# GPD fitting and risk measures
# ---------------------------------------------------------------------------


def _fit_gpd_mle(
    excesses: np.ndarray,
    shape_clip: tuple[float, float] = (-0.5, 0.99),
) -> tuple[float, float, bool]:
    """Maximum-likelihood fit of GPD parameters to non-negative excesses.

    Returns (shape, scale, converged). Initialises with method-of-moments
    for robustness.

    The shape is clipped to a finite range to guarantee numerical stability
    and a finite CVaR (xi < 1 is required for finite mean).
    """
    excesses = np.asarray(excesses, dtype=float)
    if excesses.size < 5:
        # Not enough data to fit reliably. Fall back to exponential (xi=0)
        # using sample mean as scale.
        scale = max(float(np.mean(excesses)) if excesses.size > 0 else 1e-3, 1e-6)
        return 0.0, scale, False

    # Method-of-moments initialisation (Hosking & Wallis, 1987).
    mean_e = float(np.mean(excesses))
    var_e = float(np.var(excesses, ddof=1)) if excesses.size > 1 else mean_e**2
    if var_e <= 1e-12 or mean_e <= 1e-12:
        return 0.0, max(mean_e, 1e-6), False
    shape0 = 0.5 * (1.0 - mean_e**2 / var_e)
    scale0 = 0.5 * mean_e * (1.0 + mean_e**2 / var_e)
    shape0 = float(np.clip(shape0, shape_clip[0], shape_clip[1]))
    scale0 = max(scale0, 1e-6)

    # scipy.stats.genpareto: shape parameter c == xi.
    try:
        c, loc, scale = stats.genpareto.fit(
            excesses,
            f0=shape0,  # initial guess for c
            floc=0,    # location fixed at 0 (we work with excesses)
            scale=scale0,
        )
        # If fit returns out-of-range values, clip and mark non-converged.
        c_clipped = float(np.clip(c, shape_clip[0], shape_clip[1]))
        scale_clipped = max(float(scale), 1e-6)
        converged = (
            np.isfinite(c) and np.isfinite(scale)
            and c == c_clipped and abs(scale - scale_clipped) < 1e-9
        )
        return c_clipped, scale_clipped, bool(converged)
    except Exception:
        return shape0, scale0, False


def fit_gpd(
    losses: np.ndarray,
    threshold: Optional[float] = None,
    quantile: float = 0.90,
    shape_clip: tuple[float, float] = (-0.5, 0.99),
) -> GPDFit:
    """Fit a GPD to the tail of `losses`.

    Parameters
    ----------
    losses : np.ndarray, shape (n,)
        Sample of losses (positive = loss). Use -returns if you have returns.
    threshold : float, optional
        Threshold above which to fit GPD. If None, taken as the empirical
        `quantile` of the data.
    quantile : float
        Empirical quantile to use as threshold when `threshold` is None.
        Typical values: 0.90, 0.95.
    shape_clip : tuple
        Clipping range for the shape parameter for numerical stability.

    Returns
    -------
    GPDFit
    """
    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]
    n_total = losses.size
    if n_total < 20:
        # Not enough data; return a degenerate fit with xi=0.
        return GPDFit(
            threshold=float(np.percentile(losses, 100 * quantile)) if n_total > 0 else 0.0,
            shape=0.0,
            scale=1e-3,
            n_excesses=0,
            n_total=n_total,
            converged=False,
        )

    if threshold is None:
        threshold = float(np.quantile(losses, quantile))

    excesses = losses[losses > threshold] - threshold
    shape, scale, converged = _fit_gpd_mle(excesses, shape_clip=shape_clip)
    return GPDFit(
        threshold=float(threshold),
        shape=shape,
        scale=scale,
        n_excesses=int(excesses.size),
        n_total=n_total,
        converged=converged,
    )


def gpd_var(fit: GPDFit, alpha: float) -> float:
    """Compute VaR at level `alpha` from a GPD tail fit.

    Uses the standard tail extrapolation formula:
        VaR_alpha = u + (sigma / xi) * [((1-alpha)/F_u)^(-xi) - 1]    (xi != 0)
        VaR_alpha = u - sigma * log((1-alpha)/F_u)                    (xi == 0)
    where F_u = P(X > u) is estimated empirically.

    Parameters
    ----------
    alpha : float
        Quantile level in (0, 1). Typically 0.95, 0.99.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    fu = fit.exceedance_prob
    if fu <= 0:
        return fit.threshold
    p = 1.0 - alpha
    if p >= fu:
        # Quantile lies inside the body: not extrapolated by GPD.
        # Return the threshold as a conservative lower bound.
        return fit.threshold
    ratio = p / fu
    if abs(fit.shape) < 1e-8:
        var = fit.threshold - fit.scale * np.log(ratio)
    else:
        var = fit.threshold + (fit.scale / fit.shape) * (ratio**(-fit.shape) - 1.0)
    return float(var)


def gpd_cvar(fit: GPDFit, alpha: float) -> float:
    """Compute CVaR (expected shortfall) at level `alpha` from a GPD tail fit.

    For xi < 1, the GPD tail has finite mean and:
        CVaR_alpha = VaR_alpha / (1 - xi) + (sigma - xi*u) / (1 - xi)
    For xi >= 1 the mean is infinite; we return +inf.
    For xi = 0 the formula collapses to VaR + sigma.
    """
    var = gpd_var(fit, alpha)
    if fit.shape >= 1.0:
        return float("inf")
    if abs(fit.shape) < 1e-8:
        return float(var + fit.scale)
    return float(var / (1.0 - fit.shape) + (fit.scale - fit.shape * fit.threshold) / (1.0 - fit.shape))


# ---------------------------------------------------------------------------
# Automated threshold selection (Bader, Yan & Zhang 2018, simplified)
# ---------------------------------------------------------------------------


def _anderson_darling_gpd(excesses: np.ndarray, shape: float, scale: float) -> float:
    """Anderson-Darling statistic for GPD goodness of fit.

    Returns the AD statistic (smaller = better fit). Used as a relative
    measure across candidate thresholds.
    """
    n = excesses.size
    if n < 5:
        return np.inf
    # CDF of GPD with location 0.
    if abs(shape) < 1e-8:
        u = 1.0 - np.exp(-excesses / max(scale, 1e-12))
    else:
        z = 1.0 + shape * excesses / max(scale, 1e-12)
        z = np.where(z > 0, z, np.nan)
        u = 1.0 - z**(-1.0 / shape)
    u = np.sort(u[np.isfinite(u)])
    if u.size < 5:
        return np.inf
    u = np.clip(u, 1e-12, 1 - 1e-12)
    n2 = u.size
    i = np.arange(1, n2 + 1)
    a2 = -n2 - np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))) / n2
    return float(a2)


def select_threshold_auto(
    losses: np.ndarray,
    candidate_quantiles: Optional[np.ndarray] = None,
) -> tuple[float, GPDFit]:
    """Select threshold via grid search minimising the AD statistic.

    This is a simplified, robust version of the Bader-Yan-Zhang (2018)
    procedure: we evaluate the AD goodness-of-fit statistic across a grid
    of candidate quantile thresholds and pick the one with the lowest AD
    that retains at least 30 excesses. This avoids the FDR multiple-testing
    machinery while preserving the spirit of the approach.

    Returns (threshold, fit).
    """
    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]
    if losses.size < 50:
        # Fall back to a fixed 90% quantile if data is scarce.
        return float(np.quantile(losses, 0.90)) if losses.size > 0 else 0.0, fit_gpd(losses)

    if candidate_quantiles is None:
        candidate_quantiles = np.arange(0.80, 0.98, 0.01)

    best_ad = np.inf
    best_fit: Optional[GPDFit] = None
    best_threshold = float(np.quantile(losses, 0.90))
    for q in candidate_quantiles:
        u = float(np.quantile(losses, q))
        excesses = losses[losses > u] - u
        if excesses.size < 30:
            # Need enough excesses for a reliable fit.
            continue
        shape, scale, converged = _fit_gpd_mle(excesses)
        if not converged:
            continue
        ad = _anderson_darling_gpd(excesses, shape, scale)
        if ad < best_ad:
            best_ad = ad
            best_threshold = u
            best_fit = GPDFit(
                threshold=u,
                shape=shape,
                scale=scale,
                n_excesses=int(excesses.size),
                n_total=int(losses.size),
                converged=True,
            )

    if best_fit is None:
        # No candidate worked; fall back to the 90% quantile.
        best_fit = fit_gpd(losses, quantile=0.90)
        best_threshold = best_fit.threshold

    return best_threshold, best_fit


# ---------------------------------------------------------------------------
# Block Maxima with GEV
# ---------------------------------------------------------------------------


def fit_gev_block_maxima(losses: np.ndarray, block_size: int = 5) -> GEVFit:
    """Fit a GEV distribution to block maxima of `losses`.

    Parameters
    ----------
    losses : np.ndarray
        Sample of losses (positive = loss).
    block_size : int
        Block length (e.g. 5 for weekly maxima of daily data).

    Returns
    -------
    GEVFit
    """
    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses)]
    n_blocks = losses.size // block_size
    if n_blocks < 5:
        return GEVFit(location=0.0, scale=1e-3, shape=0.0,
                      n_blocks=n_blocks, converged=False)

    truncated = losses[: n_blocks * block_size].reshape(n_blocks, block_size)
    block_max = truncated.max(axis=1)

    try:
        # scipy.stats.genextreme uses c = -xi (negated convention).
        c, loc, scale = stats.genextreme.fit(block_max)
        shape = -float(c)  # convert back to standard EVT convention
        return GEVFit(
            location=float(loc),
            scale=max(float(scale), 1e-6),
            shape=float(np.clip(shape, -0.5, 0.99)),
            n_blocks=int(n_blocks),
            converged=True,
        )
    except Exception:
        return GEVFit(
            location=float(np.mean(block_max)),
            scale=max(float(np.std(block_max, ddof=1)), 1e-6),
            shape=0.0,
            n_blocks=int(n_blocks),
            converged=False,
        )


# ---------------------------------------------------------------------------
# Convenience: features for the agent state
# ---------------------------------------------------------------------------


def evt_state_features(
    returns_window: np.ndarray,
    use_auto_threshold: bool = True,
    fixed_quantile: float = 0.90,
) -> np.ndarray:
    """Construct the EVT feature vector for the agent state.

    Operates on the rolling window of portfolio (or asset) returns and
    returns a 5-dim vector of EVT statistics computed on the loss tail:

        [shape (xi), scale (sigma), VaR_99, CVaR_99, exceedance_freq]

    The values are returned as-is (in loss units, positive = bad);
    the caller is responsible for any further normalisation.
    """
    losses = -np.asarray(returns_window, dtype=float).ravel()
    if use_auto_threshold:
        _, fit = select_threshold_auto(losses)
    else:
        fit = fit_gpd(losses, quantile=fixed_quantile)
    var99 = gpd_var(fit, 0.99)
    cvar99 = gpd_cvar(fit, 0.99)
    return np.array(
        [fit.shape, fit.scale, var99, cvar99, fit.exceedance_prob],
        dtype=np.float32,
    )
