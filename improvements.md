# Fixing DKL prediction intervals: from Winkler 666 to competitive coverage

**Your biggest wins come from three changes you can implement today: calibrate in log-space instead of original space, use the GP's own variance as a normalizer for conformal scores, and switch to per-horizon calibration.** These three changes alone should achieve ≥95% coverage and dramatically reduce your Winkler score. The current approach — discarding the GP posterior variance, using a single global conformal quantile in original space, and treating all 48 forecast horizons identically — leaves enormous gains on the table. With a Winkler penalty multiplier of **40×** for each unit of miss-distance at α=0.05, your under-coverage is almost certainly the dominant cost component, and the strategies below are ordered by expected impact.

---

## Stop throwing away your GP variance — normalized conformal is the key unlock

Your DKL model already outputs posterior mean μ(x) and standard deviation σ(x) from the variational GP, yet you only use the point prediction and apply post-hoc conformal calibration with fixed or crudely adaptive widths. This is the single largest missed opportunity. **Normalized conformal prediction** uses σ(x) as a natural scaling factor, producing intervals that are automatically wider for uncertain predictions and tighter for confident ones.

The conformity score becomes the normalized residual:

```
R_i = |y_i − μ(x_i)| / σ(x_i)
```

Instead of computing a single quantile of raw residuals (which yields constant-width intervals), you compute the quantile q̂ of these normalized scores. The prediction interval is then **μ(x) ± q̂ · σ(x)**. If the GP variance is even roughly informative about relative uncertainty, this produces dramatically tighter intervals for low-uncertainty predictions while maintaining coverage through wider intervals where needed.

```python
def normalized_conformal(cal_y, cal_mu, cal_std, test_mu, test_std, alpha=0.05):
    """Locally adaptive conformal using GP posterior std as normalizer."""
    n = len(cal_y)
    # Normalized residuals on calibration set
    norm_scores = np.abs(cal_y - cal_mu) / np.maximum(cal_std, 1e-6)
    # Finite-sample conformal quantile
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(norm_scores, q_level)
    # Variable-width intervals
    lower = test_mu - q_hat * test_std
    upper = test_mu + q_hat * test_std
    return np.maximum(lower, 0), upper, q_hat
```

This preserves the finite-sample marginal coverage guarantee of split conformal prediction regardless of whether the GP is well-calibrated. When the GP variance is informative, you get approximate conditional coverage "for free." One important caveat: **feature collapse in DKL** (documented by van Amersfoort et al., 2022) means the ELBO objective can encourage the MLP to map out-of-distribution inputs to the same features as training inputs, making GP variance unreliable for truly novel inputs. If you observe that σ(x) doesn't increase for extreme weather events, consider adding spectral normalization to the MLP layers to enforce a bi-Lipschitz constraint. You can also apply variance recalibration — find a scalar T that minimizes NLL on validation data via `calibrated_var = T * original_var` — this corrects systematic over- or under-estimation of variance from the variational approximation.

---

## Calibrate in log-space, then back-transform — natural asymmetry for free

Your model already works in log1p-transformed space internally. Currently, you transform predictions back to original space via `expm1()` before conformal calibration. **Reverse this: calibrate in log-space, then transform the interval bounds.** This single change yields three benefits that directly reduce Winkler score.

First, since `expm1()` is convex, symmetric intervals in log-space become naturally right-skewed in original space — **exactly matching the heavy-tailed distribution of outage counts**. Second, additive residuals in log-space correspond to multiplicative errors in original space, so intervals automatically scale with prediction magnitude (heteroscedasticity handled for free). Third, log-transformed residuals are typically more symmetric and closer to Gaussian, making quantile estimation more statistically efficient.

Coverage is preserved exactly because `expm1()` is monotonic: if P(z_L ≤ Z ≤ z_U) = 0.95 in log-space, then P(expm1(z_L) ≤ Y ≤ expm1(z_U)) = 0.95 in original space.

```python
def logspace_normalized_conformal(cal_log_y, cal_log_mu, cal_log_std,
                                   test_log_mu, test_log_std, alpha=0.05):
    """Normalized conformal in log1p space with back-transformation."""
    n = len(cal_log_y)
    # Normalized scores in log-space
    norm_scores = np.abs(cal_log_y - cal_log_mu) / np.maximum(cal_log_std, 1e-6)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(norm_scores, q_level)
    
    # Intervals in log-space
    log_lower = test_log_mu - q_hat * test_log_std
    log_upper = test_log_mu + q_hat * test_log_std
    
    # Back-transform to original space → naturally asymmetric
    lower = np.maximum(np.expm1(log_lower), 0)
    upper = np.expm1(log_upper)
    return lower, upper
```

For even better performance, use **asymmetric quantiles** in log-space. Instead of ±q̂, compute separate lower and upper quantiles from signed normalized residuals. This lets the calibration discover that the upper tail needs more room than the lower tail, and you can grid-search over the asymmetric split to minimize Winkler directly:

```python
def asymmetric_logspace_conformal(cal_log_y, cal_log_mu, cal_log_std,
                                   test_log_mu, test_log_std, alpha=0.05):
    """Asymmetric conformal with separate upper/lower quantiles."""
    signed_norm = (cal_log_y - cal_log_mu) / np.maximum(cal_log_std, 1e-6)
    n = len(signed_norm)
    q_low = np.quantile(signed_norm, alpha / 2)           # negative
    q_high = np.quantile(signed_norm, min(np.ceil((n+1)*(1-alpha/2))/n, 1.0))
    
    log_lower = test_log_mu + q_low * test_log_std
    log_upper = test_log_mu + q_high * test_log_std
    return np.maximum(np.expm1(log_lower), 0), np.expm1(log_upper)
```

---

## Per-horizon calibration captures how uncertainty grows over 48 steps

Applying a single conformal quantile across all 48 forecast horizons is a major source of inefficiency. Uncertainty at t+1 is fundamentally different from uncertainty at t+48 — especially in an autoregressive setup where errors compound. Per-horizon calibration maintains separate residual pools and quantiles for each step-ahead, producing **intervals that naturally widen with horizon** without manual scaling heuristics.

```python
class PerHorizonConformal:
    """Per-step-ahead conformal calibration with ACI."""
    def __init__(self, horizon=48, alpha=0.05, window=300, aci_lr=0.005):
        self.horizon = horizon
        self.alpha = alpha
        self.aci_lr = aci_lr
        # Separate residual pools per horizon step
        self.residuals = {h: deque(maxlen=window) for h in range(horizon)}
        self.alpha_h = {h: alpha for h in range(horizon)}  # adaptive alpha
    
    def update(self, log_preds, log_stds, log_actuals):
        """Feed in one complete 48-step forecast cycle."""
        for h in range(self.horizon):
            norm_res = (log_actuals[h] - log_preds[h]) / max(log_stds[h], 1e-6)
            self.residuals[h].append(norm_res)
    
    def get_intervals(self, log_preds, log_stds):
        """Return calibrated intervals in original space."""
        lower, upper = np.zeros(self.horizon), np.zeros(self.horizon)
        for h in range(self.horizon):
            res = np.array(self.residuals[h])
            n = len(res)
            if n < 20:  # pool across nearby horizons if insufficient data
                nearby = []
                for hh in range(max(0, h-3), min(self.horizon, h+4)):
                    nearby.extend(self.residuals[hh])
                res = np.array(nearby)
                n = len(res)
            a = self.alpha_h[h]
            q_lo = np.quantile(res, a / 2)
            q_hi = np.quantile(res, min(np.ceil((n+1)*(1-a/2))/n, 1.0))
            lower[h] = max(np.expm1(log_preds[h] + q_lo * log_stds[h]), 0)
            upper[h] = np.expm1(log_preds[h] + q_hi * log_stds[h])
        return lower, upper
    
    def update_aci(self, log_preds, log_stds, log_actuals):
        """ACI adaptive alpha update after observing actuals."""
        for h in range(self.horizon):
            res = np.array(self.residuals[h])
            if len(res) < 10: continue
            a = self.alpha_h[h]
            q_lo = np.quantile(res, a / 2)
            q_hi = np.quantile(res, min(1 - a/2, 1.0))
            norm = (log_actuals[h] - log_preds[h]) / max(log_stds[h], 1e-6)
            covered = (norm >= q_lo) and (norm <= q_hi)
            err = 0.0 if covered else 1.0
            self.alpha_h[h] = np.clip(a + self.aci_lr * (err - self.alpha), 0.005, 0.20)
```

You need at minimum **50–100 past forecast cycles** per horizon step for stable quantile estimation. With a window of 300, this gives a roughly 12-day history at hourly granularity. During the early warmup phase when per-horizon data is sparse, pool residuals from nearby horizons (e.g., h±3) or fall back to a global quantile.

---

## Adaptive Conformal Inference handles distribution shift automatically

Standard conformal prediction assumes exchangeability — violated by time series data. **Adaptive Conformal Inference (ACI)** from Gibbs & Candès (2021) fixes this with a remarkably simple online update rule that adjusts the effective significance level based on recent miscoverage:

```
α_{t+1} = α_t + γ · (α − err_t)
```

where err_t = 1 if the observation fell outside the interval, 0 otherwise, and γ is a step size (typically **0.005–0.01**). The intuition: if you under-covered, α decreases → wider intervals next time. If you over-covered, α increases → tighter intervals. The guarantee is distribution-free: average miscoverage converges to α as T→∞ regardless of the data-generating process.

For your 48-hour autoregressive setup, run **separate ACI instances per horizon step** with potentially different learning rates. Longer horizons benefit from larger γ (more adaptive) since their error distribution shifts more. The key advantage over your current "fixed conformal vs. adaptive conformal with 1.5×/0.8× multipliers" is that ACI learns the correct adaptation factor from data rather than requiring hand-tuned multipliers.

A more advanced alternative is **SPCI (Sequential Predictive Conformal Inference)** from Xu & Xie (2023), which predicts the conditional quantile of the next residual given recent residual history using a quantile regression model. SPCI produces tighter intervals than ACI by exploiting temporal autocorrelation in residuals. However, it requires training an additional model (a quantile regression forest on residual sequences), adding complexity. For a first pass, **ACI combined with per-horizon normalized conformal is the recommended starting point** — it delivers most of the gains with minimal implementation effort.

---

## The Winkler penalty arithmetic demands a coverage-first strategy

The Winkler score formula reveals why your current under-coverage is so costly. For a 95% interval (α=0.05), the penalty for a missed observation is **(U−L) + (2/0.05) × miss_distance = width + 40 × miss_distance**. Every unit of miss-distance costs 40× more than a unit of interval width. This means widening intervals by δ adds 2δ to the score for every observation, but eliminates 40×(miss_distance) for each previously-missed observation brought inside the interval.

**Practical implication**: if 10% of observations currently miss the interval by an average of 20 units, the penalty contribution is 0.10 × 40 × 20 = 80 per observation on average. Widening intervals to capture those observations costs far less unless the intervals must grow enormously. This is why the optimal strategy is:

1. **Phase 1**: Target 96–97% nominal coverage (slightly above 95%) to ensure actual coverage exceeds the threshold even with estimation noise
2. **Phase 2**: Once coverage is secured, grid-search for the tightest intervals maintaining ≥95%
3. **Phase 3**: Optimize asymmetric splits — instead of fixed α/2 and 1−α/2, search over (lower_q, upper_q) pairs summing to α

```python
def winkler_score(y_true, lower, upper, alpha=0.05):
    """Vectorized Winkler score."""
    width = upper - lower
    pen_below = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    pen_above = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    return np.mean(width + pen_below + pen_above)

def optimize_asymmetric_split(cal_residuals, val_preds_log, val_stds_log,
                               val_actuals, alpha=0.05):
    """Grid-search over asymmetric quantile splits to minimize Winkler."""
    best_ws, best_params = np.inf, None
    for lower_frac in np.arange(0.005, alpha, 0.005):
        upper_frac = alpha - lower_frac
        q_lo = np.quantile(cal_residuals, lower_frac)
        q_hi = np.quantile(cal_residuals, 1 - upper_frac)
        lower = np.maximum(np.expm1(val_preds_log + q_lo * val_stds_log), 0)
        upper = np.expm1(val_preds_log + q_hi * val_stds_log)
        ws = winkler_score(val_actuals, lower, upper, alpha)
        if ws < best_ws:
            best_ws = ws
            best_params = (lower_frac, upper_frac, q_lo, q_hi)
    return best_params
```

For right-skewed outage data, the optimal split typically allocates more of the α budget to the lower tail (e.g., lower_frac=0.03, upper_frac=0.02) because violations of the upper bound are more common and more costly. The Winkler score is a **proper scoring rule** — minimizing it on a validation set yields well-calibrated intervals.

---

## County-specific calibration via Mondrian conformal prediction

A single global conformal quantile produces marginal coverage across all counties, but individual counties can deviate substantially — some at 80% coverage, others at 99%. **Mondrian conformal prediction** runs separate conformal calibration per group (county), providing group-conditional coverage guarantees: P(y ∈ C(x) | county = g) ≥ 1−α for all g.

The trade-off is data efficiency. With many counties and limited forecast cycles, per-county calibration sets may be too small for reliable quantile estimation. A practical hierarchical approach addresses this:

```python
def hierarchical_conformal(cal_data, test_data, alpha=0.05, min_county_n=50):
    """Per-county calibration where data permits, fallback to cluster/global."""
    # Global quantile as fallback
    global_scores = np.abs(cal_data['log_y'] - cal_data['log_mu']) / cal_data['log_std']
    q_global = np.quantile(global_scores, min(np.ceil((len(global_scores)+1)*(1-alpha))/len(global_scores), 1.0))
    
    results = {}
    for county in test_data['county'].unique():
        cal_mask = cal_data['county'] == county
        n_county = cal_mask.sum()
        if n_county >= min_county_n:
            scores = np.abs(cal_data.loc[cal_mask, 'log_y'] - cal_data.loc[cal_mask, 'log_mu']) \
                     / cal_data.loc[cal_mask, 'log_std']
            q = np.quantile(scores, min(np.ceil((n_county+1)*(1-alpha))/n_county, 1.0))
        else:
            q = q_global  # fallback for small counties
        results[county] = q
    return results
```

For counties with **30–50 calibration points**, consider clustering by geography, climate zone, or urbanization level and calibrating per cluster. An even more sophisticated approach uses quantile regression over county features (one-hot encoded) to smoothly interpolate between county-specific and global quantiles, following the conditional conformal framework of Gibbs & Cherian (2023).

---

## Putting it all together: the recommended implementation pipeline

The strategies above are not mutually exclusive — they compose naturally. Here is the recommended priority ordering and a combined pipeline:

- **Highest impact**: Log-space calibration + normalized conformal using GP std (these two together address the largest sources of inefficiency)
- **High impact**: Per-horizon calibration + ACI adaptive alpha (captures horizon-dependent uncertainty and handles non-stationarity)
- **Medium impact**: Asymmetric quantile optimization + zero-clipping lower bounds (squeezes out remaining Winkler gains)
- **Lower impact but valuable**: County-specific Mondrian calibration (improves conditional coverage for heterogeneous counties)

```python
class FullPipeline:
    """Complete walk-forward calibration pipeline combining all strategies."""
    def __init__(self, horizon=48, alpha=0.05, window=300, aci_lr=0.005):
        self.conformal = PerHorizonConformal(horizon, alpha, window, aci_lr)
        self.alpha = alpha
    
    def calibrate_and_predict(self, model, X_test, cal_data):
        # 1. Get GP predictions in log-space (mean + std)
        log_mu, log_std = model.predict(X_test)  # shape: (48,) each
        
        # 2. Get calibrated intervals from per-horizon normalized conformal
        lower, upper = self.conformal.get_intervals(log_mu, log_std)
        
        # 3. Clip lower bounds at 0
        lower = np.maximum(lower, 0)
        return lower, upper
    
    def walk_forward(self, model, data_stream):
        all_results = []
        for batch in data_stream:
            X, y_actual = batch
            lower, upper = self.calibrate_and_predict(model, X, None)
            log_mu, log_std = model.predict(X)
            log_actual = np.log1p(y_actual)
            
            # Update residual pools
            self.conformal.update(log_mu, log_std, log_actual)
            # ACI adaptive alpha update
            self.conformal.update_aci(log_mu, log_std, log_actual)
            
            ws = winkler_score(y_actual, lower, upper, self.alpha)
            cov = np.mean((y_actual >= lower) & (y_actual <= upper))
            all_results.append({'lower': lower, 'upper': upper,
                                'winkler': ws, 'coverage': cov})
        return all_results
```

## Conclusion

The path from Winkler 666 to a significantly better score runs through three layers of improvement. The **foundation layer** — log-space calibration with GP-variance normalization — corrects the fundamental mismatch between your model's native uncertainty representation and your interval construction method. The model already speaks the language of calibrated heteroscedastic uncertainty through its GP posterior; the current pipeline just ignores it. The **temporal layer** — per-horizon calibration with ACI — acknowledges that a 1-hour-ahead and a 48-hour-ahead forecast live in different uncertainty regimes, and that outage patterns shift over time (storm seasons, grid changes). The **optimization layer** — asymmetric quantile search, Mondrian county calibration, and Winkler-direct optimization — extracts the remaining gains by precisely tailoring intervals to the scoring rule and the data's structure.

A key insight that cuts across all strategies: **the 40× penalty multiplier at α=0.05 means your under-coverage problem is almost certainly more expensive than interval width**. Every strategy should be evaluated coverage-first. Only after reaching ≥95% should you begin tightening. The asymmetric quantile search is particularly valuable for right-skewed outage data because it discovers that allocating more miscoverage budget to the rarely-violated lower tail (where outages can't go below zero anyway) frees room to tighten the expensive upper tail. Combined, these techniques should be implementable in a few hundred lines of numpy/scipy code and are designed to compose with your existing gpytorch DKL model without architectural changes.