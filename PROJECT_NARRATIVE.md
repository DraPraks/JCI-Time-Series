# IDX Next-Day Price Direction Prediction — Progress Summary

## Week 1: Business Understanding & Data Understanding

**Input:** `eren2222/complete-indonesia-stock-exchange-idx-2000-2024` (Kaggle) — 931 tickers, 2000-2024.

**Key findings from EDA:**
- OHLC columns correlated >0.95 — raw prices introduce multicollinearity without signal
- Returns are fat-tailed (high kurtosis), slightly negative skew
- Target class balance: ~63% down / ~37% up

**Sector selection:** Energy (87 tickers) + Technology (47 tickers). Contrasting dynamics: commodity-driven vs. growth/volatility. 4 suspended tickers removed, leaving 130.

**Survivorship bias:** Kept only currently-listed stocks. Correct for the use case (retail investor decision support today), would be wrong for historical backtesting.

**Data cleaning:**
- Forward-filled OHLCV gaps (limit=3) per ticker
- Removed zero-volume flat-price rows (likely non-trading days): 9,304 rows dropped
- Result: 95,932 rows, 0 missing values, 0 duplicates

**Output:** `Dataset/cleaned/energy.parquet` (66,885 rows, 83 tickers) and `technology.parquet` (29,047 rows, 47 tickers).

## Week 2: Data Preparation & Feature Engineering

**Principle:** Encode temporal context into a flat feature vector per row. The model sees one (ticker, day) at a time; all memory of the past must be explicit.

**Feature categories (77 total):**

- **Base derived (7):** daily_return, log_return, high_low_spread, close_open_gap, price_position, volume_change, log_volume
- **Lag features (30):** 5 base cols × lags [1,2,3,5,10,20]. Validated with ACF/PACF — significant autocorrelation at short lags, PACF cutoff ~5-10
- **Rolling window (29):** 4 windows [5,10,20,50] × 7 stat types (price_vs_ma, volatility, volume_ratio, volume_std, momentum, dist_from_high, dist_from_low) + autocorr_20d
- **Cross-sectional (8):** sector_avg_return (ex-self), sector_return_dispersion, relative_strength, sector_pos_frac, sector_total_volume, rel_strength_ma5/20, sector_vol_regime
- **Calendar (3):** day_of_week, month, quarter

**Design constraints enforced:**
- Scale-invariant features only (ratios, returns) — cross-ticker comparability
- All groupby + shift/rolling operations per-ticker — no cross-ticker leakage
- Cross-sectional sector averages exclude self — no self-comparison

**Validation:**
- Rolling MA overlay: `price_vs_ma20` matches manual Close/MA20-1 to 6 decimal places
- Lag shift check: correlation(`daily_return(t)`, `daily_return_lag1(t+1)`) = 1.0
- Target verification: 95,802 rows checked, 0 mismatches
- Inf values (pct_change div-by-zero on AKRA, ITMG, RIGS): replaced with NaN, dropped in Section 8

**Output:** `Dataset/processed/energy.parquet` (58,439 rows, 82 tickers, 24.4 MB) and `technology.parquet` (25,633 rows, 44 tickers, 10.7 MB). 77 features + target. 12.4% rows dropped to NaN (rolling warm-up + last-row target). Feature reference CSV exported.

## Week 3 (Planned): Evaluation & Reporting

**Split:** Temporal only. Train on earlier periods, validate on mid holdout, test on most recent. No shuffle.

**Models:** Logistic regression (baseline), gradient-boosted trees (LightGBM/XGBoost), optionally a simple MLP.

**Metrics:** Precision, recall, F1, ROC-AUC. Accuracy alone is misleading at 63/37 class balance. Confusion matrices stratified by sector and time period.

**Interpretation:** SHAP or permutation importance. Hypothesis: cross-sectional features carry weight beyond single-ticker technical indicators.

**Deliverables:** Cleaned notebooks, final report documenting pipeline + results + reasoning.
