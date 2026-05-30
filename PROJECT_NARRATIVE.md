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

## Week 3: Model Evaluation, Selection & Interpretation

**Split:** Temporal only. Train on earlier periods, validate on mid holdout, test on the most recent period. No shuffle was used, so the reported test result reflects forward-looking generalization rather than random cross-sectional mixing.

**Models compared:** Logistic regression baseline and Random Forest. Random Forest is selected as the final model because it improves the baseline on the validation and test sets across F1, ROC-AUC, PR-AUC, MCC, and macro-F1. This matters because the target is imbalanced; accuracy alone is not a useful selection metric.

**Metric summary:**

| Model | Split | F1 Up | ROC-AUC | PR-AUC | MCC | Predicted Up Rate |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Validation | 0.543 | 0.551 | 0.407 | 0.008 | 99.9% |
| Logistic Regression | Test | 0.525 | 0.562 | 0.399 | 0.000 | 100.0% |
| Random Forest | Validation | 0.552 | 0.585 | 0.439 | 0.114 | 88.0% |
| Random Forest | Test | 0.536 | 0.586 | 0.425 | 0.121 | 86.4% |

**Final selected model:** Random Forest at threshold 0.410. On the test set, it achieves F1 Up = 0.536, ROC-AUC = 0.586, PR-AUC = 0.425, and MCC = 0.121. The test confusion matrix is TP = 2,388, FP = 3,918, TN = 786, FN = 209.

**Model limitation:** Random Forest captures more signal than logistic regression, but it still has a strong upward bias. Test recall for Up is high (0.914), while precision is modest (0.377), meaning the model finds most actual Up days but also produces many false positives. Therefore, it is better interpreted as a decision-support signal for ranking or screening candidates, not as a standalone trading rule.

**Feature interpretation:** The most important Random Forest features are technical and liquidity-based rather than raw prices.

- **Volatility and intraday range:** `high_low_spread`, `volatility_5d`, `volatility_10d`, and lagged spread features suggest that short-term trading range and recent uncertainty are central to next-day direction.
- **Short-term trend and price position:** `price_vs_ma5`, `price_vs_ma10`, `price_vs_ma50`, `price_position`, `dist_from_low50`, and `dist_from_high5` indicate that the model uses where the stock sits relative to recent moving averages and recent highs/lows.
- **Volume and liquidity:** `volume_ratio_50d`, `volume_ratio_20d`, `volume_std_50d`, `volume_std_20d`, `volume_std_5d`, and `log_volume` show that abnormal volume and liquidity conditions help distinguish next-day movement regimes.
- **Recent returns and momentum:** `daily_return`, `log_return`, and `momentum_50d` capture immediate return pressure and medium-term continuation or reversal effects.
- **Sector and cross-sectional context:** Sector-relative features are part of the engineered feature set, but they are not among the top Random Forest importances. In the final report, they should be discussed as context features and tested through sector-level slices rather than overstated as the main driver.

**Qualitative analysis hooks for final report:** Connect model behavior back to market conditions by slicing results by sector (Energy vs. Technology), time period, and market regime. Useful views include false positives during weak or sideways periods, recall during broad Up regimes, and whether Energy and Technology differ in precision/recall because of commodity sensitivity versus growth-stock volatility. These should be reported only after computing the slices, not inferred from the global metrics alone.

**Deliverables:** Final report should document the full pipeline, selected Random Forest model, metric comparison against logistic regression, confusion-matrix tradeoff, feature interpretation, and qualitative sector/time/regime analysis.
