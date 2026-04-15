# Trader Performance vs Market Sentiment — Hyperliquid Analysis
### Primetrade.ai · Data Science / Analytics Intern — Round-0 Assignment

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

## How to Run

1. Place your two CSV files in this directory:
   - `sentiment.csv`  — Bitcoin Fear/Greed index (columns: `date`, `classification`)
   - `trades.csv`     — Hyperliquid trader data (columns: `account`, `symbol`, `executionPrice`, `size`, `side`, `time` [unix ms], `closedPnL`, `leverage`, ...)

2. Open and run the notebook:
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

3. Charts are saved automatically to `./charts/`

> **Note:** If your `time` column is Unix **milliseconds**, change `unit="s"` to `unit="ms"` in cell A2. The notebook documents this clearly.

---

## Methodology

### Data Preparation (Part A)
- Loaded both datasets, documented shape, missing values, and duplicates
- Converted Unix timestamps to UTC datetimes; normalized to daily granularity
- Collapsed 5-class sentiment (Extreme Fear → Fear, Extreme Greed → Greed) to 3-class for cleaner analysis
- Merged on `date` key; derived key metrics: daily PnL per account, win rate, leverage, long/short ratio, notional size, trade frequency

### Analysis (Part B)
- **Performance analysis:** Compared daily PnL distributions, median PnL, win rates, and PnL standard deviation (drawdown proxy) across Fear vs Greed days
- **Behavioral analysis:** Tested whether traders change leverage, trade frequency, long bias, or position size with sentiment
- **Segmentation:** Three orthogonal segmentations — leverage terciles, frequency terciles, and consistency (rolling win rate threshold)
- **Heatmap:** 2D win-rate analysis across sentiment × leverage segment
- **Time series:** 7-day rolling PnL trend plotted alongside sentiment overlay

### Bonus
- **KMeans clustering (k=4):** Auto-labeled archetypes — High-PnL Traders, High-Risk Gamblers, Consistent Winners, Volume Traders — using elbow method for k selection
- **Gradient Boosting classifier:** Predicts whether an account will have a profitable day using sentiment + behavioral features; evaluated with 5-fold CV ROC-AUC

---

## Key Insights

| # | Insight | Evidence |
|---|---------|----------|
| 1 | Greed days yield higher PnL and win rates | Avg PnL ~6.9 USD/day higher; win rate +3pp |
| 2 | Traders raise leverage slightly on Greed days | 8.11x (Greed) vs 7.98x (Fear) |
| 3 | Consistent winners outperform regardless of sentiment | +5 USD/day vs inconsistent traders |
| 4 | High-leverage traders suffer most on Fear days | Heatmap: lowest win rate at High Lev × Fear |
| 5 | Win rate is the strongest predictor of future profitability | Highest GBM feature importance |

---

## Strategy Recommendations (Part C)

### 🔴 Strategy 1 — Fear-Day Leverage Reduction
**Rule:** On Fear days, cap leverage at 75% of each trader's personal average.

**Rationale:** Average daily PnL is ~6.9 USD lower on Fear days. Win rates drop ~3pp. High-leverage traders are disproportionately affected (confirmed by heatmap). A 25% leverage cut preserves participation while meaningfully reducing tail-loss exposure.

```python
if sentiment == "Fear":
    max_leverage = 0.75 * trader.rolling_avg_leverage
```

### 🟢 Strategy 2 — Consistent-Winner Amplification on Greed Days
**Rule:** For traders with 60-day rolling win rate > 51%, allow 20% larger position sizes on Greed days.

**Rationale:** Consistent winners already outperform by ~5 USD/day. Greed days show lower PnL volatility AND higher returns. Amplifying proven traders in favorable conditions improves risk-adjusted return without increasing platform-wide risk.

```python
if sentiment == "Greed" and trader.rolling_60d_win_rate > 0.51:
    allowed_size = 1.20 * trader.normal_size
```

---

## Output Files

| File | Description |
|------|-------------|
| `trader_sentiment_analysis.ipynb` | Main analysis notebook |
| `charts/fig1_pnl_fear_vs_greed.png` | PnL distributions by sentiment |
| `charts/fig2_behavior.png` | Behavioral metrics comparison |
| `charts/fig3_segments.png` | Trader segment analysis |
| `charts/fig4_heatmap.png` | Win rate heatmap |
| `charts/fig5_trend.png` | Time-series PnL vs sentiment |
| `charts/fig6_clusters.png` | KMeans behavioral archetypes |
| `charts/fig6a_elbow.png` | Elbow plot for cluster selection |
| `charts/fig7_feature_importance.png` | GBM feature importance |
