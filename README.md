# Electricity Volatility Forecasting with hybrid GARCH + ML boosted trees framework

### DE–FR coupled power market

This project implements a hybrid volatility forecasting framework for 24h electricity futures in a coupled European power market (Germany–France).
**Research-grade!** Data is taken from a ML challenge posted a while ago at https://challengedata.ens.fr/participants/challenges/97/ , where you can
also retrieve it (registration required, therefore I will not upload it here, but free to download). All the variables contained in the dataset and some further info
can be found in the Jupyter notebook itself.

Currently the model is designed for **ranking and relative trading**, not point forecasts.

The objective metric is **Spearman rank correlation**, as required by the original ML challenge. 
The output is validated both statistically and through a trading strategy.

**The project is evolving**, so different metrics and more validation methods, as well as model output use cases and visualization might be added.

---

## 1. Economic Motivation

Electricity price volatility is driven by structural system stress, not only by past price moves.

Key regimes:

- Supply-demand imbalance (residual load stress)

- Cross-border congestion (flows/imbalance)

- Fuel merit order shifts (gas-coal-carbon spreads)

- Renewables penetration regimes

These regimes are persistent, interpretable and economically meaningful, making them suitable for rank-based ML.

---

## 2. Model Architecture

### GARCH Baseline

For each country:

- Rolling window GARCH(1,1) is fitted on past returns

- GARCH window -- 500 days, not set in stone, but leaves at least 300 observations and gives stable results

- One-step-ahead conditional volatility is forecast
 
- Done separately for DE and FR 

- This captures autoregressive volatility structure


### ML gradient-boosted decision trees on residual volatility

I use XGBoost in rolling windows for training on **residual volatility = realized_volatility - GARCH_forecast** with additional variables. 
Please see notebook for specific hyperparameter selection.

**Why XGBoost?**

- Nonlinear regime learning -- threshold effects, nonlinear interactions; more suitable for power markets where volatility is driven by regime shift (grid stress is important above thresholds, renewables imbalance -- in scarcity states and so on)

- Robust to outliers (heavy distribution tails) and feature scaling

- Good performance in small, noisy datasets with sufficient overfitting control

Residuals for forecasting are used as is. I also tested a 5-day-lagged rolling mean residual, which of course improves and stabilizes the ranking. But then the task becomes more of a trend-following, so for the day-ahead trades the economic usefulness may decrease.

Window size for train is set to 280 days which is approx. 1 trading year -- stabilizes regime.
Test/prediction window is 21 days -- monthly rebalancing horizon, less noisy. Generally, one would choose the windows to give the best stability-responsiveness tradeoff.

From available raw data like weather, particular fuel type etc. structural, low-noise, regime variables are constructed as 
these are most suitable for ML. Examples of such variables are:

Volatility persistence:
- vol_lag1, vol_lag3, vol_lag7
- vol_roll_std_7, vol_roll_std_30

System stress:
- DE_RESIDUAL_LOAD, FR_RESIDUAL_LOAD
- DE_RESIDUAL_STRESS, FR_RESIDUAL_STRESS

Cross-border congestion:
- LOAD_IMBALANCE
- FLOW_PRESSURE
- TOTAL_FLOW

Fuel & merit order:
- GAS_COAL_SPREAD
- CARBON_PRESSURE
- HIGH_GAS_REGIME

Renewables regime:
- REL_RENEWABLE

One should aim to optimize the number, as well as the character of variables used. Initially this project didn't have automatic feature selection to maximize the metric score, that is, test Spearman ranking, while keeping overfitting under control. Therefore, after many trials and errors of manual selection, I implemented an automatic procedure. This (very) significantly downselected the number of features to essentially the regime variables, not the exogenous drivers. For example, 1-day-lagged volatility itself, slow structural demand pressure etc. They in principle already absorbed all the effect of the exogenous factors. 

It is also important to check that the **engineered features are not forward-looking** (no future data leakage into training dataset; if you train on t-1 and predict for t, no t-data should be contained in the variables).

---

## 4. Validation Results

| Metric	      | Value |
----------------------|-------|
Train Spearman	      | ~0.44 |
Test Spearman	      | ~0.31 |
Holdout rank Spearman |	~0.62 |
Permutation p-value   |	0.002 |

Interpretation:

The model extracts statistically significant, persistent structure, performance is stable across time. The model can correctly rank volatility on ~76% of days.

## 5. Trading Strategy

Predictions are converted into a relative volatility spread strategy:

- Cross-sectional demeaning per day

- Rolling z-score per country

- Threshold-based long/short signals

- Included hypothetical transaction costs

### Results (with costs)

| Metric	     |  Value |
|--------------------|--------|
| Sharpe	     |  1.94  |
| Max drawdown	     |  -8.37 |
| Avg daily turnover |	0.82  |
| Optimized Sharpe   |	2.39  |

This reflects a realistic relative-volatility arbitrage strategy.

