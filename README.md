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

I use XGBoost in rolling windows for training on **residual volatility = realized_volatility - GARCH_forecast** with exogenous drivers. 
Please see notebook for specific hyperparameter selection.

**Why XGBoost?**

- Nonlinear regime learning -- threshold effects, nonlinear interactions; more suitable for power markets where volatility is driven by regime shift (grid stress is important above thresholds, renewables imbalance -- in scarcity states and so on)

- Robust to outliers (heavy distribution tails) and feature scaling

- Good performance in small, noisy datasets with sufficient overfitting control

Residuals are then smoothened (time-backwards) to reduce microstructure noise and improve rank stability.

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

One should aim to optimize the number, as well as the character of variables used. This project contains no automatic optimization, but it can be added in the future. 
I tested many options, starting from including all raw + engineered variables I could come up with into the training, which definitely yielded sub-optimal results in the 0.06-0.12 test Spearman range,
depending on the forecasting horizon chosen. Overfitting was also quite pronounced, keeping XGBoost hyperparameters fixed. I manually narrowed down the choice to the one you can see in the Notebook; adding or removing variables does not improve Spearman ranking.

It is also important to check that the engineered features are not forward-looking (no future data leakage into training dataset).

---

## 4. Validation Results

| Metric	      | Value |
----------------------|-------|
Train Spearman	      | ~0.55 |
Test Spearman	      | ~0.29 |
Holdout rank Spearman |	0.241 |
Pure ML (no GARCH)    |	0.091 |
Permutation p-value   |	0.006 |

Interpretation:

The model extracts statistically significant, persistent structure, including pure ML beyond GARCH.
Performance is stable across time.

## 5. Trading Strategy

Predictions are converted into a relative volatility spread strategy:

- Cross-sectional demeaning per day

- Rolling z-score per country

- Threshold-based long/short signals

- Included hypothetical transaction costs

### Results (with costs)

| Metric	     |  Value |
|--------------------|--------|
| Sharpe	     |  2.23  |
| Max drawdown	     |  −10.8 |
| Avg daily turnover |	0.66  |
| Optimized Sharpe   |	2.86  |

This reflects a realistic relative-volatility arbitrage strategy.

