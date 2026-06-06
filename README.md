# Rossmann Store Sales — Demand Forecasting

A end-to-end demand forecasting project comparing classical statistical 
models and machine learning approaches on the Rossmann retail dataset 
covering 1115 stores across Germany.

---

## Problem Statement

Rossmann operates 1115 drug stores across Germany. Store managers are 
tasked with predicting daily sales up to six weeks in advance. Accurate 
forecasts enable:
- Proactive inventory planning
- Optimized staff scheduling
- Reduced overstock and stockout costs
- Data-driven promotional planning

---

## Dataset

**Source:** [Rossmann Store Sales — Kaggle Competition](https://www.kaggle.com/competitions/rossmann-store-sales)

| File | Description | Rows |
|---|---|---|
| train.csv | Historical daily sales per store | 1,017,209 |
| store.csv | Store metadata (type, assortment, competition) | 1,115 |

**Key features:**
- `Sales` — daily turnover (target variable)
- `Promo` — whether store ran a promotion that day
- `StateHoliday` — public holiday indicator
- `SchoolHoliday` — school holiday indicator
- `StoreType` — 4 store categories (a, b, c, d)
- `Assortment` — product range level (basic, extra, extended)
- `CompetitionDistance` — distance to nearest competitor

---

---

## Methodology

### 1. Exploratory Data Analysis
- Sales trend analysis across 2013–2015
- Seasonality patterns — weekly, monthly, yearly
- Promotion impact analysis
- Store type and assortment comparisons
- Feature correlation analysis

**Key EDA findings:**
- Promotions lift average sales by ~25%
- Monday consistently peaks in weekly sales
- December shows strong yearly sales spike
- Store type B has highest average sales despite being rarest
- `Customers` is the strongest correlator with Sales but is a 
  leaky feature — excluded from modeling

### 2. Preprocessing & Feature Engineering

**Data cleaning:**
- Filtered closed store days (Open=0)
- Filled missing `CompetitionDistance` with max×2 (no nearby competitor)
- Encoded `StoreType` and `Assortment` as ordinal integers
- Converted `StateHoliday` to binary `IsHoliday`
- Dropped leaky and sparse features

**Engineered features (31 total):**

| Category | Features |
|---|---|
| Date | Year, Month, Week, DayOfMonth, Quarter |
| Calendar | IsWeekend, IsMonthStart, IsMonthEnd, DaysTillMonthEnd |
| Lag | lag_1, lag_7, lag_14, lag_28 |
| Rolling Mean | rolling_mean_7, rolling_mean_14, rolling_mean_28 |
| Rolling Std | rolling_std_7, rolling_std_14, rolling_std_28 |
| Rolling Max | rolling_max_7, rolling_max_14, rolling_max_28 |
| Store | StoreType, Assortment, CompetitionDistance, Promo2 |

**Train/test split:**
- Train: up to 2015-06-14
- Test: 2015-06-15 to 2015-07-31 (6-week holdout)
- Time-based split — never random split on time series

### 3. Models

#### SARIMA
- Classical statistical model for univariate time series
- Monthly aggregation (weekly m=52 computationally prohibitive)
- Best model selected via stepwise AIC search: `ARIMA(1,0,2)(1,0,0)[12]`
- Captures yearly seasonality (m=12)
- Evaluated on Store 1

#### Prophet
- Meta's decomposition-based forecasting model
- Handles daily data natively
- Multiplicative seasonality mode for retail data
- German public holidays added explicitly
- Promo and IsHoliday added as external regressors
- Evaluated on Store 1

#### XGBoost
- Gradient boosting on 31 engineered features
- Single global model trained on all 1115 stores
- Early stopping at best iteration (789/1000)
- SHAP analysis for feature explainability
- Evaluated on Store 1 (fair comparison) and all stores

---

## Results

### Model Comparison — Store 1

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | 378.72 | 403.39 | 8.49% |
| Prophet | 344.91 | 443.65 | 7.96% |
| **XGBoost** | **260.96** | **331.24** | **5.91%** |

### XGBoost — All 1115 Stores

| Metric | Value |
|---|---|
| MAE | 615.42 |
| RMSE | 876.18 |
| MAPE | 9.23% |
| Stores under 10% MAPE | 776 / 1115 (69.6%) |

**XGBoost outperforms SARIMA by 30% and Prophet by 26% on MAPE.**

---

## Key Insights

**1. Lag features are the strongest predictors**
> Past sales (lag_7, lag_28, rolling_mean_7) dominate feature 
> importance — weekly retail cycles are the most predictive signal.

**2. Promotions are the strongest business lever**
> Promo lifts average daily sales by ~25% across all store types. 
> Store type B sees the highest absolute promo uplift.

**3. XGBoost generalizes across store types**
> A single global model achieves under 10% MAPE on 776/1115 stores 
> without any store-specific tuning — demonstrating strong 
> generalization across diverse retail formats.

**4. Classical models are strong baselines**
> SARIMA achieves 8.49% MAPE using only historical sales values — 
> no feature engineering required. Useful when data is limited or 
> interpretability is prioritized.

**5. Prophet excels at holiday modeling**
> Prophet's explicit German holiday component and multiplicative 
> seasonality captures retail patterns more accurately than SARIMA 
> without manual feature engineering.

---

## Business Impact

- **Inventory optimization** — 5.91% MAPE on key stores enables 
  confident 6-week forward planning, reducing holding costs
- **Promotion planning** — quantified promo lift (~25%) supports 
  ROI-based promotional decisions
- **Scale** — single XGBoost model serves all 1115 stores, 
  eliminating need for per-store model maintenance
- **Explainability** — SHAP analysis identifies top drivers per 
  prediction, enabling store managers to understand and trust forecasts

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Classical Models | statsmodels, pmdarima |
| ML Model | XGBoost, Scikit-learn |
| Forecasting | Prophet (Meta) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

---

Rishimithan Kannan