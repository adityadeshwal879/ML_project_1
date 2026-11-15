# Forecasting 5-day stock volatility

ES 335 Machine Learning project, Aditya Deshwal, IIT Gandhinagar.

**Live site:** https://YOUR-USERNAME.github.io/YOUR-REPO-NAME/ (replace after enabling GitHub Pages)

## Question
Given information available at the close of day t, predict each stock's realized volatility over the next five trading days (`target_rv_5`, modelled as its log) for 30 large Indian companies, 2015 to 2026. The model forecasts the size of price moves, not their direction.

## Result
On the test period (2024 to September 2026), ridge regression's error is 10.7% lower than the training-average baseline (95% week-bootstrap range: 8.8% to 12.7%). It follows the general level of volatility but not sudden spikes. On the site you can enter your own inputs and get a forecast, explore predictions against what actually happened, and read the details and limitations.

## Repository
```
index.html, assets/            the website (GitHub Pages): forecast calculator, prediction explorer, write-up
model.js                       the fitted ridge model (weights, scaling, clipping) used by the calculator
data.js                        the model's test-period predictions used by the explorer
code/preprocessing.py          cleaning, features, time split, train-only scaling, integrity checks
code/modelling.py              metrics, baselines, ridge/linear regression (numpy), tuning, sensitivity
code/evaluation.py             final test evaluation, error analysis, bootstrap, plots, predictions
code/export_model.py           writes model.js and checks it reproduces the Python predictions
results/                       evaluation plots and test predictions CSV
requirements.txt               pandas, numpy, matplotlib
```

## Run it
Put the three scripts and `dataset.csv` in one folder, then:
```
pip install -r requirements.txt
python preprocessing.py
python modelling.py
python evaluation.py
```
Split dates and random seeds are fixed. No machine learning libraries are used.

## Data
Source, date obtained and terms of use: **TO FILL IN**. The dataset itself is not included in this repository. `model.js` holds the fitted model's numbers and five example days; `data.js` holds the model's test-period predictions that the explorer plots.

## Method in brief
- Time split: train before 24 Dec 2021, validation 2022 to 23 Dec 2023, test from 2024, with 8-day gaps so five-day target windows do not overlap.
- Capping and scaling statistics come from the training set only; lambda is chosen on validation only; the test set is used once.
- Models: training average, persistence, one-feature regression, linear regression, ridge regression.

## Limitations
Forecasts size, not direction. Overlapping target windows make errors correlated day to day. Predictions cover only a modest share of the variation and cannot reach very quiet weeks. One market regime in the test period and only 30 stocks.
