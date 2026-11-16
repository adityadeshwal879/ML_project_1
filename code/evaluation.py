"""
evaluation.py
Final TEST evaluation (run once) and error analysis.
lambda is chosen on the validation set inside this file, never on test.
Libraries: pandas, numpy, matplotlib only.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import prepare_data
from modelling import (rmse, mae, improvement, mean_baseline, persistence_baseline,
                       fit_ridge, predict, choose_lambda, LAMBDAS)

DATA_PATH = "dataset.csv"


# ---------- get the stock name back (it was one-hot encoded) ----------
def get_stock_names(df, path):
    all_tickers = sorted(pd.read_csv(path)["ticker"].unique())
    ticker_cols = ["ticker_" + t for t in all_tickers[1:]]    # the first stock was dropped
    dummies = df[ticker_cols].values
    position = dummies.argmax(axis=1) + 1                      # which column has the 1
    position[dummies.sum(axis=1) == 0] = 0                     # no 1 anywhere = dropped stock
    return np.array(all_tickers)[position]


# ---------- bootstrap by week: how uncertain is the improvement over the baseline? ----------
def bootstrap_improvement(res, n_draws=1000):
    res = res.copy()
    res["week"] = res["Date"].dt.to_period("W").astype(str)
    weekly = res.groupby("week")[["sq_err_ridge", "sq_err_mean"]].sum()
    counts = res.groupby("week").size()
    n_weeks = len(weekly)
    rng = np.random.default_rng(0)
    values = []
    for i in range(n_draws):
        pick = rng.integers(0, n_weeks, n_weeks)               # draw whole weeks with replacement
        n = counts.values[pick].sum()
        rmse_ridge = np.sqrt(weekly["sq_err_ridge"].values[pick].sum() / n)
        rmse_mean = np.sqrt(weekly["sq_err_mean"].values[pick].sum() / n)
        values.append(improvement(rmse_mean, rmse_ridge))
    return np.percentile(values, [2.5, 97.5])


# ================= STEP A: final test evaluation =================
train, val, test, feature_cols = prepare_data(DATA_PATH)

X_train, y_train = train[feature_cols].values, train["log_target"].values
X_val, y_val = val[feature_cols].values, val["log_target"].values
X_test, y_test = test[feature_cols].values, test["log_target"].values

best_lam, _ = choose_lambda(X_train, y_train, X_val, y_val, LAMBDAS)   # validation only
print("Lambda chosen on validation:", best_lam)

preds = {}
preds["Mean baseline"] = mean_baseline(train, test)
preds["Persistence baseline"] = persistence_baseline(train, test)
w_one = fit_ridge(train[["log_rv5"]].values, y_train, 0)
preds["One-feature regression"] = predict(test[["log_rv5"]].values, w_one)
w_lin = fit_ridge(X_train, y_train, 0)
preds["Linear regression"] = predict(X_test, w_lin)
w_ridge = fit_ridge(X_train, y_train, best_lam)
preds["Ridge"] = predict(X_test, w_ridge)

base = rmse(y_test, preds["Mean baseline"])
rows = []
for name, p in preds.items():
    rows.append({"model": name, "test_rmse": rmse(y_test, p), "test_mae": mae(y_test, p),
                 "improvement_vs_mean_%": improvement(base, rmse(y_test, p))})
test_results = pd.DataFrame(rows)
print("\nTEST SET RESULTS")
print(test_results.round(4).to_string(index=False))
test_results.to_csv("test_results.csv", index=False)


# ================= STEP B: error analysis =================
res = pd.DataFrame({
    "Date": test["Date"].values,
    "stock": get_stock_names(test, DATA_PATH),
    "actual": y_test,
    "ridge": preds["Ridge"],
    "mean": preds["Mean baseline"],
})
res["error"] = res["actual"] - res["ridge"]
res["sq_err_ridge"] = res["error"] ** 2
res["sq_err_mean"] = (res["actual"] - res["mean"]) ** 2
res["year"] = res["Date"].dt.year

# 1) uncertainty of the improvement (bootstrap by week)
low_ci, high_ci = bootstrap_improvement(res)
print("\nRidge improvement over the mean baseline: %.1f%%  (95%% range from week-bootstrap: %.1f%% to %.1f%%)"
      % (improvement(base, rmse(y_test, preds["Ridge"])), low_ci, high_ci))

# 2) error per year
print("\nRMSE per year (test)")
print(np.sqrt(res.groupby("year")[["sq_err_ridge", "sq_err_mean"]].mean()).round(4))

# 3) calm vs stressed days (VIX top / bottom 25% of test days)
vix = test["vix"].values
high = vix > np.quantile(vix, 0.75)
low = vix < np.quantile(vix, 0.25)
print("\nHigh-VIX days: RMSE", round(np.sqrt(res.loc[high, "sq_err_ridge"].mean()), 4),
      " average error (actual - predicted):", round(res.loc[high, "error"].mean(), 4))
print("Low-VIX days:  RMSE", round(np.sqrt(res.loc[low, "sq_err_ridge"].mean()), 4),
      " average error (actual - predicted):", round(res.loc[low, "error"].mean(), 4))

# 4) error per stock
per_stock = np.sqrt(res.groupby("stock")["sq_err_ridge"].mean()).sort_values()
print("\nBest 3 stocks :", per_stock.head(3).round(3).to_dict())
print("Worst 3 stocks:", per_stock.tail(3).round(3).to_dict())

# 5) the 10 biggest misses
res["abs_error"] = res["error"].abs()
top10 = res.sort_values("abs_error", ascending=False).head(10)
print("\n10 biggest misses (log scale)")
print(top10[["Date", "stock", "actual", "ridge"]].to_string(index=False, float_format="%.3f"))

# 6) are the errors correlated from day to day? (overlapping 5-day windows)
res = res.sort_values(["stock", "Date"])
print("\nCorrelation of a stock's error with its error k rows earlier")
for k in [1, 5, 10]:
    prev = res.groupby("stock")["error"].shift(k)
    both = pd.DataFrame({"now": res["error"], "before": prev}).dropna()
    print("  lag", k, ":", round(both["now"].corr(both["before"]), 3))

# 7) largest weights of the numeric features (these were scaled, so they can be compared)
numeric = ["log_ret", "rv_5", "rv_21", "rv_63", "ret_lag1", "abs_ret",
           "hl_range", "vol_ratio", "mkt_ret", "mkt_rv_21", "vix"]
weights = pd.Series(w_ridge[1:], index=feature_cols)[numeric]
print("\nRidge weights of the numeric features")
print(weights.sort_values(key=abs, ascending=False).round(3))


# ---------- plots (saved as image files) ----------
# Plot 1: actual vs predicted volatility for one stock
stock = "RELIANCE"
s = res[res["stock"] == stock].sort_values("Date")
plt.figure(figsize=(11, 4))
plt.plot(s["Date"], np.exp(s["actual"]), label="Actual", linewidth=1)
plt.plot(s["Date"], np.exp(s["ridge"]), label="Ridge prediction", linewidth=1)
plt.plot(s["Date"], np.exp(s["mean"]), label="Mean baseline", linestyle="--")
plt.title(stock + ": next-5-day volatility, test period")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png", dpi=150)
plt.close()

# Plot 2: predicted vs actual (a sample of 3000 test rows)
sample = res.sample(3000, random_state=0)
plt.figure(figsize=(5.5, 5.5))
plt.scatter(sample["ridge"], sample["actual"], s=4, alpha=0.4)
low_v, high_v = res["actual"].min(), res["actual"].max()
plt.plot([low_v, high_v], [low_v, high_v], color="red", label="perfect prediction")
plt.title("Predicted vs actual (log volatility)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()
plt.tight_layout()
plt.savefig("plot2_scatter.png", dpi=150)
plt.close()

# Plot 3: RMSE per stock
plt.figure(figsize=(8, 7))
plt.barh(per_stock.index, per_stock.values)
plt.axvline(rmse(y_test, preds["Ridge"]), color="red", linestyle="--", label="overall RMSE")
plt.title("Ridge RMSE per stock (test)")
plt.xlabel("RMSE (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig("plot3_rmse_per_stock.png", dpi=150)
plt.close()

# Plot 4: histogram of the errors
plt.figure(figsize=(7, 4))
plt.hist(res["error"], bins=60)
plt.title("Ridge errors on the test set (actual - predicted)")
plt.xlabel("Error (log scale)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plot4_error_histogram.png", dpi=150)
plt.close()

# ---------- save the predictions (needed later for the website) ----------
out = res.sort_values(["stock", "Date"])[["Date", "stock", "actual", "ridge", "mean"]].copy()
out["actual_vol"] = np.exp(out["actual"])
out["ridge_vol"] = np.exp(out["ridge"])
out["mean_vol"] = np.exp(out["mean"])
out.to_csv("test_predictions.csv", index=False)
print("\nSaved 4 plots, test_results.csv and test_predictions.csv")
