"""
export_model.py
Writes model.js: the fitted ridge model (weights, scaling numbers, clipping limits)
so the website can compute forecasts in the browser with the same arithmetic as Python.
Uses the training set only for every number it exports, except the example days.
Libraries: pandas, numpy only.
"""
import json
import numpy as np
import pandas as pd
from preprocessing import (load_data, remove_duplicates, remove_missing, fix_lag_feature,
                           remove_bad_rows, add_log_target, drop_redundant, encode_categories,
                           split_by_date, clip_outliers, scale_features)
from modelling import fit_ridge, predict, choose_lambda, LAMBDAS

PATH = "dataset.csv"
NUM_COLS = ["log_ret", "rv_5", "rv_21", "rv_63", "ret_lag1", "abs_ret",
            "hl_range", "vol_ratio", "mkt_ret", "mkt_rv_21", "vix"]
CLIP_COLS = ["vol_ratio", "hl_range"]
INPUTS = [c for c in NUM_COLS if c != "abs_ret"]          # abs_ret is worked out from log_ret

# ----- same steps as prepare_data, but keeping the pieces we need -----
df = load_data(PATH)
df = remove_duplicates(df)
df = remove_missing(df)
df = fix_lag_feature(df)
df = remove_bad_rows(df)
df = add_log_target(df)
df = drop_redundant(df)
raw = df.copy()                                            # raw rows (stock name, unscaled)
df = encode_categories(df)
train, val, test = split_by_date(df)
train_raw = train.copy()

bounds = {c: [float(train[c].quantile(0.001)), float(train[c].quantile(0.999))] for c in CLIP_COLS}
train, val, test = clip_outliers(train, val, test, CLIP_COLS)
means_stds = {c: (float(train[c].mean()), float(train[c].std())) for c in NUM_COLS}   # after clipping, as in scale_features
train, val, test, means, stds = scale_features(train, val, test, NUM_COLS)

not_features = ["Date", "target_rv_5", "log_target", "log_rv5"]
feature_cols = [c for c in df.columns if c not in not_features]

X_train, y_train = train[feature_cols].values, train["log_target"].values
X_val, y_val = val[feature_cols].values, val["log_target"].values
X_test, y_test = test[feature_cols].values, test["log_target"].values

best_lam, _ = choose_lambda(X_train, y_train, X_val, y_val, LAMBDAS)
w = fit_ridge(X_train, y_train, best_lam)
intercept, weights = float(w[0]), [float(v) for v in w[1:]]

# ----- forecast range from validation errors, checked on test -----
res_val = y_val - predict(X_val, w)
q10, q90 = [float(v) for v in np.quantile(res_val, [0.10, 0.90])]
res_test = y_test - predict(X_test, w)
coverage = float(np.mean((res_test >= q10) & (res_test <= q90)))

# ----- inputs: typical value and ranges from the training rows -----
stocks = sorted(raw["ticker"].unique())
inputs = {}
for c in INPUTS:
    col = train_raw[c]
    inputs[c] = {"median": float(col.median()), "p1": float(col.quantile(0.01)),
                 "p99": float(col.quantile(0.99)), "min": float(col.min()), "max": float(col.max())}

# ----- check: does the browser formula reproduce the pipeline's predictions? -----
def browser_formula(row, stock, dow):
    x = {c: float(row[c]) for c in INPUTS}
    x["abs_ret"] = abs(x["log_ret"])
    for c, (lo, hi) in bounds.items():
        x[c] = min(max(x[c], lo), hi)
    z = {c: (x[c] - means_stds[c][0]) / means_stds[c][1] for c in NUM_COLS}
    total = intercept
    for f, wt in zip(feature_cols, weights):
        if f in z: v = z[f]
        elif f.startswith("ticker_"): v = 1.0 if f == "ticker_" + stock else 0.0
        elif f.startswith("dow_"): v = 1.0 if f == "dow_" + str(dow) else 0.0
        else: v = 0.0
        total += wt * v
    return total

raw_test = raw.loc[test.index]
check = np.array([browser_formula(r, r["ticker"], int(r["dow"])) for _, r in raw_test.iterrows()])
pipeline = predict(X_test, w)
print("max |browser formula - pipeline| over", len(check), "test rows:", float(np.max(np.abs(check - pipeline))))
print("dow equals weekday of Date:", bool((raw["dow"] == raw["Date"].dt.dayofweek).all()), " dow values:", sorted(raw["dow"].unique()))
print("reference stock (dropped dummy):", stocks[0], " | best lambda:", best_lam)
print("80%% range factors: x%.3f to x%.3f ; coverage on test: %.1f%%" % (np.exp(q10), np.exp(q90), coverage * 100))

# ----- example days (chosen by rule, not by how well the model does) -----
test_raw = raw_test.copy()
test_raw["pred_log"] = pipeline
def row_for(stock, date):
    r = test_raw[(test_raw["ticker"] == stock) & (test_raw["Date"] == date)]
    return r.iloc[0] if len(r) else None

zs = pd.DataFrame({c: (test_raw[c] - train_raw[c].median()) / train_raw[c].std() for c in INPUTS})
typical_idx = (zs ** 2).sum(axis=1).idxmin()
calm_date = test_raw.loc[test_raw["vix"].idxmin(), "Date"]
stress_date = test_raw.loc[test_raw["vix"].idxmax(), "Date"]

def pick(stock, date, fallback_idx=None):
    r = row_for(stock, date)
    return r if r is not None else test_raw[test_raw["Date"] == date].iloc[0]

examples = [
    ("typical", "A typical day", test_raw.loc[typical_idx]),
    ("calm", "The calmest market day in the test period", pick("RELIANCE", calm_date)),
    ("stress", "The most stressed market day in the test period", pick("RELIANCE", stress_date)),
    ("spike", "RELIANCE, 31 May 2024: a spike", row_for("RELIANCE", pd.Timestamp("2024-05-31"))),
    ("quiet", "HDFCBANK, 10 Sep 2025: a very quiet week", row_for("HDFCBANK", pd.Timestamp("2025-09-10"))),
]
presets = []
for key, label, r in examples:
    presets.append({"id": key, "label": label, "stock": r["ticker"], "dow": int(r["dow"]),
                    "date": r["Date"].strftime("%Y-%m-%d"),
                    "values": {c: round(float(r[c]), 6) for c in INPUTS},
                    "actual": round(float(r["target_rv_5"]), 6),
                    "modelForecast": round(float(np.exp(r["pred_log"])), 6)})
    print(" ", key, r["ticker"], r["Date"].date(), "actual", round(float(r["target_rv_5"]), 4), "model", round(float(np.exp(r["pred_log"])), 4))

model = {"lambda": best_lam, "intercept": intercept, "features": feature_cols, "weights": weights,
         "numeric": NUM_COLS, "inputs": INPUTS,
         "means": {c: means_stds[c][0] for c in NUM_COLS}, "stds": {c: means_stds[c][1] for c in NUM_COLS},
         "clip": bounds, "stocks": stocks, "refStock": stocks[0],
         "baseline": float(np.exp(y_train.mean())),
         "range": {"q10": q10, "q90": q90, "testCoverage": coverage},
         "inputInfo": inputs, "presets": presets}
with open("model.js", "w") as f:
    f.write("window.VOL_MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n")
print("model.js written,", len(feature_cols), "features")
