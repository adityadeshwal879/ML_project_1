"""
modelling.py
Metrics, baselines and the from-scratch ridge/linear regression.
Importing this file only loads the functions.
Running it directly tunes lambda on the VALIDATION set and runs sensitivity checks.
The test set is never used here.
Libraries: pandas, numpy only.
"""
import numpy as np
import pandas as pd
from preprocessing import prepare_data


# ---------- metrics ----------
def rmse(y_act, y_pred):
    return np.sqrt(np.mean((y_act - y_pred) ** 2))


def mae(y_act, y_pred):
    return np.mean(abs(y_act - y_pred))


# percentage by which the model's error is lower than the baseline's (positive = better)
def improvement(baseline_error, model_error):
    return ((baseline_error - model_error) / baseline_error) * 100


# ---------- baselines ----------
def mean_baseline(train, other):
    return np.full(len(other), np.mean(train["log_target"]))


def persistence_baseline(train, other):
    offset = (train["log_target"] - train["log_rv5"]).mean()
    return (other["log_rv5"] + offset).values


# ---------- regression from scratch ----------
def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


# lam = 0 gives plain linear regression, lam > 0 gives ridge
def fit_ridge(X, y, lam):
    X = add_intercept(X)
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0                       # the intercept is not penalized
    w = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    return w


def predict(X, w):
    return add_intercept(X) @ w


# ---------- tuning ----------
# tries each lambda, scores it on the validation set, returns the best one
def choose_lambda(X_train, y_train, X_val, y_val, lambdas):
    best_lam, best_err = None, np.inf
    rows = []
    for lam in lambdas:
        w = fit_ridge(X_train, y_train, lam)
        err = rmse(y_val, predict(X_val, w))
        rows.append({"lambda": lam, "val_rmse": err})
        if err < best_err:
            best_err = err
            best_lam = lam
    return best_lam, pd.DataFrame(rows)


LAMBDAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]


if __name__ == "__main__":
    train, val, test, feature_cols = prepare_data("dataset.csv")

    X_train, y_train = train[feature_cols].values, train["log_target"].values
    X_val, y_val = val[feature_cols].values, val["log_target"].values

    # ----- choose lambda on validation -----
    best_lam, lambda_table = choose_lambda(X_train, y_train, X_val, y_val, LAMBDAS)
    print("\nValidation RMSE for each lambda")
    print(lambda_table.to_string(index=False))
    print("Best lambda:", best_lam)
    lambda_table.to_csv("lambda_curve.csv", index=False)

    # ----- validation results for every model -----
    preds = {}
    preds["Mean baseline"] = mean_baseline(train, val)
    preds["Persistence baseline"] = persistence_baseline(train, val)
    w = fit_ridge(train[["log_rv5"]].values, y_train, 0)
    preds["One-feature regression"] = predict(val[["log_rv5"]].values, w)
    w = fit_ridge(X_train, y_train, 0)
    preds["Linear regression"] = predict(X_val, w)
    w = fit_ridge(X_train, y_train, best_lam)
    preds["Ridge"] = predict(X_val, w)

    base = rmse(y_val, preds["Mean baseline"])
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "val_rmse": rmse(y_val, p), "val_mae": mae(y_val, p),
                     "improvement_vs_mean_%": improvement(base, rmse(y_val, p))})
    val_results = pd.DataFrame(rows)
    print("\nValidation results")
    print(val_results.round(4).to_string(index=False))
    val_results.to_csv("validation_results.csv", index=False)

    # ----- sensitivity checks (all scored on validation, ridge with the best lambda) -----
    def score(cols, train_part):
        w = fit_ridge(train_part[cols].values, train_part["log_target"].values, best_lam)
        return rmse(y_val, predict(val[cols].values, w))

    vol_cols = ["rv_5", "rv_21", "rv_63"]
    numeric_cols = [c for c in feature_cols if not c.startswith("ticker_") and not c.startswith("dow_")]
    train_recent = train[train["Date"] >= "2018-01-01"]

    checks = {
        "All features (reference)": score(feature_cols, train),
        "Only rv_5, rv_21, rv_63": score(vol_cols, train),
        "Numeric features only (no stock/weekday)": score(numeric_cols, train),
        "Train only on 2018 onwards": score(feature_cols, train_recent),
    }
    rows = []
    for name, err in checks.items():
        rows.append({"variant": name, "val_rmse": err, "improvement_vs_mean_%": improvement(base, err)})
    sens = pd.DataFrame(rows)
    print("\nSensitivity checks (validation set)")
    print(sens.round(4).to_string(index=False))
    sens.to_csv("sensitivity_results.csv", index=False)
