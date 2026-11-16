"""
preprocessing.py
Loads, cleans and prepares the data. Other files use prepare_data().
Running this file directly also makes the data-understanding summary and plots.
Libraries: pandas, numpy, matplotlib only.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Step 1: load the data
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return df


# Step 2: look for problems (only prints, changes nothing)
def check_data(df):
    num_df = df.select_dtypes("number")
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Infinite values:", np.isinf(num_df).sum().sum())
    print("Duplicate rows:", df.duplicated(subset=["Date", "ticker"]).sum())
    print("Rows with zero hl_range or vol_ratio:",
          ((df["hl_range"] == 0) | (df["vol_ratio"] == 0)).sum())


# Step 3: remove duplicate rows (same stock on the same date)
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["Date", "ticker"]).reset_index(drop=True)
    print("Duplicates removed:", before - len(df))
    return df


# Step 4: remove missing and infinite values
def remove_missing(df):
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    print("Missing/infinite rows removed:", before - len(df))
    return df


# Step 5: rebuild ret_lag1 (in the file it is identical to log_ret)
def fix_lag_feature(df):
    df["ret_lag1"] = df.groupby("ticker")["log_ret"].shift(1)
    # if the previous row is more than 7 days back, "yesterday" is not really yesterday
    gap = df.groupby("ticker")["Date"].diff().dt.days
    df.loc[gap > 7, "ret_lag1"] = np.nan
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print("Rows removed (first row of each stock + after date gaps):", before - len(df))
    return df


# Step 6: remove rows with impossible values (zero price range / zero volume ratio)
def remove_bad_rows(df):
    bad = (df["hl_range"] == 0) | (df["vol_ratio"] == 0)
    print("Bad rows removed:", bad.sum())
    df = df[~bad].reset_index(drop=True)
    return df


# Step 7: volatility is right-skewed, so take the log of the target
def add_log_target(df):
    df["log_target"] = np.log(df["target_rv_5"])
    df["log_rv5"] = np.log(df["rv_5"])  # helper for the baselines (not a feature)
    return df


# Step 8: neg_ret is redundant, because log_ret = abs_ret + 2 * neg_ret exactly
def drop_redundant(df):
    return df.drop(columns=["neg_ret"])


# Step 9: turn text/category columns into 0/1 columns
# drop_first=True removes one column per group (it is implied by the others)
def encode_categories(df):
    df = pd.get_dummies(df, columns=["ticker", "dow"], drop_first=True, dtype=int)
    return df


# Step 10: split by time (never randomly for time series)
# The target looks 5 trading days ahead, so each set stops 8 calendar days before
# the next one starts. This keeps their target windows from overlapping.
def split_by_date(df, train_end="2022-01-01", val_end="2024-01-01"):
    gap = pd.Timedelta(days=8)
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    train = df[df["Date"] < train_end - gap].copy()
    val = df[(df["Date"] >= train_end) & (df["Date"] < val_end - gap)].copy()
    test = df[df["Date"] >= val_end].copy()
    print("Train / Val / Test sizes:", len(train), len(val), len(test))
    return train, val, test


# Step 11: cap extreme values in heavy-tailed columns (limits come from train only)
def clip_outliers(train, val, test, clip_cols):
    for col in clip_cols:
        low = train[col].quantile(0.001)
        high = train[col].quantile(0.999)
        train[col] = train[col].clip(low, high)
        val[col] = val[col].clip(low, high)
        test[col] = test[col].clip(low, high)
    return train, val, test


# Step 12: scale numeric columns by hand (learn mean and std from train only)
def scale_features(train, val, test, num_cols):
    means = train[num_cols].mean()
    stds = train[num_cols].std()
    train[num_cols] = (train[num_cols] - means) / stds
    val[num_cols] = (val[num_cols] - means) / stds
    test[num_cols] = (test[num_cols] - means) / stds
    return train, val, test, means, stds


# Step 13: integrity checks (the program stops with an error if one fails)
def check_split(train, val, test, feature_cols):
    assert train["Date"].max() < val["Date"].min(), "train and val overlap in time"
    assert val["Date"].max() < test["Date"].min(), "val and test overlap in time"
    assert "log_target" not in feature_cols, "the target is inside the features"
    assert "target_rv_5" not in feature_cols, "the target is inside the features"
    assert "log_rv5" not in feature_cols, "baseline helper is inside the features"
    for d in (train, val, test):
        assert not d[feature_cols].isnull().any().any(), "missing values in features"
    print("Integrity checks passed")


# The whole pipeline in one function (used by modelling.py and evaluation.py)
def prepare_data(path, train_end="2022-01-01", val_end="2024-01-01"):
    num_cols = ["log_ret", "rv_5", "rv_21", "rv_63", "ret_lag1", "abs_ret",
                "hl_range", "vol_ratio", "mkt_ret", "mkt_rv_21", "vix"]
    clip_cols = ["vol_ratio", "hl_range"]

    # cleaning
    df = load_data(path)
    check_data(df)
    df = remove_duplicates(df)
    df = remove_missing(df)
    df = fix_lag_feature(df)
    df = remove_bad_rows(df)

    # preparing
    df = add_log_target(df)
    df = drop_redundant(df)
    df = encode_categories(df)
    train, val, test = split_by_date(df, train_end, val_end)
    train, val, test = clip_outliers(train, val, test, clip_cols)
    train, val, test, means, stds = scale_features(train, val, test, num_cols)

    # features = everything except Date, the targets and the baseline helper
    not_features = ["Date", "target_rv_5", "log_target", "log_rv5"]
    feature_cols = [c for c in df.columns if c not in not_features]
    check_split(train, val, test, feature_cols)
    return train, val, test, feature_cols


# Data understanding: summary table and plots (run only when this file is run directly)
def make_eda(path):
    df = load_data(path)

    summary = df.select_dtypes("number").describe().T
    summary.to_csv("data_summary.csv")
    print("\nData summary saved to data_summary.csv")
    print(summary[["mean", "std", "min", "max"]].round(4).to_string())

    # target before and after the log transform
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df["target_rv_5"], bins=80)
    axes[0].set_title("target_rv_5 (original)")
    axes[1].hist(np.log(df["target_rv_5"]), bins=80)
    axes[1].set_title("log(target_rv_5)")
    plt.tight_layout()
    plt.savefig("eda1_target_log_transform.png", dpi=150)
    plt.close()

    # volatility over time for three stocks
    plt.figure(figsize=(11, 4))
    for stock in ["RELIANCE", "TCS", "SBIN"]:
        s = df[df["ticker"] == stock]
        plt.plot(s["Date"], s["rv_21"], label=stock, linewidth=0.8)
    plt.title("21-day realized volatility over time")
    plt.xlabel("Date")
    plt.ylabel("rv_21")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda2_volatility_over_time.png", dpi=150)
    plt.close()

    # correlation of recent volatility with the future target (why persistence is weak)
    corr = np.corrcoef(np.log(df["rv_5"]), np.log(df["target_rv_5"]))[0, 1]
    print("\nCorrelation of log(rv_5) with log(target_rv_5):", round(corr, 3))
    print("EDA plots saved")


if __name__ == "__main__":
    make_eda("dataset.csv")
    train, val, test, feature_cols = prepare_data("dataset.csv")
    print("Number of features:", len(feature_cols))
