# Forecasting 5-day stock volatility

Machine learning project for ES 335 (Machine Learning), by Aditya Deshwal, IIT Gandhinagar.

- **Live website:** https://adityadeshwal879.github.io/ML_project_1/
- **Repository:** https://github.com/adityadeshwal879/ML_project_1
- **Last updated:** 17 November 2026

## Contents
1. [Overview](#1-overview)
2. [Key results](#2-key-results)
3. [Repository layout and code location](#3-repository-layout-and-code-location)
4. [Data access](#4-data-access)
5. [Setup and rebuild steps](#5-setup-and-rebuild-steps)
6. [Method in detail](#6-method-in-detail)
7. [The website](#7-the-website)
8. [Reproducibility notes](#8-reproducibility-notes)
9. [AI use disclosure](#9-ai-use-disclosure)
10. [Citations and third-party material](#10-citations-and-third-party-material)
11. [Privacy and security](#11-privacy-and-security)
12. [License](#12-license)

---

## 1. Overview

**Question.** Given information available at the close of trading day *t*, how large will a stock's price moves be over the next five trading days?

**Target.** `target_rv_5`: the square root of the sum of the squared daily log returns over the next five trading days. A value of 0.03 means those five daily moves add up, in root-sum-of-squares terms, to about 3%. The model predicts its logarithm. It measures the size of price moves, not their direction.

**Data.** One row per stock per trading day: 30 large Indian listed companies, 77,490 rows, 7 April 2015 to 11 September 2026.

**Models compared.** Training average, persistence, one-feature regression, linear regression and ridge regression. The two regressions are written from scratch with numpy (normal equation). The project uses only **pandas, numpy and matplotlib**; no machine learning library.

**Claim and its limits.** The model forecasts the size of price moves, not their direction, and is not trading advice. Conclusions apply only to these 30 stocks and to 2015 to 2026.

## 2. Key results

All numbers are on the **test period (2024 to 11 September 2026)**, with the penalty strength (lambda = 1000) chosen on the validation set. Error is RMSE on the log scale; lower is better.

| Model | Test RMSE | Test MAE | Versus training average |
|---|---|---|---|
| Training average (baseline) | 0.5503 | 0.4343 | 0% |
| Persistence | 0.6674 | 0.5227 | 21.3% worse |
| One-feature regression | 0.5253 | 0.4121 | 4.5% better |
| Linear regression | 0.4981 | 0.3903 | 9.5% better |
| **Ridge regression** | **0.4915** | **0.3847** | **10.7% better** |

- **Uncertainty.** A bootstrap that resamples whole weeks (1,000 draws) gives a 95% range of **8.8% to 12.7%** for ridge's improvement over the baseline.
- **Consistency.** Ridge beats the baseline in each test year: 2024 (0.4835 vs 0.5145), 2025 (0.5036 vs 0.5937), 2026 (0.4796 vs 0.5207).
- **Plain reading.** Ridge's squared error is about 20% below the baseline's. The model follows the general level of volatility but not sudden spikes or very quiet weeks.

## 3. Repository layout and code location

```
ML_project_1/
├── index.html              the website (forecast calculator, prediction explorer, write-up)
├── model.js                fitted ridge model used by the calculator (generated)
├── data.js                 test-period predictions used by the explorer (generated)
├── assets/                 four images shown on the website
├── code/
│   ├── preprocessing.py    cleaning, features, time split, train-only scaling, integrity checks
│   ├── modelling.py        metrics, baselines, ridge/linear regression, tuning, sensitivity checks
│   ├── evaluation.py       final test evaluation, error analysis, bootstrap, plots
│   ├── export_model.py     writes model.js and checks it against the Python predictions
│   └── export_site_data.py writes data.js from test_predictions.csv
├── requirements.txt        pandas, numpy, matplotlib
├── README.md               this file
├── .gitignore              keeps the dataset and generated files out of version control
└── .nojekyll               tells GitHub Pages to serve the files as they are
```

| Task | Where |
|---|---|
| Data cleaning and preparation | `code/preprocessing.py` (`prepare_data` runs the whole pipeline) |
| Metrics, baselines, ridge regression (numpy) | `code/modelling.py` (`fit_ridge`, `predict`, `choose_lambda`) |
| Test evaluation, error analysis, bootstrap | `code/evaluation.py` |
| Website calculator and explorer logic | `index.html` (two `<script>` blocks at the end) |

The scripts import each other, so keep the five `.py` files in the same folder.

## 4. Data access

**The dataset is not included in this repository.**

| Item | Detail |
|---|---|
| Provider / source | **[TO FILL IN: provider or website, and the exact dataset name]** |
| How it was obtained | **[TO FILL IN: download, API, or shared by the course]** |
| Date obtained | **[TO FILL IN]** |
| Terms of use / license | **[TO FILL IN]** |
| Access conditions | **[TO FILL IN: free, login needed, paid, course-only]** |
| Who computed the derived columns (`rv_*`, `target_rv_5`, `vix`, ...) | **[TO FILL IN: provider, course staff, or the author]** |

**Expected file.** `dataset.csv`, placed in the `code/` folder (the scripts read it from the folder they are run in). It has 16 columns:

| Column | Meaning | How well established |
|---|---|---|
| `Date` | Trading date | |
| `ticker` | Stock symbol (30 companies) | |
| `log_ret` | Daily log return | from the name |
| `ret_lag1` | Previous day's return | **defective in the original file**: identical to `log_ret` in every row, so it is rebuilt in preprocessing |
| `abs_ret` | Absolute value of `log_ret` | checked |
| `neg_ret` | `min(log_ret, 0)`; so `log_ret = abs_ret + 2 x neg_ret` exactly | checked; dropped as redundant |
| `rv_5`, `rv_21`, `rv_63` | Standard deviation of daily log returns over the last 5, 21, 63 days | checked on one stock (TCS): correlation 0.995, 0.978, 0.955 |
| `hl_range` | Size of the day's high-to-low price range | **inferred from the name** |
| `vol_ratio` | Trading volume relative to its recent level | **inferred from the name** |
| `mkt_ret`, `mkt_rv_21` | Market return and 21-day market volatility | **inferred from the names** |
| `vix` | A market volatility index | **which index is not stated in the file** |
| `dow` | Day of week, 0 = Monday to 4 = Friday | checked against `Date` |
| `target_rv_5` | Square root of the sum of squared daily log returns over the next 5 trading days | checked: matches in 95.5% of rows to within 0.1% |

**Derived files committed in this repository.** These are computed from the dataset and are published with the website:
- `data.js`: for each of the 17,430 test stock-days, the actual `target_rv_5` and the model's prediction, rounded to 6 decimals.
- `model.js`: the fitted weights, scaling numbers and clipping limits, plus the input values and actual outcomes of five example days.

If the dataset's terms do not allow publishing derived values like these, delete `data.js` (the explorer then switches itself off; the calculator keeps working) and remove the example days from `model.js` by editing `code/export_model.py`.

## 5. Setup and rebuild steps

Tested with **Python 3.12** (pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2); the author also ran the scripts on Python 3.14. Other recent Python 3 versions should work but were not tested. The whole pipeline takes about 10 seconds.

1. **Get the code.** `git clone https://github.com/adityadeshwal879/ML_project_1.git`, or use *Code, Download ZIP* on GitHub and unzip it.
2. **Install Python 3** from https://www.python.org if you do not have it.
3. **Install the libraries** (a virtual environment is optional but tidy):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows;  on macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Add the data.** Obtain `dataset.csv` (section 4) and put it in the `code/` folder.
5. **Run the scripts in order,** from inside `code/`. Use `python3` instead of `python` on macOS/Linux:
   ```
   cd code
   python preprocessing.py     # data summary, EDA plots, integrity checks
   python modelling.py         # tunes lambda on validation, sensitivity checks
   python evaluation.py        # final test evaluation, error analysis, plots
   python export_model.py      # writes model.js, checks it against the Python predictions
   python export_site_data.py  # writes data.js from test_predictions.csv
   ```
6. **Check you reproduced the results.** You should see:
   - `Integrity checks passed` (preprocessing and modelling);
   - `Best lambda: 1000` (modelling);
   - test table with Ridge RMSE `0.4915`, MAE `0.3847`, `10.7%` better than the baseline (evaluation);
   - `Ridge improvement over the mean baseline: 10.7%  (95% range from week-bootstrap: 8.8% to 12.7%)`;
   - `max |browser formula - pipeline| ... 1e-15` (export_model).
7. **Update the website (optional).** Copy `model.js` and `data.js` from `code/` to the repository root, replacing the old ones. The tables of numbers typed into `index.html` are not generated automatically; edit them by hand if your numbers change. Commit and push; see section 7.

**Files written into `code/` when you run the scripts**

| File | Written by | Content |
|---|---|---|
| `data_summary.csv` | preprocessing | summary statistics of every numeric column |
| `eda1_target_log_transform.png`, `eda2_volatility_over_time.png` | preprocessing | target before and after the log; volatility over time |
| `lambda_curve.csv`, `validation_results.csv`, `sensitivity_results.csv` | modelling | validation results and sensitivity checks |
| `test_results.csv`, `test_predictions.csv` | evaluation | test metrics; per-day actual and predicted volatility |
| `plot1` to `plot4` (`.png`) | evaluation | actual vs predicted, scatter, error per stock, error histogram |
| `model.js` | export_model | fitted model for the website |
| `data.js` | export_site_data | test-period predictions for the website |

**Common problems**

| Problem | Cause and fix |
|---|---|
| `FileNotFoundError: dataset.csv` | The file is missing, or you are not running from the folder that contains it. |
| `ImportError: cannot import name 'prepare_data'` | An old copy of `preprocessing.py` is being imported. Use the version from this repository and delete any `__pycache__` folder. |
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` in the same Python you run the scripts with. |
| Numbers differ slightly | Different library versions can change the last digits; the conclusions should not change. |

## 6. Method in detail

### 6.1 Cleaning (`preprocessing.py`)
Order matters: the rebuilt lag is computed before bad rows are removed.

| Step | What and why | Rows affected |
|---|---|---|
| Load and sort | Parse `Date`; sort by stock and date | |
| Checks | Missing, infinite and duplicate values | none found |
| Rebuild `ret_lag1` | The original equals `log_ret` in every row, so it is not a lag; recomputed as the previous row's return per stock. Where the previous row is more than 7 days earlier, the lag is invalid and the row is dropped | 300 (first row of each stock plus the first row after each date gap) |
| Remove bad rows | Zero `hl_range` or zero `vol_ratio` (29 of the 36 fall on 18 March 2025, which points to a bad data feed) | 36 |
| Log target | `log_target = log(target_rv_5)`; the target is right-skewed | |
| Drop `neg_ret` | Exactly redundant: `log_ret = abs_ret + 2 x neg_ret` | |

Nine date gaps of 30 days or more affect all 30 stocks at once. The first row after each gap is on 15 May 2015, 2 February 2016, 15 September 2016, 31 January 2018, 30 January 2019, 27 November 2019, 16 December 2020, 16 February 2026 and 27 July 2026. The last is the longest (88 days, from 30 April 2026). Extreme returns, such as 23 March 2020, look like real events and were kept. Total removed in cleaning: 336 of 77,490 rows (0.43%).

### 6.2 Features
44 inputs: 11 numeric (`log_ret`, `rv_5`, `rv_21`, `rv_63`, `ret_lag1`, `abs_ret`, `hl_range`, `vol_ratio`, `mkt_ret`, `mkt_rv_21`, `vix`), 29 stock columns and 4 weekday columns (one 0/1 column per category, with one category of each dropped so that no column can be rebuilt from the others; otherwise the least-squares problem is singular). The target, the helper `log_rv5` and `Date` are never features; `check_split` asserts this.

### 6.3 Time split
Never shuffled. The target looks five trading days ahead, so each set ends 8 calendar days before the next begins, so no training target overlaps a later set.

| Set | Dates | Rows |
|---|---|---|
| Train | 7 Apr 2015 to before 24 Dec 2021 | 44,786 |
| Validation | 1 Jan 2022 to before 24 Dec 2023 | 14,638 |
| Test | 1 Jan 2024 to 11 Sep 2026 | 17,430 |

`check_split` asserts that the sets do not overlap in time and that the features contain no missing values.

### 6.4 Learned preprocessing (training set only)
- **Clipping:** `vol_ratio` and `hl_range` are capped at their training 0.1% and 99.9% points (`vol_ratio` otherwise reaches 15.3).
- **Scaling:** each numeric input is standardized with the training mean and standard deviation, written by hand.
- The same numbers are applied to validation and test, so nothing from those sets leaks into preprocessing.

### 6.5 Models (`modelling.py`)
- **Training average:** the mean of the training `log_target`, used for every row.
- **Persistence:** `log(rv_5)` plus a constant (the training average gap between the target and `log(rv_5)`). The constant is needed because the target is about the square root of 5 times larger than a daily-scale volatility.
- **One-feature regression:** `log_target` on `log(rv_5)` alone.
- **Linear regression:** all 44 inputs, plain least squares.
- **Ridge regression:** least squares with an L2 penalty (intercept not penalized) [4]. Solved with the normal equation, `w = (X'X + lambda I)^-1 X'y`, using `numpy.linalg.solve`.

### 6.6 Tuning and evaluation
- Lambda is chosen from `[0.01, 0.1, 1, 10, 100, 1000, 10000]` using validation RMSE only. The best is **1000**. A finer check shows a flat minimum between about 1000 and 3000 (validation RMSE 0.4529 at 1000 and 0.4525 at 3000), so the exact value barely matters.
- The test set is used once, after lambda is fixed. `evaluation.py` re-selects lambda on validation each time it runs; it never looks at test data to choose anything.
- Metrics: RMSE (penalizes large misses) and MAE (typical miss), both on the log scale, and the percentage by which a model's RMSE is below the baseline's.

**Validation results**

| Model | Val RMSE | Val MAE | Versus average |
|---|---|---|---|
| Training average | 0.5262 | 0.4172 | 0% |
| Persistence | 0.6252 | 0.4927 | 18.8% worse |
| One-feature regression | 0.4965 | 0.3901 | 5.6% better |
| Linear regression | 0.4597 | 0.3609 | 12.6% better |
| Ridge (lambda = 1000) | 0.4529 | 0.3546 | 13.9% better |

Persistence is worse than the average because recent and future volatility correlate only about 0.3 in this data (log scale), so copying recent volatility adds more noise than it removes.

### 6.7 Uncertainty
Neighbouring days share the same five-day target window, so test errors are correlated: a stock's error correlates 0.77 with its error one row earlier, and about 0.05 at 5 and 10 rows. A plain row-by-row bootstrap would give a range that is too narrow, so `evaluation.py` resamples **whole weeks** (1,000 draws, seed 0). Result: ridge's improvement over the baseline is 10.7%, with a 95% range of 8.8% to 12.7%.

### 6.8 Sensitivity checks (validation set, ridge with the chosen lambda)

| Variant | Val RMSE | Versus average |
|---|---|---|
| All inputs (final model) | 0.4529 | 13.9% better |
| Only `rv_5`, `rv_21`, `rv_63` | 0.4640 | 11.8% better |
| No stock or weekday columns | 0.4517 | 14.1% better |
| Training only on 2018 onwards | 0.4602 | 12.5% better |

Past volatility alone gets most of the way; market and price-range inputs add about two points; stock and weekday columns add nothing. To test a different split date, call `prepare_data("dataset.csv", train_end="2021-01-01", val_end="2023-01-01")` and re-run the scripts.

### 6.9 Error analysis (test set)
- Error per stock ranges from 0.428 (TECHM) to 0.553 (ICICIBANK).
- On the calmest quarter of test days (lowest VIX) the model over-predicts by 0.146 on average on the log scale; on the most stressed quarter the average error is -0.011.
- The 10 largest errors are all over-predictions in unusually quiet weeks: HDFCBANK (8 to 10 September 2025), NTPC (11 and 14 July 2025), ICICIBANK (28 and 30 April 2025, 4 September 2025), ASIANPAINT (21 November 2025) and HCLTECH (28 April 2026). For nine of them the next five daily moves in the file were all below 0.3%, so the targets follow directly from the returns. HDFCBANK's very small moves in mid-September 2025 are unusual for a large bank; whether this is a real quiet spell or a data issue could not be settled from the file. The HCLTECH row sits just before the 88-day gap, and its target uses trading days that are missing from the file.
- Largest ridge weights among the scaled numeric inputs: `vix` 0.114, `hl_range` 0.094, `rv_21` 0.057, `rv_63` 0.051. Several inputs move together, so the ranking should be read loosely.

### 6.10 Limitations
- Forecasts size, not direction; not trading advice.
- Overlapping target windows make errors correlated, so the effective number of independent test points is smaller than the row count.
- A modest share of the variation is explained, and sudden spikes and near-zero weeks are missed.
- One market regime in the test period, only 30 large stocks, and a linear model. Nonlinear models and longer-horizon inputs were not tested.
- Some column meanings were inferred from names (section 4).

## 7. The website

- **Forecast calculator.** Enter a stock, a weekday and ten numbers; the page runs the fitted ridge model in the browser (clip, scale, multiply by the weights, exponentiate) and shows the forecast, a likely range, the comparison with the baseline and what moved the forecast. The likely range is the 10th to 90th percentile of validation errors; about 77% of real test outcomes fell inside it. `code/export_model.py` verifies that the browser formula reproduces the Python predictions on all 17,430 test rows (largest difference about 1e-15).
- **Prediction explorer.** Pick a stock and a date window to compare actual volatility, the model's prediction and the baseline over the test period.
- **Write-up.** The question, data, method, results, errors and limitations, with the numbers typed by hand from the output files.
- **No server and no tracking.** The page makes no network requests of its own, apart from loading two fonts (section 11). Everything runs in the visitor's browser.

**Publishing on GitHub Pages.** The repository must be **public** on a free account. In *Settings, Pages*, set *Source* to *Deploy from a branch*, choose branch `main` and folder `/ (root)`, and save. The site appears at the address at the top of this file after a minute or two. Any commit to `main` rebuilds it; use a hard refresh (Ctrl+Shift+R) to see changes. `index.html` must sit in the repository root, and `assets/` must keep its name and the file names.

## 8. Reproducibility notes
- Split dates are fixed in code. Random choices: the bootstrap uses `numpy.random.default_rng(0)`; the scatter-plot sample uses `random_state=0`. Nothing else is random.
- The code reads `dataset.csv` from the current folder and writes outputs there.
- The rebuild in section 5 was tested from a clean folder: it ran without errors in about 8 seconds and regenerated `model.js` and `data.js` byte-for-byte identical to the committed ones.
- Library versions are listed in section 5. The minimum versions that work were not tested.

## 9. AI use disclosure

I used an AI assistant, **Claude (Anthropic)**, through its chat interface (claude.ai) between October and November 2026. Model versions may have varied during that time. This section says what it was used for, what I did myself, and what was not independently verified.

**What the AI was used for**
- **Code.** Claude wrote the first versions of all Python scripts and revised them as I reported errors and constraints (for example, that only pandas, numpy and matplotlib were allowed, so the scaling and the regression had to be written by hand).
- **Modelling advice.** Claude suggested the log target, the time split with a gap, the baselines, tuning lambda on validation only, the week-block bootstrap, the sensitivity checks and the error analysis. I accepted these.
- **Data checks.** Claude wrote and ran code that found the faulty `ret_lag1`, the redundant `neg_ret`, the zero-value rows, the date gaps and the exact definition of `target_rv_5`.
- **Explanations.** Claude explained the code line by line and the concepts behind it.
- **Writing and website.** Claude drafted the proposal text, the report notes, this README, the website's HTML, CSS and JavaScript, and the steps for publishing on GitHub Pages.
- **Running code.** Claude ran the scripts in its own sandbox while developing and verifying them.

**What I did myself** *(edit to match what you actually did)*
**[TO FILL IN: for example, choosing the project and dataset, deciding the library restriction, running every script on my own machine, reading the outputs, asking for explanations of parts I did not understand, correcting my local files when they differed from the final versions, and publishing the site.]**

**How the AI's work was checked**
- I ran the scripts myself and reproduced the reported results (for example, Ridge test RMSE 0.4915 against 0.5503 for the baseline). Every number in this repository comes from running the code on the dataset, not from the AI's description alone.
- The website calculator was checked against the Python predictions on all test rows (section 7).
- A clean-folder rebuild reproduced the committed generated files exactly (section 8).

**What was not independently verified**
- The meanings of `hl_range`, `vol_ratio`, `mkt_ret`, `mkt_rv_21` and `vix` were inferred from the column names.
- The `rv_5`, `rv_21` and `rv_63` definitions were checked on one stock (TCS).
- Whether the unusually quiet HDFCBANK week in September 2025 is real or a data issue.
- There may be errors in AI-written code or text beyond the checks above.

**At run time:** the project calls no AI service and no external API.

**Responsibility** *(keep only if true)*: I understand the code and the results, can explain them, and take responsibility for the submitted work. The conversation with the AI is available on request.

## 10. Citations and third-party material

### Software used
- **Python** (https://www.python.org): the programming language.
- **NumPy** [1], **pandas** [2], **Matplotlib** [3]: the only libraries used by the code. Their licenses permit this use; their license texts are in their own packages.
- **GitHub Pages** (https://docs.github.com/pages): hosting for the website [8].
- **Fonts** [7]: *Schibsted Grotesk* and *Literata*, loaded by the web page from Google Fonts under the SIL Open Font License 1.1. The fonts are not stored in this repository.

### Methods (background; no code was copied from these)
- Ridge regression [4], the idea behind the penalty term. The normal-equation solution is standard textbook material.
- Block bootstrap for dependent data [5], the idea behind resampling whole weeks. The implementation here is my own simple version, not the method from the paper in full.
- Realized volatility [6], the idea of measuring volatility from squared returns.

### Code and text from other sources
No code was copied from other repositories, tutorials or forums. The code was written with AI assistance (section 9). **[Edit this sentence if you used any other source.]** All plots are generated by the scripts; the website uses no external images or icons.

### Data and course material
- Dataset: **[TO FILL IN: citation for the data source; see section 4]**.
- Course project brief and grading rubric, ES 335 Machine Learning, IIT Gandhinagar (course portal, not public).

### References
1. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2
2. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61. https://doi.org/10.25080/Majora-92bf1922-00a
3. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55
4. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634
5. Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241. https://doi.org/10.1214/aos/1176347265
6. Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579-625. https://doi.org/10.1111/1468-0262.00418
7. Google Fonts. https://fonts.google.com (Schibsted Grotesk and Literata, SIL Open Font License 1.1).
8. GitHub Docs. About GitHub Pages. https://docs.github.com/pages
9. Anthropic. Claude (AI assistant), accessed through claude.ai, October to November 2026. https://www.anthropic.com

## 11. Privacy and security

- **No credentials.** The repository contains no passwords, access tokens, API keys, private keys or `.env` files. The code does not log in to anything and calls no API.
- **No personal data.** There are no email addresses, phone numbers, roll numbers or local file paths in the code, web page or this file. The only personal information is the author's name and public GitHub username.
- **Checked.** Before publishing, the code, web page and documentation were scanned for key and token patterns, email addresses and local paths, with no matches. Re-run a similar scan before any future release, and never commit secrets: if one is committed by mistake, revoke it and remove it from the history, because deleting the file is not enough.
- **Dataset kept out.** `dataset.csv` is not committed (see `.gitignore`; note that `.gitignore` only affects Git commands, not files dragged into the GitHub web page).
- **Browser privacy.** The website does not set cookies, use local storage or send visitor data anywhere. Loading the two fonts means each visitor's browser contacts Google Fonts (fonts.googleapis.com and fonts.gstatic.com), as with any page that uses Google Fonts.
- **Derived data.** `data.js` and `model.js` contain values derived from the dataset (section 4). Publish them only if the dataset's terms allow it.

