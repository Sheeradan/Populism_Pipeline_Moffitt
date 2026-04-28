"""
Engagement Prediction Model — AfD TikTok Communication Strategy
================================================================
BA Thesis: Exploratory predictive analysis of which multimodal content
features best predict TikTok engagement (Qvalue).

Target variable : Qvalue (continuous engagement metric)
Features        : 8 binary (0/1) coded variables
Observations    : ~123 usable (after dropping uncoded rows)

Author  : Generated with Claude Code
Date    : 2026-03-22
"""

import warnings, os, sys, textwrap

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── paths ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Downloads", "Predict +Q.csv")
# Fallback: if running from the project folder, try the Downloads path directly
if not os.path.exists(DATA_PATH):
    DATA_PATH = r"C:\Users\danbe\Downloads\Predict +Q.csv"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLS = [
    "Appeal_to_the_People_pros",
    "Anti_Elitism_pros",
    "Bad_Manners_pros",
    "Crisis_Breakdown_Threat_pros",
    "Dress",
    "Setting",
    "Production",
    "Format",
]

NICE_NAMES = {
    "Appeal_to_the_People_pros": "Appeal to the People",
    "Anti_Elitism_pros": "Anti-Elitism",
    "Bad_Manners_pros": "Bad Manners",
    "Crisis_Breakdown_Threat_pros": "Crisis / Threat",
    "Dress": "Dress (formal)",
    "Setting": "Setting (institutional)",
    "Production": "Production (high)",
    "Format": "Format (edited)",
}

TARGET = "Qvalue"


def banner(title: str) -> None:
    """Print a clear section header."""
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}")


# =====================================================================
#  1. DATA LOADING & PREPROCESSING
# =====================================================================
banner("1 · DATA LOADING & PREPROCESSING")

df = pd.read_csv(DATA_PATH)

# Replace N/A strings with actual NaN
df.replace("N/A", np.nan, inplace=True)

# Ensure feature columns are numeric
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

print(f"  Raw data shape : {df.shape}")
print(f"  Columns        : {list(df.columns)}")

# Drop the ID column — not used for prediction
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)
    print("  ✓ Removed 'ID' column (not used for prediction)")

# Drop rows with any NaN (the uncoded row with N/A in Dress & Setting)
n_before = len(df)
df.dropna(subset=FEATURE_COLS + [TARGET], inplace=True)
df.reset_index(drop=True, inplace=True)
n_after = len(df)
print(f"  Rows dropped (NaN) : {n_before - n_after}")
print(f"  Usable observations : {n_after}")

# Ensure binary features are int
for col in FEATURE_COLS:
    df[col] = df[col].astype(int)

# ── summary statistics ───────────────────────────────────────────────
print("\n  Feature distributions (mean = proportion coded 1):")
for col in FEATURE_COLS:
    print(f"    {NICE_NAMES[col]:30s}  mean = {df[col].mean():.3f}  "
          f"(n₁ = {df[col].sum():3.0f}, n₀ = {(1 - df[col]).sum():3.0f})")

print(f"\n  Target variable  '{TARGET}'  —  descriptive statistics:")
desc = df[TARGET].describe()
for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
    print(f"    {stat:>5s} : {desc[stat]:>12,.1f}")

# Histogram of target
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[TARGET], bins=25, edgecolor="white", color="#4C72B0", alpha=0.85)
ax.set_xlabel("Qvalue (Engagement)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Engagement (Qvalue)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "target_distribution.png"), dpi=200)
plt.close(fig)
print(f"\n  ✓ Target histogram saved → outputs/target_distribution.png")


# =====================================================================
#  2. TRAIN / VALIDATION / TEST SPLIT  (70-15-15)
# =====================================================================
banner("2 · TRAIN / VALIDATION / TEST SPLIT (70-15-15)")

X = df[FEATURE_COLS].values
y = df[TARGET].values

# First split: 70 % train, 30 % temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
# Second split: 50-50 of the 30 % → 15 % val, 15 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

print(f"  Training set   : {X_train.shape[0]:>4d} observations  ({X_train.shape[0]/len(y)*100:.1f} %)")
print(f"  Validation set : {X_val.shape[0]:>4d} observations  ({X_val.shape[0]/len(y)*100:.1f} %)")
print(f"  Test set       : {X_test.shape[0]:>4d} observations  ({X_test.shape[0]/len(y)*100:.1f} %)")
print(f"  Total          : {len(y):>4d}")


# ── helper: evaluation metrics ───────────────────────────────────────
def evaluate(model, X_set, y_set, label=""):
    """Return dict of R², MAE, RMSE for a given set."""
    preds = model.predict(X_set)
    r2   = r2_score(y_set, preds)
    mae  = mean_absolute_error(y_set, preds)
    rmse = np.sqrt(mean_squared_error(y_set, preds))
    return {"set": label, "R²": r2, "MAE": mae, "RMSE": rmse}


def print_metrics(m: dict) -> None:
    print(f"    {m['set']:>12s}  │  R² = {m['R²']:+.4f}  │  MAE = {m['MAE']:>10,.1f}  │  RMSE = {m['RMSE']:>10,.1f}")


# =====================================================================
#  3 & 4. MODEL SELECTION & VALIDATION LOOP
# =====================================================================
banner("3 · MODEL TRAINING — Random Forest (defaults)")

rf_default = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
rf_default.fit(X_train, y_train)

for s, Xs, ys in [("Train", X_train, y_train), ("Validation", X_val, y_val)]:
    print_metrics(evaluate(rf_default, Xs, ys, s))

# ── feature importances ──────────────────────────────────────────────
print("\n  Feature importances (Random Forest — default):")
rf_imp = rf_default.feature_importances_
for col, imp in sorted(zip(FEATURE_COLS, rf_imp), key=lambda x: -x[1]):
    print(f"    {NICE_NAMES[col]:30s}  {imp:.4f}")

# ── 5-fold CV robustness ─────────────────────────────────────────────
cv_r2 = cross_val_score(rf_default, X_train, y_train, cv=5, scoring="r2")
print(f"\n  5-fold CV on training set  →  mean R² = {cv_r2.mean():.4f}  (± {cv_r2.std():.4f})")

# ── tuning Random Forest ─────────────────────────────────────────────
banner("4a · HYPERPARAMETER TUNING — Random Forest")

rf_param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    refit=True,
)
rf_grid.fit(X_train, y_train)

print(f"  Best CV R²     : {rf_grid.best_score_:.4f}")
print(f"  Best params    : {rf_grid.best_params_}")

rf_best = rf_grid.best_estimator_
for s, Xs, ys in [("Train", X_train, y_train), ("Validation", X_val, y_val)]:
    print_metrics(evaluate(rf_best, Xs, ys, s))

print("\n  Tuned feature importances (Random Forest):")
rf_imp_tuned = rf_best.feature_importances_
for col, imp in sorted(zip(FEATURE_COLS, rf_imp_tuned), key=lambda x: -x[1]):
    print(f"    {NICE_NAMES[col]:30s}  {imp:.4f}")

cv_r2_tuned = cross_val_score(rf_best, X_train, y_train, cv=5, scoring="r2")
print(f"\n  5-fold CV (tuned)  →  mean R² = {cv_r2_tuned.mean():.4f}  (± {cv_r2_tuned.std():.4f})")


# ── Gradient Boosting ────────────────────────────────────────────────
banner("3 · MODEL TRAINING — Gradient Boosting (defaults)")

gb_default = GradientBoostingRegressor(random_state=RANDOM_STATE)
gb_default.fit(X_train, y_train)

for s, Xs, ys in [("Train", X_train, y_train), ("Validation", X_val, y_val)]:
    print_metrics(evaluate(gb_default, Xs, ys, s))

print("\n  Feature importances (Gradient Boosting — default):")
gb_imp = gb_default.feature_importances_
for col, imp in sorted(zip(FEATURE_COLS, gb_imp), key=lambda x: -x[1]):
    print(f"    {NICE_NAMES[col]:30s}  {imp:.4f}")

cv_r2_gb = cross_val_score(gb_default, X_train, y_train, cv=5, scoring="r2")
print(f"\n  5-fold CV on training set  →  mean R² = {cv_r2_gb.mean():.4f}  (± {cv_r2_gb.std():.4f})")

# ── tuning Gradient Boosting ─────────────────────────────────────────
banner("4b · HYPERPARAMETER TUNING — Gradient Boosting")

gb_param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 4, 8],
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    refit=True,
)
gb_grid.fit(X_train, y_train)

print(f"  Best CV R²     : {gb_grid.best_score_:.4f}")
print(f"  Best params    : {gb_grid.best_params_}")

gb_best = gb_grid.best_estimator_
for s, Xs, ys in [("Train", X_train, y_train), ("Validation", X_val, y_val)]:
    print_metrics(evaluate(gb_best, Xs, ys, s))

print("\n  Tuned feature importances (Gradient Boosting):")
gb_imp_tuned = gb_best.feature_importances_
for col, imp in sorted(zip(FEATURE_COLS, gb_imp_tuned), key=lambda x: -x[1]):
    print(f"    {NICE_NAMES[col]:30s}  {imp:.4f}")

cv_r2_gb_tuned = cross_val_score(gb_best, X_train, y_train, cv=5, scoring="r2")
print(f"\n  5-fold CV (tuned)  →  mean R² = {cv_r2_gb_tuned.mean():.4f}  (± {cv_r2_gb_tuned.std():.4f})")


# =====================================================================
#  5. FINAL TEST EVALUATION
# =====================================================================
banner("5 · FINAL TEST EVALUATION")

# Pick the better model based on validation R²
rf_val_r2 = evaluate(rf_best, X_val, y_val, "RF-Val")["R²"]
gb_val_r2 = evaluate(gb_best, X_val, y_val, "GB-Val")["R²"]

if rf_val_r2 >= gb_val_r2:
    best_model = rf_best
    best_name = "Random Forest (tuned)"
else:
    best_model = gb_best
    best_name = "Gradient Boosting (tuned)"

print(f"  Selected model : {best_name}")
print(f"    (RF validation R² = {rf_val_r2:.4f}  |  GB validation R² = {gb_val_r2:.4f})\n")

# Final test — ONLY ONCE
test_metrics = evaluate(best_model, X_test, y_test, "TEST")
print_metrics(evaluate(best_model, X_train, y_train, "Train"))
print_metrics(evaluate(best_model, X_val, y_val, "Validation"))
print_metrics(test_metrics)

# ── interpretive note ────────────────────────────────────────────────
print("\n  Interpretation note:")
if test_metrics["R²"] < 0.05:
    print(textwrap.fill(
        "  The model exhibits negligible explanatory power on held-out data "
        "(R² ≈ 0). This suggests that the eight coded binary features, taken "
        "together, do not account for a meaningful share of the variance in "
        "TikTok engagement (Qvalue). This is a substantively valid finding: "
        "engagement on TikTok may be driven primarily by factors outside the "
        "scope of this content-level coding scheme (e.g., algorithmic "
        "recommendation, posting time, trending audio, or audience network "
        "effects). The absence of predictive power is reported transparently "
        "and does not undermine the descriptive contribution of the content "
        "analysis.", width=72, subsequent_indent="  "))
elif test_metrics["R²"] < 0.15:
    print(textwrap.fill(
        "  The model shows weak but non-trivial predictive power. Given the "
        "small sample size (N ≈ 123), eight binary features, and the inherent "
        "noise in social-media engagement, this level of explained variance is "
        "consistent with an exploratory analysis. Results should be interpreted "
        "cautiously and treated as suggestive rather than confirmatory.", width=72,
        subsequent_indent="  "))
else:
    print(textwrap.fill(
        "  The model explains a moderate share of engagement variance. While "
        "encouraging, the small sample size warrants caution; overfitting "
        "cannot be ruled out entirely. Feature importances should be "
        "interpreted as exploratory indicators, not causal effects.", width=72,
        subsequent_indent="  "))


# =====================================================================
#  6. INTERPRETATION & OUTPUTS
# =====================================================================
banner("6 · SAVING VISUALISATIONS & TABLES")

# ── 6a. Feature importance bar chart (side by side) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

nice_labels = [NICE_NAMES[c] for c in FEATURE_COLS]

for ax, importances, title, color in [
    (axes[0], rf_imp_tuned, "Random Forest (tuned)", "#4C72B0"),
    (axes[1], gb_imp_tuned, "Gradient Boosting (tuned)", "#DD8452"),
]:
    order = np.argsort(importances)
    ax.barh(np.array(nice_labels)[order], importances[order], color=color, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)

fig.suptitle("Which Content Features Best Predict TikTok Engagement (Qvalue)?",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "feature_importance_comparison.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("  ✓ feature_importance_comparison.png")

# ── 6b. Model comparison table ──────────────────────────────────────
rows = []
for name, model in [("Random Forest (tuned)", rf_best), ("Gradient Boosting (tuned)", gb_best)]:
    for s_label, Xs, ys in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        m = evaluate(model, Xs, ys, s_label)
        rows.append({"Model": name, "Set": s_label, "R²": m["R²"], "MAE": m["MAE"], "RMSE": m["RMSE"]})

comp_df = pd.DataFrame(rows)
comp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False, float_format="%.4f")
print("  ✓ model_comparison.csv")

# Pretty-print the table
print("\n  Model Comparison Table")
print("  " + "-" * 78)
print(f"  {'Model':35s} {'Set':12s} {'R²':>8s} {'MAE':>12s} {'RMSE':>12s}")
print("  " + "-" * 78)
for _, r in comp_df.iterrows():
    print(f"  {r['Model']:35s} {r['Set']:12s} {r['R²']:>+8.4f} {r['MAE']:>12,.1f} {r['RMSE']:>12,.1f}")
print("  " + "-" * 78)

# ── 6c. Partial dependence plots (top 3 features of best model) ─────
if best_model == rf_best:
    best_imp = rf_imp_tuned
else:
    best_imp = gb_imp_tuned

top3_idx = np.argsort(best_imp)[-3:][::-1]
top3_names = [FEATURE_COLS[i] for i in top3_idx]
top3_nice  = [NICE_NAMES[n] for n in top3_names]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Build a DataFrame for the display function
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)

for i, (feat, nice) in enumerate(zip(top3_names, top3_nice)):
    PartialDependenceDisplay.from_estimator(
        best_model, X_train_df, [feat], ax=axes[i],
        kind="average", grid_resolution=2,
    )
    axes[i].set_title(f"PDP: {nice}", fontsize=11)
    axes[i].set_xlabel(nice)

fig.suptitle(f"Partial Dependence — Top 3 Features ({best_name})",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "partial_dependence_top3.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("  ✓ partial_dependence_top3.png")

# ── 6d. Predicted vs. actual scatter plot ────────────────────────────
y_pred_test = best_model.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred_test, alpha=0.7, edgecolors="white", s=60, color="#4C72B0")
lo = min(y_test.min(), y_pred_test.min()) * 0.9
hi = max(y_test.max(), y_pred_test.max()) * 1.1
ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Qvalue")
ax.set_ylabel("Predicted Qvalue")
ax.set_title(f"Predicted vs. Actual — {best_name}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "predicted_vs_actual.png"), dpi=200)
plt.close(fig)
print("  ✓ predicted_vs_actual.png")


# =====================================================================
#  SUMMARY
# =====================================================================
banner("PIPELINE COMPLETE")
print(f"""
  All outputs saved to:  {OUT_DIR}

  Files produced:
    1. target_distribution.png          — histogram of Qvalue
    2. feature_importance_comparison.png — side-by-side RF vs GB importances
    3. model_comparison.csv             — train / val / test metrics table
    4. partial_dependence_top3.png      — PDP for top-3 features
    5. predicted_vs_actual.png          — scatter plot (best model)

  Best model selected : {best_name}
  Test R²             : {test_metrics['R²']:.4f}
  Test MAE            : {test_metrics['MAE']:,.1f}
  Test RMSE           : {test_metrics['RMSE']:,.1f}
""")
