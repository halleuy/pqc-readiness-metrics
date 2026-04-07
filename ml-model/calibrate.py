# calibrate.py
# Learn linear regression mapping from raw NLP cosine similarities to expert 0-5 scores.
#
# Run AFTER:
#   1. expert_scores.csv exists (with your manual scores)
#   2. main.py has been run at least once (to generate nlp_raw_scores.csv)
#   3. framework_mapping.csv exists
#
# Outputs:
#   - calibration_params.json  (loaded by main.py on next run)
#   - calibration_params.csv   (human-readable summary)
#   - calibration_summary.txt  (readable report)
#
# Then re-run main.py — it will auto-load the calibration.

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr
import os
import json
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DIMENSIONS = [
    "crypto_assets", "crypto_agility",
    "migration_planning", "risk_management",
    "standards_compliance"
]

def safe_spearmanr(a, b):
    """Compute Spearman rho, returning 0.0 if inputs are constant."""
    try:
        if len(set(a)) < 2 or len(set(b)) < 2:
            return 0.0, 1.0
        rho, p = spearmanr(a, b)
        return float(rho), float(p)
    except:
        return 0.0, 1.0

def safe_kappa(a, b):
    """Compute quadratic weighted kappa, returning 0.0 on failure."""
    try:
        if len(set(a)) < 2 or len(set(b)) < 2:
            return 0.0
        return float(cohen_kappa_score(a, b, weights='quadratic'))
    except:
        return 0.0

# ──────────────────────────────────────────────
# 1. LOAD FRAMEWORK MAPPING
# ──────────────────────────────────────────────

mapping_path = os.path.join(SCRIPT_DIR, 'framework_mapping.csv')

if not os.path.exists(mapping_path):
    print(f"❌ framework_mapping.csv not found at: {mapping_path}")
    exit(1)

mapping_df = pd.read_csv(mapping_path)
# Build lookup: PDF filename → integer ID
filename_to_id = {}
for _, row in mapping_df.iterrows():
    pdf_title = row['PDF_Title'].strip()
    framework_id = int(row['Framework ID'])
    filename_to_id[pdf_title] = framework_id

print(f"✅ Loaded framework mapping ({len(filename_to_id)} frameworks)")

# ──────────────────────────────────────────────
# 2. LOAD & MERGE DATA
# ──────────────────────────────────────────────

expert_path = os.path.join(PROJECT_DIR, 'freq-analysis', 'scripts', 'labels.csv')
raw_path = os.path.join(SCRIPT_DIR, 'nlp_raw_scores.csv')

# Check files exist
if not os.path.exists(expert_path):
    print(f"❌ Expert scores not found at: {expert_path}")
    exit(1)
if not os.path.exists(raw_path):
    print(f"❌ Raw NLP scores not found at: {raw_path}")
    print("   Run main.py first to generate nlp_raw_scores.csv")
    exit(1)

expert_df = pd.read_csv(expert_path)
raw_df = pd.read_csv(raw_path)

# If nlp_raw_scores.csv already has Framework_ID (from updated main.py), use it directly.
# Otherwise, map filenames to IDs using the mapping file.
if 'Framework_ID' not in raw_df.columns:
    raw_df['Framework_ID'] = raw_df['Framework'].apply(
        lambda f: filename_to_id.get(f.strip())
    )

# Show mapping results for debugging
print("\n📎 Filename → ID mapping:")
for _, row in raw_df.iterrows():
    fid = row.get('Framework_ID')
    status = "✅" if pd.notna(fid) else "❌ UNMAPPED"
    fname = str(row['Framework'])[:65]
    print(f"   {status} {fname}... → ID {int(fid) if pd.notna(fid) else '?'}")

raw_df = raw_df.dropna(subset=['Framework_ID'])
raw_df['Framework_ID'] = raw_df['Framework_ID'].astype(int)

# Ensure expert_df Framework_ID is also integer
expert_df['Framework_ID'] = expert_df['Framework_ID'].astype(int)

# Merge on Framework_ID (only frameworks that have BOTH expert scores and NLP scores)
merged = expert_df.merge(raw_df, on='Framework_ID', how='inner')

# Reset index so integer indexing works correctly in the loop
merged = merged.reset_index(drop=True)

print(f"\n✅ Matched {len(merged)} frameworks for calibration")
print(f"   Expert IDs:  {sorted(expert_df['Framework_ID'].tolist())}")
print(f"   NLP IDs:     {sorted(raw_df['Framework_ID'].tolist())}")
print(f"   Matched IDs: {sorted(merged['Framework_ID'].tolist())}\n")

if len(merged) < 4:
    print("❌ Need at least 4 matched frameworks. Check:")
    print("   - expert_scores.csv has 'Framework_ID' column with integers")
    print("   - nlp_raw_scores.csv has matching Framework_ID or filenames")
    print("   - framework_mapping.csv maps correctly")
    exit(1)

# ──────────────────────────────────────────────
# 3. DIAGNOSTIC: RAW DISTRIBUTIONS
# ──────────────────────────────────────────────

print("=" * 65)
print("📊 RAW SIMILARITY vs EXPERT SCORES")
print("=" * 65)

for dim in DIMENSIONS:
    raw_col = f"{dim}_raw"
    print(f"\n  {dim}:")
    print(f"    {'ID':>4} | {'Raw':>7} | {'Expert':>6}")
    print(f"    {'-'*4}-+-{'-'*7}-+-{'-'*6}")

    for _, row in merged.sort_values(raw_col).iterrows():
        print(f"    {int(row['Framework_ID']):>4} | {row[raw_col]:>7.4f} | {int(row[dim]):>6}")

    # Check rank correlation
    rho, p = safe_spearmanr(merged[raw_col].values, merged[dim].values)
    print(f"    Spearman ρ = {rho:.3f} (p={p:.3f})")

# ──────────────────────────────────────────────
# 4. LINEAR REGRESSION CALIBRATION
# ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("🎯 LINEAR REGRESSION CALIBRATION (per dimension)")
print("=" * 65)

calibration_params = {}
loo = LeaveOneOut()

overall_expert = []
overall_default = []
overall_calibrated = []
overall_loo = []

for dim in DIMENSIONS:
    raw_col = f"{dim}_raw"
    X = merged[[raw_col]].values
    y = merged[dim].values.astype(int)

    # ---- Default thresholds prediction (what original code produces) ----
    default_preds = np.digitize(X.flatten(), [0.2, 0.4, 0.6, 0.8]) + 1
    default_preds = np.clip(default_preds, 1, 5)

    # ---- Fit Ridge regression (regularised to prevent overfitting) ----
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Extract scalar values from numpy arrays
    slope = model.coef_.item()
    intercept = model.intercept_.item() if hasattr(model.intercept_, 'item') else float(model.intercept_)

    # Calibrated predictions (clipped to 0-5, rounded to integer)
    cal_preds_raw = model.predict(X)
    cal_preds = np.clip(np.round(cal_preds_raw), 0, 5).astype(int)

    # ---- Leave-One-Out cross-validation ----
    loo_preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        loo_model = Ridge(alpha=1.0)
        loo_model.fit(X[train_idx], y[train_idx])
        loo_preds[test_idx] = loo_model.predict(X[test_idx])

    loo_preds_rounded = np.clip(np.round(loo_preds), 0, 5).astype(int)

    # ---- Compute metrics ----
    default_mae = mean_absolute_error(y, default_preds)
    cal_mae = mean_absolute_error(y, cal_preds)
    loo_mae = mean_absolute_error(y, loo_preds_rounded)

    rho_cal, _ = safe_spearmanr(y, cal_preds)
    rho_loo, _ = safe_spearmanr(y, loo_preds_rounded)
    kappa_cal = safe_kappa(y, cal_preds)

    exact_default = np.mean(y == default_preds) * 100
    exact_cal = np.mean(y == cal_preds) * 100
    exact_loo = np.mean(y == loo_preds_rounded) * 100

    # ---- Print results ----
    print(f"\n  {dim}:")
    print(f"    Learned equation: score = {slope:.2f} × raw + {intercept:.2f}")
    print(f"    ")
    print(f"    {'Metric':<20} {'Default':>8} {'Calibrated':>11} {'LOO':>8}")
    print(f"    {'-'*20} {'-'*8} {'-'*11} {'-'*8}")
    print(f"    {'MAE':<20} {default_mae:>8.2f} {cal_mae:>11.2f} {loo_mae:>8.2f}")
    print(f"    {'Exact Match %':<20} {exact_default:>7.0f}% {exact_cal:>10.0f}% {exact_loo:>7.0f}%")
    print(f"    {'Spearman ρ':<20} {'—':>8} {rho_cal:>11.3f} {rho_loo:>8.3f}")
    print(f"    {'Quad. Kappa':<20} {'—':>8} {kappa_cal:>11.3f} {'—':>8}")

    # Per-framework breakdown
    print(f"\n    {'ID':>4} | {'Raw':>7} | {'Default':>7} | {'Calibrated':>10} | {'LOO':>5} | {'Expert':>6}")
    print(f"    {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*5}-+-{'-'*6}")

    for i, (_, row) in enumerate(merged.iterrows()):
        fid = int(row['Framework_ID'])
        r = row[raw_col]
        d = int(default_preds[i])
        c = int(cal_preds[i])
        l = int(loo_preds_rounded[i])
        e = int(row[dim])
        match_c = "✅" if c == e else f"(off {abs(c-e)})"
        match_l = "✅" if l == e else f"(off {abs(l-e)})"
        print(f"    {fid:>4} | {r:>7.4f} | {d:>7} | {c:>10} {match_c:<8} | {l:>5} {match_l:<8} | {e:>6}")

    # Store calibration parameters
    calibration_params[dim] = {
        'slope': round(slope, 6),
        'intercept': round(intercept, 6),
        'cal_mae': round(float(cal_mae), 3),
        'loo_mae': round(float(loo_mae), 3),
        'spearman_rho': round(rho_cal, 3)
    }

    overall_expert.extend(y.tolist())
    overall_default.extend(default_preds.tolist())
    overall_calibrated.extend(cal_preds.tolist())
    overall_loo.extend(loo_preds_rounded.tolist())

# ──────────────────────────────────────────────
# 5. OVERALL SUMMARY
# ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("📈 OVERALL RESULTS")
print("=" * 65)

overall_expert = np.array(overall_expert)
overall_default = np.array(overall_default)
overall_calibrated = np.array(overall_calibrated)
overall_loo = np.array(overall_loo)

print(f"\n  {'Metric':<25} {'Default':>8} {'Calibrated':>11} {'LOO':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*11} {'-'*8}")
print(f"  {'MAE':<25} {mean_absolute_error(overall_expert, overall_default):>8.2f} "
      f"{mean_absolute_error(overall_expert, overall_calibrated):>11.2f} "
      f"{mean_absolute_error(overall_expert, overall_loo):>8.2f}")
print(f"  {'Exact Match %':<25} {np.mean(overall_expert == overall_default)*100:>7.0f}% "
      f"{np.mean(overall_expert == overall_calibrated)*100:>10.0f}% "
      f"{np.mean(overall_expert == overall_loo)*100:>7.0f}%")

rho_def, _ = safe_spearmanr(overall_expert, overall_default)
rho_cal, _ = safe_spearmanr(overall_expert, overall_calibrated)
rho_loo, _ = safe_spearmanr(overall_expert, overall_loo)
print(f"  {'Spearman ρ':<25} {rho_def:>8.3f} {rho_cal:>11.3f} {rho_loo:>8.3f}")

improvement = (mean_absolute_error(overall_expert, overall_default) -
               mean_absolute_error(overall_expert, overall_calibrated))
print(f"\n  MAE improvement over default: {improvement:+.2f}")

loo_mae_overall = mean_absolute_error(overall_expert, overall_loo)
if loo_mae_overall < 1.0:
    print(f"  ✅ LOO MAE = {loo_mae_overall:.2f} (< 1.0 — good generalisation)")
else:
    print(f"  ⚠️  LOO MAE = {loo_mae_overall:.2f} (≥ 1.0 — consider adding more data or features)")

# ──────────────────────────────────────────────
# 6. SAVE CALIBRATION PARAMETERS
# ──────────────────────────────────────────────

# Save as JSON (loaded by main.py)
json_path = os.path.join(SCRIPT_DIR, 'calibration_params.json')
with open(json_path, 'w') as f:
    json.dump(calibration_params, f, indent=2)
print(f"\n💾 Saved calibration JSON to: {json_path}")

# Save as CSV (human-readable summary)
cal_rows = []
for dim, params in calibration_params.items():
    cal_rows.append({'dimension': dim, **params})

cal_df = pd.DataFrame(cal_rows)
cal_path = os.path.join(SCRIPT_DIR, 'calibration_params.csv')
cal_df.to_csv(cal_path, index=False)
print(f"💾 Saved calibration CSV to: {cal_path}")

# Save readable summary
summary_path = os.path.join(SCRIPT_DIR, 'calibration_summary.txt')
with open(summary_path, 'w') as f:
    f.write("PQC NLP Scorer — Calibration Summary\n")
    f.write("=" * 45 + "\n")
    f.write(f"Trained on {len(merged)} frameworks\n")
    f.write(f"Overall LOO MAE: {loo_mae_overall:.2f}\n\n")

    for dim, params in calibration_params.items():
        f.write(f"{dim}:\n")
        f.write(f"  Equation:  score = {params['slope']:.2f} × raw + {params['intercept']:.2f}\n")
        f.write(f"  Train MAE: {params['cal_mae']}\n")
        f.write(f"  LOO MAE:   {params['loo_mae']}\n")
        f.write(f"  Spearman:  {params['spearman_rho']}\n\n")

print(f"💾 Saved summary to: {summary_path}")

print("\n🎉 Done! Now re-run main.py — it will auto-load calibration_params.json")
