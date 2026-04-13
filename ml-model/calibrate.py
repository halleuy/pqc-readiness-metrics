import os
import csv
import json
import numpy as np
import warnings
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score

# --- Paths ---
ML_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERT_FILE = os.path.join(ML_DIR, '..', 'freq-analysis', 'scripts', 'labels.csv')
RAW_FILE = os.path.join(ML_DIR, 'nlp_raw_scores.csv')
DETAIL_FILE = os.path.join(ML_DIR, 'nlp_detailed_features.csv')

DIMENSIONS = ['crypto_assets', 'crypto_agility', 'migration_planning',
              'risk_management', 'standards_compliance']

# All 9 individual signals (excluding 'combined' which is derived)
SIGNALS = ['max_sim', 'top_k_mean', 'coverage_log', 'depth',
           'concentration', 'differential',
           'kw_specific', 'kw_general', 'kw_ratio']

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_expert(path):
    """Return {framework_id_str: {dim: int_score}}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fid = str(row.get('Framework_ID', '')).strip()
            try:
                fid = str(int(fid)).zfill(3)
            except ValueError:
                pass
            if fid:
                data[fid] = {d: int(row[d]) for d in DIMENSIONS}
    print(f"    Loaded {len(data)} expert scores: {sorted(data.keys())}")
    return data

def load_raw(path):
    """Return {framework_id_str: {dim_raw: float}}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fid = str(row.get('Framework_ID', '')).strip()
            if fid and fid != 'Unknown':
                data[fid] = {}
                for d in DIMENSIONS:
                    data[fid][f'{d}_raw'] = float(row[f'{d}_raw'])
    print(f"    Loaded {len(data)} NLP raw scores: {sorted(data.keys())}")
    return data

def load_detailed(path):
    """Return {framework_id_str: {dim_signal: float}}"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fid = str(row.get('Framework_ID', '')).strip()
            if fid and fid != 'Unknown':
                feats = {}
                for d in DIMENSIONS:
                    for s in SIGNALS:
                        key = f'{d}_{s}'
                        feats[key] = float(row.get(key, 0.0))
                data[fid] = feats
    print(f"    Loaded {len(data)} detailed feature sets: {sorted(data.keys())}")
    return data

# =============================================================================
# LEAVE-ONE-OUT HELPER
# =============================================================================

def loo_predict(X, y, alpha=1.0):
    """LOO Ridge Regression. Returns plain Python list of floats."""
    n = len(y)
    preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = Ridge(alpha=alpha)
        model.fit(X[mask], y[mask])
        raw = model.predict(X[~mask])
        val = raw.ravel()
        p = float(val.item()) if hasattr(val, 'item') else float(val)
        p = max(0.0, min(5.0, p))
        preds.append(p)
    return preds

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, actuals, label=""):
    """Compute and print MAE, exact-match, Spearman ρ with p-value, kappa."""
    preds_list = [x.item() if hasattr(x, 'item') else float(x) for x in preds]
    actuals_list = [x.item() if hasattr(x, 'item') else float(x) for x in actuals]

    preds_arr = np.array(preds_list)
    actuals_arr = np.array(actuals_list)

    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    rounded = np.round(preds_arr).astype(int)
    actual_int = actuals_arr.astype(int)
    exact = float(np.mean(rounded == actual_int))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(set(preds_list)) > 1 and len(set(actuals_list)) > 1:
            rho, p_val = spearmanr(preds_list, actuals_list)
        else:
            rho, p_val = 0.0, 1.0

    try:
        kappa = cohen_kappa_score(actual_int, rounded, weights='quadratic')
    except Exception:
        kappa = 0.0

    if np.isnan(rho):
        rho = 0.0
    if np.isnan(p_val):
        p_val = 1.0

    if label:
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
        print(f"    MAE: {mae:.3f}  Exact: {exact:.0%}  ρ: {rho:.3f} (p={p_val:.4f} {sig})  κ: {kappa:.3f}")
        print(f"    Preds:   {[round(p, 1) for p in preds_list]}")
        print(f"    Actuals: {[int(a) for a in actuals_list]}")

    return {'mae': mae, 'exact': exact, 'rho': rho, 'p_val': p_val, 'kappa': kappa}

# =============================================================================
# PHASE 1A — SINGLE-FEATURE RIDGE (combined raw → score)
# =============================================================================

def phase1a_single_feature(nlp_raw, expert, alpha=1.0):
    common = sorted(set(nlp_raw) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1A — SINGLE-FEATURE RIDGE  (LOO, n={n})")
    print(f"{'=' * 70}")

    if n < 3:
        print("  ❌ Need ≥ 3 overlapping frameworks.")
        return None

    all_mae = []
    cal_params = {}

    for dim in DIMENSIONS:
        X = np.array([[nlp_raw[fid][f'{dim}_raw']] for fid in common], dtype=float)
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        print(f"\n  {dim}:")
        preds = loo_predict(X, y, alpha=alpha)
        m = compute_metrics(preds, y, label=dim)

        full = Ridge(alpha=alpha).fit(X, y)
        slope = float(full.coef_.ravel().item())
        intercept = float(full.intercept_.item()) if hasattr(full.intercept_, 'item') else float(full.intercept_)
        print(f"    Equation: score = {slope:.2f} × raw + ({intercept:.2f})")

        cal_params[dim] = {'slope': slope, 'intercept': intercept}
        all_mae.append(m['mae'])

    overall_mae = float(np.mean(all_mae))
    print(f"\n  ── Overall Phase 1A ──")
    print(f"  MAE: {overall_mae:.3f}  {'✅' if overall_mae < 1.0 else '❌'}  (target < 1.0)")

    path = os.path.join(ML_DIR, 'calibration_params.json')
    with open(path, 'w') as f:
        json.dump(cal_params, f, indent=2)
    print(f"  💾 Saved {path}")

    return overall_mae

# =============================================================================
# PHASE 1B — MULTI-FEATURE RIDGE (9 signals per dimension)
# =============================================================================

def phase1b_multi_feature(detail, expert, alphas=None):
    """Try multiple alpha values and report best."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 50.0, 100.0]

    common = sorted(set(detail) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1B — MULTI-FEATURE RIDGE  (LOO, n={n}, {len(SIGNALS)} features/dim)")
    print(f"  Testing alphas: {alphas}")
    print(f"{'=' * 70}")

    best_overall_mae = 999
    best_alpha = alphas

    for alpha in alphas:
        dim_maes = []
        for dim in DIMENSIONS:
            X = np.array([[detail[fid][f'{dim}_{s}'] for s in SIGNALS] for fid in common], dtype=float)
            y = np.array([expert[fid][dim] for fid in common], dtype=float)
            preds = loo_predict(X, y, alpha=alpha)
            mae = float(np.mean(np.abs(np.array(preds) - y)))
            dim_maes.append(mae)
        overall = float(np.mean(dim_maes))
        print(f"    Alpha={alpha:<6} → MAE={overall:.3f}")
        if overall < best_overall_mae:
            best_overall_mae = overall
            best_alpha = alpha

    # Run best alpha with full output
    print(f"\n  Best alpha: {best_alpha} (MAE={best_overall_mae:.3f})")
    print(f"  Detailed results:\n")

    all_mae = []
    for dim in DIMENSIONS:
        X = np.array([[detail[fid][f'{dim}_{s}'] for s in SIGNALS] for fid in common], dtype=float)
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        print(f"  {dim}:")
        preds = loo_predict(X, y, alpha=best_alpha)
        m = compute_metrics(preds, y, label=dim)

        full = Ridge(alpha=best_alpha).fit(X, y)
        weights = {}
        for j, s in enumerate(SIGNALS):
            w = full.coef_.ravel()[j]
            weights[s] = round(float(w.item()) if hasattr(w, 'item') else float(w), 3)
        print(f"    Weights: {weights}")
        all_mae.append(m['mae'])

    overall_mae = float(np.mean(all_mae))
    print(f"\n  ── Overall Phase 1B ──")
    print(f"  MAE: {overall_mae:.3f}  {'✅' if overall_mae < 1.0 else '❌'}  (target < 1.0)")

    return overall_mae

# =============================================================================
# PHASE 1C — POOLED MULTI-FEATURE RIDGE
# =============================================================================

def phase1c_pooled(detail, expert, alphas=None):
    """Train ONE model across all dimensions with alpha tuning."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 50.0, 100.0]

    common = sorted(set(detail) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1C — POOLED RIDGE  (LOO by framework, n={n})")
    print(f"  Testing alphas: {alphas}")
    print(f"{'=' * 70}")

    # Build pooled dataset
    X_all, y_all, ids_all, dims_all = [], [], [], []
    for fid in common:
        for dim in DIMENSIONS:
            features = [detail[fid][f'{dim}_{s}'] for s in SIGNALS]
            X_all.append(features)
            y_all.append(expert[fid][dim])
            ids_all.append(fid)
            dims_all.append(dim)

    X_all = np.array(X_all, dtype=float)
    y_all = np.array(y_all, dtype=float)
    N = len(y_all)

    print(f"  Total samples: {N} ({n} frameworks × {len(DIMENSIONS)} dims)")

    best_mae = 999
    best_alpha = alphas

    for alpha in alphas:
        all_preds = np.zeros(N)
        for held_out_fid in common:
            mask = np.array([fid == held_out_fid for fid in ids_all])
            model = Ridge(alpha=alpha)
            model.fit(X_all[~mask], y_all[~mask])
            raw_preds = model.predict(X_all[mask]).ravel()
            for j, idx in enumerate(np.where(mask)):
                p = float(raw_preds[j].item()) if hasattr(raw_preds[j], 'item') else float(raw_preds[j])
                all_preds[idx] = max(0.0, min(5.0, p))

        mae = float(np.mean(np.abs(all_preds - y_all)))
        print(f"    Alpha={alpha:<6} → MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_preds = all_preds.copy()

    # Detailed output for best alpha
    print(f"\n  Best alpha: {best_alpha} (MAE={best_mae:.3f})")

    preds_list = [float(x) for x in best_preds]
    actuals_list = [float(x) for x in y_all]

    rounded = np.round(best_preds).astype(int)
    actual_int = y_all.astype(int)
    exact = float(np.mean(rounded == actual_int))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(set(preds_list)) > 1 and len(set(actuals_list)) > 1:
            rho, p_val = spearmanr(preds_list, actuals_list)
        else:
            rho, p_val = 0.0, 1.0
    if np.isnan(rho):
        rho = 0.0

    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
    print(f"\n  MAE:   {best_mae:.3f}  {'✅' if best_mae < 1.0 else '❌'}")
    print(f"  Exact: {exact:.0%}")
    print(f"  ρ:     {rho:.3f} (p={p_val:.4f} {sig})")

    # Per-dimension breakdown
    print(f"\n  Per-dimension breakdown:")
    for dim in DIMENSIONS:
        dim_mask = np.array([d == dim for d in dims_all])
        dim_preds = best_preds[dim_mask]
        dim_actuals = y_all[dim_mask]
        dim_mae = float(np.mean(np.abs(dim_preds - dim_actuals)))
        print(f"    {dim}: MAE = {dim_mae:.3f}")
        print(f"      Preds:   {[round(float(x), 1) for x in dim_preds]}")
        print(f"      Actuals: {[int(x) for x in dim_actuals]}")

    # Feature weights
    full = Ridge(alpha=best_alpha).fit(X_all, y_all)
    weights = {}
    for j, s in enumerate(SIGNALS):
        w = full.coef_.ravel()[j]
        weights[s] = round(float(w.item()) if hasattr(w, 'item') else float(w), 3)
    inter = float(full.intercept_.item()) if hasattr(full.intercept_, 'item') else float(full.intercept_)
    print(f"\n  Feature weights (pooled): {weights}")
    print(f"  Intercept: {inter:.2f}")

    return best_mae

# =============================================================================
# SUMMARY
# =============================================================================

def save_summary(results, n):
    path = os.path.join(ML_DIR, 'calibration_summary.txt')
    with open(path, 'w') as f:
        f.write("PQC Readiness — Calibration Summary\n")
        f.write("=" * 50 + "\n\n")
        for label, mae in results.items():
            if mae is not None:
                status = "PASS" if mae < 1.0 else "FAIL"
                f.write(f"{label}: MAE = {mae:.3f}  {status}\n")
        f.write(f"\nExpert-scored frameworks used: {n}\n")
        f.write(f"\nTargets:\n")
        f.write(f"  Phase 1: MAE < 1.0\n")
        f.write(f"  Phase 2: MAE < 0.8 (Random Forest + frequency features)\n")
        f.write(f"  Phase 3: MAE < 0.6 (LLM comparison)\n")
    print(f"\n  💾 Saved {path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  PQC READINESS — CALIBRATION PIPELINE (v3 — Hybrid NLP+Keywords)")
    print("=" * 70)

    print("\n  Loading data...")
    expert = load_expert(EXPERT_FILE)
    nlp_raw = load_raw(RAW_FILE)

    common = sorted(set(nlp_raw) & set(expert))
    print(f"    Overlap: {len(common)} — IDs: {common}")

    if len(common) < 3:
        print("\n  ❌ Not enough overlap. Check Framework_IDs.")
        print("     Expert IDs:", sorted(expert.keys()))
        print("     NLP IDs:   ", sorted(nlp_raw.keys()))
        return

    results = {}

    # Phase 1A
    results['Phase 1A (single-feature Ridge)'] = phase1a_single_feature(nlp_raw, expert)

    # Phase 1B + 1C
    if os.path.exists(DETAIL_FILE):
        detail = load_detailed(DETAIL_FILE)
        detail_common = sorted(set(detail) & set(expert))
        if len(detail_common) >= 3:
            results['Phase 1B (multi-feature Ridge)'] = phase1b_multi_feature(detail, expert)
            results['Phase 1C (pooled Ridge)'] = phase1c_pooled(detail, expert)
    else:
        print(f"\n  ⚠ {DETAIL_FILE} not found — skipping Phase 1B/1C")

    save_summary(results, len(common))

    print(f"\n{'=' * 70}")
    print("  CALIBRATION COMPLETE")
    print("=" * 70)

    valid = [v for v in results.values() if v is not None]
    if valid:
        best = min(valid)
        best_name = [k for k, v in results.items() if v == best]
        print(f"\n  📊 Best MAE: {best:.3f} ({best_name})")
        if best < 1.0:
            print("  ✅ Phase 1 target met!")
            print("     Next: integrate frequency features → Random Forest for Phase 2.")
        elif best < 1.2:
            print("  🔶 Close! Consider:")
            print("     1. Add frequency_scores.csv as additional features")
            print("     2. Try Random Forest instead of Ridge")
            print("     3. Refine keyword lists based on which dimensions are worst")
        else:
            print("  ❌ Phase 1 target not yet met.")

if __name__ == '__main__':
    main()
