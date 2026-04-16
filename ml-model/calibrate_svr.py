# =============================================================================
# calibrate_svr.py — SVR-based calibration for PQC Readiness Scoring
# =============================================================================
#
# Complements calibrate.py (Ridge-based) with:
#   - SVR (RBF kernel) for nonlinear patterns
#   - Automatic feature selection (forward selection)
#   - ElasticNet comparison
#   - Hyperparameter grid search with LOO
#
# Usage:  python calibrate_svr.py
# =============================================================================

import os
import csv
import json
import numpy as np
import warnings
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score

# --- Paths ---
ML_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERT_FILE = os.path.join(ML_DIR, '..', 'freq-analysis', 'scripts', 'labels.csv')
DETAIL_FILE = os.path.join(ML_DIR, 'nlp_detailed_features.csv')

DIMENSIONS = ['crypto_assets', 'crypto_agility', 'migration_planning',
              'risk_management', 'standards_compliance']

ALL_SIGNALS = ['max_sim', 'top_k_mean', 'coverage_log', 'depth',
               'concentration', 'differential',
               'kw_specific', 'kw_general', 'kw_ratio']

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_expert(path):
    """Load expert scores from labels.csv"""
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

def load_detailed(path):
    """Load detailed NLP + keyword features"""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fid = str(row.get('Framework_ID', '')).strip()
            if fid and fid != 'Unknown':
                feats = {}
                for d in DIMENSIONS:
                    for s in ALL_SIGNALS:
                        key = f'{d}_{s}'
                        feats[key] = float(row.get(key, 0.0))
                data[fid] = feats
    print(f"    Loaded {len(data)} detailed feature sets: {sorted(data.keys())}")
    return data

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, actuals, label="", verbose=True):
    """Compute MAE, exact match, Spearman ρ, and Cohen's κ"""
    preds_list = [float(x) for x in preds]
    actuals_list = [float(x) for x in actuals]

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

    if label and verbose:
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else "ns"
        print(f"    MAE: {mae:.3f}  Exact: {exact:.0%}  ρ: {rho:.3f} (p={p_val:.4f} {sig})  κ: {kappa:.3f}")
        print(f"    Preds:   {[round(p, 1) for p in preds_list]}")
        print(f"    Actuals: {[int(a) for a in actuals_list]}")

    return {'mae': mae, 'exact': exact, 'rho': rho, 'p_val': p_val, 'kappa': kappa}

# =============================================================================
# LOO HELPERS
# =============================================================================

def loo_svr(X, y, C=1.0, epsilon=0.5, gamma='scale'):
    """LOO with SVR (RBF kernel). Scales features internally."""
    n = len(y)
    preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[~mask])

        model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train, y[mask])

        raw = model.predict(X_test)
        val = raw.ravel()
        p = float(val.item()) if hasattr(val, 'item') else float(val)
        p = max(0.0, min(5.0, p))
        preds.append(p)
    return preds

def loo_elasticnet(X, y, alpha=1.0, l1_ratio=0.5):
    """LOO with ElasticNet. Scales features internally."""
    n = len(y)
    preds = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[mask])
        X_test = scaler.transform(X[~mask])

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_train, y[mask])

        raw = model.predict(X_test)
        val = raw.ravel()
        p = float(val.item()) if hasattr(val, 'item') else float(val)
        p = max(0.0, min(5.0, p))
        preds.append(p)
    return preds


# =============================================================================
# PHASE 2A — FEATURE SELECTION (Forward Selection)
# =============================================================================

def phase2a_feature_selection(detail, expert, max_features=5):
    """Find best 3-5 features per dimension using forward selection with SVR."""
    common = sorted(set(detail) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2A — FORWARD FEATURE SELECTION  (LOO SVR, n={n})")
    print(f"{'=' * 70}")

    # Use moderate SVR params for selection
    C, epsilon, gamma = 1.0, 0.5, 'scale'
    best_features = {}

    for dim in DIMENSIONS:
        print(f"\n  {dim}:")
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        selected = []
        remaining = list(ALL_SIGNALS)
        best_mae_so_far = 999

        for step in range(max_features):  # max N features
            best_candidate = None
            best_candidate_mae = 999

            for candidate in remaining:
                trial = selected + [candidate]
                X = np.array([[detail[fid][f'{dim}_{s}'] for s in trial]
                              for fid in common], dtype=float)

                preds = loo_svr(X, y, C=C, epsilon=epsilon, gamma=gamma)
                mae = float(np.mean(np.abs(np.array(preds) - y)))

                if mae < best_candidate_mae:
                    best_candidate_mae = mae
                    best_candidate = candidate

            # Only add if it improves MAE
            if best_candidate_mae < best_mae_so_far - 0.01:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                best_mae_so_far = best_candidate_mae
                print(f"    Step {step + 1}: +{best_candidate:<16s} → MAE={best_candidate_mae:.3f}")
            else:
                print(f"    Step {step + 1}: No improvement — stopping")
                break

        if len(selected) == 0:
            # Fallback to best single feature
            selected = ['max_sim']
            print(f"    ⚠ No features improved baseline — using fallback: {selected}")

        best_features[dim] = selected
        print(f"    ✓ Selected: {selected} (MAE={best_mae_so_far:.3f})")

    return best_features

# =============================================================================
# PHASE 2B — SVR WITH SELECTED FEATURES
# =============================================================================

def phase2b_svr(detail, expert, selected_features=None):
    """SVR (RBF) with hyperparameter grid search and selected features."""
    common = sorted(set(detail) & set(expert))
    n = len(common)

    # Default features if none provided
    if selected_features is None:
        selected_features = {dim: ['max_sim', 'top_k_mean', 'concentration']
                             for dim in DIMENSIONS}

    # Hyperparameter grid — appropriate for small n
    C_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    epsilon_values = [0.1, 0.3, 0.5, 0.8]
    gamma_values = ['scale', 'auto']

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2B — SVR (RBF)  (LOO, n={n})")
    print(f"  Grid: C={C_values}, ε={epsilon_values}, γ={gamma_values}")
    print(f"{'=' * 70}")

    all_mae = []
    best_params = {}

    for dim in DIMENSIONS:
        feats = selected_features[dim]
        print(f"\n  {dim} (features: {feats}):")

        X = np.array([[detail[fid][f'{dim}_{s}'] for s in feats]
                       for fid in common], dtype=float)
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        # Grid search
        best_mae = 999
        best_C, best_eps, best_gamma = 1.0, 0.5, 'scale'

        for C in C_values:
            for eps in epsilon_values:
                for gamma in gamma_values:
                    preds = loo_svr(X, y, C=C, epsilon=eps, gamma=gamma)
                    mae = float(np.mean(np.abs(np.array(preds) - y)))
                    if mae < best_mae:
                        best_mae = mae
                        best_C, best_eps, best_gamma = C, eps, gamma

        print(f"    Best params: C={best_C}, ε={best_eps}, γ={best_gamma}")
        best_params[dim] = {'C': best_C, 'epsilon': best_eps, 'gamma': best_gamma}

        # Final run with best params
        preds = loo_svr(X, y, C=best_C, epsilon=best_eps, gamma=best_gamma)
        m = compute_metrics(preds, y, label=dim)
        all_mae.append(m['mae'])

    overall_mae = float(np.mean(all_mae))
    print(f"\n  ── Overall Phase 2B (SVR) ──")
    print(f"  MAE: {overall_mae:.3f}  {'✅' if overall_mae < 1.0 else '❌'}  (target < 1.0)")

    return overall_mae, best_params, selected_features

# =============================================================================
# PHASE 2C — ELASTICNET COMPARISON
# =============================================================================

def phase2c_elasticnet(detail, expert, selected_features=None):
    """ElasticNet with L1+L2 regularisation for automatic feature zeroing."""
    common = sorted(set(detail) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2C — ELASTICNET  (LOO, n={n}, all {len(ALL_SIGNALS)} features)")
    print(f"{'=' * 70}")

    alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    all_mae = []

    for dim in DIMENSIONS:
        print(f"\n  {dim}:")

        # Use ALL features — ElasticNet will zero out the bad ones
        X = np.array([[detail[fid][f'{dim}_{s}'] for s in ALL_SIGNALS]
                       for fid in common], dtype=float)
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        # Grid search
        best_mae = 999
        best_alpha, best_l1 = 1.0, 0.5

        for alpha in alpha_values:
            for l1 in l1_ratios:
                preds = loo_elasticnet(X, y, alpha=alpha, l1_ratio=l1)
                mae = float(np.mean(np.abs(np.array(preds) - y)))
                if mae < best_mae:
                    best_mae = mae
                    best_alpha, best_l1 = alpha, l1

        print(f"    Best params: α={best_alpha}, l1_ratio={best_l1}")

        # Final run with best params
        preds = loo_elasticnet(X, y, alpha=best_alpha, l1_ratio=best_l1)
        m = compute_metrics(preds, y, label=dim)

        # Show which features survived (non-zero weights)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        en = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000)
        en.fit(X_scaled, y)

        active = []
        zeroed = []
        for j, s in enumerate(ALL_SIGNALS):
            w = float(en.coef_[j])
            if abs(w) > 0.001:
                active.append(f"{s}({w:+.3f})")
            else:
                zeroed.append(s)

        print(f"    Active features:  {active}")
        if zeroed:
            print(f"    Zeroed features:  {zeroed}")

        all_mae.append(m['mae'])

    overall_mae = float(np.mean(all_mae))
    print(f"\n  ── Overall Phase 2C (ElasticNet) ──")
    print(f"  MAE: {overall_mae:.3f}  {'✅' if overall_mae < 1.0 else '❌'}  (target < 1.0)")

    return overall_mae

# =============================================================================
# PHASE 2D — SVR WITH ALL FEATURES (no selection, for comparison)
# =============================================================================

def phase2d_svr_all_features(detail, expert):
    """SVR with all 9 features for comparison against selected features."""
    common = sorted(set(detail) & set(expert))
    n = len(common)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2D — SVR ALL FEATURES  (LOO, n={n}, {len(ALL_SIGNALS)} features)")
    print(f"{'=' * 70}")

    C_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    epsilon_values = [0.1, 0.3, 0.5, 0.8]
    gamma_values = ['scale', 'auto']

    all_mae = []

    for dim in DIMENSIONS:
        print(f"\n  {dim}:")

        X = np.array([[detail[fid][f'{dim}_{s}'] for s in ALL_SIGNALS]
                       for fid in common], dtype=float)
        y = np.array([expert[fid][dim] for fid in common], dtype=float)

        best_mae = 999
        best_C, best_eps, best_gamma = 1.0, 0.5, 'scale'

        for C in C_values:
            for eps in epsilon_values:
                for gamma in gamma_values:
                    preds = loo_svr(X, y, C=C, epsilon=eps, gamma=gamma)
                    mae = float(np.mean(np.abs(np.array(preds) - y)))
                    if mae < best_mae:
                        best_mae = mae
                        best_C, best_eps, best_gamma = C, eps, gamma

        print(f"    Best params: C={best_C}, ε={best_eps}, γ={best_gamma}")

        preds = loo_svr(X, y, C=best_C, epsilon=best_eps, gamma=best_gamma)
        m = compute_metrics(preds, y, label=dim)
        all_mae.append(m['mae'])

    overall_mae = float(np.mean(all_mae))
    print(f"\n  ── Overall Phase 2D (SVR all features) ──")
    print(f"  MAE: {overall_mae:.3f}  {'✅' if overall_mae < 1.0 else '❌'}  (target < 1.0)")

    return overall_mae

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results, selected_features, best_params, n):
    """Save all results and best configuration."""
    path = os.path.join(ML_DIR, 'calibration_svr_summary.txt')
    with open(path, 'w') as f:
        f.write("PQC Readiness — SVR Calibration Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Frameworks used: {n}\n\n")

        f.write("Results:\n")
        for label, mae in sorted(results.items(), key=lambda x: x):
            if mae is not None:
                status = "PASS ✅" if mae < 1.0 else "FAIL ❌"
                f.write(f"  {label}: MAE = {mae:.3f}  {status}\n")

        if selected_features:
            f.write(f"\nSelected features per dimension:\n")
            for dim, feats in selected_features.items():
                f.write(f"  {dim}: {feats}\n")

        if best_params:
            f.write(f"\nBest SVR params per dimension:\n")
            for dim, params in best_params.items():
                f.write(f"  {dim}: C={params['C']}, ε={params['epsilon']}, γ={params['gamma']}\n")

        f.write(f"\nTarget: MAE < 1.0\n")

    print(f"\n  💾 Saved {path}")

    # Save config as JSON for production use
    config_path = os.path.join(ML_DIR, 'svr_config.json')
    config = {
        'selected_features': selected_features,
        'best_params': best_params,
        'results': {k: round(v, 4) for k, v in results.items() if v is not None}
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  💾 Saved {config_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  PQC READINESS — SVR CALIBRATION PIPELINE")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    expert = load_expert(EXPERT_FILE)

    if not os.path.exists(DETAIL_FILE):
        print(f"\n  ❌ {DETAIL_FILE} not found.")
        print(f"     Run main.py first to generate NLP features.")
        return

    detail = load_detailed(DETAIL_FILE)

    common = sorted(set(detail) & set(expert))
    n = len(common)
    print(f"    Overlap: {n} — IDs: {common}")

    if n < 5:
        print(f"\n  ❌ Need ≥ 5 overlapping frameworks (have {n}).")
        return

    results = {}

    # Phase 2A — Feature selection
    print("\n  🔍 Starting feature selection (this may take a few minutes)...")
    selected_features = phase2a_feature_selection(detail, expert, max_features=5)

    # Phase 2B — SVR with selected features
    mae_svr, best_params, sel_feats = phase2b_svr(detail, expert, selected_features)
    results['Phase 2B (SVR selected features)'] = mae_svr

    # Phase 2C — ElasticNet comparison
    mae_en = phase2c_elasticnet(detail, expert)
    results['Phase 2C (ElasticNet)'] = mae_en

    # Phase 2D — SVR with all features
    mae_svr_all = phase2d_svr_all_features(detail, expert)
    results['Phase 2D (SVR all features)'] = mae_svr_all

    # Save
    save_results(results, selected_features, best_params, n)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SVR CALIBRATION COMPLETE")
    print(f"{'=' * 70}")

    print(f"\n  📊 Results comparison:")
    print(f"  {'Model':<40s} {'MAE':>6s}  {'Status'}")
    print(f"  {'-' * 55}")
    for label, mae in sorted(results.items(), key=lambda x: x):
        status = '✅' if mae < 1.0 else '❌'
        print(f"  {label:<40s} {mae:>6.3f}  {status}")

    best = min(results.values())
    best_name = [k for k, v in results.items() if v == best]

    print(f"\n  🏆 Best: {best_name} (MAE={best:.3f})")

    if best < 1.0:
        print("  ✅ Target met! Ready for production scoring.")
    elif best < 1.1:
        print("  🔶 Very close. Consider:")
        print("     1. Adding more expert-scored frameworks")
        print("     2. Refining keyword lists for worst dimensions")
        print("     3. Trying polynomial kernel: SVR(kernel='poly', degree=2)")
    else:
        print("  ❌ Target not met. Consider:")
        print("     1. More training data (n=30+)")
        print("     2. Better NLP reference descriptions")
        print("     3. Reviewing expert score consistency")

if __name__ == '__main__':
    main()
