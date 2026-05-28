import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# Post-Quantum Cryptography Framework Composite Scoring Script
# Based on formula: S_weighted = Σ(w_dimension × S_dimension)
# Where: 0 < w_dimension < 1 and 0 ≤ S_dimension ≤ 5
# ============================================================================

def load_dimension_weights(weights_file):
    """Load dimension weights from CSV file."""
    print(f"Loading dimension weights from {weights_file}...")
    weights_df = pd.read_csv(weights_file)
    
    weights = {}
    for _, row in weights_df.iterrows():
        weights[row['Dimension']] = row['Weight']
    
    # Verify weights sum to approximately 1.0
    total = sum(weights.values())
    print(f"Total weight: {total:.6f}")
    assert abs(total - 1.0) < 0.0001, f"Weights must sum to 1.0, got {total}"
    
    return weights

def calculate_composite_scores(scores_file, weights):
    """
    Calculate weighted composite scores for PQC readiness frameworks.
    
    Formula: S_weighted = w_assets*S_assets + w_agility*S_agility + 
                          w_migration*S_migration + w_risk*S_risk + 
                          w_standards*S_standards
    
    Args:
        scores_file (str): Path to CSV file with framework scores
        weights (dict): Dimension weights
        
    Returns:
        pd.DataFrame: Enhanced dataframe with composite scores
    """
    
    # Read the scores CSV file
    print(f"\nReading framework scores from {scores_file}...")
    df = pd.read_csv(scores_file)
    
    # Define the mapped score columns (0-5 scale)
    score_cols = {
        'crypto_assets': 'crypto_assets_mapped',
        'crypto_agility': 'crypto_agility_mapped',
        'migration_planning': 'migration_planning_mapped',
        'risk_management': 'risk_management_mapped',
        'standards_compliance': 'standards_compliance_mapped'
    }
    
    # Verify all required columns exist
    for col in score_cols.values():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # Ensure score columns are numeric
    for col in score_cols.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate weighted score (0-5 scale)
    # S_weighted = Σ(w_dimension × S_dimension)
    df['weighted_score'] = 0
    for dimension, col_name in score_cols.items():
        weight = weights[dimension]
        df['weighted_score'] += df[col_name] * weight
    
    # Calculate rebased score (0-100 scale)
    df['rebased_score'] = (df['weighted_score'] / 5) * 100
    
    # Round scores for readability
    df['weighted_score'] = df['weighted_score'].round(3)
    df['rebased_score'] = df['rebased_score'].round(1)
    
    # Identify strongest and weakest dimensions for each framework
    def get_strongest_dimension(row):
        scores = {dim: row[col] for dim, col in score_cols.items()}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'None'
    
    def get_weakest_dimension(row):
        scores = {dim: row[col] for dim, col in score_cols.items()}
        non_zero_scores = {k: v for k, v in scores.items() if v > 0}
        if not non_zero_scores:
            return 'None'
        return min(non_zero_scores, key=non_zero_scores.get)
    
    df['strongest_dimension'] = df.apply(get_strongest_dimension, axis=1)
    df['weakest_dimension'] = df.apply(get_weakest_dimension, axis=1)
    
    # Sort by weighted score (descending)
    df = df.sort_values('weighted_score', ascending=False)
    
    # Add rank
    df['rank'] = range(1, len(df) + 1)
    
    print(f"Processed {len(df)} frameworks successfully.")
    
    return df, score_cols

def display_top_frameworks(df, top_n=10):
    """Display top N frameworks in a formatted table."""
    
    print(f"\n{'='*140}")
    print(f"TOP {top_n} FRAMEWORKS BY COMPOSITE READINESS SCORE")
    print(f"{'='*140}")
    print(f"Formula: S_weighted = w_assets*S_assets + w_agility*S_agility + w_migration*S_migration + w_risk*S_risk + w_standards*S_standards")
    print(f"{'='*140}\n")
    
    top_df = df.head(top_n)
    
    # Print header
    print(f"{'Rank':<6}{'Framework':<75}{'Score (0-5)':<15}{'Score (0-100)':<15}{'Strongest':<20}{'Weakest':<20}")
    print(f"{'-'*140}")
    
    # Print rows
    for _, row in top_df.iterrows():
        # Clean framework name (remove .pdf extension)
        framework = row['Framework'].replace('.pdf', '')
        framework = framework[:72] + '...' if len(framework) > 75 else framework
        
        # Shorten dimension names for display
        strong_dim = row['strongest_dimension'].replace('crypto_', '').replace('migration_', 'migr_').replace('standards_', 'std_')
        weak_dim = row['weakest_dimension'].replace('crypto_', '').replace('migration_', 'migr_').replace('standards_', 'std_')
        
        print(f"{row['rank']:<6}{framework:<75}{row['weighted_score']:<15.3f}{row['rebased_score']:<15.1f}"
              f"{strong_dim:<20}{weak_dim:<20}")

def display_dimension_scores(df, score_cols, weights):
    """Display dimension scores for top frameworks."""
    
    print(f"\n{'='*160}")
    print("DETAILED DIMENSION SCORES FOR TOP 10 FRAMEWORKS")
    print(f"{'='*160}\n")
    
    top_10 = df.head(10)
    
    # Print header
    print(f"{'Rank':<6}{'Framework':<55}{'CA':<10}{'AG':<10}{'MP':<10}{'RM':<10}{'SC':<10}{'Weighted':<15}")
    print(f"{'-'*160}")
    w_ca = weights['crypto_assets']
    w_ag = weights['crypto_agility']
    w_mp = weights['migration_planning']
    w_rm = weights['risk_management']
    w_sc = weights['standards_compliance']
    print(f"{'':6}{'':55}{'(w=' + f'{w_ca:.2f}' + ')':<10}{'(w=' + f'{w_ag:.2f}' + ')':<10}"
          f"{'(w=' + f'{w_mp:.2f}' + ')':<10}{'(w=' + f'{w_rm:.2f}' + ')':<10}{'(w=' + f'{w_sc:.2f}' + ')':<10}{'Score':<15}")
    print(f"{'-'*160}")
    
    # Print rows
    for _, row in top_10.iterrows():
        framework = row['Framework'].replace('.pdf', '')
        framework = framework[:52] + '...' if len(framework) > 55 else framework
        
        print(f"{row['rank']:<6}{framework:<55}"
              f"{row[score_cols['crypto_assets']]:<10.2f}{row[score_cols['crypto_agility']]:<10.2f}"
              f"{row[score_cols['migration_planning']]:<10.2f}{row[score_cols['risk_management']]:<10.2f}"
              f"{row[score_cols['standards_compliance']]:<10.2f}{row['weighted_score']:<15.3f}")

def display_dimension_statistics(df, score_cols, weights):
    """Display statistical analysis of dimension scores."""
    
    print(f"\n{'='*140}")
    print("DIMENSION SCORE STATISTICS ACROSS ALL FRAMEWORKS")
    print(f"{'='*140}\n")
    
    print(f"{'Dimension':<25}{'Weight':<12}{'Mean':<10}{'Median':<10}{'Std Dev':<10}{'Min':<8}{'Max':<8}{'Weighted Mean':<15}")
    print(f"{'-'*140}")
    
    for dim, col_name in score_cols.items():
        weight = weights[dim]
        mean_score = df[col_name].mean()
        median_score = df[col_name].median()
        std_score = df[col_name].std()
        min_score = df[col_name].min()
        max_score = df[col_name].max()
        weighted_mean = mean_score * weight
        
        dim_display = dim.replace('_', ' ').title()
        
        print(f"{dim_display:<25}{weight:<12.4f}{mean_score:<10.2f}{median_score:<10.2f}"
              f"{std_score:<10.2f}{min_score:<8.2f}{max_score:<8.2f}{weighted_mean:<15.3f}")
    
    # Calculate overall statistics
    total_weighted_mean = sum(df[col].mean() * weights[dim] for dim, col in score_cols.items())
    print(f"{'-'*140}")
    print(f"{'OVERALL':<25}{'1.0000':<12}{'':<10}{'':<10}{'':<10}{'':<8}{'':<8}{total_weighted_mean:<15.3f}")

def export_results(df, score_cols, output_file):
    """Export results to CSV."""
    
    # Prepare output dataframe
    output_df = df[['rank', 'Framework', 'Framework_ID', 'weighted_score', 'rebased_score']].copy()
    
    # Add individual dimension scores
    for dim, col_name in score_cols.items():
        output_df[dim] = df[col_name]
    
    # Add strongest and weakest dimensions
    output_df['strongest_dimension'] = df['strongest_dimension']
    output_df['weakest_dimension'] = df['weakest_dimension']
    
    output_df.to_csv(output_file, index=False)
    print(f"\nResults exported to: {output_file}")

def find_file(filename, search_dirs):
    """Search for a file in multiple directories."""
    for directory in search_dirs:
        filepath = Path(directory) / filename
        if filepath.exists():
            return str(filepath)
    return None

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Define possible search directories relative to script location
    search_dirs = [
        script_dir,  # Same directory as script
        script_dir.parent,  # Parent directory
        script_dir.parent / "freq-analysis" / "scripts",  # Sibling directory
        script_dir / "data",  # Data subdirectory
        script_dir.parent / "data",  # Parent data directory
    ]
    
    print("Current working directory:", os.getcwd())
    print("Script directory:", script_dir)
    print("\nSearching for input files...")
    
    # Try to find the files
    weights_file = find_file("dimension_weights.csv", search_dirs)
    scores_file = find_file("nlp_scores.csv", search_dirs)
    
    # If not found, prompt user for paths
    if not weights_file:
        print("\n⚠️  Could not find 'dimension_weights.csv' automatically.")
        weights_file = input("Enter path to dimension_weights.csv: ").strip()
    
    if not scores_file:
        print("\n⚠️  Could not find 'nlp_scores.csv' automatically.")
        scores_file = input("Enter path to nlp_scores.csv: ").strip()
    
    # Output file in same directory as script
    OUTPUT_FILE = script_dir / "PQC_Framework_Composite_Scores.csv"
    
    try:
        print(f"\nUsing files:")
        print(f"  Weights: {weights_file}")
        print(f"  Scores:  {scores_file}")
        print(f"  Output:  {OUTPUT_FILE}")
        print()
        
        # Load dimension weights
        weights = load_dimension_weights(weights_file)
        
        print("\nDimension Weights:")
        for dim, weight in weights.items():
            print(f"  {dim}: {weight:.6f}")
        
        # Calculate composite scores
        df_scored, score_cols = calculate_composite_scores(scores_file, weights)
        
        # Display top frameworks
        display_top_frameworks(df_scored, top_n=15)
        
        # Display detailed dimension scores
        display_dimension_scores(df_scored, score_cols, weights)
        
        # Display dimension statistics
        display_dimension_statistics(df_scored, score_cols, weights)
        
        # Export results
        export_results(df_scored, score_cols, OUTPUT_FILE)
        
        # Additional analysis: frameworks by score range
        print(f"\n{'='*140}")
        print("FRAMEWORK DISTRIBUTION BY READINESS LEVEL")
        print(f"{'='*140}\n")
        
        score_ranges = [
            ('Excellent (4.0-5.0)', 4.0, 5.0),
            ('Good (3.0-3.9)', 3.0, 3.9),
            ('Moderate (2.0-2.9)', 2.0, 2.9),
            ('Limited (1.0-1.9)', 1.0, 1.9),
            ('Minimal (0.0-0.9)', 0.0, 0.9)
        ]
        
        for label, min_score, max_score in score_ranges:
            count = len(df_scored[(df_scored['weighted_score'] >= min_score) & 
                                  (df_scored['weighted_score'] <= max_score)])
            percentage = (count / len(df_scored)) * 100
            print(f"{label:<25}{count:>3} frameworks ({percentage:>5.1f}%)")
        
        print(f"\n{'='*140}")
        print(f"Overall Mean Composite Score: {df_scored['weighted_score'].mean():.3f} / 5.0 "
              f"({df_scored['rebased_score'].mean():.1f}%)")
        print(f"Median Composite Score: {df_scored['weighted_score'].median():.3f} / 5.0 "
              f"({df_scored['rebased_score'].median():.1f}%)")
        print(f"{'='*140}\n")
        
        # Display bottom 5 frameworks
        print(f"\n{'='*140}")
        print("BOTTOM 5 FRAMEWORKS BY COMPOSITE READINESS SCORE")
        print(f"{'='*140}\n")
        
        bottom_5 = df_scored.tail(5).sort_values('weighted_score')
        print(f"{'Rank':<6}{'Framework':<75}{'Score (0-5)':<15}{'Score (0-100)':<15}")
        print(f"{'-'*140}")
        
        for _, row in bottom_5.iterrows():
            framework = row['Framework'].replace('.pdf', '')
            framework = framework[:72] + '...' if len(framework) > 75 else framework
            print(f"{row['rank']:<6}{framework:<75}{row['weighted_score']:<15.3f}{row['rebased_score']:<15.1f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find required file")
        print(f"\nPlease provide the correct paths to:")
        print(f"  - dimension_weights.csv")
        print(f"  - nlp_scores.csv")
        print(f"\nDetails: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
