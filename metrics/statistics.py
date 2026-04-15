import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency, norm
from typing import Dict, List, Union, Tuple



def compute_v_test(df: pd.DataFrame, labels: np.ndarray, col_name: str, cluster_id: int) -> float:
    """
    Calculate the V-test value of a continuous variable for a given cluster.
    Measures the difference between the cluster mean and the global mean in standard deviations.

    Returns:
        float: The V-test score.
               > 2.0 : Significant over-representation (p < 0.05)
               < -2.0 : Significant under-representation
    """

    mean_global = df[col_name].mean()
    std_global = df[col_name].std(ddof=1) 
    n_global = len(df)
    
    cluster_mask = (labels == cluster_id)
    n_cluster = np.sum(cluster_mask)
    
    if n_cluster == 0 or n_cluster == n_global or std_global == 0:
        return 0.0
        
    mean_cluster = df.loc[cluster_mask, col_name].mean()
    
    # V = (mean_k - mean_global) / (std_global * sqrt((N - nk)/(N-1) * 1/nk))
    numerator = mean_cluster - mean_global
    denominator = std_global * np.sqrt((n_global - n_cluster) / ((n_global - 1) * n_cluster))
    
    return numerator / denominator

def compute_all_v_tests(df: pd.DataFrame, labels: np.ndarray, col_types: Dict[str, str]) -> pd.DataFrame:
    """
    Generate the full matrix of V-test scores for all clusters and variables.
    Automatically handles one-hot encoding for categorical variables.

    Returns:
        pd.DataFrame: Index = Cluster IDs, Columns = Variables (including expanded categories).
    """

    df_expanded = df.copy()
    multicat_cols = [c for c, t in col_types.items() if t in ['multicategorical']]
    
    if multicat_cols:
        df_expanded = pd.get_dummies(df_expanded, columns=multicat_cols, prefix_sep='=')
    

    df_numeric = df_expanded.select_dtypes(include=[np.number])
    cols = df_numeric.columns
    
    cluster_ids = sorted(np.unique(labels))
    matrix_data = []
    

    for cid in cluster_ids:
        row_scores = []
        for col in cols:
            score = compute_v_test(df_numeric, labels, col, cid)
            row_scores.append(score)
        matrix_data.append(row_scores)
        

    return pd.DataFrame(matrix_data, columns=cols, index=cluster_ids)


def get_statistical_significance(df: pd.DataFrame, labels: np.ndarray, col_name: str, col_type: str) -> float:
    """
    Compute the statistical significance (p-value) of the association between a feature and cluster assignments. 
    Kruskal-Wallis test is used for numerical features, Chi-Square test for categorical features.
    """
    data = df.copy()
    data['Cluster'] = labels
    unique_clusters = np.unique(labels)
    
    if len(unique_clusters) < 2: 
        return 1.0
    
    # --- NUMERICAL CASE: Kruskal-Wallis ---
    if col_type in ['continuous', 'ordinal']:
        groups = [data.loc[data['Cluster'] == k, col_name].dropna() for k in unique_clusters]
        if any(len(g) == 0 for g in groups): return 1.0
        stat, p_val = kruskal(*groups)
        return float(p_val)

    # --- CATEGORICAL CASE: Chi-Square ---
    elif col_type in ['binary', 'multicategorical']:
        contingency_table = pd.crosstab(data['Cluster'], data[col_name])
        if contingency_table.size == 0: return 1.0
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        return float(p_val)
    
    return 1.0

def get_sorted_features_by_impact(df: pd.DataFrame, labels: np.ndarray, col_types: Dict[str, str]) -> List[Tuple[str, float]]:
    """
    Order the variables by global "discriminative power".
    
    Instead of using the p-value (which often saturates at 0), we use the 
    Maximum Absolute V-test across all clusters.
    
    Returns:
        List[(column_name, max_abs_vtest)], sorted by descending score.
    """
    # 1. Compute all V-tests at once (optimized)
    df_vtests = compute_all_v_tests(df, labels, col_types)
    
    # 2. For each variable (column), take the max absolute value across clusters
    # Result: a pandas Series with Index=Variable, Value=MaxVtest
    max_vtests = df_vtests.abs().max(axis=0)
    
    # 3. Build and sort the list
    # Keep only columns present in col_types (real features)
    sorted_features = []
    for col in max_vtests.index:
        if col in col_types: # Safety check to avoid technical columns
            score = max_vtests[col]
            sorted_features.append((col, score))
            
    # Descending sort (highest V-test first)
    sorted_features.sort(key=lambda x: x[1], reverse=True)
    
    return sorted_features