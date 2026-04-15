import numpy as np
import pandas as pd 
from sklearn.metrics import pairwise_distances
from typing import Tuple, Dict, Optional

def robust_minmax_scale(X: np.ndarray) -> np.ndarray:
    """
    Normalize the columns of the matrix X in [0,1] in a robust way.

    Instead of using min and max, we use the 5th and 95th percentiles.
    Values outside of this interval are clipped to 0 or 1.
    """
    q_low = np.percentile(X, 5, axis=0)
    q_high = np.percentile(X, 95, axis=0)

    rng = q_high - q_low
    rng = np.where(rng == 0, 1.0, rng)

    X_scaled = (X - q_low) / rng

    return np.clip(X_scaled, 0.0, 1.0)

def compute_gower_matrix(
        df: pd.DataFrame,
        col_types: Dict[str, str],
        df2: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Compute a Gower-like dissimilarity matrix for mixed data types.
    - If df2 is None, compute intra-dataset distances.
    - If df2 is given, compute distances between df and df2.

    Args:
        df: DataFrame 
        col_types: Dict {column_name: type}, where type is in
            'continuous', 'ordinal', 'binary', 'multicategorical'

    Returns:
        D : The matrix of the distance (n_samples x n_samples)
    """
    num_cols = [c for c, t in col_types.items() if t in ["continuous", "ordinal"]]
    cat_cols = [c for c, t in col_types.items() if t in ["binary", "multicategorical"]]

    X_num = df[num_cols].to_numpy(dtype=float) if num_cols else None
    X_cat = df[cat_cols].to_numpy(dtype=float) if cat_cols else None

    X_num2 = None
    X_cat2 = None 

    if df2 is not None:
        if num_cols:
            X_num2 = df2[num_cols].to_numpy(dtype=float)
        if cat_cols:
            X_cat2 = df2[cat_cols].to_numpy(dtype=float)

    D_num = None
    if X_num is not None:
        X_num_scaled = robust_minmax_scale(X_num)

        X_num2_scaled = robust_minmax_scale(X_num2) if X_num2 is not None else None

        D_num = pairwise_distances(X_num_scaled, X_num2_scaled, metric="manhattan") / X_num.shape[1]

    D_cat = None
    if X_cat is not None:
        D_cat = pairwise_distances(X_cat, X_cat2, metric="hamming")

    # Weighted combination. Instead of using an equal beta for numeric and categorical
    # blocks, use a variance ratio to balance their contribution.

    beta = 1.0
    if D_num is not None and D_cat is not None:
        var_num = np.var(D_num)
        var_cat = np.var(D_cat)

        if var_cat > 0:
            beta = np.sqrt(var_num / var_cat)
        
        # Clip beta to prevent excessive distortion.
        beta = np.clip(beta, 0.5, 2.0)

        D = (D_num + beta * D_cat) / (1 + beta)
    elif D_num is not None:
        D = D_num
    elif D_cat is not None:
        D = D_cat
    else:
        raise ValueError("DataFrame has no usable columns. Check your column type encoding.")
    
    return D