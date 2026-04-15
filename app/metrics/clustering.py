import numpy as np
import pandas as pd
from typing import Dict
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, adjusted_rand_score
from app.metrics.distances import compute_gower_matrix, robust_minmax_scale



def compute_silhouette_gower(
        df: pd.DataFrame, 
        labels: np.ndarray,
        col_types: dict
) -> float:
    """
    Compute the silhouette score based on Gower distances.
    Note: col_types is required to compute mixed-type distances.
    """

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <2 or n_clusters>= len(df):
        return -1
    
    gower_matrix = compute_gower_matrix(df, col_types)
    silhouette = silhouette_score(gower_matrix, labels, metric="precomputed")
    return float(silhouette)



def compute_ARI_pairwise(
        model_cls: type,
        df: pd.DataFrame, 
        col_types: Dict,
        n_clusters: int,
        n_iter: int = 10,
        sub_sample_ratio: float = 0.8
)-> float:
    """
    Compute ARI-based stability by fitting on different subsets of the dataset.
    """
    ari_scores = []
    N = len(df)

    for i in range(n_iter):
        X_sample_A = resample(df, n_samples=int(N * sub_sample_ratio), replace=False, random_state=i)
        X_sample_B = resample(df, n_samples=int(N * sub_sample_ratio), replace=False, random_state=i+1000)

        model_A = model_cls(X_sample_A, col_types)
        model_B = model_cls(X_sample_B, col_types)

        model_A.fit(n_clusters)
        model_B.fit(n_clusters)

        labels_A = model_A.labels_
        labels_B = model_B.labels_

        ari_scores.append(adjusted_rand_score(labels_A, labels_B))

    return np.mean(ari_scores)


def compute_mean_profiles(
        df: pd.DataFrame,
        labels: np.ndarray,
        col_types: dict
)-> list:
    """
    Compute the mean profiles of each cluster.
    """
    unique_labels = np.unique(labels)
    unique_labels = np.sort(unique_labels)
    mean_profiles = []

    num_cols = [c for c, t in col_types.items() if t in ["continuous", "ordinal"]]
    bi_cols = [c for c, t in col_types.items() if t in ["binary"]]
    multi_cols = [c for c, t in col_types.items() if t in ["multicategorical"]]

    scaled_df = df.copy()
    for col in num_cols:
        scaled_df[col] = robust_minmax_scale(np.array(df[col]))
    
    scaled_df = pd.get_dummies(scaled_df, columns=multi_cols, drop_first=True, dtype=int)

    for c in unique_labels:
        mask = (labels == c)
        mean_profile = np.mean(scaled_df.loc[mask])
        mean_profiles.append(mean_profile)
    
    return mean_profiles

def compute_ccc_single(
        mean_profile_train: np.ndarray,
        mean_profile_test: np.ndarray
)-> float:
    """
    Compute Lin's Concordance Correlation Coefficient on two profiles.
    """
    mu_train = np.mean(mean_profile_train)
    mu_test = np.mean(mean_profile_test)

    std_train = np.std(mean_profile_train)
    std_test = np.std(mean_profile_test)

    covariance = np.mean((mean_profile_train - mu_train) * (mean_profile_test - mu_test))

    if std_train == 0 or std_test == 0:
        return 0.0
    
    rho = covariance / (std_train * std_test)

    numerator = 2 * rho * std_train * std_test
    denominator = std_train**2 + std_test**2 + (mu_train - mu_test)**2

    return numerator/denominator 

    

def compute_CCC(
        df_train: pd.DataFrame,
        labels_train: np.ndarray,
        df_test: pd.DataFrame,
        labels_test: np.ndarray,
        col_types: Dict
)-> float:
    """
    Compute the mean Lin's Concordance Correlation Coefficient using cluster mean profiles.
    CCC indicates how similar the typical test cluster profiles are to the train cluster profiles.
    """
    mean_profiles_train_list = compute_mean_profiles(df_train, labels_train, col_types)
    mean_profiles_test_list = compute_mean_profiles(df_test, labels_test, col_types)

    CCC_list = []

    if len(mean_profiles_test_list) != len(mean_profiles_train_list):
        raise ValueError("Not same length in the mean profiles lists")

    for i in range(len(mean_profiles_train_list)):
        CCC_list.append(compute_ccc_single(mean_profiles_train_list[i], mean_profiles_test_list[i]))
    
    return np.mean(CCC_list)




