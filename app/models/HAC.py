import numpy as np
import pandas as pd
from typing import Dict
from sklearn.cluster import AgglomerativeClustering
import warnings

from app.models.base import BaseClusterer
from app.metrics.distances import compute_gower_matrix

class HAC(BaseClusterer):
    """
    Wrapper for HAC clustering (bottom-up approach).
    """
    def __init__(self, df_train: pd.DataFrame, col_types: Dict[str, str]):
        super().__init__(df_train, col_types)
        self.model = None
        self.linkage = None

    def fit(self, n_clusters: int, linkage: str = "average"):
        """
        Fit the HAC model.
        Args:
            linkage: One of 'average', 'complete', 'single'.
        """

        self.n_clusters = n_clusters
        self.linkage = linkage

        D_gower = compute_gower_matrix(self.df_train, self.col_types)

        self.model = AgglomerativeClustering(
            self.n_clusters, 
            metric = "precomputed",
            linkage= self.linkage,
        )

        self.labels_ = self.model.fit_predict(D_gower)

        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit first.")
        warnings.warn(
            "There is no canonical prediction method for HAC. This implementation assigns each point to the closest cluster."
        )
        
        n_test = len(df_test)
        unique_clusters = np.unique(self.labels_)
        
        cluster_distances = np.zeros((n_test, self.n_clusters))

        for i, cluster_id in enumerate(unique_clusters):
            mask_train = (self.labels_ == cluster_id)
            df_cluster_members = self.df_train.loc[mask_train]

            if len(df_cluster_members)==0:
                cluster_distances[:, i] = np.inf
                continue
                
            D_sub = compute_gower_matrix(
                df1 = df_test,
                col_types=self.col_types,
                df2=df_cluster_members
            )
            
            if self.linkage == "average":
                score = np.mean(D_sub, axis=1)
            
            if self.linkage == 'single':
                score = np.min(D_sub, axis=1)

            elif self.linkage == 'complete':
                score = np.max(D_sub, axis=1)
            
            cluster_distances[ :, i] = score
        
        predictions = np.argmin(cluster_distances, axis=1)

        return unique_clusters[predictions]