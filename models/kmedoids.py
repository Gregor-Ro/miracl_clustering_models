import numpy as np
import pandas as pd
from typing import Dict
from sklearn_extra.cluster import KMedoids as SklearnKMedoids

from app.models.base import BaseClusterer
from app.metrics.distances import compute_gower_matrix

class KMedoids(BaseClusterer):
    """
    Wrapper for the k-medoids algorithm
    """
    def __init__(self, df_train: pd.DataFrame, col_types: Dict[str, str]):
        super().__init__(df_train, col_types)
        self.model = None
        self.medoid_indices_ = None

    def fit(self, n_clusters: int, max_iter: int = 300, random_state: int = 42):
        if self.df_train is None:
            raise ValueError("df_train has not been given")

        self.n_clusters = n_clusters

        D_gower = compute_gower_matrix(self.df_train, self.col_types)

        self.model = SklearnKMedoids(
            self.n_clusters, 
            metric="precomputed",
            method="pam",
            max_iter=max_iter,
            random_state=random_state 
        )

        self.model.fit(D_gower)
        self.medoid_indices_ = self.model.medoid_indices_
        self.labels_ = self.model.labels_

        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call fit first")
        medoids_df = self.df_train.iloc[self.medoid_indices_]

        D_gower = compute_gower_matrix(df=df_test, col_types=self.col_types, df2=medoids_df)

        predictions = np.argmin(D_gower, axis=1)

        return predictions   

