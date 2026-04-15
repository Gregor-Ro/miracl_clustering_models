import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn_extra.cluster import KMedoids

from app.models.base import BaseClusterer

class URF(BaseClusterer):
    """
    Unsupervised Random Forest (URF).

    Original method that follows these steps:
    1. We create synthetic data (noise by permutation)
    2. We train a Random Forest Classifier to distinguish real and synthetic samples
    3. The distance between two patients is determined by the frequency of co-occurrence in the same leaf.
    4. We apply K-Medoids on the dissimilarity matrix.
    """

    def __init__(self, df_train, col_types):
        super().__init__(df_train, col_types)

        self.rf_model: Optional[RandomForestClassifier] = None
        self.cluster_model: Optional[KMedoids] = None

        self.train_leaves_: Optional[np.ndarray] = None
        self.medoid_indices_: Optional[np.ndarray] = None

    def fit(self, n_clusters: int, n_trees: int = 50, max_depth: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters

        X_real = self.df_train.to_numpy()
        n_samples, n_features = X_real.shape

        rng = np.random.RandomState(random_state)
        X_synthetic = np.zeros_like(X_real)

        for j in range(n_features):
            X_synthetic[:, j] = rng.permutation(X_real[:, j])

        X_all = np.vstack([X_real, X_synthetic])

        y_all = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

        self.rf_model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features='sqrt',
            n_jobs=-1,
            random_state=random_state
        )

        self.rf_model.fit(X_all, y_all)

        leaves = self.rf_model.apply(X_real)
        self.train_leaves_ = leaves

        matches = (leaves[:, None, :] == leaves[None, :, :])
        proximity_matrix = matches.mean(axis = 2)

        dissimilarity_matrix = 1 - proximity_matrix

        np.fill_diagonal(dissimilarity_matrix, 0.0)

        self.cluster_model = KMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            method="pam",
            random_state=random_state
        )

        self.labels_ = self.cluster_model.fit_predict(dissimilarity_matrix)
        self.medoid_indices_ = self.cluster_model.medoid_indices_

        return self
    
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        if self.rf_model is None or self.cluster_model is None:
            raise ValueError("Model not trained. Call fit first.")
    
        X_test = df_test.to_numpy()

        test_leaves = self.rf_model.apply(X_test)

        medoids_leaves = self.train_leaves_[self.medoid_indices_]

        matches = (test_leaves[:, None, :] == medoids_leaves[None, :, :])
        similarity_to_medoids = matches.mean(axis=2)

        dist_to_medoids = 1.0 - similarity_to_medoids

        predictions = np.argmin(dist_to_medoids, axis=1)

        return predictions
        

    