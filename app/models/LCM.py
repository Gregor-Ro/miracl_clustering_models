import numpy as np
import pandas as pd
from typing import Dict, Optional

from app.models.base import BaseClusterer
from app.models.engines.LCM_engine import LCM_EM

class LCM(BaseClusterer):
    def __init__(self, df_train, col_types):
        super().__init__(df_train, col_types)

        self.model: Optional[LCM_EM] = None

        self.proba_: np.ndarray = None
    
    def fit(self, n_clusters: int, max_iter: int = 200, tol: float = 1e-6, random_state: int = 42):
        if self.df_train is None:
            raise ValueError("df_train has not been given")
        
        self.n_clusters = n_clusters

        self.model = LCM_EM(
            n_classes=self.n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )

        self.model.fit(self.df_train, self.col_types)

        self.labels_ = self.model.labels_
        self.proba_ = self.model.gamma_

        return self
    
    def predict(self, df_test):
        if self.model is None:
            raise ValueError("Model not trained. Call fit first.")
        X_test = df_test.to_numpy()
        labels = self.model.predict(X_test)
        return labels
    
    def evaluate(self) -> Dict[str, float]:
        metrics = super().evaluate()
        
        if self.proba_ is not None:
            entropy = self._compute_entropy_normalized()
            metrics["entropy"] = entropy
            
        return metrics  
    
    def _compute_entropy_normalized(self) -> float:
        """
        Compute the certainty of the classification (1 = sure, 0 = not sure at all).
        """

        p = np.clip(self.proba_, 1e-15, 1.0)

        entropy_per_sample = -np.sum(p * np.log(p), axis=1)
        mean_entropy = np.mean(entropy_per_sample)

        if self.n_clusters > 1:
            score = 1 - (mean_entropy / np.log(self.n_clusters))
        else:
            score = 1.0
            
        return float(score)
        