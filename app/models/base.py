from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np 
from typing import Dict, Optional, List

from app.metrics.clustering import compute_silhouette_gower, compute_ARI_pairwise, compute_CCC


class BaseClusterer(ABC):
    """
    Base class for clustering models.
    This standardizes the interface so new models can be plugged in consistently.
    """

    def __init__(self, df_train, col_types, df_test=None):
        """
        Args:
            df_train: Training DataFrame.
            col_types: Column types dictionary.
            df_test: Optional test DataFrame.
        """
        self.df_train = df_train
        self.col_types = col_types

        self.df_test = df_test

        self.labels_: Optional[np.ndarray] = None
        self.n_clusters: Optional[int] = None
        self.metrics_: Optional[Dict[str, float]] = None
        self.labels_test_: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, n_clusters: int, **kwargs):
        """
        Abstract method. Each child must implement its own fit method.
        """
        pass

    @abstractmethod
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        """
        Abstract method.
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        Compute common quality metrics (e.g., Gower-based silhouette).
        Model-based methods may require additional metrics handled by subclasses.
        """
        if self.labels_ is None:
            raise ValueError("The model has not been fitted yet (call fit first)")
        
        metrics = {}

        sil = compute_silhouette_gower(self.df_train, self.labels_, self.col_types)
        ARI = compute_ARI_pairwise(self.__class__, self.df_train, self.col_types, self.n_clusters)
        metrics["silhouette"] = sil
        metrics["ARI"] = ARI
        if self.df_test is not None and self.labels_test_ is not None:
            CCC = compute_CCC(self.df_train, self.labels_, self.df_test, self.labels_test_, self.col_types)
            metrics["CCC"] = CCC
        else:
            metrics["CCC"] = None

        self.metrics_ = metrics
        return metrics
    
    