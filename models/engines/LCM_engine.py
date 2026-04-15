import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit
from scipy.special import logsumexp

# Classes for the different variable distributions.

class BernoulliVar:
    def __init__(self): 
        self.p = None
    def fit(self, x, w): 
        self.p = np.sum(w * x) / np.sum(w)
    def pdf(self, x): 
        return np.clip(self.p**x * (1 - self.p)**(1 - x), 1e-12, 1)

class GaussianVar:
    def __init__(self): 
        self.mu = None; self.sigma = None
    def fit(self, x, w):
        mu = np.sum(w * x) / np.sum(w)
        sigma = np.sqrt(np.sum(w * (x - mu)**2) / np.sum(w))
        self.mu, self.sigma = mu, max(sigma, 1e-6)
    def pdf(self, x): 
        return np.clip(norm.pdf(x, self.mu, self.sigma), 1e-12, 1)

class CategoricalVar:
    def __init__(self, n_cat): 
        self.p = np.ones(n_cat) / n_cat
    def fit(self, x, w):
        n_cat = len(self.p)
        probs = np.zeros(n_cat)
        for c in range(n_cat):
            probs[c] = np.sum(w * (x == c))
        self.p = probs / np.sum(probs)
    def pdf(self, x): 
        return np.clip(self.p[x.astype(int)], 1e-12, 1)

class OrdinalVar:
    """
    Simplified ordinal logit model.
    Sometimes it is better to treat ordinal variables as continuous variables.
    Useful when gaps between categories are not even or when the distribution is far from normal.
    """
    def __init__(self, n_cat):
        self.n_cat = n_cat
        self.thresholds_ = np.linspace(-1, 1, n_cat - 1)
        self.loc_ = 0.0  #  latent center
    
    def fit(self, x, w):
        # Adjust the latent center
        self.loc_ = np.average(x, weights=w)
    
    def pdf(self, x):
        cdf = expit(self.thresholds_ - self.loc_)
        probs = np.diff(np.concatenate(([0], cdf, [1])))
        return np.clip(probs[x.astype(int)], 1e-12, 1)


# Main algorithm: LCM with EM

class LCM_EM:
    def __init__(self, n_classes=3, max_iter=200, tol=1e-6, random_state=42):
        """
        LCM method with standard EM optimization.
        """
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.pi_ = None
        self.gamma_ = None
        self.labels_ = None
        self.variables = None

    def _init_params(self, X, col_types):
        """Initialize distribution parameters for each variable."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        self.variables = []
        for c in X.columns:
            kind = col_types[c]
            if kind == "binary":
                    var = BernoulliVar()
            elif kind == "multicategorical":
                n_cat = int(X[c].max()) +1
                var = CategoricalVar(n_cat)
            elif kind == "ordinal":
                n_cat = int(X[c].max()) +1
                var = OrdinalVar(n_cat)
            elif kind == "continuous":
                var = GaussianVar()
            else:
                raise ValueError(f"Unknown type for {c}: {kind}")
            self.variables.append([var for i in range(self.n_classes)])
        
        # Random initialization (keep random_state for reproducibility)
        self.gamma_ = rng.rand(n_samples, self.n_classes)
        self.gamma_ /= self.gamma_.sum(axis=1, keepdims=True) 

        self.pi_ = np.ones(self.n_classes) / self.n_classes


    
    def fit(self, X, descriptor):
        self._init_params(X, descriptor)
        X_np = X.to_numpy()
        n_samples = X.shape[0]
        
        for _ in range(self.max_iter):
            # M-step
            weights = self.gamma_
            Nk = weights.sum(axis=0)
            self.pi_ = Nk / n_samples
            
            for j, col in enumerate(X.columns):
                x = X_np[:, j]
                for k in range(self.n_classes):
                    self.variables[j][k].fit(x, weights[:, k])
            
            # E-step
            log_prob = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                log_p = np.zeros(n_samples)
                for j, col in enumerate(X.columns):
                    log_p += np.log(self.variables[j][k].pdf(X_np[:, j]))
                log_prob[:, k] = np.log(self.pi_[k] + 1e-12) + log_p
            
            # Normalization (log-sum-exp)
            log_prob -= log_prob.max(axis=1, keepdims=True)
            resp = np.exp(log_prob)
            resp /= resp.sum(axis=1, keepdims=True)
            
            # Convergence
            diff = np.abs(self.gamma_ - resp).mean()
            self.gamma_ = resp
            if diff < self.tol:
                break
        
        self.labels_ = np.argmax(self.gamma_, axis=1)
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for a new dataset.
        """
        if self.pi_ is None:
            raise ValueError("Model not trained. Call fit first.")
        n_samples_test, n_features = X_test.shape
        log_prob = np.zeros(n_samples_test, self.n_classes) + np.log(self.pi_ + 1e-15)
        for j in range(n_features):
            x_col = X_test[ :, j]
            for k in range(self.n_classes):
                pdf_vals = self.variables[j][k].pdf(x_col)
                log_prob[ :, k] += np.log(pdf_vals + 1e-15)
        
        log_prob_norm = logsumexp(log_prob, axis=1, keepdims=True)
        proba = np.exp(log_prob - log_prob_norm)

        return proba


    def predict(self, X_test):
        proba = self.predict_proba(X_test)

        return np.argmax(proba, axis=1)


