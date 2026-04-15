# Clustering Models (Quick Use Guide)

This folder provides four clustering models with a shared interface:
- `fit(n_clusters=..., **params)`
- `predict(df_test)`
- `evaluate()`

## 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal structure required in your repo:
- `app/models/`
- `app/metrics/`

## 2. Common Pattern

```python
from app.models.HAC import HAC  # replace with KMedoids, LCM, or URF

model = HAC(df_train, col_types)
model.fit(n_clusters=3)              # add model-specific params if needed
labels_train = model.labels_
metrics = model.evaluate()           # silhouette, ARI, and optional CCC
labels_test = model.predict(df_test) # optional
```

`col_types` is a dictionary like:

```python
col_types = {
    "age": "continuous",
    "sex": "binary",
    "stage": "ordinal",
    "site": "multicategorical",
}
```

## 3. Models

### HAC (`HAC.py`)
- Type: Hierarchical Agglomerative Clustering on Gower distance.
- Fit: `fit(n_clusters, linkage="average")`
- `linkage`: `"average" | "complete" | "single"`

```python
from app.models.HAC import HAC
model = HAC(df_train, col_types).fit(n_clusters=4, linkage="average")
```

### K-Medoids (`kmedoids.py`)
- Type: PAM K-Medoids on precomputed Gower distance.
- Fit: `fit(n_clusters, max_iter=300, random_state=42)`

```python
from app.models.kmedoids import KMedoids
model = KMedoids(df_train, col_types).fit(n_clusters=4, max_iter=300)
```

### LCM (`LCM.py`)
- Type: Latent Class Model (EM), mixed-type variables.
- Fit: `fit(n_clusters, max_iter=200, tol=1e-6, random_state=42)`
- Adds `entropy` in `evaluate()`.

```python
from app.models.LCM import LCM
model = LCM(df_train, col_types).fit(n_clusters=4, max_iter=300, tol=1e-6)
```

### URF (`URF.py`)
- Type: Unsupervised Random Forest + K-Medoids on RF dissimilarity.
- Fit: `fit(n_clusters, n_trees=50, max_depth=6, random_state=42)`

```python
from app.models.URF import URF
model = URF(df_train, col_types).fit(n_clusters=4, n_trees=200, max_depth=8)
```

## 4. Notes

- Always call `fit(...)` before `predict(...)` or `evaluate()`.
- `model.labels_` contains training cluster assignments.
- If you set `model.df_test = df_test` and `model.labels_test_`, `evaluate()` can also compute CCC.
