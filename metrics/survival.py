import pandas as pd
import numpy as np
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test, logrank_test
from lifelines import KaplanMeierFitter
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Any

def compute_logrank_pvalue(df: pd.DataFrame, labels: np.ndarray, time_col: str, event_col: str) -> float:
    """
    Multivariate Log-Rank test across all clusters.
    Returns the p-value indicating if survival distributions differ significantly between clusters.
    """

    if time_col not in df.columns or event_col not in df.columns:
        return 1.0
        
    data = df.copy()
    data['Cluster'] = labels
    data = data.dropna(subset=[time_col, event_col])
    
    unique_clusters = data['Cluster'].unique()
    if len(unique_clusters) < 2:
        return 1.0

    results = multivariate_logrank_test(
        event_durations=data[time_col],
        event_observed=data[event_col],
        groups=data['Cluster']
    )
    return float(results.p_value)
   

def compute_pairwise_logrank(df: pd.DataFrame, labels: np.ndarray, time_col: str, event_col: str) -> List[Dict]:
    """
    Calculate the Log-Rank test for each pair of clusters.
    Returns a clean list of results for display.
    Ex: [{'c1': 0, 'c2': 1, 'p_value': 0.004, 'chi2': 5.4}, ...]
    """
    results_list = []
    
    if time_col not in df.columns or event_col not in df.columns:
        return results_list

    data = df.copy()
    data['Cluster'] = labels

    data = data.dropna(subset=[time_col, event_col])
    
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return results_list

    pw_results = pairwise_logrank_test(
        data[time_col], 
        data['Cluster'], 
        data[event_col]
    )
    
    # Extraction of useful data from Lifelines summary object
    summary = pw_results.summary
    
    for idx, row in summary.iterrows():
        # idx is a tuple (label1, label2)
        c1, c2 = idx
        results_list.append({
            'c1': c1,
            'c2': c2,
            'p_value': row['p'],
            'test_statistic': row['test_statistic']
        })
            
    return results_list


def strict_kaplan_meier_separation(
    event: pd.Series,
    time: pd.Series,
    labels: np.ndarray,
    *,
    pvalue_threshold: float = 0.05,
    max_pairwise_fail_fraction: float = 0.20,
    min_at_risk: int = 10,
    non_crossing_warmup_points_skip: int = 10,
    max_crossing_fraction: float = 0.0,
    auc_ratio_base: float = 2.0,
    auc_ratio_per_cluster: float = 0.2,
    violent_abs_drop: float = 0.05,
    violent_jump_ratio: float = 0.50,
    violent_max_drops_per_curve: int = 15,
    violent_warmup_points_skip: int = 20,
    plateau_max_fraction: float = 0.30,
    plateau_min_consecutive_points: int = 20,
    initial_size_ordering_similarity_pvalue: float = 0.05,
    enforce_initial_size_ordering: bool = True,
) -> Dict[str, Any]:
    """Clinical validator: strict separation of Kaplan–Meier curves across clusters.

    Returns a dict with:
      - approved: bool
      - reason: short failure reason if not approved
      - details: diagnostic payload (p-values, cutoff_time, aucs, etc.)
    """

    if labels is None:
        return {"approved": False, "reason": "missing_labels", "details": {}}

    labels_arr = np.asarray(labels)
    if len(event) != len(time) or len(event) != len(labels_arr):
        return {
            "approved": False,
            "reason": "length_mismatch",
            "details": {"n_event": len(event), "n_time": len(time), "n_labels": int(labels_arr.size)},
        }

    df = pd.DataFrame({"time": time, "event": event, "cluster": labels_arr})
    df = df.dropna(subset=["time", "event", "cluster"]).copy()
    if df.shape[0] == 0:
        return {"approved": False, "reason": "no_data_after_dropna", "details": {}}

    # Normalize types
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df = df.dropna(subset=["time", "event"]).copy()
    if df.shape[0] == 0:
        return {"approved": False, "reason": "no_numeric_time_event", "details": {}}

    # lifelines expects event_observed in {0,1}
    df["event"] = (df["event"].astype(float) > 0).astype(int)

    clusters = pd.unique(df["cluster"])
    n_clusters = int(len(clusters))
    if n_clusters < 2:
        return {"approved": False, "reason": "not_enough_clusters", "details": {"n_clusters": n_clusters}}

    cluster_sizes = df["cluster"].value_counts().to_dict()

    # Derive outcome name for optional AUC ratio overrides
    outcome_name = getattr(event, "name", None) or ""
    outcome_name = str(outcome_name).replace("outcome_", "")

    # Step 1: fit KM curves per cluster
    km_curves: Dict[Any, KaplanMeierFitter] = {}
    for c in clusters:
        mask = df["cluster"] == c
        if int(mask.sum()) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], event_observed=df.loc[mask, "event"], label=str(c))
        km_curves[c] = kmf

    if len(km_curves) < 2:
        return {"approved": False, "reason": "km_fit_failed", "details": {"n_km": len(km_curves)}}

    num_pairs = int(sum(1 for _ in combinations(clusters, 2)))
    if num_pairs == 0:
        return {"approved": False, "reason": "no_pairs", "details": {"n_clusters": n_clusters}}

    # Step 2: cutoff time with enough at risk in all clusters.
    # We will only evaluate separation up to this time window, by censoring everyone at cutoff_time.
    last_times = []
    for km in km_curves.values():
        et = km.event_table
        if "at_risk" not in et.columns:
            last_times.append(np.nan)
            continue
        times_ok = et.index[et["at_risk"] >= min_at_risk].to_numpy(dtype=float)
        last_times.append(times_ok.max() if times_ok.size > 0 else np.nan)

    cutoff_time = float(np.nanmin(np.asarray(last_times, dtype=float)))
    if not np.isfinite(cutoff_time) or cutoff_time <= 0:
        return {
            "approved": False,
            "reason": "no_common_cutoff_time",
            "details": {"min_at_risk": min_at_risk, "last_times": last_times},
        }

    # Step 3: pairwise log-rank tests ONLY on the common window (<= cutoff_time)
    # Implemented by administratively censoring durations at cutoff_time.
    df_c = df.copy()
    df_c["time_c"] = np.minimum(df_c["time"].to_numpy(dtype=float), cutoff_time)
    # Keep events only if they happen before (or at) cutoff_time.
    df_c["event_c"] = (df_c["event"].to_numpy(dtype=int) * (df_c["time"].to_numpy(dtype=float) <= cutoff_time)).astype(int)

    pairwise_pvalues: Dict[Tuple[Any, Any], float] = {}
    for c1, c2 in combinations(clusters, 2):
        m1 = df_c["cluster"] == c1
        m2 = df_c["cluster"] == c2
        if int(m1.sum()) == 0 or int(m2.sum()) == 0:
            continue
        test = logrank_test(
            df_c.loc[m1, "time_c"],
            df_c.loc[m2, "time_c"],
            event_observed_A=df_c.loc[m1, "event_c"],
            event_observed_B=df_c.loc[m2, "event_c"],
        )
        pairwise_pvalues[(c1, c2)] = float(test.p_value)

    n_fail_p = sum(1 for p in pairwise_pvalues.values() if p >= pvalue_threshold)
    fail_fraction = float(n_fail_p / num_pairs)
    if fail_fraction >= max_pairwise_fail_fraction:
        return {
            "approved": False,
            "reason": "pairwise_pvalues_not_significant",
            "details": {
                "pvalue_threshold": pvalue_threshold,
                "fail_fraction": fail_fraction,
                "max_fail_fraction": max_pairwise_fail_fraction,
                "cutoff_time": cutoff_time,
                "pairwise_pvalues": {str(k): v for k, v in pairwise_pvalues.items()},
            },
        }

    # Helper to read p(a,b)
    def p_between(a: Any, b: Any) -> float:
        return pairwise_pvalues.get((a, b), pairwise_pvalues.get((b, a), 1.0))

    # Step 4: non-crossing test up to cutoff_time
    crossings = 0
    for c1, c2 in combinations(clusters, 2):
        km1, km2 = km_curves.get(c1), km_curves.get(c2)
        if km1 is None or km2 is None:
            continue
        times_union = np.union1d(km1.event_table.index.to_numpy(dtype=float), km2.event_table.index.to_numpy(dtype=float))
        times_union = times_union[times_union <= cutoff_time]
        if times_union.size < 2:
            continue
        surv1 = km1.predict(times_union)
        surv2 = km2.predict(times_union)
        diff = np.asarray(surv1) - np.asarray(surv2)
        if diff.size > non_crossing_warmup_points_skip:
            diff = diff[non_crossing_warmup_points_skip:]
        tol = 1e-8
        if np.any(diff > tol) and np.any(diff < -tol):
            crossings += 1

    crossing_fraction = float(crossings / num_pairs)
    if crossing_fraction > max_crossing_fraction:
        return {
            "approved": False,
            "reason": "km_curves_crossing",
            "details": {
                "crossings": crossings,
                "num_pairs": num_pairs,
                "crossing_fraction": crossing_fraction,
                "max_crossing_fraction": max_crossing_fraction,
                "cutoff_time": cutoff_time,
            },
        }

    # Step 5: AUC ratio test (up to cutoff_time)
    aucs: Dict[Any, float] = {}
    for c, km in km_curves.items():
        times_all = km.event_table.index.to_numpy(dtype=float)
        times = times_all[times_all <= cutoff_time]
        if times.size < 2:
            times = times_all[:2] if times_all.size >= 2 else times_all
        if times.size < 2:
            aucs[c] = 0.0
            continue
        surv = np.asarray(km.predict(times), dtype=float)
        aucs[c] = float(np.trapezoid(surv, times))

    max_auc = max(aucs.values())
    min_auc = min(aucs.values())
    required_ratio = float(auc_ratio_base + auc_ratio_per_cluster * n_clusters)
    if min_auc <= 0 or (max_auc / min_auc) < required_ratio:
        return {
            "approved": False,
            "reason": "auc_ratio_too_small",
            "details": {
                "aucs": {str(k): float(v) for k, v in aucs.items()},
                "max_auc": float(max_auc),
                "min_auc": float(min_auc),
                "required_ratio": required_ratio,
                "n_clusters": n_clusters,
                "cutoff_time": cutoff_time,
                "outcome_name": outcome_name,
            },
        }

    # Step 5bis: violent drops test (per curve)
    for c, km in km_curves.items():
        et = km.event_table
        if "at_risk" not in et.columns or "observed" not in et.columns:
            continue
        times = et.index.to_numpy(dtype=float)
        m = times <= cutoff_time
        times = times[m]
        if times.size < 2:
            continue
        surv = np.asarray(km.predict(times), dtype=float)
        drops = np.maximum(surv[:-1] - surv[1:], 0.0)

        observed = et["observed"].to_numpy()[m][1:]
        at_risk = et["at_risk"].to_numpy()[m][1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            jump_ratio = np.where(at_risk > 0, observed / at_risk, 0.0)

        if drops.size > violent_warmup_points_skip:
            drops_eval = drops[violent_warmup_points_skip:]
            jump_eval = jump_ratio[violent_warmup_points_skip:]
        else:
            drops_eval = drops
            jump_eval = jump_ratio

        violent_mask = (drops_eval >= violent_abs_drop) | (jump_eval >= violent_jump_ratio)
        violent_count = int(np.count_nonzero(violent_mask))
        if violent_count > violent_max_drops_per_curve:
            return {
                "approved": False,
                "reason": "violent_drops_detected",
                "details": {
                    "cluster": str(c),
                    "violent_count": violent_count,
                    "violent_max": violent_max_drops_per_curve,
                    "cutoff_time": cutoff_time,
                },
            }

    # Step 5ter: plateau fraction on the two smallest AUC curves
    smallest_two = sorted(aucs.items(), key=lambda kv: kv[1])[:2]
    tol_eq = 1e-10
    for c, _ in smallest_two:
        km = km_curves[c]
        times_all = km.event_table.index.to_numpy(dtype=float)
        times = times_all[times_all <= cutoff_time]
        if times.size < 2:
            continue
        surv = np.asarray(km.predict(times), dtype=float)

        plateau_time = 0.0
        consecutive = 0
        for i in range(times.size - 1):
            dt = float(times[i + 1] - times[i])
            if dt <= 0:
                continue
            if abs(float(surv[i + 1] - surv[i])) <= tol_eq:
                consecutive += 1
            else:
                if consecutive > plateau_min_consecutive_points:
                    plateau_time += float(dt * consecutive)
                consecutive = 0

        total_span = float(max(cutoff_time - float(times[0]), 0.0))
        if total_span > 0:
            plateau_frac = float(plateau_time / total_span)
            if plateau_frac > plateau_max_fraction:
                return {
                    "approved": False,
                    "reason": "plateau_fraction_too_high",
                    "details": {
                        "cluster": str(c),
                        "plateau_fraction": plateau_frac,
                        "plateau_max_fraction": plateau_max_fraction,
                        "cutoff_time": cutoff_time,
                    },
                }

    # Step 6: initial size ordering consistency along AUC ordering
    if enforce_initial_size_ordering:
        ordered_clusters = [c for c, _ in sorted(aucs.items(), key=lambda kv: kv[1])]
        for idx in range(len(ordered_clusters) - 1):
            c_curr = ordered_clusters[idx]
            c_next = ordered_clusters[idx + 1]

            size_curr = int(cluster_sizes.get(c_curr, 0))
            size_next = int(cluster_sizes.get(c_next, 0))

            # if smaller AUC curve has *smaller* cluster, ok
            if size_curr < size_next:
                continue
            # if curves are clearly different (significant), allow size inversion
            if p_between(c_curr, c_next) > initial_size_ordering_similarity_pvalue:
                continue

            return {
                "approved": False,
                "reason": "initial_size_ordering_inconsistent",
                "details": {
                    "c_curr": str(c_curr),
                    "c_next": str(c_next),
                    "size_curr": size_curr,
                    "size_next": size_next,
                    "p_value": p_between(c_curr, c_next),
                },
            }

    return {
        "approved": True,
        "reason": "ok",
        "details": {
            "n_clusters": n_clusters,
            "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
            "cutoff_time": cutoff_time,
            "pairwise_pvalues": {str(k): float(v) for k, v in pairwise_pvalues.items()},
            "aucs": {str(k): float(v) for k, v in aucs.items()},
            "auc_required_ratio": required_ratio,
            "outcome_name": outcome_name,
        },
    }