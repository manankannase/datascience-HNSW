# Adaptive Cluster Budget

`ClusteredHNSW.query()` currently receives one fixed `top_clusters` value for every query. This utility provides a small experiment layer that selects the value from the nearest-centroid margin:

```text
margin = (second_nearest_centroid_distance - nearest_centroid_distance)
         / nearest_centroid_distance
```

A large margin means one cluster is clearly closest, so the policy can use a small budget. A small margin means the query lies close to a cluster boundary and should search more clusters.

## Use

```python
from adaptive_budget.adaptive_budget import AdaptiveClusterBudget

policy = AdaptiveClusterBudget(
    budgets=(1, 2, 4, 8),
    thresholds=(0.05, 0.20, 0.50),
)

distances = np.linalg.norm(clustered.centroids - query_vector, axis=1)
top_clusters = policy.choose(distances)
neighbors = clustered.query(query_vector, k=10, top_clusters=top_clusters)
```

The thresholds above are placeholders, not reported results. Tune them with development queries only:

1. For each development query, run fixed budgets such as 1, 2, 4, and 8.
2. Record the smallest budget that meets a chosen Recall@k target.
3. Choose thresholds that minimize average budget while meeting the target on development data.
4. Freeze the policy and report Recall@k, p50/p95 latency, and average clusters searched on held-out test queries.

Run the included checks with:

```bash
python -m unittest adaptive_budget/test_adaptive_budget.py
```
