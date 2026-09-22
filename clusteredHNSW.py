import os
import math
import random
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# -----------------------
# 1) Your Custom HNSW
# -----------------------
class HNSW:
    def __init__(self, dim, max_elements, M=16, ef_construction=200):
        self.dim = dim
        self.max_elements = max_elements     # informational
        self.M = M
        self.ef_construction = ef_construction

        self.layers = []     # list of dicts: layer -> {node_idx: [neighbor_idxs]}
        self.vectors = []    # list of np.array vectors
        self.entry_point = None

        # for roulette‐wheel layer assignment
        self.ml = 1.0 / math.log(self.M) if self.M > 1 else 1.0

    def _get_layer(self):
        return max(0, int(-math.log(random.random()) * self.ml))

    def _search_layer(self, query_vec, layer_idx, ep_idx, ef):
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return []
        graph = self.layers[layer_idx]
        if not graph:
            return []

        # validate entry point
        if ep_idx not in graph or ep_idx >= len(self.vectors):
            valid = [i for i in graph if i < len(self.vectors)]
            if not valid:
                return []
            ep_idx = random.choice(valid)

        visited = {ep_idx}
        candidates = []
        results = []

        d0 = np.linalg.norm(query_vec - self.vectors[ep_idx])
        heappush(candidates, (d0, ep_idx))
        heappush(results, (-d0, ep_idx))

        while candidates:
            dist_c, idx_c = heappop(candidates)
            worst = -results[0][0]
            if dist_c > worst and len(results) >= ef:
                break

            for nbr in graph.get(idx_c, []):
                if nbr in visited or nbr >= len(self.vectors):
                    continue
                visited.add(nbr)
                d = np.linalg.norm(query_vec - self.vectors[nbr])
                if len(results) < ef or d < -results[0][0]:
                    heappush(candidates, (d, nbr))
                    heappush(results, (-d, nbr))
                    if len(results) > ef:
                        heappop(results)

        out = sorted([(-d, idx) for d, idx in results])
        return out[:ef]

    def insert(self, vector):
        vec = np.array(vector, dtype=np.float32)
        idx_new = len(self.vectors)
        self.vectors.append(vec)

        lvl = self._get_layer()
        # ensure we have enough layers
        while len(self.layers) <= lvl:
            self.layers.append({})

        ep = self.entry_point
        # Phase 1: navigate from top layer down to lvl+1
        for layer in range(len(self.layers)-1, lvl, -1):
            if ep is None or not self.layers[layer]:
                continue
            res = self._search_layer(vec, layer, ep, ef=1)
            if res:
                ep = res[0][1]

        # Phase 2: for layers lvl → 0, search and link
        for layer in range(min(lvl, len(self.layers)-1), -1, -1):
            if ep is None or not self.layers[layer]:
                neighbors = []
            else:
                neighbors = self._search_layer(vec, layer, ep, self.ef_construction)
                if neighbors:
                    ep = neighbors[0][1]

            # pick top‐M
            conns = [i for _, i in neighbors[:self.M]]
            self.layers[layer][idx_new] = conns

            # backlink & prune
            for nbr in conns:
                nbr_conns = self.layers[layer].setdefault(nbr, [])
                if idx_new not in nbr_conns:
                    nbr_conns.append(idx_new)
                    if len(nbr_conns) > self.M:
                        # prune furthest
                        valid = [(np.linalg.norm(self.vectors[nbr] - self.vectors[c]), c)
                                 for c in nbr_conns]
                        valid.sort()
                        self.layers[layer][nbr] = [c for _, c in valid[:self.M]]

        # update entry point if higher level
        if self.entry_point is None or lvl > self._node_max_layer(self.entry_point):
            self.entry_point = idx_new

    def _node_max_layer(self, node):
        ml = -1
        for i, lg in enumerate(self.layers):
            if node in lg:
                ml = i
        return ml

    def search(self, query_vec, k=10):
        if self.entry_point is None:
            return []
        vec = np.array(query_vec, dtype=np.float32)
        ep = self.entry_point
        # top‐down search to layer 1
        for layer in range(len(self.layers)-1, 0, -1):
            if not self.layers[layer]:
                continue
            res = self._search_layer(vec, layer, ep, ef=1)
            if res:
                ep = res[0][1]
        # final search in layer 0 with ef = max(k, ef_construction)
        ef = max(k, self.ef_construction)
        res = self._search_layer(vec, 0, ep, ef)
        return [idx for _, idx in res[:k]]


# ---------------------------------
# 2) ClusteredHNSW using your HNSW
# ---------------------------------
class ClusteredHNSW:
    def __init__(self, n_clusters=512, dim=50, hnsw_params=None):
        self.n_clusters = n_clusters
        self.dim = dim
        self.hnsw_params = hnsw_params or {}
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.cluster_indices = defaultdict(list)
        self.cluster_hnsw = {}
        self.centroids = None

    def fit(self, data: np.ndarray):
        print("Running KMeans...")
        labels = self.kmeans.fit_predict(data)
        self.centroids = self.kmeans.cluster_centers_

        for idx, lbl in enumerate(labels):
            self.cluster_indices[lbl].append(idx)

        print("Building per-cluster HNSW indices...")
        for c in tqdm(range(self.n_clusters), desc="Clusters"):
            inds = self.cluster_indices[c]
            if not inds:
                continue
            hnsw = HNSW(dim=self.dim,
                        max_elements=len(inds),
                        M=self.hnsw_params.get("M", 16),
                        ef_construction=self.hnsw_params.get("ef_construction", 200))
            for idx in inds:
                hnsw.insert(data[idx])
            self.cluster_hnsw[c] = hnsw

    def query(self, query_vec: np.ndarray, k=10, top_clusters=1):
        dists = np.linalg.norm(self.centroids - query_vec, axis=1)
        best = np.argsort(dists)[:top_clusters]

        candidates = []
        for c in best:
            hnsw = self.cluster_hnsw.get(c)
            if hnsw is None:
                continue
            locals_ = hnsw.search(query_vec, k)
            for loc in locals_:
                gid = self.cluster_indices[c][loc]
                dist = np.linalg.norm(query_vec - hnsw.vectors[loc])
                candidates.append((dist, gid))

        candidates.sort(key=lambda x: x[0])
        return [gid for _, gid in candidates[:k]]


# ---------------------------------
# 3) Utilities & Main Execution
# ---------------------------------
def load_vecs(path):
    vecs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            vecs.append(np.array(parts[1:], dtype=np.float32))
    return np.vstack(vecs)

def load_gt(gt_dir, k):
    idx_path = os.path.join(gt_dir, 'query_knn_indices.txt')
    return np.loadtxt(idx_path, dtype=int)

def compute_recall(retrieved, gt_indices, k):
    n = gt_indices.shape[0]
    tot = 0.0
    for i in range(n):
        true = set(gt_indices[i, :k])
        pred = set(retrieved[i])
        tot += len(true & pred) / k
    return tot / n

if __name__ == "__main__":
    # Paths
    data_path  = 'data/glove_6b_50d_split/X_data.txt'
    query_path = 'data/glove_6b_50d_split/X_query.txt'
    gt_dir     = 'data/glove_6b_50d_split/knn_k10'

    # Load
    print("Loading vectors…")
    data    = load_vecs(data_path)
    queries = load_vecs(query_path)
    gt_idx  = load_gt(gt_dir, k=10)

    # Build clustered index
    clustered = ClusteredHNSW(
        n_clusters=512,
        dim=data.shape[1],
        hnsw_params={'M':16, 'ef_construction':200}
    )
    clustered.fit(data)

    # Evaluate Recall@10
    for tc in [1, 2, 4]:
        print(f"\nEvaluating top_clusters = {tc}")
        retrieved = []
        for q in tqdm(queries, desc="Querying"):
            retrieved.append(clustered.query(q, k=10, top_clusters=tc))
        retrieved = np.array(retrieved)
        rec = compute_recall(retrieved, gt_idx, k=10)
        print(f"Recall@10 = {rec:.4f}")
