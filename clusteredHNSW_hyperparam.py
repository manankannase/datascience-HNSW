import os
import math
import random
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time

# set all seed
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


# -----------------------
# Logging Setup
# -----------------------
import datetime
import logging

logfile = 'clusteredHNSW_experiment_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.info("Logging initialized.")


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

    def fit(self, data: np.ndarray, return_time=False):
        t0 = time.time()
        print("Running KMeans...")
        labels = self.kmeans.fit_predict(data)
        self.centroids = self.kmeans.cluster_centers_
        t1 = time.time()
        cluster_time = t1 - t0

        for idx, lbl in enumerate(labels):
            self.cluster_indices[lbl].append(idx)

        print("Building per-cluster HNSW indices...")
        t2 = time.time()
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
        t3 = time.time()
        hnsw_time = t3 - t2

        if return_time:
            return cluster_time, hnsw_time

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

def compute_recall_and_avg_dist(retrieved, gt_indices, queries, data, k):
    total_recall = 0.0
    total_dist = 0.0
    for i in range(len(queries)):
        true = set(gt_indices[i, :k])
        pred = set(retrieved[i])
        total_recall += len(true & pred) / k
        total_dist += sum(np.linalg.norm(queries[i] - data[j]) for j in retrieved[i]) / len(retrieved[i])
    return total_recall / len(queries), total_dist / len(queries)




# -----------------------
# Experiment Runner
# -----------------------
def run_experiments(data, queries, gt_idx, k_values, cluster_options, top_cluster_options):
    results = []
    if not os.path.exists("clusteredHNSE_experiment_results.csv"):
        with open("clusteredHNSE_experiment_results.csv", "w") as f:
            f.write("k,n_clusters,top_clusters,recall,avg_distance,cluster_time,indexing_time,query_time\n")

    for k in k_values:
        # Ground truth
        if k == 10:
            gt_k = gt_idx
        else:
            logging.info(f"Running brute-force KNN for k={k}")
            knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
            knn.fit(data)
            gt_k = knn.kneighbors(queries, return_distance=False)

        for n_clusters in cluster_options:
            logging.info(f"\nFitting KMeans with {n_clusters} clusters")
            clustered = ClusteredHNSW(n_clusters=n_clusters, dim=data.shape[1], hnsw_params={'M': 16, 'ef_construction': 200})
            cluster_time, hnsw_time = clustered.fit(data, return_time=True)
            logging.info(f"KMeans fitting time: {cluster_time:.4f} seconds")
            logging.info(f"Building HNSW indices time: {hnsw_time:.4f} seconds")

            for top_c in top_cluster_options:
                logging.info(f"\nRunning query: k={k}, clusters={n_clusters}, top_clusters={top_c}")
                start_query = time.time()
                retrieved = []
                for q in tqdm(queries, desc=f"Querying k={k}"):
                    retrieved.append(clustered.query(q, k=k, top_clusters=top_c))
                query_time = time.time() - start_query

                recall, avg_dist = compute_recall_and_avg_dist(retrieved, gt_k, queries, data, k)
                log_msg = f"Result — k={k}, clusters={n_clusters}, top_clusters={top_c} → Recall={recall:.4f}, AvgDist={avg_dist:.4f}"
                logging.info(log_msg)
                logging.info(f"Query time: {query_time:.4f} seconds")

                results.append((k, n_clusters, top_c, recall, avg_dist, cluster_time, hnsw_time, query_time))
                # Save after each config
                with open("clusteredHNSE_experiment_results.csv", "a") as f:
                    f.write(f"{k},{n_clusters},{top_c},{recall:.6f},{avg_dist:.6f},{cluster_time:.6f},{hnsw_time:.6f},{query_time:.6f}\n")

    # Save results to a file for plotting
    # np.savetxt("clusteredHNSE_experiment_results.csv", results, fmt="%.6f", delimiter=",", header="k,n_clusters,top_clusters,recall,avg_distance")
    logging.info("Saved results to clusteredHNSE_experiment_results.csv")

    return results



if __name__ == "__main__":
    # Paths
    data_path  = 'data/glove_6b_50d_split/X_data.txt'
    query_path = 'data/glove_6b_50d_split/X_query.txt'
    gt_dir     = 'data/glove_6b_50d_split/knn_k10'

    # Load
    logging.info("Loading data…")
    data = load_vecs(data_path)
    queries = load_vecs(query_path)
    gt_idx = load_gt(gt_dir, k=10)
    logging.info("Data loaded successfully.")

    # Parameters to try
    k_values = [5, 10, 20, 50, 100]
    cluster_options = [256, 512]
    top_cluster_options = [1, 2, 4, 8, 16]
    logging.info("Parameters set: k_values=%s, cluster_options=%s, top_cluster_options=%s", k_values, cluster_options, top_cluster_options)

    logging.info("Starting experiments...")
    results = run_experiments(data, queries, gt_idx, k_values, cluster_options, top_cluster_options)
    logging.info("All experiments completed.")


# nohup python Shardul/clusteredHNSW_hyperparam.py > clusteredHNSW_hyperparam.log 2>&1 &