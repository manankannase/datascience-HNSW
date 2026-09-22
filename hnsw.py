import os
import math
import random
import numpy as np
from heapq import heappush, heappop
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

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

logfile = 'HNSW_experiment_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
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

def run_hnsw_experiments(data, queries, gt_idx, k_values, hnsw_params=None):
    hnsw_params = hnsw_params or {'M': 16, 'ef_construction': 200}
    results = []
    if not os.path.exists("hnsw_experiment_results.csv"):
        with open("hnsw_experiment_results.csv", "w") as f:
            f.write("k,recall,avg_distance,indexing_time,query_time\n")

    for k in k_values:
        logging.info(f"\nRunning HNSW experiment for k={k}")
        # Build HNSW index
        t0 = datetime.datetime.now()
        hnsw = HNSW(dim=data.shape[1], max_elements=len(data), M=hnsw_params.get("M", 16), ef_construction=hnsw_params.get("ef_construction", 200))
        for idx in tqdm(range(len(data)), desc=f"Inserting to HNSW (k={k})"):
            hnsw.insert(data[idx])
        t1 = datetime.datetime.now()
        indexing_time = (t1 - t0).total_seconds()
        logging.info(f"HNSW indexing time: {indexing_time:.4f} seconds")

        # Load or compute ground truth for this k
        if k == 10 and gt_idx.shape[1] >= 10:
            gt_k = gt_idx
        else:
            logging.info(f"Running brute-force KNN for k={k}")
            knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
            knn.fit(data)
            gt_k = knn.kneighbors(queries, return_distance=False)

        # Query
        start_query = datetime.datetime.now()
        retrieved = []
        for q in tqdm(queries, desc=f"Querying HNSW (k={k})"):
            retrieved.append(hnsw.search(q, k=k))
        query_time = (datetime.datetime.now() - start_query).total_seconds()

        recall, avg_dist = compute_recall_and_avg_dist(retrieved, gt_k, queries, data, k)
        log_msg = f"Result — k={k} → Recall={recall:.4f}, AvgDist={avg_dist:.4f}"
        logging.info(log_msg)
        logging.info(f"Query time: {query_time:.4f} seconds")

        results.append((k, recall, avg_dist, indexing_time, query_time))
        with open("hnsw_experiment_results.csv", "a") as f:
            f.write(f"{k},{recall:.6f},{avg_dist:.6f},{indexing_time:.6f},{query_time:.6f}\n")

    logging.info("Saved results to hnsw_experiment_results.csv")
    return results




if __name__ == "__main__":
    # ====== Settings ======
    MAIN_DIR = "data/glove_6b_50d_split/"
    DATA_PATH = MAIN_DIR + "X_data.txt"
    QUERY_PATH = MAIN_DIR + "X_query.txt"
    GT_DIR = MAIN_DIR + "knn_k10"
    K_LIST = [10, 20, 50, 100]

    hnsw_params = {
        'M': 16,
        'ef_construction': 200
    }

    logging.info("Loading data…")
    data = load_vecs(DATA_PATH)
    queries = load_vecs(QUERY_PATH)
    gt_idx = load_gt(GT_DIR, k=10)
    logging.info("Data loaded successfully.")

    logging.info(f"Running HNSW experiments for k values: {K_LIST}")
    results = run_hnsw_experiments(data, queries, gt_idx, K_LIST, hnsw_params)
    logging.info("All HNSW experiments completed.")
