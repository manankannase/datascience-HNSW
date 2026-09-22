import hnswlib
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import os

class ClusteredHNSW:
    def __init__(self, n_clusters=512, dim=50, hnsw_params=None):
        self.n_clusters = n_clusters
        self.dim = dim
        self.hnsw_params = hnsw_params or {}
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_indices = defaultdict(list)
        self.cluster_hnsw = {}
        self.centroids = None

    def fit(self, data: np.ndarray):
        print("Running KMeans...")
        labels = self.kmeans.fit_predict(data)
        self.centroids = self.kmeans.cluster_centers_
        print("KMeans done. Building HNSW for each cluster...")

        for idx, label in enumerate(labels):
            self.cluster_indices[label].append(idx)

        for c in tqdm(range(self.n_clusters), desc="Clusters"):
            indices = self.cluster_indices[c]
            cluster_data = data[indices]
            p = hnswlib.Index(space='l2', dim=self.dim)
            p.init_index(max_elements=len(indices), **self.hnsw_params)
            p.add_items(cluster_data, np.arange(len(indices)))
            self.cluster_hnsw[c] = p

    def query(self, query_vec: np.ndarray, k=10, top_clusters=1):
        dists_to_centroids = np.linalg.norm(self.centroids - query_vec, axis=1)
        nearest_clusters = np.argsort(dists_to_centroids)[:top_clusters]

        candidates = []
        for c in nearest_clusters:
            p = self.cluster_hnsw[c]
            labels, distances = p.knn_query(query_vec, k=min(k, p.get_current_count()))
            for dist, idx in zip(distances[0], labels[0]):
                global_idx = self.cluster_indices[c][idx]
                candidates.append((dist, global_idx))

        candidates.sort(key=lambda x: x[0])
        return [idx for _, idx in candidates[:k]]


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
    n_queries = gt_indices.shape[0]
    total = 0
    for i in range(n_queries):
        true_set = set(gt_indices[i, :k])
        pred_set = set(retrieved[i])
        total += len(true_set & pred_set) / k
    return total / n_queries


if __name__ == '__main__':
    data = load_vecs('data/glove_6b_50d_split/X_data.txt')
    queries = load_vecs('data/glove_6b_50d_split/X_query.txt')
    gt_indices = load_gt('data/glove_6b_50d_split/knn_k10', k=10)

    clustered = ClusteredHNSW(
        n_clusters=512,
        dim=data.shape[1],
        hnsw_params={
            'ef_construction': 200,
            'M': 16
        }
    )

    print("Fitting clustered HNSW...")
    clustered.fit(data)

    for top_clusters in [1, 2, 4]:
        print(f'\nEvaluating with top_clusters = {top_clusters}')
        retrieved = []
        for q in tqdm(queries, desc=f'Querying (top_clusters={top_clusters})'):
            neighbors = clustered.query(q, k=10, top_clusters=top_clusters)
            retrieved.append(neighbors)
        retrieved = np.array(retrieved)

        recall = compute_recall(retrieved, gt_indices, k=10)
        print(f'Recall@10 with top_clusters={top_clusters}: {recall:.4f}')
