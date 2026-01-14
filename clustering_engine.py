# clustering_engine.py
import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # Sum of squared distances
        self.n_iter_ = 0

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Tasodifiy boshlang'ich markazlar
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices].astype(float)

        for iteration in range(self.max_iters):
            # Klasterlarga biriktirish
            old_labels = self.labels
            self.labels = self._assign_clusters(X)

            # Yangi markazlarni hisoblash
            new_centroids = self._calculate_centroids(X, self.labels)

            # Konvergensiya tekshiruvi
            if np.allclose(self.centroids, new_centroids):
                self.n_iter_ = iteration + 1
                break

            self.centroids = new_centroids

        # Inertia hisoblash
        self._calculate_inertia(X)
        return self

    def _assign_clusters(self, X):
        distances = np.zeros((len(X), self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                centroids[i] = X[np.random.choice(len(X))]
        return centroids

    def _calculate_inertia(self, X):
        """Inertia (SSE) hisoblash"""
        self.inertia_ = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)

    def predict(self, X):
        return self._assign_clusters(X)

    def get_cluster_info(self):
        """Har bir klaster haqida ma'lumot"""
        info = []
        for i in range(self.k):
            n_points = np.sum(self.labels == i)
            info.append({
                'cluster_id': i,
                'n_points': n_points,
                'centroid': self.centroids[i].tolist(),
                'percentage': (n_points / len(self.labels)) * 100
            })
        return info


class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = None
        self.core_points = []
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def fit(self, X):
        n_samples = len(X)
        self.labels = np.full(n_samples, -1)

        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_pts:
                self.labels[i] = -2  # Noise
                continue

            self.core_points.append(i)
            self._expand_cluster(X, i, neighbors, cluster_id)
            cluster_id += 1

        self.n_clusters_ = cluster_id
        self.n_noise_ = np.sum(self.labels == -2)
        return self

    def _get_neighbors(self, X, point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0].tolist()

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = list(neighbors)

        while queue:
            current_point = queue.pop(0)

            if self.labels[current_point] == -2:
                self.labels[current_point] = cluster_id

            if self.labels[current_point] != -1:
                continue

            self.labels[current_point] = cluster_id
            new_neighbors = self._get_neighbors(X, current_point)

            if len(new_neighbors) >= self.min_pts:
                self.core_points.append(current_point)
                for neighbor in new_neighbors:
                    if neighbor not in queue and self.labels[neighbor] == -1:
                        queue.append(neighbor)

    def get_cluster_info(self):
        """Har bir klaster haqida ma'lumot"""
        info = []
        for i in range(self.n_clusters_):
            n_points = np.sum(self.labels == i)
            info.append({
                'cluster_id': i,
                'n_points': n_points,
                'percentage': (n_points / len(self.labels)) * 100
            })
        return info


class ElbowMethod:
    """Optimal K ni topish uchun Elbow Method"""

    @staticmethod
    def calculate(X, max_k=10):
        inertias = []
        k_range = range(1, min(max_k + 1, len(X)))

        for k in k_range:
            kmeans = KMeans(k=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        return list(k_range), inertias