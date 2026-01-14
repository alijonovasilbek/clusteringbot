# visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

# Matplotlib backend
plt.switch_backend('Agg')

# Stil
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class Visualizer:

    @staticmethod
    def plot_kmeans(X, kmeans, title="K-Means Clustering"):
        """K-Means natijalarini chizish"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Nuqtalarni chizish
        scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels,
                             cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Markazlarni chizish
        ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                   c='red', marker='X', s=300, edgecolors='black',
                   linewidth=2, label='Markazlar', zorder=5)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Klaster ID', fontsize=10)

        plt.tight_layout()

        # BytesIO ga saqlash
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    @staticmethod
    def plot_dbscan(X, dbscan, title="DBSCAN Clustering"):
        """DBSCAN natijalarini chizish"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Noise nuqtalar
        noise_mask = dbscan.labels == -2

        # Klaster nuqtalari
        if np.any(~noise_mask):
            scatter = ax.scatter(X[~noise_mask, 0], X[~noise_mask, 1],
                                 c=dbscan.labels[~noise_mask], cmap='viridis',
                                 alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Klaster ID', fontsize=10)

        # Noise nuqtalar
        if np.any(noise_mask):
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1],
                       c='red', marker='x', s=100, alpha=0.8,
                       label=f'Shovqin ({np.sum(noise_mask)} nuqta)', linewidth=2)

        # Core points
        if len(dbscan.core_points) > 0:
            core_indices = dbscan.core_points
            ax.scatter(X[core_indices, 0], X[core_indices, 1],
                       facecolors='none', edgecolors='yellow',
                       s=150, linewidth=2, label='Core Points')

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    @staticmethod
    def plot_elbow(k_range, inertias):
        """Elbow grafigi"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Klasterlar Soni (K)', fontsize=12)
        ax.set_ylabel('Inertia (SSE)', fontsize=12)
        ax.set_title('Elbow Method - Optimal K ni Topish', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    @staticmethod
    def plot_comparison(X, kmeans, dbscan):
        """Ikkalasini taqqoslash"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # K-Means
        axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels,
                        cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                        c='red', marker='X', s=300, edgecolors='black', linewidth=2)
        axes[0].set_title('K-Means', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)

        # DBSCAN
        noise_mask = dbscan.labels == -2
        if np.any(~noise_mask):
            axes[1].scatter(X[~noise_mask, 0], X[~noise_mask, 1],
                            c=dbscan.labels[~noise_mask], cmap='viridis',
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(noise_mask):
            axes[1].scatter(X[noise_mask, 0], X[noise_mask, 1],
                            c='red', marker='x', s=100, alpha=0.8, linewidth=2)
        axes[1].set_title('DBSCAN', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf