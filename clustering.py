# clustering.py
import logging
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def parallel_agglomerative_clustering(embeddings: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative clustering using parallel processing for distance computation.

    Args:
    embeddings (np.ndarray): Array of embeddings.
    n_clusters (int): Number of clusters.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Cluster labels and linkage matrix.
    """
    n_samples = embeddings.shape[0]

    logger.info(f"Starting agglomerative clustering for {n_samples} samples")

    # Perform agglomerative clustering directly on embeddings
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(embeddings)
    logger.info(f"Finished agglomerative clustering. Number of clusters: {len(np.unique(labels))}")

    # Compute linkage matrix
    logger.info("Computing linkage matrix")
    linkage_matrix = linkage(embeddings, method='ward', metric='euclidean')
    logger.info("Finished computing linkage matrix")

    return labels, linkage_matrix

def perform_clustering(embeddings: np.ndarray, method: str = "kmeans", n_clusters: int = 10) -> Tuple[
    np.ndarray, Optional[np.ndarray]]:
    """
    Apply clustering algorithms to the embeddings.

    Args:
    embeddings (np.ndarray): Numpy array of embeddings.
    method (str): Clustering method to use ('kmeans', 'dbscan', or 'agglomerative').
    n_clusters (int): Number of clusters for KMeans and AgglomerativeClustering.

    Returns:
    Tuple[np.ndarray, Optional[np.ndarray]]: Tuple containing cluster labels and linkage matrix (for hierarchical clustering).
    """
    try:
        logger.info(f"Starting clustering with method: {method}")
        logger.info(f"Input embeddings shape: {embeddings.shape}")

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = clusterer.fit_predict(embeddings)
            logger.info(f"KMeans clustering completed. Number of clusters: {len(np.unique(clusters))}")
            return clusters, None
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            clusters = clusterer.fit_predict(embeddings)
            logger.info(f"DBSCAN clustering completed. Number of clusters: {len(np.unique(clusters))}")
            return clusters, None
        elif method == "agglomerative":
            clusters, linkage_matrix = parallel_agglomerative_clustering(embeddings, n_clusters)
            if len(np.unique(clusters)) == 1:
                logger.warning("Agglomerative clustering resulted in only one cluster")
            return clusters, linkage_matrix
        else:
            raise ValueError("Invalid clustering method")

    except Exception as e:
        logger.error(f"Error performing clustering with method {method}: {str(e)}")
        return np.array([]), None

def load_clustering_results(method: str, output_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load previously saved clustering results.

    Args:
    method (str): Clustering method ('kmeans', 'dbscan', or 'agglomerative').
    output_dir (str): Directory where clustering results are saved.

    Returns:
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Tuple containing cluster labels and linkage matrix (if applicable).
    """
    try:
        clusters_file = os.path.join(output_dir, f"{method}_clusters.npy")
        if os.path.exists(clusters_file):
            clusters = np.load(clusters_file)
            logger.info(f"Loaded existing clusters for {method}")

            linkage_matrix = None
            if method == "agglomerative":
                linkage_file = os.path.join(output_dir, f"{method}_linkage.npy")
                if os.path.exists(linkage_file):
                    linkage_matrix = np.load(linkage_file)
                    logger.info(f"Loaded existing linkage matrix for {method}")

            return clusters, linkage_matrix
        else:
            logger.info(f"No existing clustering results found for {method}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading clustering results for {method}: {str(e)}")
        return None, None

def process_clustering(embeddings: np.ndarray, method: str, n_clusters: int, output_dir: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process clustering for a given method, loading existing results if available.

    Args:
    embeddings (np.ndarray): Numpy array of embeddings.
    method (str): Clustering method to use.
    n_clusters (int): Number of clusters for KMeans and AgglomerativeClustering.
    output_dir (str): Directory to save/load clustering results.

    Returns:
    Tuple[np.ndarray, Optional[np.ndarray]]: Tuple containing cluster labels and linkage matrix (if applicable).
    """
    cached_results = load_clustering_results(method, output_dir)
    if cached_results[0] is not None:
        return cached_results

    logger.info(f"Starting clustering with method: {method}")
    clusters, linkage_matrix = perform_clustering(embeddings, method, n_clusters)

    # Save clustering results
    save_clustering_results(clusters, linkage_matrix, method, output_dir)

    logger.info(f"Finished clustering with method: {method}")
    return clusters, linkage_matrix

def save_clustering_results(clusters: np.ndarray, linkage_matrix: Optional[np.ndarray], method: str, output_dir: str):
    """Save clustering results to the specified output directory."""
    try:
        np.save(os.path.join(output_dir, f"{method}_clusters.npy"), clusters)
        if linkage_matrix is not None and method == "agglomerative":
            np.save(os.path.join(output_dir, f"{method}_linkage.npy"), linkage_matrix)
            logger.info(f"Saved linkage matrix for {method} clustering")
        logger.info(f"Saved clustering results for {method}")
    except Exception as e:
        logger.error(f"Error saving clustering results for {method}: {str(e)}")
