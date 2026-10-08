import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import traceback

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest


def check_unsupervised_readiness(processed_dir, pca_dir=None):
    """
    Checks if preprocessing artifacts exist for unsupervised learning.
    """
    required_files = ['X_train.csv', 'preprocessor.pkl']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 5 (Preprocessing) first."}
    
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)):
            missing_files.append(f)
    
    if missing_files:
        return {"ready": False, "reason": f"Missing preprocessing artifacts: {', '.join(missing_files)}. Please complete Phase 5 (Preprocessing) first."}
    
    # Check PCA availability
    pca_available = False
    if pca_dir and os.path.exists(pca_dir):
        pca_metadata_path = os.path.join(pca_dir, 'pca_metadata.json')
        if os.path.exists(pca_metadata_path):
            pca_available = True
    
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        
        return {
            "ready": True,
            "samples_count": len(X_train),
            "features_count": len(X_train.columns),
            "feature_names": X_train.columns.tolist(),
            "pca_available": pca_available
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading preprocessing artifacts: {str(e)}"}


def get_feature_stats(X, feature_names):
    """
    Calculate statistics for cluster/anomaly characteristics.
    """
    stats = {}
    for feat in feature_names:
        if feat in X.columns:
            stats[feat] = {
                "mean": float(X[feat].mean()),
                "std": float(X[feat].std()),
                "min": float(X[feat].min()),
                "max": float(X[feat].max())
            }
    return stats


def run_kmeans(processed_dir, pca_dir, n_clusters=3, use_pca=False, random_state=42):
    """
    Runs K-Means clustering.
    """
    try:
        # Load data
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        
        if use_pca and pca_dir and os.path.exists(pca_dir):
            # Use PCA-transformed data
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            X = X_train_pca.values
            feature_names = X_train_pca.columns.tolist()
            representation = "PCA Features"
        else:
            X = X_train.values
            feature_names = X_train.columns.tolist()
            representation = "Processed Features"
        
        # Validate K
        if n_clusters >= len(X):
            return {"success": False, "error": f"Number of clusters ({n_clusters}) must be less than number of samples ({len(X)})."}
        
        if n_clusters < 2:
            return {"success": False, "error": "Number of clusters must be at least 2."}
        
        # Run K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = {}
        for label, count in zip(unique_labels, counts):
            cluster_distribution[int(label)] = {
                "count": int(count),
                "percentage": round(float(count / len(X) * 100), 2)
            }
        
        # Calculate cluster characteristics (using original X_train for interpretability)
        X_train_with_labels = X_train.copy()
        X_train_with_labels['cluster'] = labels
        
        cluster_characteristics = {}
        for label in unique_labels:
            cluster_data = X_train_with_labels[X_train_with_labels['cluster'] == label]
            char_stats = {}
            
            # Calculate stats for a few key features if they exist
            key_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'DistanceFromHome']
            for feat in key_features:
                if feat in cluster_data.columns:
                    char_stats[feat] = round(float(cluster_data[feat].mean()), 2)
            
            cluster_characteristics[int(label)] = char_stats
        
        # Prepare visualization data (use PCA if available for 2D visualization)
        viz_data = None
        if use_pca and pca_dir and os.path.exists(pca_dir):
            viz_data = [
                {"pc1": float(X[i, 0]), "pc2": float(X[i, 1]), "cluster": int(labels[i])}
                for i in range(len(X))
            ]
        elif pca_dir and os.path.exists(pca_dir):
            # Even if not using PCA for clustering, use it for visualization
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            viz_data = [
                {"pc1": float(X_train_pca.iloc[i, 0]), "pc2": float(X_train_pca.iloc[i, 1]), "cluster": int(labels[i])}
                for i in range(len(X_train_pca))
            ]
        
        # Save artifacts
        unsupervised_dir = os.path.join(os.path.dirname(processed_dir), 'unsupervised')
        if not os.path.exists(unsupervised_dir):
            os.makedirs(unsupervised_dir)
        
        joblib.dump(kmeans, os.path.join(unsupervised_dir, 'kmeans_model.joblib'))
        
        metadata = {
            "algorithm": "K-Means",
            "n_clusters": int(n_clusters),
            "representation": representation,
            "use_pca": use_pca,
            "samples_count": len(X),
            "features_count": len(feature_names),
            "feature_names": feature_names,
            "cluster_distribution": cluster_distribution,
            "random_state": random_state,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(unsupervised_dir, 'kmeans_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "algorithm": "K-Means",
            "n_clusters": int(n_clusters),
            "representation": representation,
            "cluster_distribution": cluster_distribution,
            "cluster_characteristics": cluster_characteristics,
            "visualization_data": viz_data[:500] if viz_data else None,  # Limit for UI
            "viz_total": len(viz_data) if viz_data else 0
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"K-Means clustering failed: {str(e)}"}


def run_dbscan(processed_dir, pca_dir, eps=0.5, min_samples=5, use_pca=False):
    """
    Runs DBSCAN clustering.
    """
    try:
        # Load data
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        
        if use_pca and pca_dir and os.path.exists(pca_dir):
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            X = X_train_pca.values
            feature_names = X_train_pca.columns.tolist()
            representation = "PCA Features"
        else:
            X = X_train.values
            feature_names = X_train.columns.tolist()
            representation = "Processed Features"
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Calculate statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise
        n_noise = int(counts[unique_labels == -1][0]) if -1 in unique_labels else 0
        noise_percentage = round(n_noise / len(X) * 100, 2)
        
        cluster_distribution = {}
        for label, count in zip(unique_labels, counts):
            cluster_distribution[int(label)] = {
                "count": int(count),
                "percentage": round(float(count / len(X) * 100), 2)
            }
        
        # Prepare visualization data
        viz_data = None
        if use_pca and pca_dir and os.path.exists(pca_dir):
            viz_data = [
                {"pc1": float(X[i, 0]), "pc2": float(X[i, 1]), "cluster": int(labels[i])}
                for i in range(len(X))
            ]
        elif pca_dir and os.path.exists(pca_dir):
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            viz_data = [
                {"pc1": float(X_train_pca.iloc[i, 0]), "pc2": float(X_train_pca.iloc[i, 1]), "cluster": int(labels[i])}
                for i in range(len(X_train_pca))
            ]
        
        # Save artifacts
        unsupervised_dir = os.path.join(os.path.dirname(processed_dir), 'unsupervised')
        if not os.path.exists(unsupervised_dir):
            os.makedirs(unsupervised_dir)
        
        joblib.dump(dbscan, os.path.join(unsupervised_dir, 'dbscan_model.joblib'))
        
        metadata = {
            "algorithm": "DBSCAN",
            "eps": float(eps),
            "min_samples": int(min_samples),
            "representation": representation,
            "use_pca": use_pca,
            "samples_count": len(X),
            "features_count": len(feature_names),
            "feature_names": feature_names,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percentage": noise_percentage,
            "cluster_distribution": cluster_distribution,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(unsupervised_dir, 'dbscan_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "algorithm": "DBSCAN",
            "eps": float(eps),
            "min_samples": int(min_samples),
            "representation": representation,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_percentage": noise_percentage,
            "cluster_distribution": cluster_distribution,
            "visualization_data": viz_data[:500] if viz_data else None,
            "viz_total": len(viz_data) if viz_data else 0
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"DBSCAN clustering failed: {str(e)}"}


def run_isolation_forest(processed_dir, pca_dir, contamination=0.1, use_pca=False, random_state=42):
    """
    Runs Isolation Forest for anomaly detection.
    """
    try:
        # Load data
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        
        if use_pca and pca_dir and os.path.exists(pca_dir):
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            X = X_train_pca.values
            feature_names = X_train_pca.columns.tolist()
            representation = "PCA Features"
        else:
            X = X_train.values
            feature_names = X_train.columns.tolist()
            representation = "Processed Features"
        
        # Validate contamination
        if contamination <= 0 or contamination >= 1:
            return {"success": False, "error": "Contamination must be between 0 and 1 (exclusive)."}
        
        # Run Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        labels = iso_forest.fit_predict(X)  # 1 for normal, -1 for anomaly
        
        # Calculate statistics
        n_normal = int(np.sum(labels == 1))
        n_anomalies = int(np.sum(labels == -1))
        anomaly_percentage = round(n_anomalies / len(X) * 100, 2)
        
        # Prepare visualization data
        viz_data = None
        if use_pca and pca_dir and os.path.exists(pca_dir):
            viz_data = [
                {"pc1": float(X[i, 0]), "pc2": float(X[i, 1]), "label": int(labels[i])}
                for i in range(len(X))
            ]
        elif pca_dir and os.path.exists(pca_dir):
            X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
            viz_data = [
                {"pc1": float(X_train_pca.iloc[i, 0]), "pc2": float(X_train_pca.iloc[i, 1]), "label": int(labels[i])}
                for i in range(len(X_train_pca))
            ]
        
        # Save artifacts
        unsupervised_dir = os.path.join(os.path.dirname(processed_dir), 'unsupervised')
        if not os.path.exists(unsupervised_dir):
            os.makedirs(unsupervised_dir)
        
        joblib.dump(iso_forest, os.path.join(unsupervised_dir, 'isolation_forest_model.joblib'))
        
        metadata = {
            "algorithm": "Isolation Forest",
            "contamination": float(contamination),
            "representation": representation,
            "use_pca": use_pca,
            "samples_count": len(X),
            "features_count": len(feature_names),
            "feature_names": feature_names,
            "n_normal": n_normal,
            "n_anomalies": n_anomalies,
            "anomaly_percentage": anomaly_percentage,
            "random_state": random_state,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(unsupervised_dir, 'isolation_forest_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "algorithm": "Isolation Forest",
            "contamination": float(contamination),
            "representation": representation,
            "n_normal": n_normal,
            "n_anomalies": n_anomalies,
            "anomaly_percentage": anomaly_percentage,
            "visualization_data": viz_data[:500] if viz_data else None,
            "viz_total": len(viz_data) if viz_data else 0
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Isolation Forest failed: {str(e)}"}
