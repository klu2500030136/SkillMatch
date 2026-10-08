import os
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import traceback


def safe_json(val):
    """
    Recursively convert numpy/pandas types to standard Python types for JSON serialization.
    Handles nested dictionaries and lists.
    """
    # Handle None
    if val is None:
        return None
    
    # Handle NaN and Infinity
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    
    # Handle numpy integers
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    
    # Handle numpy floats
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    
    # Handle numpy booleans
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    
    # Handle numpy arrays - convert to list recursively
    if isinstance(val, np.ndarray):
        return [safe_json(item) for item in val]
    
    # Handle pandas Series/DataFrame
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return safe_json(val.values)
    
    # Handle pandas scalars
    if hasattr(val, 'item'):  # pandas scalar
        return safe_json(val.item())
    
    # Handle dictionaries - recurse
    if isinstance(val, dict):
        return {safe_json(k): safe_json(v) for k, v in val.items()}
    
    # Handle lists/tuples - recurse
    if isinstance(val, (list, tuple)):
        return [safe_json(item) for item in val]
    
    # Handle standard Python types
    if isinstance(val, (int, float, str, bool)):
        return val
    
    # Fallback for unknown types
    return str(val)


def check_pca_readiness(processed_dir):
    """
    Checks if preprocessing artifacts exist for PCA.
    PCA requires the preprocessed training data from Phase 5.
    """
    required_files = ['X_train.csv', 'y_train.csv', 'preprocessor.pkl']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 5 (Preprocessing) first."}
    
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)):
            missing_files.append(f)
    
    if missing_files:
        return {"ready": False, "reason": f"Missing preprocessing artifacts: {', '.join(missing_files)}. Please complete Phase 5 (Preprocessing) first."}
    
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        
        return {
            "ready": True,
            "features_count": len(X_train.columns),
            "samples_count": len(X_train),
            "feature_names": X_train.columns.tolist()
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading preprocessing artifacts: {str(e)}"}


def identify_modeling_features(processed_dir, target_col='Attrition'):
    """
    Identifies which features are used for modeling vs excluded.
    This uses the preprocessed feature names from the fitted preprocessor.
    """
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        
        # Load preprocessor to get original feature mapping if possible
        preprocessor_path = os.path.join(processed_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        
        # The preprocessed features are already the modeling features
        # (after encoding, scaling, and feature selection)
        modeling_features = X_train.columns.tolist()
        
        # Determine feature types from the preprocessed data
        feature_info = []
        for feat in modeling_features:
            feat_info = {
                "feature": feat,
                "type": "Numerical (Preprocessed)",  # After encoding/scaling, all are numerical
                "role": "Modeling Feature",
                "used_for_pca": True,
                "reason": "Preprocessed feature ready for modeling"
            }
            feature_info.append(feat_info)
        
        # Target info
        target_info = {
            "feature": target_col,
            "type": "Categorical (Target)",
            "role": "Target",
            "used_for_pca": False,
            "reason": "Target variable - excluded from PCA to prevent leakage"
        }
        
        return {
            "success": True,
            "dataset_columns": int(len(modeling_features) + 1),  # features + target
            "target": target_col,
            "modeling_features_count": int(len(modeling_features)),
            "excluded_features_count": 1,  # target
            "numerical_features_count": int(len(modeling_features)),  # all preprocessed features are numerical
            "categorical_features_count": 0,  # after encoding, no categorical features remain
            "feature_screening": [target_info] + feature_info
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Feature identification failed: {str(e)}"}


def apply_pca(processed_dir, pca_dir, variance_threshold=0.90):
    """
    Applies PCA to the preprocessed training data.
    Saves PCA artifacts to pca_dir.
    """
    try:
        if not os.path.exists(pca_dir):
            os.makedirs(pca_dir)
        
        # Load preprocessed training data
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        
        feature_names = X_train.columns.tolist()
        n_original_features = len(feature_names)
        
        # Fit PCA on all components first to analyze variance
        pca_full = PCA()
        X_train_pca_full = pca_full.fit_transform(X_train)
        
        # Calculate cumulative variance
        explained_variance = pca_full.explained_variance_
        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Determine number of components to retain based on threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        # Ensure at least 2 components for visualization
        n_components = max(n_components, 2)
        
        # Fit PCA with selected number of components
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        
        # Calculate component loadings
        # loadings = components * sqrt(explained_variance)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Create loadings dataframe for each component
        loadings_data = []
        for i in range(n_components):
            component_loadings = []
            for j, feat in enumerate(feature_names):
                component_loadings.append({
                    "feature": feat,
                    "loading": safe_json(loadings[j, i]),
                    "absolute_loading": safe_json(abs(float(loadings[j, i]))),
                    "direction": "Positive" if loadings[j, i] >= 0 else "Negative"
                })
            # Sort by absolute loading
            component_loadings.sort(key=lambda x: x['absolute_loading'], reverse=True)
            loadings_data.append(component_loadings)
        
        # Calculate 2D projection data (PC1 vs PC2)
        projection_2d = []
        for i in range(len(X_train_pca)):
            projection_2d.append({
                "pc1": safe_json(X_train_pca[i, 0]),
                "pc2": safe_json(X_train_pca[i, 1]),
                "target": safe_json(y_train.iloc[i, 0]) if len(y_train.columns) > 0 else None
            })
        
        # Save PCA artifacts
        with open(os.path.join(pca_dir, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        with open(os.path.join(pca_dir, 'pca_metadata.json'), 'w') as f:
            import json
            metadata = {
                "original_features": int(n_original_features),
                "selected_components": int(n_components),
                "variance_threshold": float(variance_threshold),
                "variance_retained": safe_json(cumulative_variance[n_components - 1]),
                "variance_lost": safe_json(1.0 - cumulative_variance[n_components - 1]),
                "compression_ratio": safe_json(n_components / n_original_features),
                "feature_names": feature_names,
                "explained_variance": [safe_json(v) for v in explained_variance[:n_components]],
                "explained_variance_ratio": [safe_json(v) for v in explained_variance_ratio[:n_components]],
                "cumulative_variance": [safe_json(v) for v in cumulative_variance[:n_components]]
            }
            json.dump(metadata, f, indent=2)
        
        # Save transformed data
        X_train_pca_df = pd.DataFrame(
            X_train_pca,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X_train.index
        )
        X_train_pca_df.to_csv(os.path.join(pca_dir, 'X_train_pca.csv'), index=False)
        
        # Return summary
        result = {
            "success": True,
            "summary": {
                "original_features": n_original_features,
                "selected_components": n_components,
                "variance_retained": round(float(cumulative_variance[n_components - 1]) * 100, 2),
                "variance_lost": round((1.0 - float(cumulative_variance[n_components - 1])) * 100, 2),
                "compression_ratio": round(n_components / n_original_features, 2)
            },
            "explained_variance": {
                "individual": [round(float(v) * 100, 2) for v in explained_variance_ratio[:n_components]],
                "cumulative": [round(float(v) * 100, 2) for v in cumulative_variance[:n_components]]
            },
            "components": [
                {
                    "name": f"PC{i+1}",
                    "explained_variance": round(float(explained_variance[i]), 6),
                    "explained_variance_ratio": round(float(explained_variance_ratio[i]) * 100, 2),
                    "cumulative_variance": round(float(cumulative_variance[i]) * 100, 2)
                }
                for i in range(n_components)
            ],
            "loadings": loadings_data,
            "projection_2d": projection_2d[:500],  # Limit to 500 points for visualization
            "projection_2d_total": len(projection_2d)
        }
        
        # Recursively convert all numpy/pandas types to JSON-safe types
        return safe_json(result)
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"PCA failed: {str(e)}"}


def get_pca_results(pca_dir):
    """
    Loads previously computed PCA results.
    """
    try:
        if not os.path.exists(pca_dir):
            return {"success": False, "error": "PCA results not found. Please run PCA first."}
        
        metadata_path = os.path.join(pca_dir, 'pca_metadata.json')
        if not os.path.exists(metadata_path):
            return {"success": False, "error": "PCA metadata not found. Please re-run PCA."}
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load transformed data for projection
        X_train_pca = pd.read_csv(os.path.join(pca_dir, 'X_train_pca.csv'))
        
        # Load target labels for coloring
        # processed_dir is the parent of pca_dir
        base_dir = os.path.dirname(pca_dir)
        processed_dir = os.path.join(base_dir, 'processed')
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        
        # Recalculate projection data
        projection_2d = []
        for i in range(len(X_train_pca)):
            projection_2d.append({
                "pc1": safe_json(X_train_pca.iloc[i, 0]),
                "pc2": safe_json(X_train_pca.iloc[i, 1]),
                "target": safe_json(y_train.iloc[i, 0]) if len(y_train.columns) > 0 else None
            })
        
        # Reconstruct component and loading data from metadata
        n_components = metadata['selected_components']
        feature_names = metadata['feature_names']
        
        # Load PCA model to get loadings
        with open(os.path.join(pca_dir, 'pca_model.pkl'), 'rb') as f:
            pca = pickle.load(f)
        
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_data = []
        for i in range(n_components):
            component_loadings = []
            for j, feat in enumerate(feature_names):
                component_loadings.append({
                    "feature": feat,
                    "loading": safe_json(loadings[j, i]),
                    "absolute_loading": safe_json(abs(float(loadings[j, i]))),
                    "direction": "Positive" if loadings[j, i] >= 0 else "Negative"
                })
            component_loadings.sort(key=lambda x: x['absolute_loading'], reverse=True)
            loadings_data.append(component_loadings)
        
        result = {
            "success": True,
            "summary": {
                "original_features": metadata['original_features'],
                "selected_components": metadata['selected_components'],
                "variance_retained": round(metadata['variance_retained'] * 100, 2),
                "variance_lost": round(metadata['variance_lost'] * 100, 2),
                "compression_ratio": metadata['compression_ratio']
            },
            "explained_variance": {
                "individual": [round(v * 100, 2) for v in metadata['explained_variance_ratio']],
                "cumulative": [round(v * 100, 2) for v in metadata['cumulative_variance']]
            },
            "components": [
                {
                    "name": f"PC{i+1}",
                    "explained_variance": metadata['explained_variance'][i],
                    "explained_variance_ratio": round(metadata['explained_variance_ratio'][i] * 100, 2),
                    "cumulative_variance": round(metadata['cumulative_variance'][i] * 100, 2)
                }
                for i in range(n_components)
            ],
            "loadings": loadings_data,
            "projection_2d": projection_2d[:500],
            "projection_2d_total": len(projection_2d)
        }
        
        # Recursively convert all numpy/pandas types to JSON-safe types
        return safe_json(result)
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Failed to load PCA results: {str(e)}"}
