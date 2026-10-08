import os
import json
import pandas as pd
import pickle
import joblib
from datetime import datetime


def get_project_summary(upload_dir, processed_dir, models_dir, pca_dir, unsupervised_dir):
    """
    Generates a comprehensive project summary for the Phase 9 dashboard.
    Reads actual project state from artifacts.
    """
    summary = {
        "project": {
            "name": "SkillMatch",
            "subtitle": "Employee Intelligence & Attrition Analysis",
            "description": "An end-to-end machine learning application for exploring employee data, identifying patterns associated with attrition, evaluating machine learning models, and generating employee-level predictions."
        },
        "dataset": get_dataset_status(upload_dir),
        "phases": {
            "foundation": get_foundation_status(),
            "dataset": get_dataset_status(upload_dir),
            "eda": get_eda_status(upload_dir),
            "preprocessing": get_preprocessing_status(processed_dir),
            "pca": get_pca_status(pca_dir),
            "ml": get_ml_status(models_dir, unsupervised_dir),
            "evaluation": get_evaluation_status(models_dir),
            "cross_validation": get_cv_status(models_dir),
            "prediction": get_prediction_status(processed_dir, models_dir)
        },
        "models": get_model_inventory(models_dir, unsupervised_dir),
        "technology": {
            "backend": "Flask",
            "data": ["Pandas", "NumPy"],
            "ml": "Scikit-learn",
            "visualization": "Chart.js",
            "persistence": "joblib",
            "dataset": "IBM HR Analytics Employee Attrition & Performance",
            "target": "Attrition",
            "secondary_target": "YearsAtCompany",
            "cv": "5-Fold Stratified Cross Validation"
        }
    }
    
    return summary


def get_dataset_status(upload_dir):
    """Check dataset status."""
    if not os.path.exists(upload_dir):
        return {"status": "NOT_STARTED", "rows": None, "columns": None, "target": None}
    
    dataset_files = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
    if not dataset_files:
        return {"status": "NOT_STARTED", "rows": None, "columns": None, "target": None}
    
    # Load the most recent dataset
    dataset_file = sorted(dataset_files)[-1]
    try:
        df = pd.read_csv(os.path.join(upload_dir, dataset_file))
        target = "Attrition" if "Attrition" in df.columns else None
        
        return {
            "status": "COMPLETED",
            "rows": len(df),
            "columns": len(df.columns),
            "target": target,
            "filename": dataset_file
        }
    except:
        return {"status": "ERROR", "rows": None, "columns": None, "target": None}


def get_foundation_status():
    """Foundation is always completed if the app is running."""
    return {"status": "COMPLETED"}


def get_eda_status(upload_dir):
    """Check EDA status based on dataset availability."""
    dataset_status = get_dataset_status(upload_dir)
    if dataset_status["status"] == "COMPLETED":
        return {"status": "READY"}
    return {"status": "NOT_STARTED"}


def get_preprocessing_status(processed_dir):
    """Check preprocessing status."""
    if not os.path.exists(processed_dir):
        return {"status": "NOT_STARTED"}
    
    if os.path.exists(os.path.join(processed_dir, 'preprocessor.pkl')):
        return {"status": "COMPLETED"}
    
    return {"status": "NOT_STARTED"}


def get_pca_status(pca_dir):
    """Check PCA status."""
    if not os.path.exists(pca_dir):
        return {"status": "NOT_STARTED"}
    
    if os.path.exists(os.path.join(pca_dir, 'pca_model.pkl')):
        # Try to read metadata
        metadata_path = os.path.join(pca_dir, 'pca_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    "status": "COMPLETED",
                    "components": metadata.get("n_components", None),
                    "variance": metadata.get("cumulative_variance", None)
                }
            except:
                pass
        return {"status": "COMPLETED"}
    
    return {"status": "NOT_STARTED"}


def get_ml_status(models_dir, unsupervised_dir):
    """Check ML status."""
    supervised_status = "NOT_STARTED"
    unsupervised_status = "NOT_STARTED"
    
    # Check supervised models
    if os.path.exists(models_dir):
        if os.path.exists(os.path.join(models_dir, 'best_model.joblib')):
            supervised_status = "COMPLETED"
        elif os.path.exists(os.path.join(models_dir, 'linear_regression_model.joblib')):
            supervised_status = "COMPLETED"
    
    # Check unsupervised models
    if os.path.exists(unsupervised_dir):
        if os.path.exists(os.path.join(unsupervised_dir, 'kmeans_model.joblib')):
            unsupervised_status = "COMPLETED"
    
    if supervised_status == "COMPLETED" or unsupervised_status == "COMPLETED":
        return {
            "status": "COMPLETED",
            "supervised": supervised_status,
            "unsupervised": unsupervised_status
        }
    
    return {"status": "NOT_STARTED", "supervised": supervised_status, "unsupervised": unsupervised_status}


def get_evaluation_status(models_dir):
    """Check evaluation status."""
    if not os.path.exists(models_dir):
        return {"status": "NOT_STARTED"}
    
    # Check if evaluation results exist (we infer from model metadata)
    if os.path.exists(os.path.join(models_dir, 'model_metadata.json')):
        return {"status": "COMPLETED"}
    
    return {"status": "NOT_STARTED"}


def get_cv_status(models_dir):
    """Check cross-validation status."""
    if not os.path.exists(models_dir):
        return {"status": "NOT_STARTED"}
    
    # Check for CV JSON files
    cv_files = [f for f in os.listdir(models_dir) if f.endswith('_cv.json')]
    if cv_files:
        return {
            "status": "COMPLETED",
            "models_evaluated": len(cv_files)
        }
    
    return {"status": "NOT_STARTED"}


def get_prediction_status(processed_dir, models_dir):
    """Check prediction status."""
    if not os.path.exists(processed_dir) or not os.path.exists(models_dir):
        return {"status": "NOT_STARTED"}
    
    has_preprocessor = os.path.exists(os.path.join(processed_dir, 'preprocessor.pkl'))
    has_model = os.path.exists(os.path.join(models_dir, 'best_model.joblib'))
    
    if has_preprocessor and has_model:
        return {"status": "READY"}
    
    return {"status": "NOT_STARTED"}


def get_model_inventory(models_dir, unsupervised_dir):
    """Get actual model inventory."""
    inventory = {
        "supervised": [],
        "unsupervised": []
    }
    
    # Supervised models (from Phase 5 implementation)
    supervised_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "Gradient Boosting",
        "Linear Regression"
    ]
    
    # Check which supervised models have CV results (indicates they were implemented)
    if os.path.exists(models_dir):
        cv_files = [f for f in os.listdir(models_dir) if f.endswith('_cv.json')]
        for model in supervised_models:
            safe_name = model.lower().replace(' ', '_').replace('-', '_')
            if f"{safe_name}_cv.json" in cv_files:
                inventory["supervised"].append({"name": model, "status": "IMPLEMENTED"})
            else:
                inventory["supervised"].append({"name": model, "status": "AVAILABLE"})
    
    # Check for Linear Regression
    if os.path.exists(os.path.join(models_dir, 'linear_regression_model.joblib')):
        inventory["supervised"].append({"name": "Linear Regression", "status": "IMPLEMENTED"})
    
    # Unsupervised models
    unsupervised_models = [
        "K-Means",
        "DBSCAN",
        "Isolation Forest"
    ]
    
    if os.path.exists(unsupervised_dir):
        if os.path.exists(os.path.join(unsupervised_dir, 'kmeans_model.joblib')):
            inventory["unsupervised"].append({"name": "K-Means", "status": "IMPLEMENTED"})
        else:
            inventory["unsupervised"].append({"name": "K-Means", "status": "AVAILABLE"})
        
        if os.path.exists(os.path.join(unsupervised_dir, 'dbscan_model.joblib')):
            inventory["unsupervised"].append({"name": "DBSCAN", "status": "IMPLEMENTED"})
        else:
            inventory["unsupervised"].append({"name": "DBSCAN", "status": "AVAILABLE"})
        
        if os.path.exists(os.path.join(unsupervised_dir, 'isolation_forest_model.joblib')):
            inventory["unsupervised"].append({"name": "Isolation Forest", "status": "IMPLEMENTED"})
        else:
            inventory["unsupervised"].append({"name": "Isolation Forest", "status": "AVAILABLE"})
    
    return inventory
