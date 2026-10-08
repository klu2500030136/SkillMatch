import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import traceback

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def safe_json(val):
    """Convert numpy types to standard Python types for JSON serialization."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        # Handle infinity and NaN
        if np.isinf(val) or np.isnan(val):
            return None
        return round(float(val), 4)
    return str(val)


def check_evaluation_readiness(processed_dir, models_dir):
    """
    Checks if evaluation artifacts exist.
    Requires: test data, preprocessing, and at least one trained model.
    """
    required_files = ['X_test.csv', 'y_test.csv', 'preprocessor.pkl']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 3 (Preprocessing) first."}
    
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)):
            missing_files.append(f)
    
    if missing_files:
        return {"ready": False, "reason": f"Missing preprocessing artifacts: {', '.join(missing_files)}. Please complete Phase 3 (Preprocessing) first."}
    
    if not os.path.exists(models_dir):
        return {"ready": False, "reason": "Models directory not found. Please train at least one model in Phase 5 (Machine Learning) first."}
    
    # Check for at least one model
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    if not model_files:
        return {"ready": False, "reason": "No trained models found. Please train at least one model in Phase 5 (Machine Learning) first."}
    
    try:
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))
        
        target_name = y_test.columns[0]
        
        # Detect available models
        available_models = []
        for model_file in model_files:
            if model_file == 'best_model.joblib':
                # Load metadata to get the actual model name
                metadata_path = os.path.join(models_dir, 'model_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_name = metadata.get('model', 'Best Model')
                else:
                    model_name = 'Best Model'
            else:
                model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title()
            available_models.append({
                "name": model_name,
                "file": model_file
            })
        
        return {
            "ready": True,
            "test_rows": len(X_test),
            "features_count": len(X_test.columns),
            "target": target_name,
            "available_models": available_models
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading artifacts: {str(e)}"}


def evaluate_classification_model(model_name, processed_dir, models_dir, pca_dir=None):
    """
    Evaluates a single classification model on the held-out test set.
    Does NOT retrain the model.
    Uses existing preprocessing and test data.
    """
    try:
        # Load test data
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).iloc[:, 0]
        
        target_name = y_test.name
        
        # Load preprocessing
        preprocessor_path = os.path.join(processed_dir, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            return {"success": False, "error": "Preprocessing artifact not found. Please re-run Phase 3 (Preprocessing)."}
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load model
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_path = os.path.join(models_dir, f'{safe_model_name}_model.joblib')
        
        # If individual model file doesn't exist, try best_model.joblib
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, 'best_model.joblib')
            if not os.path.exists(model_path):
                return {"success": False, "error": f"Model '{model_name}' not found. Please train it in Phase 5 (Machine Learning)."}
        
        model = joblib.load(model_path)
        
        # Check if model expects PCA features
        use_pca = False
        if pca_dir and os.path.exists(pca_dir):
            pca_metadata_path = os.path.join(pca_dir, 'pca_metadata.json')
            if os.path.exists(pca_metadata_path):
                with open(pca_metadata_path, 'r') as f:
                    pca_metadata = json.load(f)
                # Check if model was trained with PCA (feature count matches PCA components)
                if len(X_test.columns) == pca_metadata.get('selected_components', 0):
                    use_pca = True
        
        # Transform test data using existing preprocessing
        # Note: For Phase 5, X_test is already preprocessed, so we don't transform again
        # If preprocessing was applied to raw data and saved, we would transform here
        # Current architecture: X_test.csv is already preprocessed
        
        # Apply PCA if needed
        if use_pca:
            pca_model_path = os.path.join(pca_dir, 'pca_model.pkl')
            if os.path.exists(pca_model_path):
                pca = joblib.load(pca_model_path)
                X_test = pca.transform(X_test)
            else:
                return {"success": False, "error": "PCA model not found for PCA-based evaluation."}
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Generate probabilities/decision scores for ROC-AUC
        y_prob = None
        roc_auc = None
        fpr = None
        tpr = None
        thresholds = None
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_prob)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score)
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Format classification report for UI
        class_report_formatted = []
        for class_label in class_report:
            if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            metrics = class_report[class_label]
            class_report_formatted.append({
                "class": str(class_label),
                "precision": safe_json(metrics['precision']),
                "recall": safe_json(metrics['recall']),
                "f1_score": safe_json(metrics['f1-score']),
                "support": int(metrics['support'])
            })
        
        # Add averages
        if 'macro avg' in class_report:
            class_report_formatted.append({
                "class": "Macro Average",
                "precision": safe_json(class_report['macro avg']['precision']),
                "recall": safe_json(class_report['macro avg']['recall']),
                "f1_score": safe_json(class_report['macro avg']['f1-score']),
                "support": int(class_report['macro avg']['support'])
            })
        if 'weighted avg' in class_report:
            class_report_formatted.append({
                "class": "Weighted Average",
                "precision": safe_json(class_report['weighted avg']['precision']),
                "recall": safe_json(class_report['weighted avg']['recall']),
                "f1_score": safe_json(class_report['weighted avg']['f1-score']),
                "support": int(class_report['weighted avg']['support'])
            })
        
        return {
            "success": True,
            "model_name": model_name,
            "target": target_name,
            "test_samples": len(X_test),
            "metrics": {
                "accuracy": safe_json(accuracy),
                "precision": safe_json(precision),
                "recall": safe_json(recall),
                "f1_score": safe_json(f1),
                "roc_auc": safe_json(roc_auc) if roc_auc is not None else None
            },
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "matrix": cm.tolist()
            },
            "classification_report": class_report_formatted,
            "roc_curve": {
                "fpr": [safe_json(x) for x in fpr] if fpr is not None else None,
                "tpr": [safe_json(x) for x in tpr] if tpr is not None else None,
                "thresholds": [safe_json(x) for x in thresholds] if thresholds is not None else None
            },
            "evaluated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Evaluation failed: {str(e)}"}


def evaluate_regression_model(model_name, processed_dir, models_dir):
    """
    Evaluates a regression model (Linear Regression) on the held-out test set.
    """
    try:
        # Load test features
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        
        # Load regression target
        target_col = 'YearsAtCompany'
        y_test_file = os.path.join(processed_dir, 'y_test_years.csv')
        
        if not os.path.exists(y_test_file):
            return {"success": False, "error": f"Regression target file not found. Please re-run Phase 3 (Preprocessing) to generate regression targets."}
        
        y_test = pd.read_csv(y_test_file).iloc[:, 0]
        
        # Remove target column from features if present
        if target_col in X_test.columns:
            X_test = X_test.drop(columns=[target_col])
        
        # Load model
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        model_path = os.path.join(models_dir, f'{safe_model_name}_model.joblib')
        
        # If individual model file doesn't exist, try best_model.joblib
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, 'best_model.joblib')
            if not os.path.exists(model_path):
                return {"success": False, "error": f"Model '{model_name}' not found. Please train it in Phase 5 (Machine Learning)."}
        
        model = joblib.load(model_path)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "success": True,
            "model_name": model_name,
            "target": target_col,
            "test_samples": len(X_test),
            "metrics": {
                "mae": safe_json(mae),
                "mse": safe_json(mse),
                "rmse": safe_json(rmse),
                "r2": safe_json(r2)
            },
            "evaluated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Regression evaluation failed: {str(e)}"}


def evaluate_all_models(processed_dir, models_dir, pca_dir=None):
    """
    Evaluates all available classification models for comparison.
    Returns a summary of metrics for each model.
    """
    try:
        readiness = check_evaluation_readiness(processed_dir, models_dir)
        if not readiness['ready']:
            return {"success": False, "error": readiness['reason']}
        
        available_models = readiness['available_models']
        
        comparison_results = []
        
        for model_info in available_models:
            model_name = model_info['name']
            
            # Skip regression model for classification comparison
            if 'regression' in model_name.lower():
                continue
            
            result = evaluate_classification_model(model_name, processed_dir, models_dir, pca_dir)
            
            if result['success']:
                comparison_results.append({
                    "model": model_name,
                    "accuracy": result['metrics']['accuracy'],
                    "precision": result['metrics']['precision'],
                    "recall": result['metrics']['recall'],
                    "f1_score": result['metrics']['f1_score'],
                    "roc_auc": result['metrics']['roc_auc']
                })
        
        return {
            "success": True,
            "comparison": comparison_results,
            "total_models_evaluated": len(comparison_results)
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Model comparison failed: {str(e)}"}
