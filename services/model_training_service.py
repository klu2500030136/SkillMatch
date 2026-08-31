import os
import json
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def check_training_readiness(processed_dir):
    """Checks if Phase 6 artifacts exist."""
    required_files = ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv', 'preprocessor.pkl']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 6."}
        
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)):
            missing_files.append(f)
            
    if missing_files:
        return {"ready": False, "reason": f"Missing artifacts: {', '.join(missing_files)}. Please re-run Phase 6."}
        
    # Get metadata for UI
    try:
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv'))
        y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv'))
        y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv'))
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        
        target_name = y_train.columns[0]
        classes = y_train[target_name].value_counts().to_dict()
        
        return {
            "ready": True,
            "train_rows": len(X_train),
            "val_rows": len(y_val),
            "test_rows": len(y_test),
            "features_count": len(X_train.columns),
            "target": target_name,
            "class_distribution": classes
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading artifacts: {str(e)}"}

def get_model_instance(name, config, class_weight, random_state):
    """Factory method to get a configured scikit-learn estimator."""
    if name == 'Logistic Regression':
        cw = class_weight if class_weight == 'balanced' else None
        return LogisticRegression(
            C=float(config.get('lr_C', 1.0)),
            max_iter=int(config.get('lr_max_iter', 1000)),
            class_weight=cw,
            random_state=random_state
        )
    elif name == 'Decision Tree':
        cw = class_weight if class_weight == 'balanced' else None
        md = config.get('dt_max_depth')
        md = int(md) if md and md != 'None' else None
        return DecisionTreeClassifier(
            max_depth=md,
            min_samples_split=int(config.get('dt_min_samples_split', 2)),
            criterion=config.get('dt_criterion', 'gini'),
            class_weight=cw,
            random_state=random_state
        )
    elif name == 'Random Forest':
        cw = class_weight if class_weight == 'balanced' else None
        md = config.get('rf_max_depth')
        md = int(md) if md and md != 'None' else None
        return RandomForestClassifier(
            n_estimators=int(config.get('rf_n_estimators', 100)),
            max_depth=md,
            min_samples_split=int(config.get('rf_min_samples_split', 2)),
            class_weight=cw,
            random_state=random_state
        )
    elif name == 'K-Nearest Neighbors':
        # KNN does not natively support class_weight in sklearn in the same way, but it handles standard configs.
        return KNeighborsClassifier(
            n_neighbors=int(config.get('knn_n_neighbors', 5)),
            weights=config.get('knn_weights', 'uniform')
        )
    elif name == 'Support Vector Machine':
        cw = class_weight if class_weight == 'balanced' else None
        return SVC(
            C=float(config.get('svm_C', 1.0)),
            kernel=config.get('svm_kernel', 'rbf'),
            class_weight=cw,
            probability=True,  # Needed for ROC-AUC
            random_state=random_state
        )
    elif name == 'Gradient Boosting':
        md = config.get('gb_max_depth')
        md = int(md) if md and md != 'None' else 3
        return GradientBoostingClassifier(
            n_estimators=int(config.get('gb_n_estimators', 100)),
            learning_rate=float(config.get('gb_learning_rate', 0.1)),
            max_depth=md,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

def train_models(selected_models, config, class_weight, random_state, processed_dir, models_dir):
    """
    Trains selected models on X_train/y_train, evaluates on X_val/y_val.
    Saves best model and metadata to models_dir.
    """
    try:
        # Load datasets
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
        X_val = pd.read_csv(os.path.join(processed_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(processed_dir, 'y_val.csv')).iloc[:, 0]
        
        target_name = y_train.name
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        results = []
        best_model_obj = None
        best_f1 = -1
        best_roc_auc = -1
        best_model_name = None
        
        for name in selected_models:
            start_time = time.time()
            try:
                # 1. Instantiate
                model = get_model_instance(name, config, class_weight, random_state)
                
                # 2. Train (on Train only)
                model.fit(X_train, y_train)
                
                # 3. Predict (on Validation only)
                y_pred = model.predict(X_val)
                
                # Probability / Decision scores for ROC-AUC
                roc_auc = None
                if hasattr(model, 'predict_proba'):
                    # Some binary classifiers return (n_samples, 2), some may return single array depending on classes
                    y_prob = model.predict_proba(X_val)
                    if y_prob.shape[1] == 2:
                        roc_auc = roc_auc_score(y_val, y_prob[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_val, y_prob) # fallback
                elif hasattr(model, 'decision_function'):
                    y_prob = model.decision_function(X_val)
                    roc_auc = roc_auc_score(y_val, y_prob)
                    
                # 4. Calculate metrics
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                train_time = time.time() - start_time
                
                model_result = {
                    "name": name,
                    "status": "success",
                    "accuracy": round(acc * 100, 2),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "roc_auc": round(roc_auc * 100, 2) if roc_auc is not None else "N/A",
                    "training_time": round(train_time, 4)
                }
                
                # Track best model
                # Primary: F1, Secondary: ROC_AUC (if available), Tertiary: Recall
                val_roc_auc = roc_auc if roc_auc is not None else -1
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_roc_auc = val_roc_auc
                    best_model_name = name
                    best_model_obj = model
                elif f1 == best_f1:
                    if val_roc_auc > best_roc_auc:
                        best_f1 = f1
                        best_roc_auc = val_roc_auc
                        best_model_name = name
                        best_model_obj = model
                    elif val_roc_auc == best_roc_auc and val_roc_auc == -1:
                        # Fallback to recall if ROC-AUC missing and F1 is tied
                        if rec > results[-1].get('recall', -1)/100: 
                            best_model_name = name
                            best_model_obj = model
                
                results.append(model_result)
                
            except Exception as ex:
                traceback.print_exc()
                results.append({
                    "name": name,
                    "status": "failed",
                    "reason": str(ex)
                })
        
        # Determine best overall
        best_model_summary = None
        if best_model_obj:
            best_model_summary = {
                "name": best_model_name,
                "selection_metric": "F1-score",
                "score": round(best_f1 * 100, 2)
            }
            
            # Save artifacts
            joblib.dump(best_model_obj, os.path.join(models_dir, 'best_model.joblib'))
            
            # Metadata
            metadata = {
                "project": "SkillMatch",
                "task": "classification",
                "target": target_name,
                "model": best_model_name,
                "selection_metric": "f1",
                "validation_metrics": [r for r in results if r['name'] == best_model_name][0],
                "random_state": random_state,
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "feature_count": len(X_train.columns),
                "trained_at": datetime.now().isoformat()
            }
            with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
                
            # Copy preprocessing pipeline pointer or ensure it's kept in sync (it's already saved by Phase 6)
            # The Phase 8/9 pipeline will load `data/processed/preprocessor.pkl` along with `data/models/best_model.joblib`.
            
        return {
            "success": True,
            "models": results,
            "best_model": best_model_summary
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Model training failed: {str(e)}"}
