import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import traceback

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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


def check_cv_readiness(processed_dir):
    """
    Checks if cross-validation artifacts exist.
    Requires: X_train.csv, y_train.csv, preprocessor.pkl
    """
    required_files = ['X_train.csv', 'y_train.csv', 'preprocessor.pkl']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 3 (Preprocessing) first."}
    
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)):
            missing_files.append(f)
    
    if missing_files:
        return {"ready": False, "reason": f"Missing preprocessing artifacts: {', '.join(missing_files)}. Please complete Phase 3 (Preprocessing) first."}
    
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
        
        target_name = y_train.name
        unique_classes = y_train.nunique()
        
        if unique_classes < 2:
            return {"ready": False, "reason": f"Target '{target_name}' has only {unique_classes} class. Cross-validation requires at least 2 classes."}
        
        if len(X_train) < 5:
            return {"ready": False, "reason": f"Training data has only {len(X_train)} samples. 5-fold cross-validation requires at least 5 samples."}
        
        return {
            "ready": True,
            "train_rows": len(X_train),
            "features_count": len(X_train.columns),
            "target": target_name,
            "unique_classes": unique_classes,
            "class_distribution": y_train.value_counts().to_dict()
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading artifacts: {str(e)}"}


def get_model_instance(name, random_state=42):
    """Factory method to get a configured scikit-learn estimator for CV."""
    if name == 'Logistic Regression':
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )
    elif name == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            criterion='gini',
            class_weight='balanced',
            random_state=random_state
        )
    elif name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=random_state
        )
    elif name == 'K-Nearest Neighbors':
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform'
        )
    elif name == 'Support Vector Machine':
        return SVC(
            C=1.0,
            kernel='rbf',
            class_weight='balanced',
            random_state=random_state
        )
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def run_cross_validation(model_name, processed_dir, models_dir, n_splits=5, random_state=42):
    """
    Runs 5-fold stratified cross-validation on the training data.
    Uses the original dataset and fits preprocessing inside each fold to prevent data leakage.
    """
    try:
        # Load the original dataset file from session or find it
        upload_folder = os.path.join(os.path.dirname(processed_dir), 'uploads')
        dataset_files = [f for f in os.listdir(upload_folder) if f.endswith('.csv') and 'HR' in f]
        
        if not dataset_files:
            return {"success": False, "error": "Original dataset file not found in uploads directory."}
        
        # Load the most recent HR dataset
        dataset_file = sorted(dataset_files)[-1]
        dataset_path = os.path.join(upload_folder, dataset_file)
        df = pd.read_csv(dataset_path)
        
        # Load training indices to reconstruct the training set
        X_train_processed = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
        
        target_name = y_train.name
        
        # Get the original training indices
        train_indices = X_train_processed.index.tolist()
        
        # Reconstruct the original training dataset
        X_train_raw = df.loc[train_indices]
        y_train_raw = df.loc[train_indices, target_name]
        
        # Map Yes/No to 1/0 if needed
        if set(y_train_raw.dropna().unique()).issubset({'Yes', 'No'}):
            y_train_raw = y_train_raw.map({'Yes': 1, 'No': 0})
        
        # Remove target from features
        X_train_raw = X_train_raw.drop(columns=[target_name])
        
        # Load the preprocessor configuration to reconstruct it
        preprocessor_path = os.path.join(processed_dir, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            return {"success": False, "error": "Preprocessor artifact not found. Please re-run Phase 3 (Preprocessing)."}
        
        with open(preprocessor_path, 'rb') as f:
            old_preprocessor = pickle.load(f)
        
        # Reconstruct the preprocessing pipeline based on the old preprocessor
        # This is needed because we need to fit it inside each CV fold
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        # Identify feature types from original data
        ordinal_features = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 
                           'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']
        numeric_features = [f for f in X_train_raw.select_dtypes(include=[np.number]).columns if f not in ordinal_features]
        categorical_features = [f for f in X_train_raw.select_dtypes(exclude=[np.number]).columns if f not in ordinal_features]
        
        # Build new preprocessor with same configuration
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        if ordinal_features:
            transformers.append(('ord', ordinal_transformer, ordinal_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Create a pipeline that includes preprocessor + model
        model = get_model_instance(model_name, random_state)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0),
            'roc_auc': 'roc_auc'
        }
        
        # Create stratified k-fold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Run cross-validation
        cv_results = cross_validate(
            pipeline,
            X_train_raw,
            y_train_raw,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            return_estimator=True
        )
        
        # Process fold-level results
        fold_results = []
        for fold_idx in range(n_splits):
            fold_result = {
                "fold": fold_idx + 1,
                "accuracy": safe_json(cv_results['test_accuracy'][fold_idx]),
                "precision": safe_json(cv_results['test_precision'][fold_idx]),
                "recall": safe_json(cv_results['test_recall'][fold_idx]),
                "f1": safe_json(cv_results['test_f1'][fold_idx]),
                "roc_auc": safe_json(cv_results['test_roc_auc'][fold_idx]) if 'test_roc_auc' in cv_results else None
            }
            fold_results.append(fold_result)
        
        # Calculate mean and standard deviation
        summary = {
            "accuracy": {
                "mean": safe_json(np.mean(cv_results['test_accuracy'])),
                "std": safe_json(np.std(cv_results['test_accuracy']))
            },
            "precision": {
                "mean": safe_json(np.mean(cv_results['test_precision'])),
                "std": safe_json(np.std(cv_results['test_precision']))
            },
            "recall": {
                "mean": safe_json(np.mean(cv_results['test_recall'])),
                "std": safe_json(np.std(cv_results['test_recall']))
            },
            "f1": {
                "mean": safe_json(np.mean(cv_results['test_f1'])),
                "std": safe_json(np.std(cv_results['test_f1']))
            }
        }
        
        # ROC-AUC if available
        if 'test_roc_auc' in cv_results:
            summary["roc_auc"] = {
                "mean": safe_json(np.mean(cv_results['test_roc_auc'])),
                "std": safe_json(np.std(cv_results['test_roc_auc']))
            }
        
        # Save the CV results
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        cv_metadata = {
            "project": "SkillMatch",
            "task": "cross_validation",
            "target": target_name,
            "model": model_name,
            "n_splits": n_splits,
            "cv_type": "StratifiedKFold",
            "random_state": random_state,
            "train_rows": len(X_train_raw),
            "features_count": len(X_train_raw.columns),
            "fold_results": fold_results,
            "summary": summary,
            "executed_at": datetime.now().isoformat()
        }
        
        cv_filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_cv.json"
        with open(os.path.join(models_dir, cv_filename), 'w') as f:
            json.dump(cv_metadata, f, indent=4)
        
        return {
            "success": True,
            "model_name": model_name,
            "target": target_name,
            "n_splits": n_splits,
            "train_samples": len(X_train_raw),
            "fold_results": fold_results,
            "summary": summary,
            "cv_metadata_file": cv_filename
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Cross-validation failed: {str(e)}"}


def run_all_models_cv(processed_dir, models_dir, n_splits=5, random_state=42):
    """
    Runs 5-fold cross-validation for all available classification models.
    """
    try:
        readiness = check_cv_readiness(processed_dir)
        if not readiness['ready']:
            return {"success": False, "error": readiness['reason']}
        
        # Models to evaluate
        models_to_evaluate = [
            'Logistic Regression',
            'Decision Tree',
            'Random Forest',
            'K-Nearest Neighbors',
            'Support Vector Machine',
            'Gradient Boosting'
        ]
        
        comparison_results = []
        
        for model_name in models_to_evaluate:
            result = run_cross_validation(model_name, processed_dir, models_dir, n_splits, random_state)
            
            if result['success']:
                comparison_results.append({
                    "model": model_name,
                    "mean_accuracy": result['summary']['accuracy']['mean'],
                    "mean_precision": result['summary']['precision']['mean'],
                    "mean_recall": result['summary']['recall']['mean'],
                    "mean_f1": result['summary']['f1']['mean'],
                    "mean_roc_auc": result['summary'].get('roc_auc', {}).get('mean', None)
                })
        
        return {
            "success": True,
            "comparison": comparison_results,
            "total_models_evaluated": len(comparison_results)
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Model comparison failed: {str(e)}"}
