import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
import traceback

# Known Ordinal Features and their mappings
ORDINAL_MAPPINGS = {
    'Education': {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5},
    'EnvironmentSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'JobInvolvement': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'JobSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'PerformanceRating': {'Low': 1, 'Good': 2, 'Excellent': 3, 'Outstanding': 4},
    'RelationshipSatisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
    'WorkLifeBalance': {'Bad': 1, 'Good': 2, 'Better': 3, 'Best': 4}
}

def safe_json(val):
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return round(float(val), 2)
    return str(val)

def get_preprocessing_metadata(df):
    """Generates metadata for the preprocessing UI configuration step."""
    total_rows = len(df)
    total_cols = len(df.columns)
    
    duplicates_count = int(df.duplicated().sum())
    
    # Missing values
    missing_counts = df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0].to_dict()
    
    # Constant features
    constant_features = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_features.append({
                "feature": col,
                "value": str(df[col].iloc[0]) if not df.empty else "N/A"
            })
            
    # Identifier candidates (high cardinality, non-numeric or evenly spaced ints like ID)
    identifier_candidates = []
    for col in df.columns:
        if 'id' in col.lower() or 'number' in col.lower():
            identifier_candidates.append(col)
        elif df[col].nunique() == total_rows and pd.api.types.is_numeric_dtype(df[col]):
            identifier_candidates.append(col)
            
    # Target overview
    target_class_dist = {}
    if 'Attrition' in df.columns:
        target_class_dist = df['Attrition'].value_counts().to_dict()
        
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricals = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return {
        "rows": total_rows,
        "columns": total_cols,
        "duplicates": duplicates_count,
        "missing_cols": {k: int(v) for k, v in missing_cols.items()},
        "constant_features": constant_features,
        "identifier_candidates": identifier_candidates,
        "target_class_dist": target_class_dist,
        "numeric_features_count": len(numerics),
        "categorical_features_count": len(categoricals),
        "all_columns": df.columns.tolist()
    }

def apply_preprocessing(df, config, processed_dir):
    """
    Applies the full preprocessing pipeline according to config.
    Saves outputs to processed_dir.
    """
    try:
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        original_rows = len(df)
        original_cols = len(df.columns)
        
        processed_df = df.copy()
        
        # 1. Duplicates
        if config.get('remove_duplicates', True):
            processed_df = processed_df.drop_duplicates()
            
        # 2. Target Extraction & Validation
        target_col = config.get('target_column', 'Attrition')
        if target_col not in processed_df.columns:
            return {"success": False, "error": f"Target column '{target_col}' not found."}
            
        target_type = config.get('target_type', 'classification')
        
        # Drop rows where target is missing
        processed_df = processed_df.dropna(subset=[target_col])
        
        y = processed_df[target_col].copy()
        if target_type == 'classification' and target_col == 'Attrition':
            # Map Yes/No to 1/0
            if set(y.dropna().unique()).issubset({'Yes', 'No'}):
                y = y.map({'Yes': 1, 'No': 0})
            else:
                # Fallback categorical mapping
                y = y.astype('category').cat.codes
        else:
            # Ensure numeric for regression
            y = pd.to_numeric(y, errors='coerce')
            
        # 3. Feature Selection
        X = processed_df.drop(columns=[target_col])
        features_to_remove = config.get('features_to_remove', [])
        features_to_remove = [f for f in features_to_remove if f in X.columns]
        if features_to_remove:
            X = X.drop(columns=features_to_remove)
            
        # Identify feature types
        ordinal_features = [f for f in X.columns if f in ORDINAL_MAPPINGS]
        numeric_features = [f for f in X.select_dtypes(include=[np.number]).columns if f not in ordinal_features]
        categorical_features = [f for f in X.select_dtypes(exclude=[np.number]).columns if f not in ordinal_features]
        
        # 4. Pipeline Construction
        
        # Numeric transformer
        num_imputer_strategy = config.get('num_imputation', 'median')
        scaler_type = config.get('scaling', 'standard')
        
        num_steps = [('imputer', SimpleImputer(strategy=num_imputer_strategy))]
        if scaler_type == 'standard':
            num_steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            num_steps.append(('scaler', MinMaxScaler()))
            
        numeric_transformer = Pipeline(steps=num_steps)
        
        # Categorical transformer (Nominal)
        cat_imputer_strategy = config.get('cat_imputation', 'most_frequent')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=cat_imputer_strategy)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Ordinal transformer
        # We manually map ordinals if they are strings. If they are already numeric, we just impute and pass through.
        # But we'll rely on simple imputation, then passthrough (assuming they are already properly encoded numerically in IBM HR dataset, 
        # or we rely on a custom FunctionTransformer if we needed string mapping).
        # In the IBM dataset, Education, JobSatisfaction etc are already 1, 2, 3, 4 integers! 
        # So we just treat them as passthrough after imputation.
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')) # ordinal uses most_frequent
        ])
        
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if categorical_features:
            transformers.append(('cat', categorical_transformer, categorical_features))
        if ordinal_features:
            transformers.append(('ord', ordinal_transformer, ordinal_features))
            
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # 5. Train/Val/Test Split
        train_size = float(config.get('train_size', 70)) / 100.0
        val_size = float(config.get('val_size', 15)) / 100.0
        test_size = float(config.get('test_size', 15)) / 100.0
        
        # Normalize just in case of float errors
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 0.01:
            return {"success": False, "error": "Split ratios must sum to 100%."}
            
        random_state = int(config.get('random_state', 42))
        
        stratify = y if target_type == 'classification' else None
        
        # First split: train vs (val + test)
        temp_test_size = val_size + test_size
        
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=temp_test_size, random_state=random_state, stratify=stratify
            )
            
            # Second split: val vs test
            # Calculate proportion of val relative to (val + test)
            if temp_test_size == 0:
                X_val, X_test, y_val, y_test = X_temp, X_temp, y_temp, y_temp
            else:
                val_ratio = val_size / temp_test_size
                stratify_temp = y_temp if target_type == 'classification' else None
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, train_size=val_ratio, random_state=random_state, stratify=stratify_temp
                )
        except ValueError as e:
            return {"success": False, "error": f"Data splitting failed (possibly due to too few samples or imbalanced classes): {str(e)}"}
        
        # 6. Fit Preprocessor ONLY on Training Data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names if possible
        feature_names = []
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = preprocessor.get_feature_names_out()
                # Clean up names (remove 'num__', 'cat__', 'ord__')
                feature_names = [f.replace('num__', '').replace('cat__', '').replace('ord__', '') for f in feature_names]
            except:
                pass
                
        if not feature_names:
            feature_names = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]
            
        # 7. Convert back to DataFrame for easier preview/saving
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_val_df = pd.DataFrame(X_val_processed, columns=feature_names, index=X_val.index)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
        # 8. Save artifacts
        with open(os.path.join(processed_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(preprocessor, f)
            
        X_train_df.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
        X_val_df.to_csv(os.path.join(processed_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(processed_dir, 'y_val.csv'), index=False)
        X_test_df.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
        
        # Generate summary
        return {
            "success": True,
            "summary": {
                "original_rows": original_rows,
                "original_cols": original_cols,
                "processed_rows": len(processed_df),
                "processed_features": X_train_processed.shape[1],
                "duplicates_removed": original_rows - len(processed_df),
                "features_removed": len(features_to_remove),
                "target": target_col,
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test)
            },
            "preview_columns": feature_names[:50], # Limit preview columns
            "preview_data": X_train_df.head(5).round(4).to_dict('records'),
            "target_preview": y_train.head(5).tolist()
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Preprocessing failed: {str(e)}"}
