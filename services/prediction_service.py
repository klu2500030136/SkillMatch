import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import traceback
import warnings
from datetime import datetime

# Suppress sklearn feature name warning for prediction
warnings.filterwarnings('ignore', message='X does not have valid feature names')


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


def check_prediction_readiness(processed_dir, models_dir):
    """
    Checks if prediction artifacts exist.
    Requires: preprocessor.pkl, best_model.joblib, model_metadata.json
    """
    required_files = ['preprocessor.pkl', 'best_model.joblib', 'model_metadata.json']
    missing_files = []
    
    if not os.path.exists(processed_dir):
        return {"ready": False, "reason": "Preprocessing directory not found. Please complete Phase 3 (Preprocessing) first."}
    
    if not os.path.exists(models_dir):
        return {"ready": False, "reason": "Models directory not found. Please complete Phase 5 (Machine Learning) first."}
    
    for f in required_files:
        if not os.path.exists(os.path.join(processed_dir, f)) and not os.path.exists(os.path.join(models_dir, f)):
            # Check both directories
            if f.endswith('.pkl') and not os.path.exists(os.path.join(processed_dir, f)):
                missing_files.append(f)
            elif f.endswith('.joblib') and not os.path.exists(os.path.join(models_dir, f)):
                missing_files.append(f)
            elif f.endswith('.json') and not os.path.exists(os.path.join(models_dir, f)):
                missing_files.append(f)
    
    if missing_files:
        return {"ready": False, "reason": f"Missing prediction artifacts: {', '.join(missing_files)}. Please complete Phase 3 (Preprocessing) and Phase 5 (Machine Learning) first."}
    
    try:
        # Load model metadata
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
        
        return {
            "ready": True,
            "model_name": model_metadata.get("model", "Unknown"),
            "target": model_metadata.get("target", "Attrition"),
            "feature_count": model_metadata.get("feature_count", 0),
            "trained_at": model_metadata.get("trained_at", "Unknown")
        }
    except Exception as e:
        return {"ready": False, "reason": f"Error reading model metadata: {str(e)}"}


def get_categorical_values(upload_dir):
    """
    Returns valid categorical values from the uploaded dataset.
    """
    try:
        # Find the most recent HR dataset
        dataset_files = [f for f in os.listdir(upload_dir) if 'HR' in f and f.endswith('.csv')]
        if not dataset_files:
            return None
        
        dataset_file = sorted(dataset_files)[-1]
        df = pd.read_csv(os.path.join(upload_dir, dataset_file))
        
        categorical_fields = {
            'BusinessTravel': sorted(df['BusinessTravel'].unique().tolist()) if 'BusinessTravel' in df.columns else [],
            'Department': sorted(df['Department'].unique().tolist()) if 'Department' in df.columns else [],
            'EducationField': sorted(df['EducationField'].unique().tolist()) if 'EducationField' in df.columns else [],
            'Gender': sorted(df['Gender'].unique().tolist()) if 'Gender' in df.columns else [],
            'JobRole': sorted(df['JobRole'].unique().tolist()) if 'JobRole' in df.columns else [],
            'MaritalStatus': sorted(df['MaritalStatus'].unique().tolist()) if 'MaritalStatus' in df.columns else [],
            'OverTime': sorted(df['OverTime'].unique().tolist()) if 'OverTime' in df.columns else []
        }
        
        return categorical_fields
    except Exception as e:
        traceback.print_exc()
        return None


def get_ordinal_mappings():
    """
    Returns ordinal field mappings as defined in preprocessing.
    """
    return {
        'Education': {
            1: 'Below College',
            2: 'College',
            3: 'Bachelor',
            4: 'Master',
            5: 'Doctor'
        },
        'EnvironmentSatisfaction': {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        },
        'JobInvolvement': {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        },
        'JobSatisfaction': {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        },
        'PerformanceRating': {
            1: 'Low',
            2: 'Good',
            3: 'Excellent',
            4: 'Outstanding'
        },
        'RelationshipSatisfaction': {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        },
        'WorkLifeBalance': {
            1: 'Bad',
            2: 'Good',
            3: 'Better',
            4: 'Best'
        }
    }


def predict_attrition(employee_data, processed_dir, models_dir):
    """
    Predicts attrition for a single employee using the saved preprocessing pipeline and model.
    
    Args:
        employee_data: dict with employee feature values
        processed_dir: path to processed directory containing preprocessor.pkl
        models_dir: path to models directory containing best_model.joblib
    
    Returns:
        dict with prediction results
    """
    try:
        # Load preprocessor
        preprocessor_path = os.path.join(processed_dir, 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            return {"success": False, "error": "Preprocessor artifact not found. Please re-run Phase 3 (Preprocessing)."}
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load model
        model_path = os.path.join(models_dir, 'best_model.joblib')
        if not os.path.exists(model_path):
            return {"success": False, "error": "Model artifact not found. Please re-run Phase 5 (Machine Learning)."}
        
        model = joblib.load(model_path)
        
        # Load model metadata
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
        
        target = model_metadata.get("target", "Attrition")
        model_name = model_metadata.get("model", "Unknown")
        
        # Create DataFrame from employee data
        # Expected features based on preprocessing:
        # Numeric: Age, DailyRate, DistanceFromHome, HourlyRate, JobLevel, MonthlyIncome, MonthlyRate, 
        #          NumCompaniesWorked, PercentSalaryHike, StockOptionLevel, TotalWorkingYears, 
        #          TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
        # Categorical: BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
        # Ordinal: Education, EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, PerformanceRating, 
        #          RelationshipSatisfaction, WorkLifeBalance
        
        expected_features = [
            'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
            'MaritalStatus', 'OverTime', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
            'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'
        ]
        
        # Filter to only expected features
        employee_dict = {k: v for k, v in employee_data.items() if k in expected_features}
        
        # Create DataFrame
        employee_df = pd.DataFrame([employee_dict])
        
        # Ensure all expected features are present
        missing_features = set(expected_features) - set(employee_dict.keys())
        if missing_features:
            return {"success": False, "error": f"Missing required features: {', '.join(missing_features)}"}
        
        # Apply preprocessing
        employee_processed = preprocessor.transform(employee_df)
        
        # Do NOT add feature names - the model was trained without them
        # or with specific naming that sklearn handles internally
        
        # Make prediction
        prediction = model.predict(employee_processed)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(employee_processed)[0]
                probability = safe_json(proba[1]) if len(proba) > 1 else None  # Probability of positive class
            except:
                pass
        
        # Map prediction to readable class
        # Project uses: No -> 0, Yes -> 1
        if prediction == 1:
            predicted_class = "Yes"
        elif prediction == 0:
            predicted_class = "No"
        else:
            predicted_class = str(prediction)
        
        return {
            "success": True,
            "prediction": predicted_class,
            "probability": probability,
            "model_name": model_name,
            "target": target,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"Prediction failed: {str(e)}"}
