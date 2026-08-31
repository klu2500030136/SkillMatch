import pandas as pd

def validate_column_pair(col1, col2, error_msg="Please select two different features."):
    if col1 and col2 and col1 == col2:
        return {"valid": False, "error": error_msg}
    return {"valid": True}

def validate_column_types(df, cols, expected_type='numeric', error_msg="Invalid column types."):
    for col in cols:
        if col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            if expected_type == 'numeric' and not is_num:
                return {"valid": False, "error": error_msg}
            elif expected_type == 'categorical' and is_num:
                return {"valid": False, "error": error_msg}
    return {"valid": True}

def validate_target_leakage(target, predictors, error_msg="The target column cannot be used as an input feature."):
    if not target or not predictors:
        return {"valid": True}
    if target in predictors:
        return {"valid": False, "error": error_msg}
    return {"valid": True}
