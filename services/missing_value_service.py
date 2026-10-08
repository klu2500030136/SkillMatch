import pandas as pd
import numpy as np

# Configurable thresholds
COMPLETE_THRESHOLD = 0
LOW_MISSING_THRESHOLD = 5
MODERATE_MISSING_THRESHOLD = 20

def safe_json(val):
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        # Handle infinity and NaN
        if np.isinf(val) or np.isnan(val):
            return None
        return round(float(val), 2)
    return str(val)

def classify_missingness(pct):
    if pct == COMPLETE_THRESHOLD:
        return "Complete"
    elif pct <= LOW_MISSING_THRESHOLD:
        return "Low Missingness"
    elif pct <= MODERATE_MISSING_THRESHOLD:
        return "Moderate Missingness"
    else:
        return "High Missingness"

def get_missing_summary(df):
    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing_cells = int(df.isna().sum().sum())
    
    missing_by_col = df.isna().sum()
    affected_cols = int((missing_by_col > 0).sum())
    complete_cols = total_cols - affected_cols
    
    highest_missing_pct = 0
    highest_missing_col = None
    
    if total_rows > 0:
        pct_series = (missing_by_col / total_rows) * 100
        highest_missing_pct = float(pct_series.max())
        if highest_missing_pct > 0:
            highest_missing_col = pct_series.idxmax()
            
    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_missing_cells": total_missing_cells,
        "affected_columns": affected_cols,
        "complete_columns": complete_cols,
        "highest_missing_pct": safe_json(highest_missing_pct),
        "highest_missing_col": highest_missing_col
    }

def get_missing_statistics(df):
    total_rows = len(df)
    stats = []
    
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
        completeness_pct = 100 - missing_pct
        
        status = classify_missingness(missing_pct)
        
        stats.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_pct": safe_json(missing_pct),
            "completeness_pct": safe_json(completeness_pct),
            "status": status,
            "non_missing_count": total_rows - missing_count
        })
    
    return stats

def get_missing_chart_data(df):
    total_rows = len(df)
    missing_by_col = df.isna().sum()
    affected = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    if affected.empty:
        return {
            "has_missing": False,
            "labels": [],
            "counts": [],
            "percentages": []
        }
        
    labels = affected.index.tolist()
    counts = [int(x) for x in affected.values]
    percentages = [safe_json((x / total_rows) * 100) for x in affected.values]
    
    return {
        "has_missing": True,
        "labels": labels,
        "counts": counts,
        "percentages": percentages
    }

def get_missing_heatmap_data(df, sample_size=100):
    # For performance, only sample if rows > sample_size
    sampled_df = df.sample(min(len(df), sample_size), random_state=42) if len(df) > sample_size else df
    
    # We only care about missing vs present (1 vs 0)
    matrix = sampled_df.isna().astype(int).values.tolist()
    
    return {
        "sampled": len(df) > sample_size,
        "sample_size": len(sampled_df),
        "total_features": len(df.columns),
        "features": df.columns.tolist(),
        "matrix": matrix
    }

def get_missing_insights(df):
    total_rows = len(df)
    missing_by_col = df.isna().sum()
    affected_cols = int((missing_by_col > 0).sum())
    
    insights = []
    
    if affected_cols == 0:
        insights.append("No missing values were detected in the active dataset.")
        insights.append("Your data is 100% complete across all features.")
    else:
        insights.append(f"{affected_cols} feature{'s' if affected_cols != 1 else ''} contain{'s' if affected_cols == 1 else ''} missing values and should be considered during preprocessing.")
        
        complete = len(df.columns) - affected_cols
        insights.append(f"{complete} of {len(df.columns)} features are complete.")
        
        pct_series = (missing_by_col / total_rows) * 100
        highest_col = pct_series.idxmax()
        highest_pct = pct_series.max()
        
        insights.append(f"'{highest_col}' has the highest missing-value percentage at {highest_pct:.2f}%.")
        
        high_missing = (pct_series > 20).sum()
        if high_missing > 0:
            insights.append(f"{high_missing} feature(s) have High Missingness (>20%). Features with significant missing values may require targeted treatment before model training.")
        
        insights.append("Missing values can reduce the number of usable observations for some algorithms. Missingness should be reviewed during Phase 6 - Preprocessing.")
        
    return insights

def get_feature_missing_details(df, feature):
    if feature not in df.columns:
        return {"error": "Feature not found"}
        
    total_rows = len(df)
    missing_count = int(df[feature].isna().sum())
    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
    non_missing_count = total_rows - missing_count
    completeness_pct = 100 - missing_pct
    
    return {
        "feature": feature,
        "dtype": str(df[feature].dtype),
        "total_records": total_rows,
        "missing_count": missing_count,
        "missing_pct": safe_json(missing_pct),
        "non_missing_count": non_missing_count,
        "completeness_pct": safe_json(completeness_pct),
        "status": classify_missingness(missing_pct)
    }
