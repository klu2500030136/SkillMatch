import pandas as pd
import numpy as np
from services.analysis_service import MAPPINGS

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

def get_dataset_overview(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numerical_features": len(num_cols),
        "categorical_features": len(cat_cols),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

def get_numerical_statistics(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
            
        q1 = series.quantile(0.25)
        q2 = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        skewness = series.skew()
        kurtosis = series.kurt()
        
        stats.append({
            "feature": col,
            "count": safe_json(series.count()),
            "mean": safe_json(series.mean()),
            "median": safe_json(q2),
            "std_dev": safe_json(series.std()),
            "min": safe_json(series.min()),
            "q1": safe_json(q1),
            "q2": safe_json(q2),
            "q3": safe_json(q3),
            "max": safe_json(series.max()),
            "iqr": safe_json(iqr),
            "skewness": safe_json(skewness),
            "kurtosis": safe_json(kurtosis)
        })
    return stats

def get_categorical_statistics(df):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    stats = []
    for col in cat_cols:
        series = df[col].dropna()
        if series.empty:
            continue
            
        unique_vals = series.nunique()
        most_freq = series.mode().iloc[0] if not series.mode().empty else None
        freq = series.value_counts().iloc[0] if not series.value_counts().empty else 0
        percent = (freq / len(df)) * 100 if len(df) > 0 else 0
        
        stats.append({
            "feature": col,
            "unique_values": safe_json(unique_vals),
            "most_frequent": str(most_freq) if most_freq is not None else "N/A",
            "frequency": safe_json(freq),
            "percentage": round(percent, 2)
        })
    return stats

def get_column_distribution(df, column):
    if column not in df.columns:
        return {"error": "Column not found"}
        
    series = df[column].dropna()
    is_num = pd.api.types.is_numeric_dtype(series)
    
    # Apply Mappings for presentation
    if column in MAPPINGS:
        series = series.map(MAPPINGS[column]).fillna(series)
        is_num = False # Treat mapped ordinals as categorical for charting
        
    if is_num:
        counts, bins = np.histogram(series, bins='auto')
        labels = [f"{round(bins[i], 1)} - {round(bins[i+1], 1)}" for i in range(len(bins)-1)]
        return {
            "type": "numerical",
            "labels": labels,
            "data": [int(x) for x in counts],
            "title": f"Distribution of {column}"
        }
    else:
        val_counts = series.value_counts().head(20)
        labels = val_counts.index.astype(str).tolist()
        data = [int(x) for x in val_counts.values]
        percentages = [round((v / len(df)) * 100, 2) for v in data]
        
        table_data = []
        for l, d, p in zip(labels, data, percentages):
            table_data.append({"Category": l, "Count": d, "Percentage": p})
            
        return {
            "type": "categorical",
            "labels": labels,
            "data": data,
            "table_data": table_data,
            "title": f"Distribution of {column}"
        }

def get_column_statistics(df, column):
    if column not in df.columns:
        return {"error": "Column not found"}
        
    series = df[column].dropna()
    is_num = pd.api.types.is_numeric_dtype(series)
    
    if column in MAPPINGS:
        series = series.map(MAPPINGS[column]).fillna(series)
        is_num = False
        
    missing = int(df[column].isna().sum())
    
    if is_num:
        q1 = series.quantile(0.25)
        q2 = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return {
            "type": "numerical",
            "name": column,
            "dtype": str(df[column].dtype),
            "count": safe_json(series.count()),
            "missing": missing,
            "mean": safe_json(series.mean()),
            "median": safe_json(q2),
            "std_dev": safe_json(series.std()),
            "min": safe_json(series.min()),
            "q1": safe_json(q1),
            "q2": safe_json(q2),
            "q3": safe_json(q3),
            "max": safe_json(series.max()),
            "iqr": safe_json(iqr),
            "skewness": safe_json(series.skew()),
            "kurtosis": safe_json(series.kurt())
        }
    else:
        unique_vals = series.nunique()
        most_freq = series.mode().iloc[0] if not series.mode().empty else None
        freq = series.value_counts().iloc[0] if not series.value_counts().empty else 0
        percent = (freq / len(df)) * 100 if len(df) > 0 else 0
        return {
            "type": "categorical",
            "name": column,
            "dtype": str(df[column].dtype),
            "count": safe_json(series.count()),
            "missing": missing,
            "unique_values": safe_json(unique_vals),
            "most_frequent": str(most_freq) if most_freq is not None else "N/A",
            "frequency": safe_json(freq),
            "percentage": round(percent, 2)
        }
