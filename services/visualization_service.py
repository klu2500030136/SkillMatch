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

def _is_numeric(df, col):
    return pd.api.types.is_numeric_dtype(df[col])

def get_filter_options(df):
    """Returns unique values for categorical columns to be used in filters."""
    options = {}
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df[col].nunique() < 20: # Only provide filters for manageable cardinality
            options[col] = df[col].dropna().unique().tolist()
    return options

def apply_filters(df, filters):
    """Applies a dictionary of filters (col: value) to the dataframe."""
    temp_df = df.copy()
    if filters:
        for col, val in filters.items():
            if col in temp_df.columns:
                temp_df = temp_df[temp_df[col] == val]
    return temp_df

def map_ordinal_series(series, col_name):
    if col_name in MAPPINGS:
        return series.map(MAPPINGS[col_name]).fillna(series)
    return series

def _format_label(col):
    """Format camelCase or PascalCase to Title Case with spaces."""
    import re
    s = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', col)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s).replace('_', ' ').title()

def generate_histogram(df, x_col):
    if not _is_numeric(df, x_col):
        return {"error": "Histogram requires a numerical feature."}
        
    s = df[x_col].dropna()
    if s.empty:
        return {"error": "No usable values were found for this visualization."}
        
    counts, bins = np.histogram(s, bins='auto')
    labels = [f"{round(bins[i], 1)} - {round(bins[i+1], 1)}" for i in range(len(bins)-1)]
    
    return {
        "chart_type": "bar",
        "title": f"{_format_label(x_col)} Distribution",
        "labels": labels,
        "datasets": [{
            "label": "Count",
            "data": [int(c) for c in counts],
            "backgroundColor": "#3b82f6"
        }],
        "x_label": _format_label(x_col),
        "y_label": "Frequency",
        "insight": f"The values range from {safe_json(s.min())} to {safe_json(s.max())} with a median of {safe_json(s.median())}."
    }

def generate_bar_chart(df, x_col):
    s = df[x_col].dropna()
    s = map_ordinal_series(s, x_col)
    
    if s.empty:
        return {"error": "No usable values were found for this visualization."}
        
    val_counts = s.value_counts().head(20) # Limit to top 20
    
    return {
        "chart_type": "bar",
        "title": f"{_format_label(x_col)} Frequencies",
        "labels": val_counts.index.astype(str).tolist(),
        "datasets": [{
            "label": "Count",
            "data": val_counts.values.tolist(),
            "backgroundColor": "#14b8a6"
        }],
        "x_label": _format_label(x_col),
        "y_label": "Count",
        "insight": f"The highest employee count is in '{val_counts.index[0]}'."
    }

def generate_scatter_plot(df, x_col, y_col):
    if not _is_numeric(df, x_col) or not _is_numeric(df, y_col):
        return {"error": "Scatter plot requires two numerical features."}
        
    temp_df = df[[x_col, y_col]].dropna()
    if temp_df.empty:
        return {"error": "No usable values were found for this visualization."}
        
    data_points = [{"x": safe_json(row[x_col]), "y": safe_json(row[y_col])} for _, row in temp_df.iterrows()]
    
    corr = temp_df.corr().iloc[0,1]
    corr_str = f"a Pearson correlation of {corr:.2f}" if not pd.isna(corr) else "no correlation"
    
    return {
        "chart_type": "scatter",
        "title": f"{_format_label(y_col)} vs {_format_label(x_col)}",
        "labels": [],
        "datasets": [{
            "label": f"{_format_label(y_col)}",
            "data": data_points,
            "backgroundColor": "#8b5cf6"
        }],
        "x_label": _format_label(x_col),
        "y_label": _format_label(y_col),
        "insight": f"The plotted values show the observed relationship between {_format_label(x_col)} and {_format_label(y_col)} with {corr_str}."
    }

def generate_grouped_visualization(df, feature, group_col, agg='mean'):
    if feature not in df.columns or group_col not in df.columns:
        return {"error": "Invalid features selected."}
        
    temp_df = df[[feature, group_col]].dropna()
    
    # Map ordinals for grouping column
    temp_df[group_col] = map_ordinal_series(temp_df[group_col], group_col)
    
    if _is_numeric(df, feature):
        # Numerical grouped by categorical
        if agg not in ['mean', 'median', 'sum', 'count']:
            agg = 'mean'
            
        grouped = temp_df.groupby(group_col)[feature].agg(agg).reset_index()
        
        # Sort values
        grouped = grouped.sort_values(by=feature, ascending=False).head(20)
        
        return {
            "chart_type": "bar",
            "title": f"{agg.capitalize()} {_format_label(feature)} by {_format_label(group_col)}",
            "labels": grouped[group_col].astype(str).tolist(),
            "datasets": [{
                "label": f"{agg.capitalize()} {_format_label(feature)}",
                "data": [safe_json(v) for v in grouped[feature]],
                "backgroundColor": "#f59e0b"
            }],
            "x_label": _format_label(group_col),
            "y_label": f"{agg.capitalize()} {_format_label(feature)}",
            "insight": f"'{grouped.iloc[0][group_col]}' has the highest {agg} {_format_label(feature)}."
        }
    else:
        # Categorical vs Categorical (Stacked/Grouped Bar)
        temp_df[feature] = map_ordinal_series(temp_df[feature], feature)
        crosstab = pd.crosstab(temp_df[group_col], temp_df[feature])
        
        datasets = []
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#14b8a6']
        
        for i, col in enumerate(crosstab.columns):
            datasets.append({
                "label": str(col),
                "data": crosstab[col].tolist(),
                "backgroundColor": colors[i % len(colors)]
            })
            
        return {
            "chart_type": "bar",
            "title": f"{_format_label(feature)} by {_format_label(group_col)}",
            "labels": crosstab.index.astype(str).tolist(),
            "datasets": datasets,
            "x_label": _format_label(group_col),
            "y_label": "Count",
            "insight": f"Observed distribution of {_format_label(feature)} across {_format_label(group_col)}.",
            "stacked": True
        }

def generate_correlation_matrix(df):
    numerics = df.select_dtypes(include=[np.number])
    if numerics.empty:
        return {"error": "No numerical features available for correlation."}
        
    corr = numerics.corr()
    
    labels = corr.columns.tolist()
    matrix = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            matrix.append({
                "x": labels[j],
                "y": labels[i],
                "v": safe_json(val) if not pd.isna(val) else 0
            })
            
    return {
        "chart_type": "heatmap",
        "title": "Numerical Features Correlation Matrix",
        "labels": labels,
        "data": matrix,
        "insight": "Pearson correlation between numerical features. Values closer to 1 or -1 indicate strong relationships."
    }

def get_box_plot_data(df, feature, group_col=None):
    if not _is_numeric(df, feature):
        return {"error": "Box plot requires a numerical feature."}
        
    temp_df = df.copy()
    if group_col:
        temp_df[group_col] = map_ordinal_series(temp_df[group_col], group_col)
        temp_df = temp_df[[feature, group_col]].dropna()
        groups = temp_df.groupby(group_col)
    else:
        temp_df = temp_df[[feature]].dropna()
        groups = [("All Data", temp_df)]
        
    labels = []
    datasets = []
    
    for name, group in groups:
        s = group[feature]
        q1 = s.quantile(0.25)
        median = s.median()
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        min_val = s[s >= (q1 - 1.5 * iqr)].min()
        max_val = s[s <= (q3 + 1.5 * iqr)].max()
        
        labels.append(str(name))
        datasets.append([min_val, q1, median, q3, max_val])
        
    return {
        "chart_type": "boxplot", # Requires a custom chart.js implementation or plugin
        "title": f"Box Plot of {_format_label(feature)}" + (f" by {_format_label(group_col)}" if group_col else ""),
        "labels": labels,
        "data": datasets,
        "insight": "Shows the median, quartiles, and range to identify distribution and outliers."
    }
