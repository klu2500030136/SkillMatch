import pandas as pd
import numpy as np

# Mapping dictionaries for HR presentation
MAPPINGS = {
    'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
    'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
    'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
}

def safe_json(val):
    """Converts numpy types to standard python types for JSON serialization."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return round(float(val), 2)
    return str(val)

def _is_numeric(df, feature):
    return pd.api.types.is_numeric_dtype(df[feature])

def analyze_distribution(df, feature):
    if feature not in df.columns:
        return {"success": False, "error": f"Feature {feature} not found."}
        
    result = {
        "success": True,
        "analysis_type": "distribution",
        "title": f"{feature} Distribution",
        "chart": None,
        "table": None,
        "summary": "Distribution analysis."
    }
    
    s = df[feature].dropna()
    
    if _is_numeric(df, feature):
        # Histogram data
        counts, bins = np.histogram(s, bins=10)
        labels = [f"{round(bins[i], 1)} - {round(bins[i+1], 1)}" for i in range(len(bins)-1)]
        
        result['chart'] = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{"label": "Count", "data": [int(x) for x in counts], "backgroundColor": "#3b82f6"}]
            }
        }
        result['summary'] = f"The values range from {safe_json(s.min())} to {safe_json(s.max())} with a median of {safe_json(s.median())}."
    else:
        # Categorical frequencies
        val_counts = s.value_counts().head(15) # Top 15 to prevent huge charts
        labels = val_counts.index.astype(str).tolist()
        data = val_counts.values.tolist()
        
        result['chart'] = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{"label": "Count", "data": [int(x) for x in data], "backgroundColor": "#14b8a6"}]
            }
        }
        
        table_rows = [{"Category": str(k), "Count": safe_json(v), "Percentage": f"{round((v/len(s))*100, 1)}%"} for k, v in val_counts.items()]
        result['table'] = {"columns": ["Category", "Count", "Percentage"], "rows": table_rows}
        result['summary'] = f"The most frequent category is '{labels[0]}' with {data[0]} occurrences."
        
    return result

def analyze_summary(df, feature):
    if feature not in df.columns:
        return {"success": False, "error": f"Feature {feature} not found."}
        
    s = df[feature].dropna()
    
    rows = []
    if _is_numeric(df, feature):
        rows = [
            {"Metric": "Count", "Value": safe_json(s.count())},
            {"Metric": "Mean", "Value": safe_json(s.mean())},
            {"Metric": "Median", "Value": safe_json(s.median())},
            {"Metric": "Std Dev", "Value": safe_json(s.std())},
            {"Metric": "Minimum", "Value": safe_json(s.min())},
            {"Metric": "25th Percentile", "Value": safe_json(s.quantile(0.25))},
            {"Metric": "75th Percentile", "Value": safe_json(s.quantile(0.75))},
            {"Metric": "Maximum", "Value": safe_json(s.max())},
        ]
    else:
        rows = [
            {"Metric": "Count", "Value": safe_json(s.count())},
            {"Metric": "Unique Values", "Value": safe_json(s.nunique())},
            {"Metric": "Most Frequent", "Value": str(s.mode().iloc[0]) if not s.mode().empty else "N/A"},
            {"Metric": "Frequency", "Value": safe_json(s.value_counts().iloc[0]) if not s.value_counts().empty else "N/A"},
        ]
        
    return {
        "success": True,
        "analysis_type": "summary",
        "title": f"{feature} Summary",
        "table": {"columns": ["Metric", "Value"], "rows": rows},
        "summary": "Basic summary statistics."
    }

def analyze_missingness(df, feature):
    if feature not in df.columns:
        return {"success": False, "error": f"Feature {feature} not found."}
        
    total = len(df)
    missing = int(df[feature].isna().sum())
    non_missing = total - missing
    pct = round((missing / total) * 100, 2)
    
    rows = [
        {"Metric": "Total Rows", "Value": total},
        {"Metric": "Missing Count", "Value": missing},
        {"Metric": "Non-Missing Count", "Value": non_missing},
        {"Metric": "Missing Percentage", "Value": f"{pct}%"},
    ]
    
    return {
        "success": True,
        "analysis_type": "missingness",
        "title": f"{feature} Missingness",
        "table": {"columns": ["Metric", "Value"], "rows": rows},
        "summary": f"{missing} missing values detected ({pct}%)."
    }

def analyze_target_relationship(df, feature, target_col, target_type):
    if feature not in df.columns or target_col not in df.columns:
        return {"success": False, "error": "Required columns not found."}
        
    result = {
        "success": True,
        "analysis_type": "target_relationship",
        "title": f"{feature} vs {target_col}",
        "chart": None,
        "table": None,
        "summary": "Target relationship analysis."
    }
    
    # Drop NAs
    subset = df[[feature, target_col]].dropna()
    if subset.empty:
        return {"success": False, "error": "No valid data to compare."}
        
    if target_type == "classification" and target_col == "Attrition":
        # Group by Attrition
        if _is_numeric(df, feature):
            grouped = subset.groupby(target_col)[feature].agg(['count', 'mean', 'median']).reset_index()
            rows = []
            for _, r in grouped.iterrows():
                rows.append({"Attrition": str(r[target_col]), "Count": safe_json(r['count']), "Mean": safe_json(r['mean']), "Median": safe_json(r['median'])})
            
            result['table'] = {"columns": ["Attrition", "Count", "Mean", "Median"], "rows": rows}
            
            # Bar chart of Means
            labels = grouped[target_col].astype(str).tolist()
            means = [safe_json(x) for x in grouped['mean']]
            result['chart'] = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": f"Average {feature}", "data": means, "backgroundColor": "#8b5cf6"}]
                }
            }
            if len(means) == 2:
                diff = means[1] - means[0] if means[0] != 0 else 0
                word = "higher" if diff > 0 else "lower"
                result['summary'] = f"The average {feature} is {abs(diff):.2f} {word} when Attrition is {labels[1]} compared to {labels[0]}."
        else:
            # Categorical vs Attrition
            crosstab = pd.crosstab(subset[feature], subset[target_col])
            # Add total and percentage if "Yes" exists
            rows = []
            labels = crosstab.index.astype(str).tolist()
            yes_rates = []
            
            for idx in crosstab.index:
                row_dict = {feature: str(idx)}
                total = 0
                yes_count = 0
                for col in crosstab.columns:
                    val = int(crosstab.loc[idx, col])
                    row_dict[f"Attrition = {col}"] = val
                    total += val
                    if str(col).lower() == 'yes':
                        yes_count = val
                
                row_dict["Total"] = total
                rate = round((yes_count / total) * 100, 1) if total > 0 else 0
                row_dict["Attrition Rate"] = f"{rate}%"
                yes_rates.append(rate)
                rows.append(row_dict)
                
            cols = [feature] + [f"Attrition = {c}" for c in crosstab.columns] + ["Total", "Attrition Rate"]
            result['table'] = {"columns": cols, "rows": rows}
            
            # Chart
            result['chart'] = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "Attrition Rate (%)", "data": yes_rates, "backgroundColor": "#ec4899"}]
                }
            }
    
    elif target_type == "regression" and target_col == "YearsAtCompany":
        if _is_numeric(df, feature):
            corr = subset.corr().iloc[0, 1]
            result['summary'] = f"The Pearson correlation between {feature} and {target_col} is {safe_json(corr)}."
        else:
            # Categorical vs YearsAtCompany
            grouped = subset.groupby(feature)[target_col].agg(['count', 'mean', 'median']).reset_index()
            rows = []
            for _, r in grouped.iterrows():
                rows.append({feature: str(r[feature]), "Count": safe_json(r['count']), "Mean Years": safe_json(r['mean']), "Median Years": safe_json(r['median'])})
            
            result['table'] = {"columns": [feature, "Count", "Mean Years", "Median Years"], "rows": rows}
            
            labels = grouped[feature].astype(str).tolist()
            means = [safe_json(x) for x in grouped['mean']]
            result['chart'] = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "Average Years", "data": means, "backgroundColor": "#10b981"}]
                }
            }
            
    return result

def analyze_group_comparison(df, feature, group_col):
    if feature not in df.columns or group_col not in df.columns:
        return {"success": False, "error": "Required columns not found."}
        
    subset = df[[feature, group_col]].dropna()
    
    if _is_numeric(df, feature):
        grouped = subset.groupby(group_col)[feature].agg(['count', 'mean', 'median']).reset_index()
        rows = []
        labels = grouped[group_col].astype(str).tolist()
        means = []
        for _, r in grouped.iterrows():
            m = safe_json(r['mean'])
            means.append(m)
            rows.append({group_col: str(r[group_col]), "Count": safe_json(r['count']), "Mean": m, "Median": safe_json(r['median'])})
            
        return {
            "success": True,
            "title": f"{feature} grouped by {group_col}",
            "table": {"columns": [group_col, "Count", "Mean", "Median"], "rows": rows},
            "chart": {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{"label": f"Average {feature}", "data": means, "backgroundColor": "#6366f1"}]
                }
            }
        }
    else:
        # categorical vs categorical
        crosstab = pd.crosstab(subset[group_col], subset[feature])
        rows = []
        for idx in crosstab.index:
            row_dict = {group_col: str(idx)}
            for col in crosstab.columns:
                row_dict[str(col)] = int(crosstab.loc[idx, col])
            rows.append(row_dict)
            
        cols = [group_col] + [str(c) for c in crosstab.columns]
        return {
            "success": True,
            "title": f"{feature} grouped by {group_col}",
            "table": {"columns": cols, "rows": rows}
        }

def analyze_correlation(df, feature):
    if feature not in df.columns:
        return {"success": False, "error": "Feature not found."}
    if not _is_numeric(df, feature):
        return {"success": False, "error": "Correlation requires a numerical feature."}
        
    numerics = df.select_dtypes(include=[np.number])
    if feature not in numerics.columns:
        return {"success": False, "error": "Correlation requires a numerical feature."}
        
    corr_matrix = numerics.corr()
    corrs = corr_matrix[feature].drop(feature).dropna()
    corrs = corrs.sort_values(key=abs, ascending=False)
    
    rows = []
    labels = []
    data = []
    
    for idx, val in corrs.items():
        v = safe_json(val)
        rows.append({"Feature": str(idx), "Correlation": v})
        labels.append(str(idx))
        data.append(v)
        
    return {
        "success": True,
        "title": f"Correlations with {feature}",
        "table": {"columns": ["Feature", "Correlation"], "rows": rows},
        "chart": {
            "type": "bar",
            "data": {
                "labels": labels[:10], # Top 10 for chart
                "datasets": [{"label": "Pearson Correlation", "data": data[:10], "backgroundColor": "#f43f5e"}]
            }
        },
        "summary": "Showing absolute correlation strength."
    }

# =======================================================
# HR QUESTION EXPLORER
# =======================================================

def _map_ordinal(series, mapping):
    return series.map(mapping).fillna(series)

def hrq_distance_by_role_attrition(df):
    req = ['DistanceFromHome', 'JobRole', 'Attrition']
    for c in req:
        if c not in df.columns:
            return {"success": False, "error": f"Missing required column: {c}"}
            
    grouped = df.groupby(['JobRole', 'Attrition'])['DistanceFromHome'].agg(['count', 'mean', 'median']).reset_index()
    rows = []
    for _, r in grouped.iterrows():
        rows.append({
            "JobRole": str(r['JobRole']),
            "Attrition": str(r['Attrition']),
            "Employee Count": safe_json(r['count']),
            "Average Distance": safe_json(r['mean']),
            "Median Distance": safe_json(r['median'])
        })
        
    return {
        "success": True,
        "title": "Distance from home by job role and attrition",
        "table": {"columns": ["JobRole", "Attrition", "Employee Count", "Average Distance", "Median Distance"], "rows": rows}
    }

def hrq_income_by_education_attrition(df):
    req = ['MonthlyIncome', 'Education', 'Attrition']
    for c in req:
        if c not in df.columns:
            return {"success": False, "error": f"Missing required column: {c}"}
            
    temp_df = df.copy()
    temp_df['Education'] = _map_ordinal(temp_df['Education'], MAPPINGS['Education'])
    
    grouped = temp_df.groupby(['Education', 'Attrition'])['MonthlyIncome'].agg(['count', 'mean', 'median']).reset_index()
    rows = []
    for _, r in grouped.iterrows():
        rows.append({
            "Education": str(r['Education']),
            "Attrition": str(r['Attrition']),
            "Employee Count": safe_json(r['count']),
            "Average MonthlyIncome": safe_json(r['mean']),
            "Median MonthlyIncome": safe_json(r['median'])
        })
        
    return {
        "success": True,
        "title": "Average monthly income by education and attrition",
        "table": {"columns": ["Education", "Attrition", "Employee Count", "Average MonthlyIncome", "Median MonthlyIncome"], "rows": rows}
    }

def hrq_attrition_by_generic(df, feature, title, mapping=None, bin_col=False):
    req = [feature, 'Attrition']
    for c in req:
        if c not in df.columns:
            return {"success": False, "error": f"Missing required column: {c}"}
            
    temp_df = df.copy()
    if mapping:
        temp_df[feature] = _map_ordinal(temp_df[feature], mapping)
    
    if bin_col and pd.api.types.is_numeric_dtype(temp_df[feature]):
        bins = [-1, 2, 5, 10, 20, 100]
        labels = ['0-2', '3-5', '6-10', '11-20', '21+']
        temp_df[feature] = pd.cut(temp_df[feature], bins=bins, labels=labels)
        
    crosstab = pd.crosstab(temp_df[feature], temp_df['Attrition'])
    rows = []
    labels_chart = crosstab.index.astype(str).tolist()
    yes_rates = []
    
    for idx in crosstab.index:
        total = crosstab.loc[idx].sum()
        yes = crosstab.loc[idx, 'Yes'] if 'Yes' in crosstab.columns else 0
        no = crosstab.loc[idx, 'No'] if 'No' in crosstab.columns else 0
        rate = round((yes / total) * 100, 1) if total > 0 else 0
        
        rows.append({
            feature: str(idx),
            "Employees": safe_json(total),
            "Attrition = Yes": safe_json(yes),
            "Attrition = No": safe_json(no),
            "Attrition Rate": f"{rate}%"
        })
        yes_rates.append(rate)
        
    return {
        "success": True,
        "title": title,
        "table": {"columns": [feature, "Employees", "Attrition = Yes", "Attrition = No", "Attrition Rate"], "rows": rows},
        "chart": {
            "type": "bar",
            "data": {
                "labels": labels_chart,
                "datasets": [{"label": "Attrition Rate (%)", "data": yes_rates, "backgroundColor": "#f59e0b"}]
            }
        },
        "summary": "Observation of attrition rates across groups."
    }

def hrq_attrition_by_overtime(df):
    return hrq_attrition_by_generic(df, "OverTime", "Attrition by Overtime")

def hrq_attrition_by_jobrole(df):
    return hrq_attrition_by_generic(df, "JobRole", "Attrition by Job Role")

def hrq_attrition_by_jobsatisfaction(df):
    return hrq_attrition_by_generic(df, "JobSatisfaction", "Attrition by Job Satisfaction", MAPPINGS['JobSatisfaction'])

def hrq_attrition_by_worklifebalance(df):
    return hrq_attrition_by_generic(df, "WorkLifeBalance", "Attrition by Work-Life Balance", MAPPINGS['WorkLifeBalance'])

def hrq_attrition_by_tenure(df):
    return hrq_attrition_by_generic(df, "YearsAtCompany", "Attrition by Years at Company", bin_col=True)

# Question Registry
QUESTIONS = {
    "q1": hrq_distance_by_role_attrition,
    "q2": hrq_income_by_education_attrition,
    "q3": hrq_attrition_by_overtime,
    "q4": hrq_attrition_by_jobrole,
    "q5": hrq_attrition_by_jobsatisfaction,
    "q6": hrq_attrition_by_worklifebalance,
    "q7": hrq_attrition_by_tenure
}

def run_hr_question(df, question_id):
    if question_id not in QUESTIONS:
        return {"success": False, "error": "Unknown question ID."}
    return QUESTIONS[question_id](df)
