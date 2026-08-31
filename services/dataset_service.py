import os
import pandas as pd
from werkzeug.utils import secure_filename
import uuid

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file, upload_folder):
    """
    Validates and saves the uploaded file securely.
    Returns (stored_filepath, original_filename, file_type, error_message).
    """
    if file.filename == '':
        return None, None, None, "No selected file."
    
    if file and allowed_file(file.filename):
        original_filename = file.filename
        file_type = original_filename.rsplit('.', 1)[1].lower()
        
        # Generate a safe, unique filename to prevent overwriting
        secure_name = secure_filename(original_filename)
        unique_id = str(uuid.uuid4())[:8]
        stored_filename = f"{unique_id}_{secure_name}"
        
        stored_filepath = os.path.join(upload_folder, stored_filename)
        file.save(stored_filepath)
        
        return stored_filepath, original_filename, file_type, None
    
    return None, None, None, "Unsupported file type. Please upload a CSV or Excel file."

def load_dataset(filepath, file_type):
    """
    Loads the dataset into a pandas DataFrame.
    Returns (DataFrame, error_message).
    """
    try:
        if file_type == 'csv':
            df = pd.read_csv(filepath)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        else:
            return None, "Unsupported file type."
        
        # Basic validation
        if df.empty:
            return None, "The uploaded dataset is completely empty."
        if df.shape[0] == 0:
            return None, "The uploaded dataset has no rows."
        if df.shape[1] == 0:
            return None, "The uploaded dataset has no columns."
        if len(df.columns) != len(set(df.columns)):
            return None, "The uploaded dataset contains duplicate column names."
            
        return df, None
        
    except pd.errors.EmptyDataError:
        return None, "The uploaded dataset is empty."
    except pd.errors.ParserError:
        return None, "We couldn't parse this file. Please verify that it is a valid CSV."
    except Exception as e:
        return None, "We couldn't read this dataset. Please verify that the file is a valid CSV or Excel file."

def get_metadata(df):
    """Extracts basic metadata from the DataFrame."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

def get_schema(df):
    """Generates column-level schema information."""
    schema = []
    
    # Map pandas dtypes to user-friendly strings
    dtype_mapping = {
        'object': 'Categorical / Text',
        'int64': 'Integer',
        'float64': 'Decimal',
        'bool': 'Boolean',
        'datetime64[ns]': 'Date/Time'
    }
    
    for col in df.columns:
        pd_dtype = str(df[col].dtype)
        friendly_type = dtype_mapping.get(pd_dtype, pd_dtype)
        
        # Check if object might actually be categorical with few uniques
        if pd_dtype == 'object' and df[col].nunique() < 10:
            friendly_type = 'Categorical'
            
        schema.append({
            "column": col,
            "dtype": friendly_type,
            "non_null": int(df[col].notna().sum()),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True))
        })
    return schema

def detect_targets(df):
    """Identifies primary (classification) and secondary (regression) targets."""
    targets = {
        "classification": None,
        "regression": None
    }
    
    # Primary objective detection: Attrition
    if 'Attrition' in df.columns:
        targets['classification'] = {
            "name": "Attrition",
            "type": "Classification",
            "description": "Employee leave / stay prediction"
        }
        
    # Secondary objective detection: YearsAtCompany
    if 'YearsAtCompany' in df.columns:
        targets['regression'] = {
            "name": "YearsAtCompany",
            "type": "Regression",
            "description": "Employee tenure estimation"
        }
        
    return targets

def get_preview(df):
    """Returns the first 10 rows for preview."""
    # Convert to dict, handling NaN values safely for JSON serialization
    preview_df = df.head(10).fillna('')
    columns = list(preview_df.columns)
    data = preview_df.to_dict('records')
    return {
        "columns": columns,
        "data": data
    }
