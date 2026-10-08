from flask import Blueprint, render_template, redirect, url_for, request, session, flash, current_app, jsonify
from services import dataset_service
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return redirect(url_for('main.upload'))

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'dataset_file' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)
            
        file = request.files['dataset_file']
        
        stored_filepath, original_filename, file_type, error = dataset_service.save_file(file, current_app.config['UPLOAD_FOLDER'])
        
        if error:
            flash(error, 'danger')
            return redirect(request.url)
            
        df, load_error = dataset_service.load_dataset(stored_filepath, file_type)
        
        if load_error:
            # Clean up the saved file if it can't be loaded
            if os.path.exists(stored_filepath):
                os.remove(stored_filepath)
            flash(load_error, 'danger')
            return redirect(request.url)
            
        # Extract metadata
        metadata = dataset_service.get_metadata(df)
        schema = dataset_service.get_schema(df)
        targets = dataset_service.detect_targets(df)
        preview = dataset_service.get_preview(df)
        
        # Save to session
        session['dataset'] = {
            'loaded': True,
            'original_filename': original_filename,
            'stored_filepath': stored_filepath,
            'file_type': file_type,
            'metadata': metadata,
            'schema': schema,
            'targets': targets,
            'preview': preview
        }
        session.pop('preprocessing_state', None)
        session.pop('training_state', None)
        
        flash('Dataset uploaded successfully.', 'success')
        return redirect(url_for('main.upload'))
        
    # GET request
    dataset_state = session.get('dataset', {'loaded': False})
    return render_template('upload.html', dataset=dataset_state)

@main_bp.route('/remove-dataset', methods=['POST'])
def remove_dataset():
    dataset_state = session.get('dataset')
    if dataset_state and dataset_state.get('loaded'):
        filepath = dataset_state.get('stored_filepath')
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass # Ignore if file is locked or already gone
                
        session.pop('dataset', None)
        session.pop('preprocessing_state', None)
        session.pop('training_state', None)
        flash('Dataset removed.', 'success')
    return redirect(url_for('main.upload'))

@main_bp.route('/analyse', methods=['GET'])
def analyse():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('analyse.html', dataset=dataset_state)
    
    # We pass the schema to populate feature dropdowns
    schema = dataset_state.get('schema', [])
    targets = dataset_state.get('targets', {})
    
    return render_template('analyse.html', dataset=dataset_state, schema=schema, targets=targets)

from flask import jsonify
from services import analysis_service
from services import validation_service

@main_bp.route('/analyse/run', methods=['POST'])
def analyse_run():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"success": False, "error": "No dataset loaded."})
        
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        return jsonify({"success": False, "error": load_error})
        
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON request."})
        
    question_id = data.get('question_id')
    if question_id:
        return jsonify(analysis_service.run_hr_question(df, question_id))
        
    analysis_type = data.get('analysis_type')
    feature = data.get('feature')
    
    if not feature or not analysis_type:
        return jsonify({"success": False, "error": "Feature and analysis method are required."})
        
    if analysis_type == 'distribution':
        return jsonify(analysis_service.analyze_distribution(df, feature))
    elif analysis_type == 'summary':
        return jsonify(analysis_service.analyze_summary(df, feature))
    elif analysis_type == 'missingness':
        return jsonify(analysis_service.analyze_missingness(df, feature))
    elif analysis_type == 'target_relationship':
        target_col = data.get('target_col', 'Attrition')
        target_type = data.get('target_type', 'classification')
        
        # Validation: target_col and feature should be different
        val_res = validation_service.validate_column_pair(feature, target_col, "Feature and target must be different.")
        if not val_res['valid']:
            return jsonify({"success": False, "error": val_res['error']})
            
        return jsonify(analysis_service.analyze_target_relationship(df, feature, target_col, target_type))
    elif analysis_type == 'group_comparison':
        group_col = data.get('group_col')
        if not group_col:
            return jsonify({"success": False, "error": "Grouping column required."})
            
        val_res = validation_service.validate_column_pair(feature, group_col, "Please select a grouping column different from the primary feature.")
        if not val_res['valid']:
            return jsonify({"success": False, "error": val_res['error']})
            
        return jsonify(analysis_service.analyze_group_comparison(df, feature, group_col))
    elif analysis_type == 'correlation':
        val_res = validation_service.validate_column_types(df, [feature], 'numeric', "Correlation analysis requires a numerical feature.")
        if not val_res['valid']:
            return jsonify({"success": False, "error": val_res['error']})
            
        return jsonify(analysis_service.analyze_correlation(df, feature))
        
    return jsonify({"success": False, "error": "Unknown analysis method."})

from services import statistics_service

@main_bp.route('/statistics')
def statistics():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('statistics.html', dataset=dataset_state)
    
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        flash(load_error, 'danger')
        return render_template('statistics.html', dataset={'loaded': False})
        
    overview = statistics_service.get_dataset_overview(df)
    numerical_stats = statistics_service.get_numerical_statistics(df)
    categorical_stats = statistics_service.get_categorical_statistics(df)
    columns = list(df.columns)
    
    return render_template(
        'statistics.html',
        dataset=dataset_state,
        overview=overview,
        numerical_stats=numerical_stats,
        categorical_stats=categorical_stats,
        columns=columns
    )

@main_bp.route('/statistics/column', methods=['POST'])
def statistics_column():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
        
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        return jsonify({"error": load_error})
        
    data = request.get_json()
    if not data or 'column' not in data:
        return jsonify({"error": "Column is required"})
        
    column = data['column']
    if column not in df.columns:
        return jsonify({"error": "Column not found in the active dataset."})
        
    stats = statistics_service.get_column_statistics(df, column)
    distribution = statistics_service.get_column_distribution(df, column)
    
    return jsonify({
        "success": True,
        "stats": stats,
        "distribution": distribution
    })

from services import missing_value_service

@main_bp.route('/missing-values')
def missing_values():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('missing.html', dataset=dataset_state)
    
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        flash(load_error, 'danger')
        return render_template('missing.html', dataset={'loaded': False})
        
    summary = missing_value_service.get_missing_summary(df)
    stats = missing_value_service.get_missing_statistics(df)
    chart_data = missing_value_service.get_missing_chart_data(df)
    heatmap_data = missing_value_service.get_missing_heatmap_data(df)
    insights = missing_value_service.get_missing_insights(df)
    columns = list(df.columns)
    
    return render_template(
        'missing.html',
        dataset=dataset_state,
        summary=summary,
        stats=stats,
        chart_data=chart_data,
        heatmap_data=heatmap_data,
        insights=insights,
        columns=columns
    )

@main_bp.route('/missing-values/feature', methods=['POST'])
def missing_values_feature():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
        
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        return jsonify({"error": load_error})
        
    data = request.get_json()
    if not data or 'feature' not in data:
        return jsonify({"error": "Feature is required"})
        
    feature = data['feature']
    if feature not in df.columns:
        return jsonify({"error": "Feature not found in the active dataset."})
        
    details = missing_value_service.get_feature_missing_details(df, feature)
    return jsonify({"success": True, "details": details})

from services import visualization_service

@main_bp.route('/visualization')
def visualization():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('visualization.html', dataset=dataset_state, filter_options={})
    
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        flash(load_error, 'danger')
        return render_template('visualization.html', dataset={'loaded': False}, filter_options={})
        
    columns = list(df.columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    filter_options = visualization_service.get_filter_options(df)
    
    return render_template(
        'visualization.html',
        dataset=dataset_state,
        columns=columns,
        num_cols=num_cols,
        cat_cols=cat_cols,
        filter_options=filter_options
    )

@main_bp.route('/visualization/generate', methods=['POST'])
def visualization_generate():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
        
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        return jsonify({"error": load_error})
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request."})
        
    chart_type = data.get('chart_type')
    x_column = data.get('x_column')
    y_column = data.get('y_column')
    group_column = data.get('group_column')
    aggregation = data.get('aggregation', 'mean')
    filter_column = data.get('filter_column')
    filter_value = data.get('filter_value')
    
    if filter_column and filter_value:
        if filter_column in df.columns:
            # Cast filter_value to numeric if column is numeric
            if pd.api.types.is_numeric_dtype(df[filter_column]):
                try:
                    filter_value = float(filter_value)
                except ValueError:
                    pass
            df = visualization_service.apply_filters(df, {filter_column: filter_value})
            
    if df.empty:
        return jsonify({"error": "No data available for visualization (possibly excluded by filter or empty dataset)."})
        
    # Backend Validation: Required columns exist
    required_cols = []
    if chart_type in ['histogram', 'bar', 'scatter', 'grouped_bar', 'boxplot']:
        required_cols.append(x_column)
    if chart_type == 'scatter':
        required_cols.append(y_column)
    if chart_type in ['grouped_bar', 'boxplot']:
        required_cols.append(group_column)
        
    for col in required_cols:
        if not col or col not in df.columns:
            return jsonify({"error": f"Required column '{col}' is missing or not provided."})
            
    if chart_type == 'histogram':
        val_res = validation_service.validate_column_types(df, [x_column], 'numeric', "Histogram requires a numerical feature.")
        if not val_res['valid']:
            return jsonify({"error": val_res['error']})
        result = visualization_service.generate_histogram(df, x_column)
    elif chart_type == 'bar':
        result = visualization_service.generate_bar_chart(df, x_column)
    elif chart_type == 'scatter':
        val_res = validation_service.validate_column_pair(x_column, y_column, "Please select two different features for scatter plot axes.")
        if not val_res['valid']:
            return jsonify({"error": val_res['error']})
            
        val_res2 = validation_service.validate_column_types(df, [x_column, y_column], 'numeric', "Scatter plots require numerical features for both axes.")
        if not val_res2['valid']:
            return jsonify({"error": val_res2['error']})
            
        result = visualization_service.generate_scatter_plot(df, x_column, y_column)
    elif chart_type == 'grouped_bar':
        val_res = validation_service.validate_column_pair(x_column, group_column, "Feature and grouping column must be different.")
        if not val_res['valid']:
            return jsonify({"error": val_res['error']})
            
        result = visualization_service.generate_grouped_visualization(df, x_column, group_column, aggregation)
    elif chart_type == 'boxplot':
        val_res = validation_service.validate_column_pair(x_column, group_column, "Feature and grouping column must be different.")
        if not val_res['valid']:
            return jsonify({"error": val_res['error']})
            
        result = visualization_service.get_box_plot_data(df, x_column, group_column)
    elif chart_type == 'heatmap':
        result = visualization_service.generate_correlation_matrix(df)
    else:
        return jsonify({"error": "Invalid chart type."})
        
    if "error" in result:
        return jsonify({"error": result["error"]})
        
    if "error" in result:
        return jsonify({"error": result["error"]})
        
    return jsonify({"success": True, "data": result})

from services import preprocessing_service

@main_bp.route('/preprocessing')
def preprocessing():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('preprocessing.html', dataset=dataset_state)
    
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        flash(load_error, 'danger')
        return render_template('preprocessing.html', dataset={'loaded': False})
        
    metadata = preprocessing_service.get_preprocessing_metadata(df)
    preprocessing_state = session.get('preprocessing_state', None)
    
    return render_template('preprocessing.html', dataset=dataset_state, metadata=metadata, preprocessing_state=preprocessing_state)

@main_bp.route('/preprocessing/apply', methods=['POST'])
def preprocessing_apply():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
        
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        return jsonify({"error": load_error})
        
    config = request.get_json()
    if not config:
        return jsonify({"error": "Invalid request."})
        
    # Backend Validation: Target cannot be excluded
    target_col = config.get('target_column')
    features_to_remove = config.get('features_to_remove', [])
    val_res = validation_service.validate_target_leakage(target_col, features_to_remove, "The target column cannot be excluded from the dataset.")
    if not val_res['valid']:
        return jsonify({"error": val_res['error']})
        
    # Ensure processed directory exists near uploaded dataset
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    
    result = preprocessing_service.apply_preprocessing(df, config, processed_dir)
    
    if result.get("success"):
        session['preprocessing_state'] = {
            'config': config,
            'summary': result['summary']
        }
        session.pop('training_state', None)
        
    return jsonify(result)


from services import pca_service

@main_bp.route('/pca')
def pca():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('pca.html', dataset=dataset_state)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    readiness = pca_service.check_pca_readiness(processed_dir)
    preprocessing_state = session.get('preprocessing_state', None)
    
    return render_template('pca.html', dataset=dataset_state, readiness=readiness, preprocessing_state=preprocessing_state)

@main_bp.route('/pca/features', methods=['POST'])
def pca_features():
    """Returns feature screening information."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    
    target_col = dataset_state.get('targets', {}).get('classification', {}).get('name', 'Attrition')
    result = pca_service.identify_modeling_features(processed_dir, target_col)
    
    return jsonify(result)

@main_bp.route('/pca/apply', methods=['POST'])
def pca_apply():
    """Applies PCA to the preprocessed data."""
    try:
        dataset_state = session.get('dataset')
        if not dataset_state or not dataset_state.get('loaded'):
            return jsonify({"success": False, "error": "No dataset loaded"})
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Invalid request."})
        
        variance_threshold = float(data.get('variance_threshold', 0.90))
        
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
        processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
        pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
        
        # Check readiness
        readiness = pca_service.check_pca_readiness(processed_dir)
        if not readiness.get('ready'):
            return jsonify({"success": False, "error": readiness.get('reason')})
        
        result = pca_service.apply_pca(processed_dir, pca_dir, variance_threshold)
        
        if result.get("success"):
            session['pca_state'] = {
                'timestamp': datetime.now().isoformat(),
                'variance_threshold': variance_threshold
            }
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"PCA failed: {str(e)}"})

@main_bp.route('/pca/results', methods=['POST'])
def pca_results():
    """Returns previously computed PCA results."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    result = pca_service.get_pca_results(pca_dir)
    return jsonify(result)


from services import model_training_service
from services import unsupervised_service

@main_bp.route('/training')
def training():
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('training.html', dataset=dataset_state)
        
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    
    readiness = model_training_service.check_training_readiness(processed_dir)
    training_state = session.get('training_state', None)
    
    return render_template('training.html', dataset=dataset_state, readiness=readiness, training_state=training_state)

@main_bp.route('/training/train', methods=['POST'])
def training_train():
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
        
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request."})
        
    selected_models = req.get('selected_models', [])
    if not selected_models:
        return jsonify({"error": "No models selected for training."})
        
    config = req.get('config', {})
    class_weight = req.get('class_weight', 'balanced')
    random_state = int(req.get('random_state', 42))
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    # Check readiness again securely
    readiness = model_training_service.check_training_readiness(processed_dir)
    if not readiness.get('ready'):
        return jsonify({"error": readiness.get('reason')})
        
    result = model_training_service.train_models(
        selected_models, config, class_weight, random_state, processed_dir, models_dir
    )
    
    if result.get("success"):
        session['training_state'] = {
            'timestamp': datetime.now().isoformat(),
            'best_model': result.get('best_model')
        }
        
    return jsonify(result)


@main_bp.route('/evaluation')
def evaluation():
    """Model Evaluation page - Phase 6."""
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('evaluation.html', dataset=dataset_state)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    # Check evaluation readiness
    from services import evaluation_service
    readiness = evaluation_service.check_evaluation_readiness(processed_dir, models_dir)
    
    return render_template(
        'evaluation.html',
        dataset=dataset_state,
        readiness=readiness
    )


@main_bp.route('/evaluation/model', methods=['POST'])
def evaluate_model():
    """Evaluate a specific model on the test set."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request"})
    
    model_name = req.get('model_name')
    model_type = req.get('model_type', 'classification')  # 'classification' or 'regression'
    
    if not model_name:
        return jsonify({"error": "Model name is required"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    from services import evaluation_service
    
    if model_type == 'regression':
        result = evaluation_service.evaluate_regression_model(model_name, processed_dir, models_dir)
    else:
        result = evaluation_service.evaluate_classification_model(model_name, processed_dir, models_dir, pca_dir)
    
    return jsonify(result)


@main_bp.route('/evaluation/compare', methods=['POST'])
def compare_models():
    """Evaluate all available models for comparison."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    from services import evaluation_service
    result = evaluation_service.evaluate_all_models(processed_dir, models_dir, pca_dir)
    
    return jsonify(result)

# Phase 8: Employee Prediction

@main_bp.route('/predict')
def predict():
    """Employee Prediction page - Phase 8."""
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('prediction.html', dataset=dataset_state)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    # Check prediction readiness
    from services import prediction_service
    readiness = prediction_service.check_prediction_readiness(processed_dir, models_dir)
    
    # Get categorical values for dropdowns
    categorical_values = prediction_service.get_categorical_values(upload_folder)
    
    # Get ordinal mappings
    ordinal_mappings = prediction_service.get_ordinal_mappings()
    
    return render_template(
        'prediction.html',
        dataset=dataset_state,
        readiness=readiness,
        categorical_values=categorical_values,
        ordinal_mappings=ordinal_mappings
    )


@main_bp.route('/predict/predict', methods=['POST'])
def predict_employee():
    """Predict attrition for an employee."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request"})
    
    employee_data = req.get('employee_data')
    if not employee_data:
        return jsonify({"error": "Employee data is required"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    from services import prediction_service
    result = prediction_service.predict_attrition(employee_data, processed_dir, models_dir)
    
    return jsonify(result)


# Phase 9: Project Summary & Review

@main_bp.route('/project-summary')
def project_summary():
    """Project Summary & Review Dashboard - Phase 9."""
    dataset_state = session.get('dataset', {'loaded': False})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    unsupervised_dir = os.path.join(os.path.dirname(upload_folder), 'unsupervised')
    
    from services import project_summary_service
    summary = project_summary_service.get_project_summary(upload_folder, processed_dir, models_dir, pca_dir, unsupervised_dir)
    
    return render_template('project_summary.html', summary=summary, dataset=dataset_state)


# Phase 7: 5-Fold Cross Validation

@main_bp.route('/cross-validation')
def cross_validation():
    """5-Fold Cross Validation page - Phase 7."""
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('cross_validation.html', dataset=dataset_state)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    # Check CV readiness
    from services import cross_validation_service
    readiness = cross_validation_service.check_cv_readiness(processed_dir)
    
    return render_template(
        'cross_validation.html',
        dataset=dataset_state,
        readiness=readiness
    )


@main_bp.route('/cross-validation/run', methods=['POST'])
def run_cross_validation():
    """Run 5-fold cross-validation for a specific model."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request"})
    
    model_name = req.get('model_name')
    n_splits = int(req.get('n_splits', 5))
    random_state = int(req.get('random_state', 42))
    
    if not model_name:
        return jsonify({"error": "Model name is required"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    from services import cross_validation_service
    result = cross_validation_service.run_cross_validation(model_name, processed_dir, models_dir, n_splits, random_state)
    
    return jsonify(result)


@main_bp.route('/cross-validation/compare', methods=['POST'])
def compare_cv_models():
    """Run 5-fold cross-validation for all models and compare."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    from services import cross_validation_service
    result = cross_validation_service.run_all_models_cv(processed_dir, models_dir)
    
    return jsonify(result)


# New Phase 5 routes for extended ML functionality

@main_bp.route('/ml')
def ml():
    """Main Machine Learning page with supervised and unsupervised tabs."""
    dataset_state = session.get('dataset', {'loaded': False})
    if not dataset_state.get('loaded'):
        return render_template('ml.html', dataset=dataset_state)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    # Check readiness for both supervised and unsupervised
    supervised_readiness = model_training_service.check_training_readiness(processed_dir)
    unsupervised_readiness = unsupervised_service.check_unsupervised_readiness(processed_dir, pca_dir)
    
    training_state = session.get('training_state', None)
    
    return render_template(
        'ml.html',
        dataset=dataset_state,
        supervised_readiness=supervised_readiness,
        unsupervised_readiness=unsupervised_readiness,
        training_state=training_state
    )


@main_bp.route('/ml/regression/train', methods=['POST'])
def ml_regression_train():
    """Train Linear Regression for YearsAtCompany prediction."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request."})
    
    target_col = req.get('target_col', 'YearsAtCompany')
    random_state = int(req.get('random_state', 42))
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    models_dir = os.path.join(os.path.dirname(upload_folder), 'models')
    
    result = model_training_service.train_regression_model(target_col, processed_dir, models_dir, random_state)
    
    if result.get("success"):
        session['regression_state'] = {
            'timestamp': datetime.now().isoformat(),
            'target': target_col
        }
    
    return jsonify(result)


@main_bp.route('/ml/unsupervised/kmeans', methods=['POST'])
def ml_unsupervised_kmeans():
    """Run K-Means clustering."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request."})
    
    n_clusters = int(req.get('n_clusters', 3))
    use_pca = req.get('use_pca', False)
    random_state = int(req.get('random_state', 42))
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    result = unsupervised_service.run_kmeans(processed_dir, pca_dir, n_clusters, use_pca, random_state)
    
    return jsonify(result)


@main_bp.route('/ml/unsupervised/dbscan', methods=['POST'])
def ml_unsupervised_dbscan():
    """Run DBSCAN clustering."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request."})
    
    eps = float(req.get('eps', 0.5))
    min_samples = int(req.get('min_samples', 5))
    use_pca = req.get('use_pca', False)
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    result = unsupervised_service.run_dbscan(processed_dir, pca_dir, eps, min_samples, use_pca)
    
    return jsonify(result)


@main_bp.route('/ml/unsupervised/anomaly', methods=['POST'])
def ml_unsupervised_anomaly():
    """Run Isolation Forest anomaly detection."""
    dataset_state = session.get('dataset')
    if not dataset_state or not dataset_state.get('loaded'):
        return jsonify({"error": "No dataset loaded"})
    
    req = request.get_json()
    if not req:
        return jsonify({"error": "Invalid request."})
    
    contamination = float(req.get('contamination', 0.1))
    use_pca = req.get('use_pca', False)
    random_state = int(req.get('random_state', 42))
    
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'data/uploads')
    processed_dir = os.path.join(os.path.dirname(upload_folder), 'processed')
    pca_dir = os.path.join(os.path.dirname(upload_folder), 'pca')
    
    result = unsupervised_service.run_isolation_forest(processed_dir, pca_dir, contamination, use_pca, random_state)
    
    return jsonify(result)
