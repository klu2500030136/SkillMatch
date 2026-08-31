from flask import Blueprint, render_template, redirect, url_for, request, session, flash, current_app, jsonify
from services import dataset_service
import os
import pandas as pd
import numpy as np
from datetime import datetime

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
        return render_template('visualization.html', dataset=dataset_state)
    
    filepath = dataset_state.get('stored_filepath')
    file_type = dataset_state.get('file_type')
    
    df, load_error = dataset_service.load_dataset(filepath, file_type)
    if load_error:
        flash(load_error, 'danger')
        return render_template('visualization.html', dataset={'loaded': False})
        
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


from services import model_training_service

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
    return render_template('placeholders/evaluation.html')

@main_bp.route('/predict')
def predict():
    return render_template('placeholders/prediction.html')
