# SkillMatch

Employee Intelligence & Machine Learning Platform

## Current Phase
## Phase 0 — Application Foundation

The foundation phase established the application shell, including the Flask architecture, Jinja2 template inheritance, responsive CSS layout, and the overall UI design system.

## Project Status

**Currently Implemented: Phase 1 & Phase 2**

### Phase 1 — Upload Dataset
*   Upload CSV and Excel files (up to 10MB).
*   Automatic validation and ingestion using `pandas`.
*   Calculation of dataset metadata (rows, columns, missing values).
*   Automatic detection of `Attrition` (Classification Target) and `YearsAtCompany` (Regression Target).
*   Schema extraction and datatype mapping.
*   *Note: Phase 1 performs dataset ingestion and basic structural profiling only. No preprocessing or machine-learning model training is performed.*

### Phase 2 — Analyse Features
*   Interactive Feature Analysis Workspace.
*   Feature selection with dynamically mapped column schemas.
*   Support for multiple analysis methods: Distribution, Summary, Missingness, Target Relationship, Group Comparison, Correlation.
*   Predefined HR Question Explorer containing deterministic queries based on the IBM HR Analytics dataset.
*   Result visualization using Chart.js.
*   *Note: Phase 2 provides feature-level exploratory analysis. No preprocessing or machine-learning model training is performed.* 

## Purpose
Phase 0 establishes the Flask application shell and navigation architecture for SkillMatch. It provides a polished, production-style, responsive frontend and routing foundation that future machine-learning phases will build upon. 

**Important:** Machine-learning functionality is intentionally not implemented in Phase 0.

## Tech Stack
- **Python** (Backend)
- **Flask** (Web Framework)
- **HTML5** (Structure)
- **CSS3** (Styling with custom variables, no external heavy CSS frameworks)
- **Vanilla JavaScript** (Lightweight UI interactions)

## Structure
```
skillmatch/
├── app.py                  # Main application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── routes/                 # Flask routes and blueprints
│   ├── __init__.py
│   └── routes.py           # Defines the 9 stages
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Main application shell
│   ├── upload.html         # Data intake page
│   └── placeholders/       # Empty states for future ML phases
└── static/                 # Static assets
    ├── css/                # Modular CSS files (variables, main, layout, components)
    └── js/                 # UI behavior scripts
```

## Routes
1. `/upload` - Upload Dataset (Functional UI)
2. `/analyse` - Analyse Features (Placeholder)
3. `/statistics` - Descriptive Statistics (Placeholder)
4. `/missing-values` - Missing Value Analysis (Placeholder)
5. `/visualization` - Data Visualization (Placeholder)
6. `/preprocessing` - Preprocessing (Placeholder)
7. `/training` - Model Training (Placeholder)
8. `/evaluation` - Model Evaluation (Placeholder)
9. `/predict` - Predict Employee (Placeholder)

## Running the Application

Follow these steps to run the application locally:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Current Limitations
- ML functionality is intentionally not implemented yet.
- File upload UI is purely frontend; the backend does not process the dataset yet.
- Placeholder pages are shown for stages 2-9.
- No user authentication or real notifications are implemented.

## Phase 3 � Descriptive Statistics

Phase 3 provides descriptive statistical analysis of the active employee dataset. It does not perform data cleaning, preprocessing, or machine-learning training.

Features include:
- Dataset-level summary (rows, columns, feature counts, missing values, duplicates)
- Numerical statistics (Count, Mean, Median, Std Dev, Min, Q1, Q2, Q3, Max, IQR, Skewness, Kurtosis)
- Categorical statistics (Unique Values, Most Frequent, Frequency, Percentage)
- Interactive column inspection (Detailed statistics per column)
- Distribution inspection (Histograms and Bar charts)
- Filtering by feature type (Numerical vs Categorical)
- Search functionality by feature name


## Phase 4 � Missing Value Analysis

Phase 4 detects missing values, calculates missing percentages, classifies affected features, visualizes missingness, provides data-driven completeness insights, and identifies features that should be considered during preprocessing.

Phase 4 does not modify, impute, remove, or otherwise treat missing values. Missing-value treatment is performed later during Phase 6 � Preprocessing.


## Phase 5 � Data Visualization

Phase 5 provides data visualization capabilities to explore employee patterns, distributions, relationships, and attrition trends.
Features include univariate visualization (histograms, box plots, bar charts), bivariate visualization (scatter plots, grouped bar charts), HR-specific quick visualizations, numerical correlation analysis, and dynamic filtering.

Phase 5 visualizes the active dataset without modifying the underlying data. Data preprocessing and transformation are performed later in Phase 6.


## Phase 6 � Data Preprocessing

Phase 6 provides data preprocessing capabilities to transform the raw employee dataset into a clean, reproducible, machine-learning-ready dataset. 
Features include data cleaning (duplicate and constant feature removal), missing-value treatment, categorical/ordinal encoding, feature scaling (Standardization), feature selection, target preparation (Classification/Regression), and train/validation/test splitting.

Phase 6 transforms the raw employee dataset into machine-learning-ready data without training or evaluating a model. Model training begins in Phase 7.


## Phase 7 � Model Training

Phase 7 trains and compares classification models using the preprocessed training data and validation data. Features include evaluating models such as Logistic Regression (baseline), Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, and Gradient Boosting.
Models are evaluated on Accuracy, Precision, Recall, F1-score, and ROC-AUC. 
The best model is selected via the highest Validation F1-score, protecting the untouched test set for final unbiased evaluation in Phase 8.

