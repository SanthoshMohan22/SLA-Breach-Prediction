# AI-Powered SLA Violation Prediction Model - Complete Documentation

## Project Overview
This project builds a machine learning model to predict SLA (Service Level Agreement) violations in a SaaS support system. The model analyzes customer support tickets and identifies which ones are likely to breach the 48-hour resolution SLA threshold.

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── Customer_support_tickets_Dataset.csv
│   └── processed/
│       └── cleaned_tickets.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_building.ipynb
├── artifacts/
│   └── best_model.pkl
├── DOCUMENTATION.md
└── README.md
```

---

## Phase 1: Data Cleaning (01_data_cleaning.ipynb)

### Objectives
- Load raw customer support tickets dataset
- Handle missing values and data inconsistencies
- Remove duplicates
- Prepare data for exploratory analysis

### Key Operations
- Loads raw data from `data/raw/Customer_support_tickets_Dataset.csv`
- Performs data validation and cleaning
- Outputs cleaned dataset to `data/processed/cleaned_tickets.csv`

### Output
- **Cleaned dataset**: 5,022 records with 58 features

---

## Phase 2: Exploratory Data Analysis (02_eda.ipynb)

### Objectives
- Understand data distributions and patterns
- Identify correlations between features
- Detect outliers and anomalies
- Generate visualizations for insights

### Key Analysis Areas

#### 1. Dataset Overview
- Total Records: 5,022
- Total Features: 58
- Complete dataset with minimal missing values

#### 2. Data Distribution
- **ID**: Range from 0 to 30M (unique identifier)
- **Issue Number**: Count from 0 to 3500
- **Issue Control Count**: Heavy skew towards lower values
- **Comments Count**: Right-skewed distribution

#### 3. Correlation Insights
Top correlations identified:
- Strong relationships between workflow status fields
- Issue control count correlates with multiple workflow states
- Temporal features show meaningful relationships

#### 4. Categorical Features
- Multiple categorical columns with varying unique values
- Issue types and workflow statuses are primary categorical features

#### 5. Data Quality
- **Completeness**: >99.5% of data is non-null
- **No duplicates**: All records are unique
- **Proper data types**: Mixed numeric and categorical data

#### 6. Statistical Properties
- Skewness varies across features (some right-skewed, some symmetric)
- Multiple features with moderate to high kurtosis
- Ready for modeling with minimal preprocessing

### Visualizations Generated
1. **Correlation Matrix Heatmap**: Shows relationships between all numeric features
2. **Distribution Plots**: Histograms for key numeric variables
3. **Statistical Summary**: Descriptive statistics for all columns

---

## Phase 3: Model Building & Evaluation (03_model_building.ipynb)

### 1. Feature Engineering

#### Target Variable Creation
- **SLA Threshold**: 48 hours
- **Definition**: SLA breach = ticket resolution time > 48 hours
- **Implementation**: Calculate resolution_time_hours from 'started' and 'ended' timestamps
- **Balance**: Class distribution analyzed for imbalanced data handling

#### Feature Selection
- **Selected Features**: 38 numeric features (after NaN handling)
- **Removed**: 'sla_breach' target variable from feature set
- **Data Sampling**: 30% of dataset (1,507 samples) used for faster training
- **Feature Cleaning**: NaN values filled with column means

### 2. Data Preparation

#### Train-Test Split
- **Test Size**: 20%
- **Stratification**: Used to maintain class balance
- **Random State**: 42 (for reproducibility)
- Training set: Remaining 80%

#### Feature Scaling
- StandardScaler applied to training data
- Scaler fit on training set, applied to test set
- Ensures equal feature importance in distance-based models

### 3. Models Trained

#### Model 1: Logistic Regression
- **Type**: Linear classifier using scaled features
- **Parameters**: max_iter=1000, random_state=42
- **Best For**: Interpretable predictions and baseline performance
- **Characteristics**: Works well with scaled numerical features

#### Model 2: Random Forest
- **Type**: Ensemble (20 trees for fast training)
- **Parameters**: n_estimators=20, max_depth=10, random_state=42
- **Best For**: Feature importance, reduced overfitting
- **Characteristics**: No scaling required, handles non-linearity

#### Model 3: XGBoost
- **Type**: Gradient boosting classifier (20 trees)
- **Parameters**: n_estimators=20, max_depth=5, learning_rate=0.1, random_state=42
- **Best For**: High performance on complex patterns
- **Characteristics**: Sequential tree building, optimal for predictions

### 4. Model Performance Metrics

#### Evaluation Metrics Used
1. **Accuracy**: Overall correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under the Receiver Operating Characteristic curve
6. **Confusion Matrix**: True/False positives and negatives

#### Performance Summary
- **Best Performing Model**: Random Forest
- **Model Comparison**: All three models evaluated on same test set
- **Visualization**: Bar graph showing all metrics for side-by-side comparison

### Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 93.71% | 97.01% | 95.94% | 96.47% |
| **Random Forest** | **100%** | **100%** | **100%** | **100%** |
| **XGBoost** | **100%** | **100%** | **100%** | **100%** |

**🎯 Best Model: Random Forest with 100% Accuracy**

### 5. Final Deliverables

#### Best Model
- **Model**: Best performing classifier selected
- **Save Location**: `artifacts/best_model.pkl`
- **Format**: Pickled Python object for production deployment

#### Performance Visualization
- **Bar Graph**: Shows accuracy, precision, recall, and F1-score for all models
- **ROC Curves**: Displays AUC for each model's probabilistic predictions
- **Comparison**: Easy identification of best-performing model

---

## Key Findings & Insights

### Data Insights (from EDA)
1. ✓ Dataset is well-structured with 5,022 support tickets
2. ✓ Minimal missing data ensures robust analysis
3. ✓ Numeric features show varied distributions (mix of normal and skewed)
4. ✓ Strong correlations between workflow-related features
5. ✓ Multiple categorical variables provide rich context for predictions

### Model Insights
1. All three models demonstrate viable performance on SLA prediction
2. XGBoost typically performs best on complex business problems
3. Precision/Recall trade-off visible across models
4. ROC-AUC provides good discrimination capability

### Business Implications
- Model can identify high-risk tickets early
- Enables proactive SLA management
- Helps allocate resources to critical tickets
- Supports data-driven support team decisions

---

## How to Use the Models

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

### Load Best Model
```python
import pickle
best_model = pickle.load(open('artifacts/best_model.pkl', 'rb'))
```

### Make Predictions
```python
predictions = best_model.predict(X_new)
probabilities = best_model.predict_proba(X_new)
```

---

## Requirements & Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization

### Python Version
- Python 3.13.12 or higher

---

## Next Steps & Improvements

1. **Model Tuning**: GridSearchCV for hyperparameter optimization
2. **Feature Engineering**: Create advanced features from timestamps
3. **Class Imbalance**: Apply SMOTE if breach class is underrepresented
4. **Cross-Validation**: K-fold validation for robust evaluation
5. **Explainability**: SHAP values for model interpretation
6. **Deployment**: ML pipeline for automated predictions
7. **Monitoring**: Track model performance on new data

---

## Troubleshooting

### Common Issues

**Issue**: File not found error
- **Solution**: Ensure `data/processed/` directory exists before running

**Issue**: Package import errors
- **Solution**: Install all required packages using pip

**Issue**: Out of memory
- **Solution**: Reduce batch size or use subset of data for testing

---

## Contact & Support

For issues or questions regarding this project, contact the data science team.

**Last Updated**: April 8, 2026
**Version**: 1.0
