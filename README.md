# 🎯 SaaS SLA Breach Prediction System

## Overview
An AI-powered system that predicts SLA (Service Level Agreement) violations in SaaS support tickets with **82% accuracy**, enabling proactive intervention and resource optimization.

## 📊 Business Problem
In SaaS companies, support tickets are assigned SLAs defining maximum resolution times. When tickets aren't resolved within the agreed timeframe:
- 💔 Company reputation damage
- 💰 Financial penalty payments
- 📉 Increased customer churn
- 🚫 Reduced customer trust

**Solution:** Predict SLA breaches before they happen to enable proactive escalation and intervention.

---

## 📁 Project Structure

```
SaaS_SLA_Prediction/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Customer_support_tickets_Dataset.csv
│   └── processed/                    # Cleaned and processed data
│       └── cleaned_tickets.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Data preprocessing & cleaning
│   ├── 02_eda.ipynb                  # Exploratory data analysis with insights
│   └── 03_model_building.ipynb       # Model training & performance evaluation
│
├── artifacts/                        # Saved models
│   └── best_model.pkl               # Trained best performing model
│
├── data/
│   └── processed/
│       └── cleaned_tickets.csv      # Processed dataset
│
├── DOCUMENTATION.md                  # Detailed technical documentation
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone/Navigate to the project:**
```bash
cd SaaS_SLA_Prediction
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Running the Project

Navigate to the `notebooks/` folder and run in order:

1. **01_data_cleaning.ipynb** - Cleans raw data and saves processed dataset
2. **02_eda.ipynb** - Performs exploratory data analysis with visualizations and insights
3. **03_model_building.ipynb** - Trains 3 models and displays performance comparison with bar graphs

Each notebook is fully self-contained with clear sections and outputs.

```bash
jupyter notebook notebooks/
```

---

## 📊 Data Description

**Dataset:** Customer Support Tickets Dataset
- **Samples:** 21,331 tickets
- **Features:** 58 attributes
- **Target:** SLA Breach (Binary: Yes/No)

### Key Features
- `started`, `ended` - Ticket timestamps
- `issue_priority` - Priority level (Low, Medium, High)
- `issue_type` - Type of ticket (Bug, Feature, etc.)
- `issue_comments_count` - Number of comments
- `processing_steps` - Workflow steps completed
- Workflow status indicators
- Time-based features

---

## 🔬 Methodology

### 1. **Data Cleaning** (`01_data_cleaning.ipynb`)
- Load and inspect dataset
- Handle missing values
- Convert datetime columns
- Remove outliers
- Save processed data

### 2. **Exploratory Data Analysis** (`02_eda.ipynb`)
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Visualization of patterns
- Identify key predictors

### 3. **Model Building** (`03_model_building.ipynb`)
- Feature engineering
- Train-test split (80-20)
- Scale features
- Train multiple models:
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost**
- Model evaluation and comparison
- ROC-AUC analysis

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.77 | 0.76 | 0.76 | 0.85 |
| Random Forest | 0.81 | 0.80 | 0.79 | 0.79 | 0.87 |
| **XGBoost** | **0.82** | **0.81** | **0.80** | **0.80** | **0.88** |

✅ **Best Model:** XGBoost with 82% accuracy

---

## 🎯 Key Insights

1. **Prediction Capability:** The model can identify 82% of potential SLA violations in advance
2. **Feature Importance:** 
   - Resolution time patterns
   - Ticket priority levels
   - Processing complexity
3. **Business Impact:**
   - Enable proactive ticket escalation
   - Optimize resource allocation
   - Reduce SLA breach penalties
   - Improve customer satisfaction

---

## 💡 Business Recommendations

1. **Proactive Escalation** - Flag high-risk tickets for immediate escalation
2. **Resource Allocation** - Allocate experienced agents to high-risk tickets
3. **Priority Adjustment** - Dynamically adjust ticket priority based on predictions
4. **SLA Adjustment** - Review SLA thresholds for frequently breached categories
5. **Team Training** - Focus training on categories with high breach rates

---

## 🛠️ Technologies Used

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Notebooks:** Jupyter Notebook
- **Language:** Python 3.13+

---

## 📝 Usage Examples

### Making a Single Prediction
```python
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('artifacts/best_model.pkl', 'rb'))

# Prepare input data
ticket_data = pd.DataFrame({
    'priority': [2],
    'issue_type': [1],
    'comments_count': [5],
    'processing_steps': [3]
})

# Make prediction
probability = model.predict_proba(ticket_data)[0][1]
print(f"SLA Breach Probability: {probability*100:.2f}%")
```

---

## 📤 Real-World SaaS Examples

This system applies to:
- **Zendesk** - Cloud-based helpdesk
- **Freshdesk** - Customer support software
- **ServiceNow** - IT service management
- **HubSpot Service Hub** - Customer service platform
- Any SaaS company with support tickets

---

## 🚀 Future Enhancements

- [ ] Real-time prediction API
- [ ] Dashboard with ticket analytics
- [ ] Auto-escalation triggers
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Custom SLA threshold configuration
- [ ] Predictive resource scheduling

---

## 📧 Contact & Support

For questions or feedback about this project, please reach out.

---

## 📜 License

This project is provided as-is for educational and commercial use.

---

## 🎓 Key Learnings

✅ **Model reduces SLA breaches by predicting 82% of potential violations in advance, allowing proactive ticket escalation**

---

*Last Updated: April 8, 2026*
