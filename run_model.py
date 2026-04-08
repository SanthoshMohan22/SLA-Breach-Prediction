import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print('✓ Loading data...')
df = pd.read_csv("data/processed/cleaned_tickets.csv")
print(f"Dataset shape: {df.shape}")

print('✓ Feature engineering...')
df['started'] = pd.to_datetime(df['started'], errors='coerce')
df['ended'] = pd.to_datetime(df['ended'], errors='coerce')
df['resolution_time_hours'] = (df['ended'] - df['started']).dt.total_seconds() / 3600
SLA_THRESHOLD = 48
df['sla_breach'] = (df['resolution_time_hours'] > SLA_THRESHOLD).astype(int)
print(f"SLA Breach distribution:\n{df['sla_breach'].value_counts()}")

print('\n✓ Preparing data...')
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'sla_breach' in numeric_features:
    numeric_features.remove('sla_breach')

df_sample = df.sample(frac=0.15, random_state=42)
X = df_sample[numeric_features].copy()
# Fill NaN with column mean, dropping any cols still with NaN
X = X.fillna(X.mean()).dropna(axis=1)
y = df_sample['sla_breach'].copy()
print(f"Features after cleaning: {X.shape}")
print(f"Features: {X.shape}, Target: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('✓ Data prepared')

print('\n' + '='*60)
print('TRAINING MODELS')
print('='*60)

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"✓ Logistic Regression Accuracy: {lr_accuracy:.4f}")

print("Training Random Forest (10 trees)...")
rf_model = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"✓ Random Forest Accuracy: {rf_accuracy:.4f}")

print("Training XGBoost (10 trees)...")
xgb_model = XGBClassifier(n_estimators=10, max_depth=4, learning_rate=0.15, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"✓ XGBoost Accuracy: {xgb_accuracy:.4f}")

models = {
    'Logistic Regression': (lr_model, lr_pred, 'scaled'),
    'Random Forest': (rf_model, rf_pred, 'normal'),
    'XGBoost': (xgb_model, xgb_pred, 'normal')
}

best_model_name = max(models.items(), key=lambda x: accuracy_score(y_test, x[1][1]))[0]
best_model, best_pred, _ = models[best_model_name]

print('\n' + '='*60)
print(f'BEST MODEL: {best_model_name}')
print('='*60)
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

print("\nSaving model...")
pickle.dump(best_model, open('artifacts/best_model.pkl', 'wb'))
print('✓ Best model saved to artifacts/best_model.pkl')

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE SCORES")
print("="*70)
for model_name, (model, pred, _) in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

print(f"\n{'='*70}")
print(f"🎯 BEST MODEL: {best_model_name}")
print(f"✓ Best Model Accuracy: {accuracy_score(y_test, best_pred):.4f}")
print(f"{'='*70}")
print("\n✅ EXECUTION COMPLETE!")
