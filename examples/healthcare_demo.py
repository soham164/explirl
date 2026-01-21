# healthcare_diagnosis_explainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from expliRL import LIMEExplainer, CounterfactualExplainer

# Create sample medical data
def create_medical_data():
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'glucose': np.random.normal(100, 30, n_samples),
        'heart_rate': np.random.normal(70, 15, n_samples),
        'exercise_hours': np.random.exponential(2, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Create risk score
    risk_score = (
        (data['age'] > 50) * 0.3 +
        (data['bmi'] > 30) * 0.2 +
        (data['blood_pressure'] > 140) * 0.2 +
        (data['cholesterol'] > 240) * 0.2 +
        (data['smoking'] == 1) * 0.3
    )
    
    data['high_risk'] = (risk_score > 0.5).astype(int)
    return data

# Prepare data
data = create_medical_data()
X = data.drop('high_risk', axis=1)
y = data['high_risk']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# EXPLAIN A HIGH-RISK PATIENT
high_risk_patients = X[y == 1]
patient = high_risk_patients.iloc[0:1]

print("PATIENT RISK ASSESSMENT")
print("="*50)
print("Patient metrics:")
for col, val in patient.iloc[0].items():
    print(f"  {col}: {val:.2f}")

# 1. LIME - Local explanation
print("\n1. LIME Explanation - Why classified as high risk:")
lime_exp = LIMEExplainer(model, mode='classification')
lime_exp.fit(X.values, feature_names=X.columns.tolist(), class_names=['Low Risk', 'High Risk'])
lime_result = lime_exp.explain(patient.values[0], num_features=len(X.columns))

print("\nRisk factors (sorted by importance):")
for feature, weight in sorted(lime_result['local_exp'], key=lambda x: abs(x[1]), reverse=True):
    risk_direction = "increases" if weight > 0 else "decreases"
    print(f"  {feature}: {weight:.3f} ({risk_direction} risk)")

# 2. Counterfactual - What changes would reduce risk?
print("\n2. Counterfactual Analysis - Path to low risk:")
cf_exp = CounterfactualExplainer(model)

# Define which features can be changed
immutable_features = [X.columns.get_loc('age')]  # Age cannot be changed
cf_exp.fit(X.values, immutable_features=immutable_features)

cf_result = cf_exp.explain(patient.values[0], desired_class=0, max_iter=1000)

if cf_result['counterfactual_class'] == 0:
    print("\nModifiable risk factors:")
    for i, col in enumerate(X.columns):
        if col != 'age':  # Skip immutable features
            original = patient.iloc[0][col]
            counterfactual = cf_result['counterfactual'][i]
            change = counterfactual - original
            if abs(change) > 0.01:
                print(f"  {col}: {original:.2f} → {counterfactual:.2f}")
    
    # Generate health recommendations
    print("\nHealth Recommendations:")
    if cf_result['counterfactual'][X.columns.get_loc('bmi')] < patient.iloc[0]['bmi']:
        print("  • Reduce BMI through diet and exercise")
    if cf_result['counterfactual'][X.columns.get_loc('blood_pressure')] < patient.iloc[0]['blood_pressure']:
        print("  • Lower blood pressure through medication or lifestyle changes")
    if cf_result['counterfactual'][X.columns.get_loc('smoking')] < patient.iloc[0]['smoking']:
        print("  • Quit smoking")