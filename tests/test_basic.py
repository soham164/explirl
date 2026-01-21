# test_basic.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=100, n_features=4, n_informative=3, 
                          n_redundant=1, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
print("Training model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

# Test each explainer
print("\n" + "="*50)
print("Testing SHAP Explainer...")
from expliRL import SHAPExplainer
shap_exp = SHAPExplainer(model, explainer_type='tree')
shap_exp.fit(X_train)
explanation = shap_exp.explain(X_test[0])
print("✓ SHAP explanation generated")

print("\n" + "="*50)
print("Testing LIME Explainer...")
from expliRL import LIMEExplainer
lime_exp = LIMEExplainer(model)
lime_exp.fit(X_train)
explanation = lime_exp.explain(X_test[0])
print("✓ LIME explanation generated")

print("\n" + "="*50)
print("Testing Counterfactual Explainer...")
from expliRL import CounterfactualExplainer
cf_exp = CounterfactualExplainer(model)
cf_exp.fit(X_train)
explanation = cf_exp.explain(X_test[0])
print(f"✓ Counterfactual found: Class {explanation['original_class']} → {explanation['counterfactual_class']}")

print("\n" + "="*50)
print("Testing RL Counterfactual Explainer...")
from expliRL import RLCounterfactualExplainer
rl_exp = RLCounterfactualExplainer(model)
rl_exp.fit(X_train)
explanation = rl_exp.explain(X_test[0], num_episodes=10)  # Few episodes for quick test
print(f"✓ RL Counterfactual found: {explanation['num_changes']} features changed")

print("\n✅ All explainers working!")