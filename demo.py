# demo.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from expliRL import (
    SHAPExplainer,
    LIMEExplainer,
    CounterfactualExplainer,
    RLCounterfactualExplainer
)
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target


y_binary = (y == 0).astype(int)  # Setosa vs others


X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)


print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}\n")


idx = 5
instance = X_test[idx]
prediction = model.predict([instance])[0]
print(f"Explaining instance {idx}:")
print(f"Features: {instance}")
print(f"Prediction: {'Setosa' if prediction == 1 else 'Not Setosa'}\n")
print("="*50)
print("SHAP Explanation")
print("="*50)
shap_exp = SHAPExplainer(model, explainer_type='tree')
shap_exp.fit(X_train, feature_names=iris.feature_names)
shap_result = shap_exp.explain(instance)
print("Feature importance:")
for name, value in zip(iris.feature_names, shap_result['shap_values'][0]):
    print(f"  {name}: {value:.3f}")

try:
    shap_exp.visualize(plot_type='waterfall')
except:
    print("(Visualization skipped - no display available)")

# 2. RL Counterfactual
print("\n" + "="*50)
print("RL-Driven Counterfactual")
print("="*50)
rl_exp = RLCounterfactualExplainer(model)
rl_exp.fit(X_train)
rl_result = rl_exp.explain(instance, num_episodes=20)

print(f"Original prediction: {rl_result['original_class']}")
print(f"Counterfactual prediction: {rl_result['counterfactual_class']}")
print(f"Features changed: {rl_result['num_changes']}")
print("\nMinimal changes needed:")
for i, (name, change) in enumerate(zip(iris.feature_names, rl_result['changes'])):
    if abs(change) > 0.01:
        print(f"  {name}: {change:+.3f}")

print("\n✅ Demo complete!")