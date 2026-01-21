 """
expliRL Quickstart Example
==========================
This example demonstrates basic usage of all explainers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Import expliRL components
from expliRL import (
    SHAPExplainer,
    LIMEExplainer, 
    CounterfactualExplainer,
    RLCounterfactualExplainer
)

def main():
    # Load sample data
    print("Loading iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Binary classification (setosa vs non-setosa)
    y_binary = (y == 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    
    # Train a model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Select an instance to explain
    instance_idx = 0
    instance = X_test[instance_idx:instance_idx+1]
    
    print(f"\nExplaining instance {instance_idx}:")
    print(f"Features: {instance[0]}")
    print(f"Prediction: {model.predict(instance)[0]}")
    
    # 1. SHAP Explanation
    print("\n" + "="*50)
    print("SHAP EXPLANATION")
    print("="*50)
    
    try:
        shap_explainer = SHAPExplainer(model, explainer_type='tree')
        shap_explainer.fit(X_train, feature_names=iris.feature_names)
        shap_exp = shap_explainer.explain(instance)
        
        print("Feature contributions:")
        shap_values = shap_exp['shap_values']
        
        # Handle different shapes of SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        for i, (name, value) in enumerate(zip(iris.feature_names, shap_values)):
            print(f"  {name}: {value:.4f}")
    except Exception as e:
        print(f"SHAP error (this is okay for testing): {e}")
    
    # 2. LIME Explanation
    print("\n" + "="*50)
    print("LIME EXPLANATION")
    print("="*50)
    
    try:
        lime_explainer = LIMEExplainer(model, mode='classification')
        lime_explainer.fit(X_train, feature_names=iris.feature_names)
        lime_exp = lime_explainer.explain(instance, num_features=4)
        
        print("Local feature weights:")
        for feat, weight in lime_exp['local_exp']:
            print(f"  {feat}: {weight:.4f}")
    except Exception as e:
        print(f"LIME error (this is okay for testing): {e}")
    
    # 3. Traditional Counterfactual
    print("\n" + "="*50)
    print("COUNTERFACTUAL EXPLANATION")
    print("="*50)
    
    try:
        cf_explainer = CounterfactualExplainer(model)
        cf_explainer.fit(X_train)
        cf_exp = cf_explainer.explain(instance, max_iter=1000)
        
        print(f"Original class: {cf_exp['original_class']}")
        print(f"Counterfactual class: {cf_exp['counterfactual_class']}")
        print(f"Number of features changed: {cf_exp['num_changes']}")
        print("\nFeature changes needed:")
        for i, (name, change) in enumerate(zip(iris.feature_names, cf_exp['changes'])):
            if abs(change) > 1e-5:
                print(f"  {name}: {change:+.4f}")
    except Exception as e:
        print(f"Counterfactual error (this is okay for testing): {e}")
    
    # 4. RL-based Counterfactual
    print("\n" + "="*50)
    print("RL-BASED COUNTERFACTUAL EXPLANATION")
    print("="*50)
    
    try:
        rl_explainer = RLCounterfactualExplainer(model)
        rl_explainer.fit(X_train)
        rl_exp = rl_explainer.explain(instance, num_episodes=50)
        
        print(f"Original class: {rl_exp['original_class']}")
        print(f"Counterfactual class: {rl_exp['counterfactual_class']}")
        print(f"Number of features changed: {rl_exp['num_changes']}")
        print("\nFeature changes (RL-optimized):")
        for i, (name, change) in enumerate(zip(iris.feature_names, rl_exp['changes'])):
            if abs(change) > 1e-5:
                print(f"  {name}: {change:+.4f}")
        
        if rl_exp['training_log']:
            last_episode = rl_exp['training_log'][-1]
            print(f"\nRL Training: Converged in episode {last_episode['episode']} "
                  f"with reward {last_episode['reward']:.2f}")
    except Exception as e:
        print(f"RL Counterfactual error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Quickstart complete! Try visualizing the explanations by calling:")
    print("  explainer.visualize()")

if __name__ == "__main__":
 