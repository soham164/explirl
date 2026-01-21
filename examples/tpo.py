"""
expliRL Visualization Demo
=========================
This example demonstrates the visualization capabilities of expliRL
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Import expliRL components
from expliRL import SHAPExplainer, LIMEExplainer, CounterfactualExplainer
from expliRL.utils.visualization import ExplanationVisualizer

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
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Select an instance to explain
    instance_idx = 0
    instance = X_test[instance_idx:instance_idx+1]
    
    print(f"Explaining instance {instance_idx}:")
    print(f"Features: {instance[0]}")
    print(f"Prediction: {model.predict(instance)[0]}")
    
    # Initialize visualizer
    viz = ExplanationVisualizer()
    
    # 1. Generate and visualize SHAP explanation
    print("\nGenerating SHAP explanation...")
    try:
        shap_explainer = SHAPExplainer(model, explainer_type='tree')
        shap_explainer.fit(X_train, feature_names=iris.feature_names)
        explanation = shap_explainer.explain(instance)
        
        shap_values = explanation['shap_values']
        features = X_test[instance_idx]
        feature_names = iris.feature_names
        
        # Handle different shapes of SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create SHAP summary plot
        fig = viz.plot_shap_summary(shap_values, features, feature_names)
        plt.show()
        
    except Exception as e:
        print(f"SHAP visualization error: {e}")
    
    # 2. Generate and visualize LIME explanation
    print("\nGenerating LIME explanation...")
    try:
        lime_explainer = LIMEExplainer(model, mode='classification')
        lime_explainer.fit(X_train, feature_names=iris.feature_names)
        lime_explanation = lime_explainer.explain(instance, num_features=4)
        
        fig = viz.plot_lime_explanation(lime_explanation)
        plt.show()
        
    except Exception as e:
        print(f"LIME visualization error: {e}")
    
    # 3. Generate and visualize Counterfactual explanation
    print("\nGenerating Counterfactual explanation...")
    try:
        cf_explainer = CounterfactualExplainer(model)
        cf_explainer.fit(X_train)
        cf_explanation = cf_explainer.explain(instance, max_iter=1000)
        
        fig = viz.plot_counterfactual_comparison(cf_explanation)
        fig.show()  # Plotly uses .show() instead of plt.show()
        
    except Exception as e:
        print(f"Counterfactual visualization error: {e}")
    
    print("\nVisualization demo complete!")

if __name__ == "__main__":
    main()