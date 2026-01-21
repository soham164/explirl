"""
expliRL Complete Demo Script
============================
A comprehensive demonstration of all expliRL capabilities
Run this for a complete walkthrough of the library
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time

from expliRL import (
    SHAPExplainer,
    LIMEExplainer,
    CounterfactualExplainer,
    RLCounterfactualExplainer
)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def demo_intro():
    """Introduction to expliRL"""
    print_section("Welcome to expliRL Demo")
    print("""
expliRL is a unified explainable AI framework that combines:
  1. SHAP - Feature importance via game theory
  2. LIME - Local interpretable model-agnostic explanations
  3. Traditional Counterfactuals - Optimization-based changes
  4. RL Counterfactuals - Reinforcement learning optimized
  
Let's see them in action!
    """)
    time.sleep(2)

def demo_data_setup():
    """Setup demo data and model"""
    print_section("Data Setup")
    
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Binary classification
    y_binary = (y == 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"✓ Model trained successfully!")
    print(f"✓ Accuracy: {accuracy:.2%}")
    
    # Select instance to explain
    instance_idx = 0
    instance = X_test[instance_idx]
    prediction = model.predict([instance])[0]
    
    print(f"\n📍 Explaining instance {instance_idx}:")
    print(f"   Features: {instance}")
    print(f"   Prediction: {'Setosa' if prediction == 1 else 'Not Setosa'} (class {prediction})")
    
    return iris, X_train, X_test, y_train, y_test, model, instance

def demo_shap(model, X_train, instance, feature_names):
    """Demonstrate SHAP explainer"""
    print_section("1️⃣  SHAP Explanation")
    
    print("Generating SHAP values...")
    explainer = SHAPExplainer(model, explainer_type='tree')
    explainer.fit(X_train, feature_names=feature_names)
    explanation = explainer.explain(instance)
    
    shap_values = explanation['shap_values']
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]
    
    # Flatten to 1D array
    shap_values = np.asarray(shap_values).flatten()
    
    print("\n Feature Contributions (SHAP values):")
    for name, value in zip(feature_names, shap_values):
        direction = "↑" if value > 0 else "↓"
        print(f"   {name:25s}: {value:+.4f} {direction}")
    
    print("\n💡 Interpretation:")
    print("   Positive values push prediction toward class 1 (Setosa)")
    print("   Negative values push prediction toward class 0 (Not Setosa)")
    
    return explanation

def demo_lime(model, X_train, instance, feature_names):
    """Demonstrate LIME explainer"""
    print_section("2️⃣  LIME Explanation")
    
    print("Generating LIME local explanation...")
    explainer = LIMEExplainer(model, mode='classification')
    explainer.fit(X_train, feature_names=feature_names)
    explanation = explainer.explain(instance, num_features=4)
    
    print("\n Local Feature Weights:")
    for feat, weight in explanation['local_exp']:
        direction = "↑" if weight > 0 else "↓"
        print(f"   {feat:40s}: {weight:+.4f} {direction}")
    
    print("\n💡 Interpretation:")
    print("   LIME shows which features matter most for THIS specific instance")
    print("   It builds a local linear model around the prediction")
    
    return explanation

def demo_counterfactual(model, X_train, instance, feature_names):
    """Demonstrate traditional counterfactual"""
    print_section("3️⃣  Traditional Counterfactual")
    
    print("Finding counterfactual explanation...")
    print("(Using optimization to find minimal changes)")
    
    explainer = CounterfactualExplainer(model)
    explainer.fit(X_train)
    explanation = explainer.explain(instance, max_iter=1000)
    
    print(f"\n Original prediction: Class {explanation['original_class']}")
    print(f" Counterfactual prediction: Class {explanation['counterfactual_class']}")
    print(f"🔄 Features changed: {explanation['num_changes']}")
    
    print("\n Required Changes:")
    for i, (name, change) in enumerate(zip(feature_names, explanation['changes'])):
        if abs(change) > 1e-5:
            original = instance[i]
            new_value = original + change
            print(f"   {name:25s}: {original:.2f} → {new_value:.2f} ({change:+.2f})")
    
    return explanation

def demo_rl_counterfactual(model, X_train, instance, feature_names):
    """Demonstrate RL-based counterfactual"""
    print_section("4️⃣  RL-Optimized Counterfactual")
    
    print("Training RL agent to find optimal counterfactual...")
    print("(Using Deep Q-Network to minimize changes)")
    
    explainer = RLCounterfactualExplainer(model)
    explainer.fit(X_train)
    
    start_time = time.time()
    explanation = explainer.explain(instance, num_episodes=50)
    elapsed = time.time() - start_time
    
    print(f"\n Original prediction: Class {explanation['original_class']}")
    print(f" Counterfactual prediction: Class {explanation['counterfactual_class']}")
    print(f" Features changed: {explanation['num_changes']}")
    print(f"⏱  Training time: {elapsed:.2f}s")
    
    print("\n RL-Optimized Changes:")
    for i, (name, change) in enumerate(zip(feature_names, explanation['changes'])):
        if abs(change) > 1e-5:
            original = instance[i]
            new_value = original + change
            print(f"   {name:25s}: {original:.2f} → {new_value:.2f} ({change:+.2f})")
    
    if explanation['training_log']:
        last_log = explanation['training_log'][-1]
        print(f"\n🎓 RL Training Stats:")
        print(f"   Episodes: {last_log['episode'] + 1}")
        print(f"   Final reward: {last_log['reward']:.2f}")
        print(f"   Converged: {'Yes' if last_log['converged'] else 'No'}")
    
    return explanation

def demo_comparison(cf_explanation, rl_explanation):
    """Compare traditional vs RL counterfactuals"""
    print_section("  Comparison: Traditional vs RL")
    
    cf_changes = cf_explanation['num_changes']
    rl_changes = rl_explanation['num_changes']
    
    print(f"\n Efficiency Comparison:")
    print(f"   Traditional CF: {cf_changes} features changed")
    print(f"   RL-Optimized CF: {rl_changes} features changed")
    
    if rl_changes < cf_changes:
        improvement = ((cf_changes - rl_changes) / cf_changes) * 100
        print(f"\n RL found a {improvement:.1f}% more efficient solution!")
        print(f"   (Fewer changes = more actionable explanations)")
    elif rl_changes == cf_changes:
        print(f"\n✓ Both methods found equally efficient solutions")
    
    print("\n Key Insight:")
    print("   RL learns to find minimal changes through reinforcement learning")
    print("   This makes explanations more practical and actionable")

def demo_summary():
    """Print demo summary"""
    print_section(" Demo Complete!")
    
    print("""
    What we demonstrated:

1. SHAP - Global feature importance using game theory
   → Shows which features matter most overall

2. LIME - Local explanations for individual predictions
   → Explains why THIS specific prediction was made

3. Traditional Counterfactuals - What needs to change
   → Shows minimal changes to flip the prediction

4. RL Counterfactuals - Optimized changes using AI
   → Finds even more efficient explanations

    Next Steps:

• Try with your own models: Works with any scikit-learn model
• Explore visualizations: explainer.visualize()
• Use the CLI: explirl explain -m model.pkl -d data.csv
• Deploy the API: explirl serve
• Check documentation: README.md

📦 Installation:
   pip install expliRL  (coming soon to PyPI!)

🌟 Star us on GitHub: github.com/explirl/expliRL
    """)

def main():
    """Run complete demo"""
    try:
        # Introduction
        demo_intro()
        
        # Setup
        iris, X_train, X_test, y_train, y_test, model, instance = demo_data_setup()
        feature_names = iris.feature_names
        
        # Run all explainers
        shap_exp = demo_shap(model, X_train, instance, feature_names)
        lime_exp = demo_lime(model, X_train, instance, feature_names)
        cf_exp = demo_counterfactual(model, X_train, instance, feature_names)
        rl_exp = demo_rl_counterfactual(model, X_train, instance, feature_names)
        
        # Comparison
        demo_comparison(cf_exp, rl_exp)
        
        # Summary
        demo_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
