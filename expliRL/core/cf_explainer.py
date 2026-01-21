import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from typing import Any, Dict, Optional, Union, List
from .base_explainer import BaseExplainer

class CounterfactualExplainer(BaseExplainer):
    """Traditional counterfactual explainer using optimization"""
    
    def __init__(self, model: Any, data: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        super().__init__(model, data)
        self.feature_ranges = None
        self.categorical_features = []
        self.immutable_features = []
        
    def fit(self, X, y=None, feature_ranges=None, categorical_features=None, 
            immutable_features=None):
        """Initialize counterfactual explainer"""
        self.data = self._validate_instance(X)
        self.feature_ranges = feature_ranges or self._compute_feature_ranges(X)
        self.categorical_features = categorical_features or []
        self.immutable_features = immutable_features or []
        return self
    
    def _compute_feature_ranges(self, X):
        """Compute min/max ranges for each feature"""
        X = self._validate_instance(X)
        return [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], 
                desired_class: Optional[int] = None, max_iter: int = 1000, **kwargs) -> Dict:
        """Generate counterfactual explanation"""
        instance = self._validate_instance(instance)
        
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        original_pred = self.model.predict(instance)[0]
        
        if desired_class is None:
            desired_class = 1 - original_pred if original_pred in [0, 1] else None
            if desired_class is None:
                raise ValueError("Please specify desired_class for multi-class problems")
        
        # Simple gradient-free optimization
        cf = self._optimize_counterfactual(instance[0], desired_class, max_iter)
        
        self.explanation = {
            'original': instance[0],
            'counterfactual': cf,
            'original_class': original_pred,
            'counterfactual_class': self.model.predict(cf.reshape(1, -1))[0],
            'changes': cf - instance[0],
            'num_changes': np.sum(np.abs(cf - instance[0]) > 1e-5)
        }
        
        return self.explanation
    
    def _optimize_counterfactual(self, instance, desired_class, max_iter):
        """Simple optimization to find counterfactual"""
        cf = instance.copy()
        
        for _ in range(max_iter):
            # Random perturbation
            feature_idx = np.random.choice([i for i in range(len(cf)) 
                                           if i not in self.immutable_features])
            
            if feature_idx in self.categorical_features:
                # For categorical, sample from data distribution
                cf[feature_idx] = np.random.choice(self.data[:, feature_idx])
            else:
                # For continuous, perturb within range
                min_val, max_val = self.feature_ranges[feature_idx]
                perturbation = np.random.normal(0, (max_val - min_val) * 0.1)
                cf[feature_idx] = np.clip(cf[feature_idx] + perturbation, min_val, max_val)
            
            # Check if we achieved desired class
            if self.model.predict(cf.reshape(1, -1))[0] == desired_class:
                return cf
        
        return cf
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs):
        """Visualize counterfactual explanation"""
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        import matplotlib.pyplot as plt
        
        # Create comparison table
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original vs counterfactual
        features = list(range(len(explanation['original'])))
        x = np.arange(len(features))
        width = 0.35
        
        ax1.bar(x - width/2, explanation['original'], width, label='Original')
        ax1.bar(x + width/2, explanation['counterfactual'], width, label='Counterfactual')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Feature Value')
        ax1.set_title('Original vs Counterfactual')
        ax1.legend()
        
        # Plot changes
        changes = explanation['changes']
        colors = ['red' if c < 0 else 'green' for c in changes]
        ax2.bar(x, changes, color=colors)
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Change Required')
        ax2.set_title('Feature Changes for Counterfactual')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
