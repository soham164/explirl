import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union
from .base_explainer import BaseExplainer

class SHAPExplainer(BaseExplainer):
    """SHAP-based explainer wrapper"""
    
    def __init__(self, model: Any, data: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                 explainer_type: str = 'tree'):
        super().__init__(model, data)
        self.explainer_type = explainer_type
        self.explainer = None
        self.feature_names = None
        
    def fit(self, X, y=None, feature_names=None):
        """Initialize SHAP explainer"""
        self.feature_names = feature_names
        
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model, X)
        elif self.explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, X)
        elif self.explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X)
        else:
            # Default to Explainer which auto-selects
            self.explainer = shap.Explainer(self.model, X)
        
        return self
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict:
        """Generate SHAP explanation"""
        instance = self._validate_instance(instance)
        
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(instance)
        
        # Handle multi-class outputs
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) == 1 else shap_values
        
        # Ensure shap_values is properly formatted
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        
        # Convert to numpy array if needed
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        self.explanation = {
            'shap_values': shap_values,
            'base_value': self.explainer.expected_value,
            'instance': instance,
            'feature_names': self.feature_names
        }
        
        return self.explanation
    
    def visualize(self, explanation: Optional[Dict] = None, plot_type: str = 'waterfall', **kwargs):
        """Visualize SHAP explanations"""
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        shap_values = explanation['shap_values']
        
        if plot_type == 'waterfall':
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                base_values=explanation['base_value'],
                data=explanation['instance'][0],
                feature_names=explanation['feature_names']
            ))
        elif plot_type == 'force':
            shap.force_plot(
                explanation['base_value'],
                shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                explanation['instance'][0],
                feature_names=explanation['feature_names']
            )
        elif plot_type == 'summary':
            shap.summary_plot(shap_values, explanation['instance'], 
                             feature_names=explanation['feature_names'])
        
        plt.show()
