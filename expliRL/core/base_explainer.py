from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union

class BaseExplainer(ABC):
    """Abstract base class for all explainers"""
    
    def __init__(self, model: Any, data: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        self.model = model
        self.data = data
        self.explanation = None
    
    def fit(self, X, y=None):
        """Fit the explainer to training data"""
        # Default implementation - can be overridden
        self.data = self._validate_instance(X)
        return self
    
    @abstractmethod
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], **kwargs) -> Dict:
        """Generate explanation for a single instance"""
        pass
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs):
        """Visualize the explanation"""
        # Default implementation - can be overridden
        import matplotlib.pyplot as plt
        
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        print("Visualization not implemented for this explainer type.")
        print("Explanation summary:", explanation.keys())
    
    def _validate_instance(self, instance):
        """Validate input instance"""
        if isinstance(instance, pd.DataFrame):
            return instance.values
        elif isinstance(instance, np.ndarray):
            return instance
        else:
            raise ValueError("Instance must be numpy array or pandas DataFrame")