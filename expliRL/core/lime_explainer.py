import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union
from .base_explainer import BaseExplainer

class LIMEExplainer(BaseExplainer):
    """LIME-based explainer wrapper"""
    
    def __init__(self, model: Any, data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 mode: str = 'classification'):
        super().__init__(model, data)
        self.mode = mode
        self.explainer = None
        self.feature_names = None
        self.class_names = None
        
    def fit(self, X, y=None, feature_names=None, class_names=None):
        """Initialize LIME explainer"""
        X = self._validate_instance(X)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.class_names = class_names or ['class_0', 'class_1']
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=True
        )
        
        return self
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], 
                num_features: int = 10, **kwargs) -> Dict:
        """Generate LIME explanation"""
        instance = self._validate_instance(instance)
        
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        if self.mode == 'classification':
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        exp = self.explainer.explain_instance(
            instance[0],
            predict_fn,
            num_features=num_features,
            **kwargs
        )
        
        self.explanation = {
            'lime_exp': exp,
            'local_exp': exp.as_list(),
            'instance': instance,
            'prediction': self.model.predict(instance)[0],
            'feature_names': self.feature_names
        }
        
        return self.explanation
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs):
        """Visualize LIME explanation"""
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        exp = explanation['lime_exp']
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.show()
