import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, List, Any

class ExplanationEvaluator:
    """Metrics for evaluating explanation quality"""
    
    @staticmethod
    def fidelity_score(model, explainer, X_test, sample_size=100):
        """Measure how well the explainer approximates the model"""
        indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
        
        fidelity_scores = []
        for idx in indices:
            instance = X_test[idx:idx+1]
            
            # Get model prediction
            model_pred = model.predict(instance)[0]
            
            # Get explanation and try to predict from it
            explanation = explainer.explain(instance)
            
            # For LIME, we can use the local model
            if hasattr(explanation.get('lime_exp'), 'predict_proba'):
                exp_pred = explanation['lime_exp'].predict_proba(instance)[0]
                fidelity = 1 - abs(model_pred - exp_pred)
                fidelity_scores.append(fidelity)
        
        return np.mean(fidelity_scores) if fidelity_scores else 0.0
    
    @staticmethod
    def proximity_score(original, counterfactual, metric='l2'):
        """Measure distance between original and counterfactual"""
        if metric == 'l2':
            return np.linalg.norm(original - counterfactual)
        elif metric == 'l1':
            return np.sum(np.abs(original - counterfactual))
        elif metric == 'linf':
            return np.max(np.abs(original - counterfactual))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def sparsity_score(original, counterfactual, threshold=1e-5):
        """Count number of changed features"""
        changes = np.abs(original - counterfactual) > threshold
        return np.sum(changes)
    
    @staticmethod
    def validity_score(model, counterfactuals, desired_class):
        """Percentage of valid counterfactuals"""
        valid_count = 0
        for cf in counterfactuals:
            pred = model.predict(cf.reshape(1, -1))[0]
            if pred == desired_class:
                valid_count += 1
        return valid_count / len(counterfactuals) if counterfactuals else 0.0
    
    @staticmethod
    def evaluate_all(explanation_dict, model=None):
        """Comprehensive evaluation of an explanation"""
        metrics = {}
        
        if 'counterfactual' in explanation_dict:
            original = explanation_dict['original']
            counterfactual = explanation_dict['counterfactual']
            
            metrics['proximity_l2'] = ExplanationEvaluator.proximity_score(
                original, counterfactual, 'l2')
            metrics['sparsity'] = ExplanationEvaluator.sparsity_score(
                original, counterfactual)
            
            if model and 'counterfactual_class' in explanation_dict:
                metrics['validity'] = float(
                    explanation_dict['counterfactual_class'] == 
                    explanation_dict.get('desired_class', 1 - explanation_dict['original_class'])
                )
        
        return metrics
