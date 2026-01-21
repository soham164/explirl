import numpy as np

class CounterfactualReward:
    """Configurable reward functions for counterfactual generation"""
    
    def __init__(self, weights=None):
        """Initialize reward function with custom weights"""
        self.weights = weights or {
            'classification': 10.0,
            'proximity': -0.1,
            'sparsity': -0.05,
            'realism': -0.02
        }
    
    def calculate(self, original, current, model, desired_class, data_distribution=None):
        """Calculate total reward"""
        rewards = {}
        
        # Classification reward
        current_pred = model.predict(current.reshape(1, -1))[0]
        rewards['classification'] = self.weights['classification'] if current_pred == desired_class else -1.0
        
        # Proximity penalty (L2 distance)
        distance = np.linalg.norm(current - original)
        rewards['proximity'] = self.weights['proximity'] * distance
        
        # Sparsity penalty (number of changed features)
        num_changes = np.sum(np.abs(current - original) > 1e-5)
        rewards['sparsity'] = self.weights['sparsity'] * num_changes
        
        # Realism penalty (distance from data distribution)
        if data_distribution is not None:
            dist_to_data = np.min([np.linalg.norm(current - x) for x in data_distribution])
            rewards['realism'] = self.weights['realism'] * dist_to_data
        
        return sum(rewards.values()), rewards
