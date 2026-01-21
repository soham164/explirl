import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

class ExplanationVisualizer:
    """Visualization utilities for explanations"""
    
    @staticmethod
    def plot_shap_summary(shap_values, features, feature_names=None):
        """Create SHAP summary plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_shap)[::-1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(mean_shap))]
        
        # Plot
        y_pos = np.arange(len(indices[:10]))
        ax.barh(y_pos, mean_shap[indices[:10]])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices[:10]])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('Feature Importance (SHAP)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_lime_explanation(explanation_dict):
        """Visualize LIME explanation"""
        local_exp = explanation_dict['local_exp']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = [x[0] for x in local_exp]
        weights = [x[1] for x in local_exp]
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Weight')
        ax.set_title('LIME Local Explanation')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_counterfactual_comparison(explanation_dict):
        """Create interactive counterfactual comparison"""
        original = explanation_dict['original']
        counterfactual = explanation_dict['counterfactual']
        changes = explanation_dict['changes']
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Add original values
        fig.add_trace(go.Bar(
            name='Original',
            x=list(range(len(original))),
            y=original,
            marker_color='lightblue'
        ))
        
        # Add counterfactual values
        fig.add_trace(go.Bar(
            name='Counterfactual',
            x=list(range(len(counterfactual))),
            y=counterfactual,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Original vs Counterfactual Instance',
            xaxis_title='Feature Index',
            yaxis_title='Feature Value',
            barmode='group',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_rl_training_curve(training_log):
        """Plot RL training progress"""
        if not training_log:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        episodes = [log['episode'] for log in training_log]
        rewards = [log['reward'] for log in training_log]
        steps = [log['steps'] for log in training_log]
        
        # Reward curve
        ax1.plot(episodes, rewards, 'b-', alpha=0.6)
        ax1.plot(episodes, pd.Series(rewards).rolling(10, min_periods=1).mean(), 
                'r-', linewidth=2, label='Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Reward Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Steps to convergence
        ax2.plot(episodes, steps, 'g-', alpha=0.6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps to Convergence')
        ax2.set_title('Convergence Speed')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
