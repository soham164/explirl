import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Any, Dict, Optional, Union
from .base_explainer import BaseExplainer
from ..rl_engine.envs import CounterfactualEnv
from ..rl_engine.agents import DQNAgent
from ..rl_engine.rewards import CounterfactualReward

class RLCounterfactualExplainer(BaseExplainer):
    """RL-driven counterfactual explainer"""
    
    def __init__(self, model: Any, data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 agent_type: str = 'dqn'):
        super().__init__(model, data)
        self.agent_type = agent_type
        self.env = None
        self.agent = None
        self.training_log = []
    
    def fit(self, X, y=None, feature_ranges=None, **kwargs):
        """Initialize RL environment and agent"""
        self.data = self._validate_instance(X)
        
        # Create environment
        self.env = CounterfactualEnv(
            model=self.model,
            data=self.data,
            feature_ranges=feature_ranges
        )
        
        # Create agent
        state_size = self.data.shape[1]
        action_size = state_size * 2  # +/- for each feature
        
        if self.agent_type == 'dqn':
            self.agent = DQNAgent(state_size, action_size)
        
        return self
    
    def explain(self, instance: Union[np.ndarray, pd.DataFrame], 
                desired_class: Optional[int] = None, 
                num_episodes: int = 100, **kwargs) -> Dict:
        """Generate RL-based counterfactual - CONCRETE IMPLEMENTATION"""
        instance = self._validate_instance(instance)
        
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        original_pred = self.model.predict(instance)[0]
        
        if desired_class is None:
            desired_class = 1 - original_pred if original_pred in [0, 1] else None
        
        # Reset environment with target instance
        state, _ = self.env.reset(instance[0], desired_class)
        
        # Train agent
        best_cf = None
        best_reward = -float('inf')
        
        for episode in range(num_episodes):
            state, _ = self.env.reset(instance[0], desired_class)
            total_reward = 0
            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated) and steps < 50:
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if (terminated or truncated) and reward > best_reward:
                    best_reward = reward
                    best_cf = state.copy()
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
            
            self.training_log.append({
                'episode': episode,
                'steps': steps,
                'reward': total_reward,
                'converged': terminated
            })
        
        if best_cf is None:
            best_cf = state
        
        self.explanation = {
            'original': instance[0],
            'counterfactual': best_cf,
            'original_class': original_pred,
            'counterfactual_class': self.model.predict(best_cf.reshape(1, -1))[0],
            'changes': best_cf - instance[0],
            'num_changes': np.sum(np.abs(best_cf - instance[0]) > 1e-5),
            'training_log': self.training_log[-10:]
        }
        
        return self.explanation
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs):
        """Visualize RL counterfactual and training progress - CONCRETE IMPLEMENTATION"""
        if explanation is None:
            explanation = self.explanation
        
        if explanation is None:
            raise ValueError("No explanation to visualize. Run explain() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(15, 5))
            
            # Plot 1: Original vs Counterfactual
            ax1 = plt.subplot(131)
            features = list(range(len(explanation['original'])))
            x = np.arange(len(features))
            width = 0.35
            
            ax1.bar(x - width/2, explanation['original'], width, label='Original')
            ax1.bar(x + width/2, explanation['counterfactual'], width, label='Counterfactual')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Value')
            ax1.set_title('Original vs RL-Counterfactual')
            ax1.legend()
            
            # Plot 2: Feature Changes
            ax2 = plt.subplot(132)
            changes = explanation['changes']
            colors = ['red' if c < 0 else 'green' for c in changes]
            ax2.bar(x, changes, color=colors)
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Change')
            ax2.set_title('Feature Changes')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Plot 3: Training Progress
            ax3 = plt.subplot(133)
            if self.training_log:
                episodes = [log['episode'] for log in self.training_log[-20:]]
                rewards = [log['reward'] for log in self.training_log[-20:]]
                ax3.plot(episodes, rewards, 'b-')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Reward')
                ax3.set_title('RL Training Progress')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available. Showing text summary:")
            print(f"Original class: {explanation['original_class']}")
            print(f"Counterfactual class: {explanation['counterfactual_class']}")
            print(f"Number of changes: {explanation['num_changes']}")