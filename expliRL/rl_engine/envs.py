# expliRL/rl_engine/envs.py
import gymnasium as gym  # Changed from 'import gym'
import numpy as np
from gymnasium import spaces  # Changed from 'gym import spaces'

class CounterfactualEnv(gym.Env):
    """Gym environment for counterfactual generation"""
    
    def __init__(self, model, data, feature_ranges=None):
        super(CounterfactualEnv, self).__init__()
        
        self.model = model
        self.data = data
        self.n_features = data.shape[1]
        
        # Feature ranges
        if feature_ranges is None:
            self.feature_ranges = [(data[:, i].min(), data[:, i].max()) 
                                  for i in range(self.n_features)]
        else:
            self.feature_ranges = feature_ranges
        
        # Action space: modify each feature up or down
        self.action_space = spaces.Discrete(self.n_features * 2)
        
        # State space: feature vector
        self.observation_space = spaces.Box(
            low=np.array([r[0] for r in self.feature_ranges], dtype=np.float32),
            high=np.array([r[1] for r in self.feature_ranges], dtype=np.float32),
            dtype=np.float32
        )
        
        self.original_instance = None
        self.current_state = None
        self.desired_class = None
        self.steps = 0
        self.max_steps = 50
        
    def reset(self, instance=None, desired_class=None, seed=None, options=None):
        """Reset environment with new instance - updated for gymnasium"""
        super().reset(seed=seed)
        
        if instance is not None:
            self.original_instance = instance.copy()
            self.current_state = instance.copy()
            self.desired_class = desired_class
        else:
            # If no instance provided, use stored one
            self.current_state = self.original_instance.copy()
        
        self.steps = 0
        return self.current_state, {}  # Gymnasium returns (obs, info)
    
    def step(self, action):
        """Execute action and return new state"""
        self.steps += 1
        
        # Decode action
        feature_idx = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Modify feature
        step_size = (self.feature_ranges[feature_idx][1] - 
                    self.feature_ranges[feature_idx][0]) * 0.1
        
        self.current_state[feature_idx] += direction * step_size
        
        # Clip to valid range
        self.current_state[feature_idx] = np.clip(
            self.current_state[feature_idx],
            self.feature_ranges[feature_idx][0],
            self.feature_ranges[feature_idx][1]
        )
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        current_pred = self.model.predict(self.current_state.reshape(1, -1))[0]
        terminated = (current_pred == self.desired_class)
        truncated = (self.steps >= self.max_steps)
        
        info = {
            'current_class': current_pred,
            'desired_class': self.desired_class,
            'distance': np.linalg.norm(self.current_state - self.original_instance)
        }
        
        # Gymnasium uses terminated and truncated instead of done
        return self.current_state, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """Calculate reward for current state"""
        current_pred = self.model.predict(self.current_state.reshape(1, -1))[0]
        
        # Reward components
        class_reward = 10.0 if current_pred == self.desired_class else -1.0
        
        # Distance penalty
        distance = np.linalg.norm(self.current_state - self.original_instance)
        distance_penalty = -0.1 * distance
        
        # Sparsity bonus (fewer changes)
        num_changes = np.sum(np.abs(self.current_state - self.original_instance) > 1e-5)
        sparsity_bonus = -0.05 * num_changes
        
        return class_reward + distance_penalty + sparsity_bonus