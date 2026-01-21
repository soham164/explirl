import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Union, List, Optional, Tuple

class DataPreprocessor:
    """Utility class for data preprocessing"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoders = {}
        self.feature_types = {}
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            categorical_features: Optional[List] = None):
        """Fit preprocessor to data"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        
        # Fit scaler on numerical features
        if categorical_features:
            numerical_mask = [i for i in range(X_array.shape[1]) 
                            if i not in categorical_features]
            if numerical_mask:
                self.scaler.fit(X_array[:, numerical_mask])
        else:
            self.scaler.fit(X_array)
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if self.scaler:
            X_transformed = self.scaler.transform(X_array)
        else:
            X_transformed = X_array
        
        return X_transformed
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data"""
        if self.scaler:
            return self.scaler.inverse_transform(X)
        return X

    def encode_constraints(self, constraints: dict) -> dict:
        """Encode user constraints into actionable rules"""
        encoded = {
            'immutable_features': constraints.get('immutable_features', []),
            'feature_ranges': constraints.get('feature_ranges', {}),
            'categorical_values': constraints.get('categorical_values', {}),
            'max_changes': constraints.get('max_changes', None)
        }
        return encoded
