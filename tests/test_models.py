import unittest
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import pandas as pd


class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for each test"""
        self.X, self.y = make_classification(n_samples=100, n_features=4, random_state=42)
        self.model = RandomForestClassifier(random_state=42)
    
    def test_model_training(self):
        """Test that model can be trained successfully"""
        self.model.fit(self.X, self.y)
        self.assertTrue(hasattr(self.model, 'feature_importances_'))
        self.assertEqual(len(self.model.feature_importances_), 4)
    
    def test_model_prediction(self):
        """Test that trained model can make predictions"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_model_serialization(self):
        """Test that model can be saved and loaded"""
        self.model.fit(self.X, self.y)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(self.model, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Test that loaded model works
            original_pred = self.model.predict(self.X)
            loaded_pred = loaded_model.predict(self.X)
            self.assertTrue((original_pred == loaded_pred).all())
        finally:
            os.unlink(temp_path)
    
    def test_data_processing(self):
        """Test data processing functionality"""
        df = pd.DataFrame(self.X, columns=['f1', 'f2', 'f3', 'f4'])
        df['target'] = self.y
        
        self.assertEqual(len(df.columns), 5)
        self.assertEqual(len(df), 100)
        self.assertTrue(all(col in df.columns for col in ['f1', 'f2', 'f3', 'f4', 'target']))


if __name__ == '__main__':
    unittest.main()
