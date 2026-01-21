# expliRL - Unified Explainable AI Framework 🧠

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 What is expliRL?

expliRL is an open-source Python library that unifies multiple Explainable AI (XAI) techniques into one comprehensive framework. It combines:

**For Tabular Data:**
- **SHAP** (SHapley Additive exPlanations) - Feature importance via game theory
- **LIME** (Local Interpretable Model-agnostic Explanations) - Local surrogate models
- **Traditional Counterfactuals** - Optimization-based minimal input changes
- **RL-driven Counterfactuals** - Reinforcement Learning optimized explanations

**For Image Data:** 🆕
- **Grad-CAM** - Gradient-weighted Class Activation Mapping for CNNs
- **LIME for Images** - Superpixel-based explanations for any image classifier
- **RL Image Counterfactuals** - Coming soon!

Perfect for researchers, data scientists, and enterprises who need to understand black-box model predictions through interactive explanations, visualizations, and structured logs.

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Detailed Setup Guide](#-detailed-setup-guide)
4. [Usage Examples](#-usage-examples)
5. [API Reference](#-api-reference)
6. [CLI Tool](#-cli-tool)
7. [REST API](#-rest-api)
8. [Troubleshooting](#-troubleshooting)
9. [Contributing](#-contributing)

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv explirl-env

# Activate on Windows
explirl-env\Scripts\activate

# Activate on macOS/Linux
source explirl-env/bin/activate
```

### Step 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/expliRL.git
cd expliRL

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Step 3: Install from PyPI (When Available)

```bash
pip install expliRL
```

### Step 4: Verify Installation

```bash
python -c "import expliRL; print(expliRL.__version__)"
# Should output: 0.1.0
```

## ⚡ Quick Start

```python
from expliRL import RLCounterfactualExplainer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train your model
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] > 0).astype(int)
model = RandomForestClassifier().fit(X_train, y_train)

# Initialize explainer
explainer = RLCounterfactualExplainer(model)
explainer.fit(X_train)

# Generate counterfactual for a test instance
test_instance = np.array([1.5, -0.5, 0.8, -1.2])
explanation = explainer.explain(test_instance)

# View results
print(f"Original prediction: {explanation['original_class']}")
print(f"Counterfactual prediction: {explanation['counterfactual_class']}")
print(f"Features to change: {explanation['num_changes']}")

# Visualize
explainer.visualize()
```

## 📚 Detailed Setup Guide

### Complete Installation from Scratch

#### 1. System Requirements

Ensure you have:
- Python 3.7+ installed
- Git installed
- 4GB+ RAM (8GB recommended for RL training)
- 2GB free disk space

#### 2. Full Installation Process

```bash
# Step 1: Create project directory
mkdir my-xai-project
cd my-xai-project

# Step 2: Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Clone expliRL
git clone https://github.com/yourusername/expliRL.git
cd expliRL

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Install expliRL
pip install -e .

# Step 7: Run tests (optional)
python -m pytest tests/
```

#### 3. Docker Installation (Alternative)

Create a `Dockerfile`:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "examples/quickstart.py"]
```

Build and run:

```bash
docker build -t explirl .
docker run -it explirl
```

## 🎯 Usage Examples

### Example 1: SHAP Explanations

```python
from expliRL import SHAPExplainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Create explainer
shap_exp = SHAPExplainer(model, explainer_type='tree')
shap_exp.fit(X, feature_names=iris.feature_names)

# Explain a prediction
instance = X[0]
explanation = shap_exp.explain(instance)

# Visualize
shap_exp.visualize(plot_type='waterfall')
```

### Example 2: LIME Explanations

```python
from expliRL import LIMEExplainer
from sklearn.ensemble import RandomForestClassifier

# Assume X_train, y_train are ready
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create LIME explainer
lime_exp = LIMEExplainer(model, mode='classification')
lime_exp.fit(X_train, feature_names=['f1', 'f2', 'f3', 'f4'])

# Explain
explanation = lime_exp.explain(X_test[0], num_features=4)

# Print local weights
for feature, weight in explanation['local_exp']:
    print(f"{feature}: {weight:.3f}")

# Visualize
lime_exp.visualize()
```

### Example 3: Traditional Counterfactuals

```python
from expliRL import CounterfactualExplainer

# Initialize explainer
cf_exp = CounterfactualExplainer(model)
cf_exp.fit(
    X_train,
    categorical_features=[2, 5],  # Indices of categorical features
    immutable_features=[0]        # Features that cannot be changed
)

# Generate counterfactual
explanation = cf_exp.explain(
    instance=X_test[0],
    desired_class=1,
    max_iter=1000
)

print(f"Change features: {explanation['changes']}")
print(f"Achieved class: {explanation['counterfactual_class']}")
```

### Example 4: RL-Driven Counterfactuals (Advanced)

```python
from expliRL import RLCounterfactualExplainer
import numpy as np

# Configure RL explainer
rl_exp = RLCounterfactualExplainer(model, agent_type='dqn')
rl_exp.fit(X_train)

# Generate with custom parameters
explanation = rl_exp.explain(
    instance=X_test[0],
    desired_class=1,
    num_episodes=100  # More episodes = better optimization
)

# Access training logs
for log in explanation['training_log'][-5:]:
    print(f"Episode {log['episode']}: Reward={log['reward']:.2f}, Steps={log['steps']}")

# Visualize with RL training curve
rl_exp.visualize()
```

### Example 5: Batch Processing

```python
from expliRL import SHAPExplainer
import pandas as pd

# Process multiple instances
instances_to_explain = X_test[:10]
explanations = []

explainer = SHAPExplainer(model)
explainer.fit(X_train)

for i, instance in enumerate(instances_to_explain):
    exp = explainer.explain(instance)
    explanations.append({
        'instance_id': i,
        'prediction': model.predict([instance])[0],
        'top_feature': np.argmax(np.abs(exp['shap_values']))
    })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(explanations)
print(results_df)
```

### Example 6: Credit Scoring Demo

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from expliRL import RLCounterfactualExplainer

# Load credit data
credit_data = pd.read_csv('credit_data.csv')
X = credit_data.drop('approved', axis=1)
y = credit_data['approved']

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_scaled, y)

# Explain why loan was rejected
rejected_application = X_scaled[y == 0][0]

explainer = RLCounterfactualExplainer(model)
explainer.fit(X_scaled)

explanation = explainer.explain(
    rejected_application,
    desired_class=1  # Want approval
)

# Interpret results
feature_names = X.columns
for i, (name, change) in enumerate(zip(feature_names, explanation['changes'])):
    if abs(change) > 0.1:  # Significant changes only
        original = scaler.inverse_transform([explanation['original']])[0][i]
        new = scaler.inverse_transform([explanation['counterfactual']])[0][i]
        print(f"{name}: {original:.2f} → {new:.2f}")
```

### Example 7: Image XAI with Grad-CAM 🆕

```python
from expliRL import GradCAMExplainer
import torchvision.models as models
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Create explainer
explainer = GradCAMExplainer(model)

# Load and explain image
image = Image.open('cat.jpg')
explanation = explainer.explain(image)

# Visualize heatmap showing where model focuses
explainer.visualize(explanation)
```

### Example 8: LIME for Images 🆕

```python
from expliRL import LIMEImageExplainer
import torchvision.models as models

# Load model
model = models.resnet50(pretrained=True)

# Create explainer
explainer = LIMEImageExplainer(model)

# Explain which image regions matter
explanation = explainer.explain(
    'medical_scan.jpg',
    num_samples=1000,
    num_features=10
)

# Show important regions
explainer.visualize(explanation)
```

## 📖 API Reference

### Core Classes

#### `SHAPExplainer`

```python
class SHAPExplainer(model, data=None, explainer_type='tree')
```

**Parameters:**
- `model`: Trained ML model
- `data`: Training data (optional)
- `explainer_type`: One of ['tree', 'kernel', 'linear', 'auto']

**Methods:**
- `fit(X, y=None, feature_names=None)`: Initialize explainer
- `explain(instance, **kwargs)`: Generate SHAP values
- `visualize(plot_type='waterfall')`: Create visualizations

#### `LIMEExplainer`

```python
class LIMEExplainer(model, data=None, mode='classification')
```

**Parameters:**
- `model`: Trained ML model
- `mode`: 'classification' or 'regression'

**Methods:**
- `fit(X, y=None, feature_names=None, class_names=None)`: Initialize
- `explain(instance, num_features=10)`: Generate local explanation
- `visualize()`: Display LIME plot

#### `CounterfactualExplainer`

```python
class CounterfactualExplainer(model, data=None)
```

**Methods:**
- `fit(X, feature_ranges=None, categorical_features=None, immutable_features=None)`
- `explain(instance, desired_class=None, max_iter=1000)`
- `visualize()`: Show original vs counterfactual comparison

#### `RLCounterfactualExplainer`

```python
class RLCounterfactualExplainer(model, data=None, agent_type='dqn')
```

**Parameters:**
- `agent_type`: Currently supports 'dqn'

**Methods:**
- `fit(X, feature_ranges=None)`: Initialize RL environment
- `explain(instance, desired_class=None, num_episodes=100)`: Train and generate
- `visualize()`: Show counterfactual and training curves

## 🖥️ CLI Tool

### Installation

The CLI tool is automatically installed with expliRL:

```bash
explirl --help
```

### Commands

#### Generate Explanation

```bash
# SHAP explanation
explirl explain -m model.pkl -d data.csv -i 0 -t shap -o explanation.json

# LIME explanation
explirl explain -m model.pkl -d data.csv -i 5 -t lime

# Counterfactual
explirl explain -m model.pkl -d data.csv -i 10 -t cf

# RL Counterfactual
explirl explain -m model.pkl -d data.csv -i 0 -t rl-cf
```

**Parameters:**
- `-m, --model-path`: Path to pickled model
- `-d, --data-path`: Path to CSV data
- `-i, --instance-idx`: Index of instance to explain
- `-t, --method`: Explanation method
- `-o, --output`: Save explanation to file

#### Start API Server

```bash
# Start on default port (8000)
explirl serve

# Custom host and port
explirl serve --host 0.0.0.0 --port 5000
```

## 🌐 REST API

### Starting the Server

```python
from expliRL.api.service import start_server
start_server(host="0.0.0.0", port=8000)
```

Or via CLI:
```bash
explirl serve
```

### API Endpoints

#### Health Check
```http
GET /health
```

#### Load Model
```http
POST /load_model
{
  "model_id": "rf_model",
  "model_path": "/path/to/model.pkl"
}
```

#### Generate Explanation
```http
POST /explain
{
  "model_id": "rf_model",
  "instance": [1.5, -0.5, 0.8, -1.2],
  "method": "shap",
  "params": {}
}
```

### Example API Usage with Python

```python
import requests

# Load model
response = requests.post(
    "http://localhost:8000/load_model",
    json={"model_id": "my_model", "model_path": "model.pkl"}
)

# Get explanation
response = requests.post(
    "http://localhost:8000/explain",
    json={
        "model_id": "my_model",
        "instance": [1.5, -0.5, 0.8, -1.2],
        "method": "rl_counterfactual"
    }
)

explanation = response.json()
print(explanation)
```

### Example with cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate explanation
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model1",
    "instance": [1.5, -0.5, 0.8, -1.2],
    "method": "shap"
  }'
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue 1: ImportError with SHAP

**Error:** `ImportError: cannot import name 'TreeExplainer' from 'shap'`

**Solution:**
```bash
pip install --upgrade shap
pip install numpy<1.24  # SHAP compatibility
```

#### Issue 2: PyTorch Installation Issues

**Error:** `No module named 'torch'`

**Solution:**
```bash
# CPU version
pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# GPU version (CUDA 11.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Issue 3: Memory Issues with RL Training

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Use CPU for RL training
import torch
torch.set_default_tensor_type('torch.FloatTensor')

# Reduce batch size
explainer = RLCounterfactualExplainer(model)
explainer.agent.batch_size = 16  # Default is 32
```

#### Issue 4: API Server Not Starting

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
explirl serve --port 8001
```

#### Issue 5: Visualization Not Showing

**Error:** Plots not displaying

**Solution:**
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode

# Your expliRL code here
explainer.visualize()
plt.show()
```

### Performance Optimization

#### 1. Faster SHAP Calculations
```python
# Use TreeExplainer for tree-based models (much faster)
explainer = SHAPExplainer(model, explainer_type='tree')

# Limit data samples for KernelExplainer
sample_data = shap.sample(X_train, 100)  # Use only 100 samples
explainer = SHAPExplainer(model, data=sample_data, explainer_type='kernel')
```

#### 2. Optimize RL Training
```python
# Reduce episodes for quick results
explanation = rl_explainer.explain(instance, num_episodes=50)

# Use saved agent for multiple explanations
rl_explainer.agent.save('trained_agent.pth')
rl_explainer.agent.load('trained_agent.pth')
```

#### 3. Batch Processing
```python
# Process multiple instances efficiently
from joblib import Parallel, delayed

def explain_instance(explainer, instance):
    return explainer.explain(instance)

# Parallel processing
explanations = Parallel(n_jobs=4)(
    delayed(explain_instance)(explainer, inst) 
    for inst in instances
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/expliRL.git
cd expliRL

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black expliRL/
flake8 expliRL/
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_explainers.py

# With coverage
pytest --cov=expliRL tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- SHAP library by Scott Lundberg
- LIME library by Marco Ribeiro
- OpenAI Gym for RL environments
- PyTorch team for deep learning framework

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/expliRL/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/expliRL/discussions)
- **Email:** support@explirl.ai

## 🚦 Project Status

- ✅ SHAP Integration
- ✅ LIME Integration
- ✅ Traditional Counterfactuals
- ✅ RL-based Counterfactuals
- ✅ CLI Tool
- ✅ REST API
- 🚧 GUI Dashboard (Coming Soon)
- 🚧 More RL Algorithms (PPO, A2C)
- 🚧 Image/Text Support

---

**Made with ❤️ by the expliRL Team**