# Contributing to expliRL 🤝

Thank you for your interest in contributing to expliRL! We're excited to have you join our community of developers working to make explainable AI more accessible and powerful.

## 🌟 Ways to Contribute

- **Bug Reports** - Help us identify and fix issues
- **Feature Requests** - Suggest new explainability methods or improvements
- **Code Contributions** - Implement new features, fix bugs, or improve performance
- **Documentation** - Improve guides, examples, and API documentation
- **Testing** - Add test cases and improve test coverage
- **Examples** - Create demos and tutorials for different use cases

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/expliRL.git
cd expliRL

# Add upstream remote
git remote add upstream https://github.com/originalowner/expliRL.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black expliRL/ tests/ examples/

# Check style with flake8
flake8 expliRL/ tests/ examples/

# Sort imports with isort
isort expliRL/ tests/ examples/
```

### Code Quality Standards

- **Type Hints**: Use type hints for all public functions
- **Docstrings**: Follow Google-style docstrings
- **Error Handling**: Include proper exception handling
- **Testing**: Write tests for new features (aim for >80% coverage)

Example function with proper style:

```python
from typing import Optional, Dict, Any
import numpy as np

def explain_instance(
    self, 
    instance: np.ndarray, 
    desired_class: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Generate explanation for a single instance.
    
    Args:
        instance: Input instance to explain (shape: [n_features])
        desired_class: Target class for counterfactual (optional)
        **kwargs: Additional parameters for explanation method
        
    Returns:
        Dictionary containing explanation results with keys:
        - 'prediction': Original model prediction
        - 'explanation': Method-specific explanation data
        - 'metadata': Additional information about the explanation
        
    Raises:
        ValueError: If instance has wrong shape or invalid parameters
        RuntimeError: If explanation generation fails
    """
    if instance.ndim != 1:
        raise ValueError(f"Instance must be 1D array, got shape {instance.shape}")
    
    try:
        # Implementation here
        return {
            'prediction': prediction,
            'explanation': explanation_data,
            'metadata': {'method': 'example', 'timestamp': time.time()}
        }
    except Exception as e:
        raise RuntimeError(f"Explanation failed: {str(e)}") from e
```

### Testing Guidelines

We use pytest for testing. Tests should be:

- **Fast** - Unit tests should run quickly
- **Isolated** - Tests shouldn't depend on each other
- **Deterministic** - Use fixed random seeds when needed
- **Comprehensive** - Cover edge cases and error conditions

Example test structure:

```python
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from expliRL import SHAPExplainer

class TestSHAPExplainer:
    """Test suite for SHAP explainer functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model for testing."""
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model
    
    def test_explainer_initialization(self, trained_model):
        """Test that explainer initializes correctly."""
        explainer = SHAPExplainer(trained_model)
        assert explainer.model is trained_model
        assert explainer.explainer_type == 'auto'
    
    def test_explain_single_instance(self, trained_model, sample_data):
        """Test explanation generation for single instance."""
        X, _ = sample_data
        explainer = SHAPExplainer(trained_model)
        explainer.fit(X)
        
        explanation = explainer.explain(X[0])
        
        assert 'shap_values' in explanation
        assert len(explanation['shap_values']) == X.shape[1]
        assert isinstance(explanation['shap_values'], np.ndarray)
    
    def test_invalid_instance_shape(self, trained_model, sample_data):
        """Test error handling for invalid input shapes."""
        X, _ = sample_data
        explainer = SHAPExplainer(trained_model)
        explainer.fit(X)
        
        with pytest.raises(ValueError, match="Instance must be 1D"):
            explainer.explain(X[:2])  # 2D input should fail
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_explainers.py

# Run with coverage report
pytest --cov=expliRL --cov-report=html tests/

# Run tests matching pattern
pytest -k "test_shap" tests/
```

## 🐛 Bug Reports

When reporting bugs, please include:

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load model with '...'
2. Call explainer.explain() with '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened, including full error traceback.

**Environment**
- OS: [e.g. Windows 10, macOS 12.1, Ubuntu 20.04]
- Python version: [e.g. 3.8.5]
- expliRL version: [e.g. 0.1.0]
- Dependencies: [output of `pip freeze`]

**Additional Context**
- Sample data/model if possible
- Screenshots if applicable
- Any workarounds you've found
```

## ✨ Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this feature needed?
3. **Propose an API** - how should it work?
4. **Consider alternatives** - are there other ways to achieve this?

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Describe the problem this feature would solve.

**Proposed API**
```python
# Example of how the feature would be used
explainer = NewExplainer(model)
result = explainer.new_method(data, param1=value1)
```

**Alternatives Considered**
Other approaches you've considered and why this is better.

**Implementation Notes**
Any thoughts on how this could be implemented.
```

## 🔧 Code Contributions

### Pull Request Process

1. **Create an issue first** for significant changes
2. **Keep PRs focused** - one feature/fix per PR
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Follow the code style** guidelines

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines (black, flake8, isort)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] PR description clearly explains the changes

### PR Template

```markdown
**Description**
Brief description of changes made.

**Type of Change**
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing performed

**Related Issues**
Fixes #(issue number)

**Screenshots** (if applicable)
Add screenshots to help explain your changes.
```

## 📚 Documentation Contributions

### Types of Documentation

- **API Documentation** - Docstrings in code
- **User Guides** - README.md and tutorials
- **Examples** - Code examples in `/examples`
- **Developer Docs** - This file and setup guides

### Documentation Style

- Use clear, concise language
- Include code examples
- Test all code snippets
- Use consistent formatting
- Add screenshots for UI features

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View in browser
open _build/html/index.html
```

## 🧪 Adding New Explainers

To add a new explanation method:

### 1. Create the Explainer Class

```python
# expliRL/explainers/your_explainer.py
from typing import Dict, Any, Optional
import numpy as np
from .base import BaseExplainer

class YourExplainer(BaseExplainer):
    """Your explanation method implementation.
    
    This explainer implements [method name] for generating
    explanations of machine learning model predictions.
    
    Args:
        model: Trained ML model with predict() method
        param1: Description of parameter
        param2: Description of parameter
    """
    
    def __init__(self, model, param1: float = 1.0, param2: str = 'default'):
        super().__init__(model)
        self.param1 = param1
        self.param2 = param2
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """Fit the explainer to training data.
        
        Args:
            X: Training features
            y: Training labels (optional)
            **kwargs: Additional parameters
        """
        # Implementation here
        self._fitted = True
    
    def explain(self, instance: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanation for instance.
        
        Args:
            instance: Instance to explain
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with explanation results
        """
        if not self._fitted:
            raise RuntimeError("Explainer must be fitted before explaining")
        
        # Implementation here
        return {
            'explanation_type': 'your_method',
            'instance': instance,
            'explanation_data': explanation_result
        }
    
    def visualize(self, explanation: Optional[Dict] = None, **kwargs) -> None:
        """Create visualization of explanation.
        
        Args:
            explanation: Explanation dict from explain()
            **kwargs: Visualization parameters
        """
        # Implementation here
        pass
```

### 2. Add Tests

```python
# tests/test_your_explainer.py
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from expliRL.explainers.your_explainer import YourExplainer

class TestYourExplainer:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        return X, y
    
    @pytest.fixture
    def model(self, sample_data):
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model
    
    def test_initialization(self, model):
        explainer = YourExplainer(model)
        assert explainer.model is model
    
    def test_fit_and_explain(self, model, sample_data):
        X, y = sample_data
        explainer = YourExplainer(model)
        explainer.fit(X, y)
        
        explanation = explainer.explain(X[0])
        assert 'explanation_type' in explanation
        assert explanation['explanation_type'] == 'your_method'
```

### 3. Update Imports

```python
# expliRL/__init__.py
from .explainers.your_explainer import YourExplainer

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer', 
    'CounterfactualExplainer',
    'RLCounterfactualExplainer',
    'YourExplainer'  # Add your explainer
]
```

### 4. Add Example

```python
# examples/your_explainer_demo.py
"""
Your Explainer Demo
==================
Demonstrates usage of YourExplainer
"""

from expliRL import YourExplainer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Create explainer
    explainer = YourExplainer(model)
    explainer.fit(X, y)
    
    # Generate explanation
    explanation = explainer.explain(X[0])
    print(f"Explanation: {explanation}")
    
    # Visualize
    explainer.visualize(explanation)

if __name__ == "__main__":
    main()
```

## 🎯 Specific Contribution Areas

### High Priority

- **Performance Optimization** - Speed up SHAP/LIME calculations
- **Memory Efficiency** - Reduce memory usage for large datasets
- **New RL Algorithms** - Implement PPO, A2C for counterfactuals
- **Visualization Improvements** - Interactive plots, better styling
- **Documentation** - More examples, tutorials, API docs

### Medium Priority

- **Image/Text Support** - Extend explainers to other data types
- **Model Support** - Add support for more ML frameworks
- **CLI Enhancements** - More commands, better UX
- **API Features** - Batch processing, async support
- **Testing** - Increase coverage, add integration tests

### Future Ideas

- **GUI Dashboard** - Web interface for explanations
- **Explanation Comparison** - Compare different methods
- **Automated Reporting** - Generate explanation reports
- **Cloud Integration** - Deploy explanations at scale
- **Fairness Metrics** - Integrate bias detection

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub Releases** - Release descriptions
- **Documentation** - Author credits

## 📞 Getting Help

- **GitHub Discussions** - Ask questions, share ideas
- **GitHub Issues** - Report bugs, request features
- **Code Review** - Get feedback on your contributions
- **Discord/Slack** - Real-time chat (if available)

## 📋 Contributor Agreement

By contributing to expliRL, you agree that:

- Your contributions will be licensed under the MIT License
- You have the right to submit the contributions
- You understand the project's goals and coding standards

## 🙏 Thank You

Every contribution, no matter how small, helps make expliRL better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, your efforts are appreciated!

---

**Happy Contributing! 🚀**

*Made with ❤️ by the expliRL community*