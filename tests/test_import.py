"""Test basic imports and version info for PyPI validation"""

def test_import_main_package():
    """Test that main package imports correctly"""
    import expliRL
    assert hasattr(expliRL, '__version__')
    assert expliRL.__version__ == "0.1.0"

def test_import_all_explainers():
    """Test that all explainers can be imported"""
    from expliRL import (
        SHAPExplainer,
        LIMEExplainer, 
        CounterfactualExplainer,
        RLCounterfactualExplainer
    )
    
    # Check classes exist
    assert SHAPExplainer is not None
    assert LIMEExplainer is not None
    assert CounterfactualExplainer is not None
    assert RLCounterfactualExplainer is not None

def test_version_consistency():
    """Test version consistency across files"""
    import expliRL
    
    # Read version from setup.py
    with open('setup.py', 'r') as f:
        setup_content = f.read()
        assert '0.1.0' in setup_content
    
    # Check __init__.py version
    assert expliRL.__version__ == "0.1.0"