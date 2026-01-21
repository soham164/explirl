# expliRL/__init__.py

# Version info
__version__ = "0.1.0"

# Import main classes
from expliRL.core.shap_explainer import SHAPExplainer
from expliRL.core.lime_explainer import LIMEExplainer
from expliRL.core.cf_explainer import CounterfactualExplainer
from expliRL.core.rl_cf_explainer import RLCounterfactualExplainer

# Import image explainers
from expliRL.core.gradcam_explainer import GradCAMExplainer
from expliRL.core.lime_image_explainer import LIMEImageExplainer

# Define what's available when someone does "from expliRL import *"
__all__ = [
    "SHAPExplainer", 
    "LIMEExplainer", 
    "CounterfactualExplainer", 
    "RLCounterfactualExplainer",
    "GradCAMExplainer",
    "LIMEImageExplainer",
    "__version__"
]