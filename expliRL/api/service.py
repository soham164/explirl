from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from typing import List, Dict, Optional
import uvicorn

app = FastAPI(title="expliRL API", version="0.1.0")

# Global variables to store models and explainers
models = {}
explainers = {}

class ExplainRequest(BaseModel):
    model_id: str
    instance: List[float]
    method: str = "shap"
    params: Optional[Dict] = None

class ExplainResponse(BaseModel):
    explanation: Dict
    metrics: Optional[Dict] = None

@app.get("/")
async def root():
    return {"message": "expliRL API - Unified XAI Framework"}

@app.post("/load_model")
async def load_model(model_id: str, model_path: str):
    """Load a model from file"""
    try:
        with open(model_path, 'rb') as f:
            models[model_id] = pickle.load(f)
        return {"status": "success", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """Generate explanation for an instance"""
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[request.model_id]
    instance = np.array(request.instance)
    
    try:
        if request.method == "shap":
            from expliRL import SHAPExplainer
            explainer = SHAPExplainer(model)
            explanation = explainer.explain(instance)
        elif request.method == "lime":
            from expliRL import LIMEExplainer
            explainer = LIMEExplainer(model)
            explanation = explainer.explain(instance)
        elif request.method == "counterfactual":
            from expliRL import CounterfactualExplainer
            explainer = CounterfactualExplainer(model)
            explanation = explainer.explain(instance)
        elif request.method == "rl_counterfactual":
            from expliRL import RLCounterfactualExplainer
            explainer = RLCounterfactualExplainer(model)
            explanation = explainer.explain(instance)
        else:
            raise HTTPException(status_code=400, detail="Unknown method")
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in explanation.items():
            if isinstance(value, np.ndarray):
                explanation[key] = value.tolist()
        
        return ExplainResponse(explanation=explanation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def start_server(host="0.0.0.0", port=8000):
    """Start the API server"""
    uvicorn.run(app, host=host, port=port)
