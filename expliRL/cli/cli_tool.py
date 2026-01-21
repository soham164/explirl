import click
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from expliRL import (
    SHAPExplainer, 
    LIMEExplainer, 
    CounterfactualExplainer, 
    RLCounterfactualExplainer
)

@click.group()
def cli():
    """expliRL CLI - Generate XAI explanations from the command line"""
    pass

@cli.command()
@click.option('--model-path', '-m', required=True, help='Path to saved model')
@click.option('--data-path', '-d', required=True, help='Path to data file (CSV)')
@click.option('--instance-idx', '-i', default=0, help='Index of instance to explain')
@click.option('--method', '-t', default='shap', 
              type=click.Choice(['shap', 'lime', 'cf', 'rl-cf']))
@click.option('--output', '-o', help='Output file path')
def explain(model_path, data_path, instance_idx, method, output):
    """Generate explanation for a single instance"""
    
    # Load model
    click.echo(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    click.echo(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    X = data.drop(columns=['target'], errors='ignore').values
    
    # Get instance
    instance = X[instance_idx]
    
    # Generate explanation
    click.echo(f"Generating {method} explanation...")
    
    if method == 'shap':
        explainer = SHAPExplainer(model)
        explainer.fit(X)
        explanation = explainer.explain(instance)
        click.echo(f"SHAP values: {explanation['shap_values']}")
    
    elif method == 'lime':
        explainer = LIMEExplainer(model)
        explainer.fit(X)
        explanation = explainer.explain(instance)
        click.echo("LIME explanation:")
        for feat, weight in explanation['local_exp']:
            click.echo(f"  {feat}: {weight:.4f}")
    
    elif method == 'cf':
        explainer = CounterfactualExplainer(model)
        explainer.fit(X)
        explanation = explainer.explain(instance)
        click.echo(f"Original class: {explanation['original_class']}")
        click.echo(f"Counterfactual class: {explanation['counterfactual_class']}")
        click.echo(f"Number of changes: {explanation['num_changes']}")
    
    elif method == 'rl-cf':
        explainer = RLCounterfactualExplainer(model)
        explainer.fit(X)
        explanation = explainer.explain(instance, num_episodes=50)
        click.echo(f"Original class: {explanation['original_class']}")
        click.echo(f"Counterfactual class: {explanation['counterfactual_class']}")
        click.echo(f"Number of changes: {explanation['num_changes']}")
        if explanation['training_log']:
            last_log = explanation['training_log'][-1]
            click.echo(f"RL convergence: Episode {last_log['episode']}, "
                      f"Reward {last_log['reward']:.2f}")
    
    # Save output if requested
    if output:
        import json
        with open(output, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for key, value in explanation.items():
                if isinstance(value, np.ndarray):
                    explanation[key] = value.tolist()
            json.dump(explanation, f, indent=2)
        click.echo(f"Explanation saved to {output}")

@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, help='API server port')
def serve(host, port):
    """Start the API server"""
    click.echo(f"Starting expliRL API server on {host}:{port}...")
    from expliRL.api.service import start_server
    start_server(host, port)

def main():
    cli()

if __name__ == '__main__':
    main()
