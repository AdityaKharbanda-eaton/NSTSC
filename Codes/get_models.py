import pickle
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_model():
    # Get dataset name from command line or use default
    Dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Coffee"
    print(f"Using dataset: {Dataset_name}")
    
    # Load the learned tree
    file_path = os.path.join("../Tree_Models", f"{Dataset_name}_learned_tree.pkl")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    with open(file_path, "rb") as f:
        Tree = pickle.load(f)
    
    # Dictionary to store parameters for each node
    node_parameters = {}
    
    # Process each node
    for node_idx, node in Tree.items():
        print(f"\n=== Node {node_idx} ===")
        
        # Print requested attributes if they exist
        for attr in ['bestmodel', 'predcls', 'bstmdlclass', 'prntnb', 'childtype']:
            if hasattr(node, attr):
                print(f"{attr}: {getattr(node, attr)}")
        
        # If node has a model, extract its parameters
        if hasattr(node, 'bestmodel'):
            node_parameters[node_idx] = extract_parameters(node.bestmodel, node_idx)
    
    # Create combined plot for all nodes with models
    if node_parameters:
        create_combined_plot(node_parameters, Dataset_name)

def extract_parameters(model, node_idx):
    """Extract parameter values from a model."""
    print(f"\nExtracting parameters for Node {node_idx}:")
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Dictionary to store parameter values
    parameter_dict = {}
    
    # Extract values for each parameter
    for name, param in state_dict.items():
        # Skip parameters ending with _2
        if name.endswith('_2'):
            continue
            
        # Convert parameter to numpy array
        param_data = param.cpu().detach().numpy()
        
        # Store parameter values
        parameter_dict[name] = param_data
        
        # Print min and max values
        print(f"{name}: Min = {np.min(param_data):.6f}, Max = {np.max(param_data):.6f}")
    
    return parameter_dict

def create_combined_plot(node_parameters, dataset_name):
    """Create a combined plot of t, b, and A parameters for all nodes."""
    nodes = sorted(node_parameters.keys())
    
    # Create a big figure
    fig = plt.figure(figsize=(15, 5 * len(nodes)))
    
    # Plot parameters for each node
    for i, node_idx in enumerate(nodes):
        params = node_parameters[node_idx]
        
        # Group parameters by type
        t_params = {k: v for k, v in params.items() if k.startswith('t') and not k.endswith('_2')}
        b_params = {k: v for k, v in params.items() if k.startswith('b') and not k.endswith('_2')}
        a_params = {k: v for k, v in params.items() if k.startswith('A') and not k.endswith('_2')}
        
        # Plot t parameters
        ax1 = fig.add_subplot(len(nodes), 3, 3*i + 1)
        plot_parameter_group(ax1, t_params, f'Node {node_idx} - t Parameters')
        
        # Plot b parameters
        ax2 = fig.add_subplot(len(nodes), 3, 3*i + 2)
        plot_parameter_group(ax2, b_params, f'Node {node_idx} - b Parameters')
        
        # Plot A parameters
        ax3 = fig.add_subplot(len(nodes), 3, 3*i + 3)
        plot_parameter_group(ax3, a_params, f'Node {node_idx} - A Parameters')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_parameters.png")
    plt.close()
    print(f"Combined plot saved as {dataset_name}_parameters.png")

def plot_parameter_group(ax, params, title):
    """Plot a group of parameters on a given axis."""
    for name, values in params.items():
        # Handle different parameter shapes
        if len(values.shape) == 2:
            # For matrices, plot max value at each time step
            time_steps = range(1, values.shape[1] + 1)
            ax.plot(time_steps, np.max(values, axis=0), label=name)
        elif len(values.shape) == 1:
            # For vectors, plot the values directly
            time_steps = range(1, len(values) + 1)
            ax.plot(time_steps, values, label=name)
        else:
            # For scalars, plot a horizontal line
            ax.axhline(y=float(values), label=f"{name}={float(values):.4f}", linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add legend if there are parameters to show
    if params:
        ax.legend()

if __name__ == "__main__":
    load_and_analyze_model()