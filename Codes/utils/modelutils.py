import torch
from Models_node_copy import TL_NN1, TL_NN2, TL_NN3, TL_NN4
import os
import pickle
import numpy as np
import torch.nn.functional as F


def get_node_properties(model, node_index):
    """
    Get properties of a specific node in the model.
    
    :param model: The model containing nodes.
    :param node_index: Index of the node to inspect.
    :return: Nothing, just print information.
    """
    print(f"Node {node_index} Properties:")
    node = model[node_index]
    for attr in dir(node):
        if not attr.startswith('__') and not callable(getattr(node, attr)):
            try:
                value = getattr(node, attr)
                print(f"{attr}: {value}")
            except:
                print(f"{attr}: [Unable to display]")

    print("\nChecking stoptrain flag:")
    print(f"Node {node_index} stoptrain: {getattr(node, 'stoptrain', 'Not set')}")

def print_tree_structure(tree):
    """
    Print the structure of the NSTSC tree and details of each node.
    
    Parameters:
    -----------
    tree : dict
        Dictionary representing the NSTSC tree model
    """
    
    
    def get_node_type(model):
        """Identify the type of neural network model"""
        if isinstance(model, TL_NN1):
            return "TL_NN1 (Conjunction/AND)"
        elif isinstance(model, TL_NN2):
            return "TL_NN2 (Disjunction/OR)"
        elif isinstance(model, TL_NN3):
            return "TL_NN3 (Always/Globally)"
        elif isinstance(model, TL_NN4):
            return "TL_NN4 (Eventually/Finally)"
        else:
            return "Unknown Model"
    
    def print_node_info(node_id, node, prefix="", is_last=True):
        # Print node information
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}Node {node_id}")
        
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # Print predicted class if available
        if hasattr(node, 'predcls'):
            print(f"{next_prefix}Predicted Class: {node.predcls}")
        
        # Print best model class if available
        if hasattr(node, 'bstmdlclass'):
            print(f"{next_prefix}Best Model Class: {node.bstmdlclass}")
            
        # Print best model info if available
        if hasattr(node, 'bestmodel'):
            model_type = get_node_type(node.bestmodel)
            print(f"{next_prefix}Best Model: {model_type}")
            
        # Print Gini index if available
        if hasattr(node, 'ginis'):
            print(f"{next_prefix}Gini Index: {node.ginis:.6f}")
            
        # Print children recursively
        has_left = hasattr(node, 'leftchd')
        has_right = hasattr(node, 'rightchd')
        
        if has_left:
            print_node_info(node.leftchd, tree[node.leftchd], next_prefix, not has_right)
        
        if has_right:
            print_node_info(node.rightchd, tree[node.rightchd], next_prefix, True)
    
    # Start printing from root (node 0)
    print("NSTSC Tree Structure:")
    print("====================\n")
    print_node_info(0, tree[0])
    
    # Print summary of the tree
    print("\nTree Summary:")
    print(f"Total Nodes: {len(tree)}")
    
    # Count leaf nodes
    leaf_nodes = sum(1 for node_id, node in tree.items() 
                   if not hasattr(node, 'leftchd') and not hasattr(node, 'rightchd'))
    print(f"Leaf Nodes: {leaf_nodes}")
    print(f"Internal Nodes: {len(tree) - leaf_nodes}")

def load_preprocessed_data(dataset_name):
    """
    @brief Loads preprocessed data for a given dataset.
    @param dataset_name The name identifier of the dataset to load.
    @return The data loaded from the preprocessed file if it exists; otherwise, None.
    @details This function constructs a relative file path to a preprocessed dataset stored in a pickle file.
    It checks whether the file exists. If the file is found, it opens the file in binary read mode,
    loads the data using the pickle module, prints a success message, and returns the data.
    If the file does not exist, it prints an error message and returns None.
    """
    
    file_path = os.path.join('../../Preprocessed_data', f'{dataset_name}_data.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from {file_path}")
        return data
    else:
        print(f"File {file_path} does not exist.")
        return None

def print_labels(y):
    """
    @brief Prints each row from the provided list or iterable along with its index.
    @param y An iterable (e.g., list, tuple) where each element represents a row to be printed.
    @details This function iterates over the elements of 'y', printing each one with its corresponding index. 
    It is useful for debugging or simply displaying the contents of a dataset. 
    @example:
        labels = ['label1', 'label2', 'label3']
        print_labels(labels)
    """

    for idx, row in enumerate(y):
        print(f"Row {idx}: {row}")

def print_tree_summary(tree):
    print(f"Total Nodes: {len(tree)}")
    
    leaf_nodes = []
    internal_nodes = []
    
    for node_id, node in tree.items():
        if hasattr(node, 'leftchd') or hasattr(node, 'rightchd'):
            internal_nodes.append(node_id)
        else:
            leaf_nodes.append(node_id)
    
    print(f"Internal Nodes: {len(internal_nodes)} {internal_nodes}")
    print(f"Leaf Nodes: {len(leaf_nodes)} {leaf_nodes}")
    
    # Print paths from root to leaf
    print("\nPaths from Root to Leaf Nodes:")
    
    def trace_path(node_id, path=None):
        if path is None:
            path = []
        
        path.append(node_id)
        node = tree[node_id]
        
        # If leaf node, return the path
        if not hasattr(node, 'leftchd') and not hasattr(node, 'rightchd'):
            return [path]
        
        # Otherwise, collect paths from children
        paths = []
        if hasattr(node, 'leftchd'):
            left_paths = trace_path(node.leftchd, path.copy())
            paths.extend(left_paths)
        
        if hasattr(node, 'rightchd'):
            right_paths = trace_path(node.rightchd, path.copy())
            paths.extend(right_paths)
        
        return paths
    
    paths = trace_path(0)
    for i, path in enumerate(paths):
        leaf_predcls = tree[path[-1]].predcls if hasattr(tree[path[-1]], 'predcls') else "N/A"
        print(f"Path {i+1} (Predicts Class {leaf_predcls}): {path}")

# Function to get the rule from a node's model
def extract_rule_from_model(node, feature_names=None):
    """
    Extract a readable rule from a node's best model.
    
    Args:
        node: The tree node containing a bestmodel
        feature_names: Optional list of feature names
    
    Returns:
        A string representing the rule
    """
    if not hasattr(node, 'bestmodel'):
        return "No model available"
    
    model = node.bestmodel
    
    # Get model type
    if isinstance(model, TL_NN1):
        operator = "AND"
    elif isinstance(model, TL_NN2):
        operator = "OR"
    elif isinstance(model, TL_NN3):
        operator = "Always"
    elif isinstance(model, TL_NN4):
        operator = "Eventually"
    else:
        operator = "Unknown"
    
    # Extract parameters (thresholds and weights)
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().numpy()
    
    # Generate rule based on model type
    rule = f"Model Type: {operator}\n"
    
    if hasattr(node, 'bstmdlclass'):
        rule += f"Separates class {node.bstmdlclass}\n"
    
    return rule

# Function to get the complete path rule for a leaf node
def get_path_rule(tree, leaf_id):
    """
    Get the complete rule path from root to a leaf node.
    
    Args:
        tree: The tree model
        leaf_id: The ID of the leaf node
    
    Returns:
        A list of (node_id, branch_type, rule) tuples representing the path
    """
    path = []
    current_id = leaf_id
    
    # Traverse up the tree until we reach the root
    while hasattr(tree[current_id], 'prntnb'):
        parent_id = tree[current_id].prntnb
        parent_node = tree[parent_id]
        
        # Determine if this node is a left (true) or right (false) child
        if hasattr(parent_node, 'leftchd') and parent_node.leftchd == current_id:
            branch_type = "true"
        else:
            branch_type = "false"
        
        # Get the rule for this node
        if hasattr(parent_node, 'bestmodel'):
            rule = extract_rule_from_model(parent_node)
        else:
            rule = "No model"
        
        path.append((parent_id, branch_type, rule))
        current_id = parent_id
    
    # Reverse the path so it goes from root to leaf
    return path[::-1]

# Get rules for all leaf nodes
def print_all_leaf_rules(tree):
    """
    Get rules for all leaf nodes in the tree.
    
    Args:
        tree: The tree model
    
    Returns:
        A dictionary mapping leaf node IDs to path rules
    """
    leaf_rules = {}
    
    for node_id, node in tree.items():
        if not hasattr(node, 'leftchd') and not hasattr(node, 'rightchd'):
            # This is a leaf node
            path_rule = get_path_rule(tree, node_id)
            leaf_rules[node_id] = {
                'predcls': getattr(node, 'predcls', None),
                'path_rule': path_rule
            }

    # Display rules for each leaf node
    for leaf_id, rule_info in leaf_rules.items():
        print(f"Leaf Node {leaf_id} (Predicts Class {rule_info['predcls']}):")
        print("-" * 50)
        
        for node_id, branch_type, rule in rule_info['path_rule']:
            print(f"Node {node_id} ({branch_type} branch):")
            print(rule)
            print()
        
        print("=" * 50)

# Extract key features from a model
def extract_key_features(model, view_names=["Original", "FFT", "Derivative"]):
    """
    Extract the most important features from a model based on weights
    
    Args:
        model: The neural network model
        view_names: Names of the three views
    
    Returns:
        Dictionary with key features for each view
    """
    key_features = {}
    
    # Extract parameters
    t1 = model.t1.detach().numpy()
    # t2 = model.t2.detach().numpy()
    # t3 = model.t3.detach().numpy()
    b1 = model.b1.detach().numpy()
    # b2 = model.b2.detach().numpy()
    # b3 = model.b3.detach().numpy()
    
    #calculate threshold values
    u1 = b1/t1
    # u2 = b2/t2
    # u3 = b3/t3

    # Apply softmax on each view using torch.nn.functional
    A1 = F.softmax(model.A1, dim=1).detach().numpy()
    # A2 = F.softmax(model.A2, dim=1).detach().numpy()
    # A3 = F.softmax(model.A3, dim=1).detach().numpy()

    # Final aggregation weights using softmax
    # A4 = F.softmax(model.A4, dim=1).detach().numpy()
    
    # Store parameters
    key_features = {
        view_names[0]: {"t": t1, "b": b1, "u": u1, "A": A1},
        # view_names[1]: {"t": t2, "b": b2, "u": u2, "A": A2},
        # view_names[2]: {"t": t3, "b": b3, "u": u3, "A": A3},
        # "agg": A4
    }
    
    # For each view, identify top K features by weight
    for view_name in view_names:
        weights = key_features[view_name]["A"]
        top_indices = np.argsort(weights.flatten())[-20:]  # top 10 features
        key_features[view_name]["top_indices"] = top_indices
    
    return key_features

