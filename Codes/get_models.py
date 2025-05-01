import pickle
import os
import sys
def load_model():
    #Default value for Dataset_name
    Dataset_name = "Coffee"
    if len(sys.argv) > 1:
        Dataset_name = sys.argv[1]
    else:
        print("No Dataset_name provided. Using default value: 'Coffee'")

    # Load the learned tree
    file_path = os.path.join("../Tree_Models", f"{Dataset_name}_learned_tree.pkl")
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    print(f"Looking for file: {file_path}")
    with open(file_path, "rb") as f:
        Tree = pickle.load(f)

    # Retrieve the bestmodel from each node (if available)
    best_models = {}
    for node_idx, node in Tree.items():
        if hasattr(node, 'bestmodel'):
            best_models[node_idx] = node.bestmodel
            print(f"Node {node_idx} best model: {node.bestmodel}")
        else:
            print(f"Node {node_idx} does not have a bestmodel attribute.")
        if hasattr(node, 'predcls'):
            print(f"Node {node_idx} predcls: {node.predcls}")
        else:
            print(f"Node {node_idx} does not have a predcls attribute")
        if hasattr(node, 'bstmdlclass'):
            print(f"Node {node_idx} bstmdlclass: {node.bstmdlclass}")
        else:
            print(f"Node {node_idx} does not have a bstmdlclass attribute")
        if hasattr(node, 'prntnb'):
            best_models[node_idx] = node.prntnb
            print(f"Node {node_idx} parent node: {node.prntnb}")
        else:
            print(f"Node {node_idx} does not have a prntnb attribute.")

if __name__ == "__main__":
    load_model()