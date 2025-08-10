# -*- coding: utf-8 -*-
"""
@file inference_kentucky.py
@brief Inference script for KentuckyDecisionTreeData dataset with FLOPS and CPU time profiling.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountMode, flop_count
from Models_node import *
from utils.datautils import *
from utils.train_utils import *
import sys
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_test_data(dataset_path, dataset_name):
    """
    Load and preprocess test data for inference.
    """
    dataset_folder = dataset_path + dataset_name + '/'
    Xtest = pd.read_csv(dataset_folder + dataset_name + '_TEST.tsv', header=None, sep='\t').values
    ytest = Xtest[:, 0]
    Xtest = Xtest[:, 1:]
    
    # Shuffle data
    Xtest, ytest = Shuffle(Xtest, ytest)
    
    return Xtest, ytest

def preprocess_data_for_inference(Xtest_raw):
    """
    Preprocess test data using the same pipeline as training.
    """
    N, T = calculate_dataset_metrics(Xtest_raw)
    
    # Apply multi-view processing
    # For inference, we use the same processing pipeline but only on test data
    Xtest_raw_processed, Xtest_fft, Xtest_derv = Splitview(Xtest_raw, T)
    
    intvlen, nintv = Get_intinfo(T)
    
    # Extract interval features (this function expects train/val/test but we only have test)
    # We'll create dummy train/val data for the function call
    dummy_train = np.zeros_like(Xtest_raw_processed[:1])  # Single dummy sample
    dummy_val = np.zeros_like(Xtest_raw_processed[:1])    # Single dummy sample
    
    dummy_train_fft = np.zeros_like(Xtest_fft[:1])
    dummy_val_fft = np.zeros_like(Xtest_fft[:1])
    
    dummy_train_derv = np.zeros_like(Xtest_derv[:1])
    dummy_val_derv = np.zeros_like(Xtest_derv[:1])
    
    # Extract features
    _, _, _, _, _, _, Xtest_raw_processed, Xtest_fft, Xtest_derv = Extract_intfea(
        dummy_train, dummy_train_fft, dummy_train_derv, 
        dummy_val, dummy_val_fft, dummy_val_derv, 
        Xtest_raw_processed, Xtest_fft, Xtest_derv, 
        nintv, intvlen
    )
    
    # Concatenate views
    Xtest = np.concatenate((Xtest_raw_processed, Xtest_fft, Xtest_derv), 1)
    
    # Note: We can't apply standardization without training data statistics
    # In a real scenario, we would save the scaler during training and apply it here
    print("Warning: Standardization skipped - would need training data statistics in practice")
    
    return Xtest, T

def inference_single_sample(model_nodes, x_sample, T):
    """
    Perform inference on a single sample through the tree structure.
    """
    with torch.no_grad():
        x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        x1 = x_tensor[:, :T]
        x2 = x_tensor[:, T:2*T]
        x3 = x_tensor[:, 2*T:]
        
        # Start from root node (node 0)
        current_node = 0
        prediction = 0
        
        # Traverse the tree
        while current_node in model_nodes and hasattr(model_nodes[current_node], 'bestmodel'):
            node = model_nodes[current_node]
            
            # Get prediction from current node
            with record_function("model_forward"):
                output = node.bestmodel(x1, x2, x3)
                pred = torch.round(output).item()
            
            if pred == 1:  # True case
                if hasattr(node, 'leftchd'):
                    current_node = node.leftchd
                else:
                    prediction = getattr(node, 'bstmdlclass', 0)
                    break
            else:  # False case
                if hasattr(node, 'rightchd'):
                    current_node = node.rightchd
                else:
                    # Find the most common class in false cases
                    prediction = getattr(node, 'bstmdlclass', 0)
                    break
        
        return prediction

def count_flops_single_sample(model_nodes, x_sample, T):
    """
    Count FLOPS for inference on a single sample.
    """
    x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
    x1 = x_tensor[:, :T]
    x2 = x_tensor[:, T:2*T]
    x3 = x_tensor[:, 2*T:]
    
    total_flops = 0
    current_node = 0
    
    # Count FLOPS for each node in the inference path
    while current_node in model_nodes and hasattr(model_nodes[current_node], 'bestmodel'):
        node = model_nodes[current_node]
        model = node.bestmodel
        
        # Count FLOPS for this model
        try:
            flops = flop_count(model, (x1, x2, x3), supported_ops=None)[0]
            total_flops += flops
        except Exception as e:
            print(f"FLOP counting failed for node {current_node}: {e}")
            # Estimate FLOPS manually if automatic counting fails
            total_flops += estimate_flops_manually(model, T)
        
        # Determine next node (simplified - assumes we follow the path)
        output = model(x1, x2, x3)
        pred = torch.round(output).item()
        
        if pred == 1 and hasattr(node, 'leftchd'):
            current_node = node.leftchd
        elif pred == 0 and hasattr(node, 'rightchd'):
            current_node = node.rightchd
        else:
            break
    
    return total_flops

def estimate_flops_manually(model, T):
    """
    Manually estimate FLOPS for a model when automatic counting fails.
    """
    # This is a rough estimation based on typical neural network operations
    # For the TL_NN models, estimate based on parameter operations
    total_params = sum(p.numel() for p in model.parameters())
    # Rough estimate: each parameter involves a multiply-add operation
    estimated_flops = total_params * 2  # multiply + add
    return estimated_flops

def profile_cpu_time(model_nodes, x_sample, T, num_runs=100):
    """
    Profile CPU time for inference on a single sample.
    """
    x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
    
    # Warm up
    for _ in range(10):
        _ = inference_single_sample(model_nodes, x_sample, T)
    
    # Profile with torch.profiler
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("inference_full"):
            for _ in range(num_runs):
                _ = inference_single_sample(model_nodes, x_sample, T)
    
    # Also measure simple time
    start_time = time.time()
    for _ in range(num_runs):
        _ = inference_single_sample(model_nodes, x_sample, T)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    return prof, avg_time

def main():
    """
    Main inference function.
    """
    # Dataset configuration
    dataset_name = "KentuckyDecisionTreeData"
    dataset_path = "../UCRArchive_2018/"
    model_path = f"../Tree_Models/{dataset_name}_learned_tree.pkl"
    
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    try:
        with open(model_path, "rb") as model_file:
            model_nodes = pickle.load(model_file)
        print("Model loaded successfully!")
        print(f"Number of nodes in tree: {len(model_nodes)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    print(f"Loading test data for {dataset_name}...")
    try:
        Xtest_raw, ytest = load_test_data(dataset_path, dataset_name)
        print(f"Test data shape: {Xtest_raw.shape}")
        print(f"Number of test samples: {len(ytest)}")
        print(f"Number of classes: {len(np.unique(ytest))}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Preprocess test data
    print("Preprocessing test data...")
    try:
        Xtest, T = preprocess_data_for_inference(Xtest_raw)
        print(f"Preprocessed test data shape: {Xtest.shape}")
        print(f"Time series length T: {T}")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return
    
    # Select a few samples for profiling
    num_samples_to_profile = min(10, len(Xtest))
    sample_indices = np.random.choice(len(Xtest), num_samples_to_profile, replace=False)
    
    print(f"\nProfiling {num_samples_to_profile} samples...")
    
    total_flops = 0
    total_time = 0
    successful_profiles = 0
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"\nProfiling sample {i+1}/{num_samples_to_profile} (index {sample_idx})")
        
        x_sample = Xtest[sample_idx]
        true_label = ytest[sample_idx]
        
        try:
            # Perform inference
            prediction = inference_single_sample(model_nodes, x_sample, T)
            print(f"True label: {true_label}, Predicted: {prediction}")
            
            # Count FLOPS
            flops = count_flops_single_sample(model_nodes, x_sample, T)
            print(f"FLOPS for this sample: {flops:,}")
            
            # Profile CPU time
            prof, avg_time = profile_cpu_time(model_nodes, x_sample, T, num_runs=50)
            print(f"Average CPU time per inference: {avg_time*1000:.4f} ms")
            
            total_flops += flops
            total_time += avg_time
            successful_profiles += 1
            
            # Print profiler summary for the first sample
            if i == 0:
                print("\nDetailed profiler output for first sample:")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            
        except Exception as e:
            print(f"Error profiling sample {sample_idx}: {e}")
            continue
    
    if successful_profiles > 0:
        avg_flops = total_flops / successful_profiles
        avg_time_ms = (total_time / successful_profiles) * 1000
        
        print(f"\n{'='*60}")
        print(f"PROFILING SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Successfully profiled samples: {successful_profiles}")
        print(f"Average FLOPS per sample: {avg_flops:,.0f}")
        print(f"Average CPU time per sample: {avg_time_ms:.4f} ms")
        print(f"FLOPS per millisecond: {avg_flops/avg_time_ms:,.0f}")
        print(f"{'='*60}")
        
        # Run full evaluation on all test data
        print(f"\nRunning full evaluation on all {len(Xtest)} test samples...")
        correct_predictions = 0
        start_time = time.time()
        
        for i in range(len(Xtest)):
            try:
                prediction = inference_single_sample(model_nodes, Xtest[i], T)
                if prediction == ytest[i]:
                    correct_predictions += 1
            except Exception as e:
                print(f"Error in inference for sample {i}: {e}")
        
        end_time = time.time()
        
        accuracy = correct_predictions / len(Xtest)
        total_inference_time = end_time - start_time
        avg_inference_time = total_inference_time / len(Xtest)
        
        print(f"\nFULL EVALUATION RESULTS:")
        print(f"Test Accuracy: {accuracy:.4f} ({correct_predictions}/{len(Xtest)})")
        print(f"Total inference time: {total_inference_time:.2f} seconds")
        print(f"Average time per sample: {avg_inference_time*1000:.4f} ms")
        print(f"Throughput: {len(Xtest)/total_inference_time:.2f} samples/second")
    else:
        print("No samples were successfully profiled!")

if __name__ == "__main__":
    main()
