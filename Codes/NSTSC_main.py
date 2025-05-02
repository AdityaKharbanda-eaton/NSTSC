# -*- coding: utf-8 -*-
"""
@file NSTSC_main.py
@brief Main script to train and evaluate the NSTSC model on a dataset.
"""

from Models_node import *
from utils.datautils import *
from utils.train_utils import *
import pickle
import sys
import os
import time
def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    # Get Dataset_name from system arguments or use default value
    Dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Coffee"
    Max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"Using dataset: {Dataset_name}")
    print(f"Max epoch: {Max_epoch}")
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    #Max_epoch = 10
    # model training
    preprocess_time_start = time.time()
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)
    N, T = calculate_dataset_metrics(Xtrain)
    preprocess_time_end = time.time()
    print(f"Preprocessing time: {preprocess_time_end - preprocess_time_start:.2f} seconds")
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    train_time_start = time.time()
    Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    train_time_end = time.time()
    print(f"Training time: {train_time_end - train_time_start:.2f} seconds")
    # Save the learned tree to a file in the Tree_Models folder
    path = f"../Tree_Models"
    output_path = os.path.join(path, f"{Dataset_name}_learned_tree.pkl")
    if not os.path.exists(path):
        os.makedirs(path)
    with open(output_path, "wb") as f:
        pickle.dump(Tree, f)
    # model testing
    evaluate_time_start = time.time()
    testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    evaluate_time_end = time.time()
    print(f"Evaluation time: {evaluate_time_end - evaluate_time_start:.2f} seconds")
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))


if __name__ == "__main__":
    main()

