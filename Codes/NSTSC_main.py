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


def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    # Get Dataset_name from system arguments or use default value
    Dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Coffee"
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    Max_epoch = 10
    # model training
    Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)
    N, T = calculate_dataset_metrics(Xtrain)
    Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    # Save the learned tree to a file in the Tree_Models folder
    output_path = f"Tree_Models/{Dataset_name}_learned_tree.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(Tree, f)
    # model testing
    testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))


if __name__ == "__main__":
    main()

