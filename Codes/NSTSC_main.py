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
import time

def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    if len(sys.argv) < 3:
        print("Usage: python NSTSC_main.py <Dataset_name> <Max_epoch>")
        sys.exit(1)
    Dataset_name = sys.argv[1]
    Max_epoch = int(sys.argv[2])
    print("Start Training ---" + str(Dataset_name) + " ---dataset\n")
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    # model training
    # Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    # Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)
    start_time = time.time()
    Xtrain, ytrain, Xval, yval, Xtest, ytest = Readdataset(dataset_path_, Dataset_name)
    N, T = calculate_dataset_metrics(Xtrain)
    end_time = time.time()
    print("Preprocessing time: {:.2f} seconds".format(end_time - start_time))
    # Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    training_start_time = time.time()
    Tree = Train_model(Xtrain, Xval, ytrain, yval, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    training_end_time = time.time()
    print("Training time: {:.2f} seconds".format(training_end_time - training_start_time))
    with open(f"../Tree_Models/{Dataset_name}_model.pkl", "wb") as model_file:
        pickle.dump(Tree, model_file)
        print(f"Model saved to ../Tree_Models/{Dataset_name}_model.pkl")
    # model testing
    # testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    testing_start_time = time.time()
    testaccu = Evaluate_model(Tree, Xtest, ytest)
    testing_end_time = time.time()
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))
    print("Testing time: {:.2f} seconds".format(testing_end_time - testing_start_time))


if __name__ == "__main__":
    main()

