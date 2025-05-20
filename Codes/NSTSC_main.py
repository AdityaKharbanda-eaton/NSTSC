# -*- coding: utf-8 -*-
"""
@file NSTSC_main.py
@brief Main script to train and evaluate the NSTSC model on a dataset.
"""

from Models_node import *
from utils.datautils import *
from utils.train_utils import *
import pickle


def main():
    """
    @brief Main function to train and evaluate the NSTSC model.
    """
    Dataset_name = "BeetleFly"
    print('Start Training ---' + str(Dataset_name) + ' ---dataset\n')
    dataset_path_ = "../UCRArchive_2018/"
    normalize_dataset = True
    Max_epoch = 100
    # model training
    # Xtrain_raw, ytrain_raw, Xval_raw, yval_raw, Xtest_raw, ytest_raw = Readdataset(dataset_path_, Dataset_name)
    # Xtrain, Xval, Xtest = Multi_view(Xtrain_raw, Xval_raw, Xtest_raw)

    Xtrain, ytrain, Xval, yval, Xtest, ytest = Readdataset(dataset_path_, Dataset_name)

    N, T = calculate_dataset_metrics(Xtrain)
    # Tree = Train_model(Xtrain, Xval, ytrain_raw, yval_raw, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    Tree = Train_model(Xtrain, Xval, ytrain, yval, epochs=Max_epoch, normalize_timeseries=normalize_dataset)
    with open(f"{Dataset_name}_model.pkl", "wb") as model_file:
        pickle.dump(Tree, model_file)
    # model testing
    # testaccu = Evaluate_model(Tree, Xtest, ytest_raw)
    testaccu = Evaluate_model(Tree, Xtest, ytest)
    print("Test accuracy for dataset {} is --- {}".format(Dataset_name, testaccu))


if __name__ == "__main__":
    main()

