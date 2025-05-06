# -*- coding: utf-8 -*-
"""
@file datautils.py
@brief Data utility functions for NSTSC, including shuffling, loading, feature extraction, and preprocessing.
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


# Shuffle data
def Shuffle(X, y):
    # print("Running Shuffle")
    """
    @brief Shuffle the data and labels in unison.
    @param X: Input data array.
    @param y: Labels array.
    @return Tuple of shuffled (X, y).
    """
    nums = list(range(len(y)))
    random.shuffle(nums)
    random.shuffle(nums)
    random.shuffle(nums)
    X = X[nums,:]
    y = y[nums]
    return X, y


# Load Dataset given a dataset's path
def Readdataset(dataset_path_, Dataset_name, standalize=True, val=False, shuffle = True):
    # print("Running Readdataset")
    """
    @brief Load and preprocess dataset from the given path.
    @param dataset_path_: Path to the dataset directory.
    @param Dataset_name: Name of the dataset.
    @param standalize: Whether to standardize the data.
    @param val: Whether to split validation from test set.
    @return Xtrain, ytrain, Xval, yval, Xtest, ytest
    """
    
    Dataset_folder = dataset_path_ + Dataset_name + '/'
    Xtrain = pd.read_csv(Dataset_folder + Dataset_name + '_TRAIN.tsv', header=None, sep='\t').values
    Xtest = pd.read_csv(Dataset_folder + Dataset_name + '_TEST.tsv', header=None, sep='\t').values
    ytrain = Xtrain[:,0]
    ytest = Xtest[:,0]
    Xtrain = Xtrain[:,1:]
    Xtest = Xtest[:,1:]
    if shuffle:
        Xtrain, ytrain = Shuffle(Xtrain, ytrain)
        Xtest, ytest = Shuffle(Xtest, ytest)

    #fixing the data leakage issue, commented the earlier approach of standardizing data below
    Xtrain_fft = np.fft.fft(Xtrain)
    Xtrain_fft = np.abs(Xtrain_fft)
    Xtrain_dif = Xtrain[:,1:] - Xtrain[:,:-1]
    Xtrain_dif = np.concatenate((Xtrain_dif[:,0].reshape([-1,1]),Xtrain_dif),1)
    Xtrain = np.concatenate((Xtrain, Xtrain_fft, Xtrain_dif),1)
    Xtest_fft = np.fft.fft(Xtest)
    Xtest_fft = np.abs(Xtest_fft)
    Xtest_dif = Xtest[:,1:] - Xtest[:,:-1]
    Xtest_dif = np.concatenate((Xtest_dif[:,0].reshape([-1,1]),Xtest_dif),1)
    Xtest = np.concatenate((Xtest, Xtest_fft, Xtest_dif),1)
    
    ss = StandardScaler()
    if standalize:
        Xtrain = ss.fit_transform(Xtrain)
        Xtest = ss.transform(Xtest)


    Ntrain = Xtrain.shape[0]
    Xall, yall = np.concatenate((Xtrain, Xtest)), np.concatenate((ytrain, ytest))
        
    yset = np.array(list(set(yall))).astype(int)
    classnum = len(yset)    
    for ci in range(classnum):
        yall[yall == yset[ci]] = ci
    
    # ss = StandardScaler()
    # if standalize:
    #     Xall = ss.fit_transform(Xall)
        
    # Xall_fft = np.fft.fft(Xall)
    # Xall_fft = np.abs(Xall_fft)
    # Xall_dif = Xall[:,1:] - Xall[:,:-1]
    # Xall_dif = np.concatenate((Xall_dif[:,0].reshape([-1,1]),Xall_dif),1)
    # Xall = np.concatenate((Xall, Xall_fft, Xall_dif),1)
    # if standalize:
    #     Xall = ss.fit_transform(Xall)
    Xtrain, Xtest = Xall[:Ntrain,:], Xall[Ntrain:,:] 
    ytrain, ytest = yall[:Ntrain,], yall[Ntrain:,]
    
    if val:
        Ntest = Xtest.shape[0]
        Nval = int(Ntest * 0.5)
        Xval, yval = Xtest[:Nval, :], ytest[:Nval,]
        Xtest, ytest = Xtest[Nval:, :], ytest[Nval:,]
    else:
        Xval = Xtest - 0
        yval = ytest - 0
    
    return Xtrain, ytrain, Xval, yval, Xtest, ytest


# Dimension of data
def calculate_dataset_metrics(Xtrain):
    # print("Running calculate_dataset_metrics")
    """
    @brief Calculate the number of samples and time steps in the training data.
    @param Xtrain: Training data array.
    @return Tuple (N, T) where N is the number of samples and T is the number of time steps.
    """
    
    N, T = Xtrain.shape[0], int(Xtrain.shape[1]/3)
    
    return N, T


# Compute interval length
def Get_intinfo(T):
    # print("Running Get_intinfo")
    """
    @brief Compute the interval length and number of intervals for a given time series length.
    @param T: Number of time steps.
    @return Tuple (intvlen, nintv) where intvlen is interval length and nintv is number of intervals.
    """
    
    if T > 40:
        nintv = 20
        intvlen = int(T//nintv)
    else:
        intvlen = 5
        nintv = int(T//intvlen)
    
    return intvlen, nintv


# Multi-view representation
def Splitview(X, T):
    # print("Running Splitview")
    """
    @brief Split the input data into original, FFT, and special feature views.
    @param X: Input data array.
    @param T: Number of time steps per view.
    @return Tuple (Xori, Xfft, Xspe) of split data views.
    """
    
    Xori = X[:,:T]
    Xfft = X[:,T:2*T]
    Xspe = X[:,2*T:]
    
    return Xori, Xfft, Xspe


# Interval feature extraction
def Extract_intfea(
    Xtrain_raw, Xtrain_fft, Xtrain_derv,
    Xval_raw, Xval_fft, Xval_derv,
    Xtest_raw, Xtest_fft, Xtest_derv,
    nintv, intvlen):
    # print("Running Extract_intfea")
    """
    @brief Extract interval features for all data splits and views.
    @param Xtrain_raw: Raw training data.
    @param Xtrain_fft: FFT features for training data.
    @param Xtrain_derv: Derivative features for training data.
    @param Xval_raw: Raw validation data.
    @param Xval_fft: FFT features for validation data.
    @param Xval_derv: Derivative features for validation data.
    @param Xtest_raw: Raw test data.
    @param Xtest_fft: FFT features for test data.
    @param Xtest_derv: Derivative features for test data.
    @param nintv: Number of intervals.
    @param intvlen: Length of each interval.
    @return Tuple of all processed data splits and views with interval features.
    """
    
    Xtrain_raw = Addstatfea(Xtrain_raw, nintv, intvlen)
    Xtrain_fft = Addstatfea(Xtrain_fft, nintv, intvlen)
    Xtrain_derv = Addstatfea(Xtrain_derv, nintv, intvlen)
    Xval_raw = Addstatfea(Xval_raw, nintv, intvlen)
    Xval_fft = Addstatfea(Xval_fft, nintv, intvlen)
    Xval_derv = Addstatfea(Xval_derv, nintv, intvlen)
    Xtest_raw = Addstatfea(Xtest_raw, nintv, intvlen)
    Xtest_fft = Addstatfea(Xtest_fft, nintv, intvlen)
    Xtest_derv = Addstatfea(Xtest_derv, nintv, intvlen)
    
    return Xtrain_raw, Xtrain_fft, Xtrain_derv, Xval_raw, Xval_fft, \
        Xval_derv, Xtest_raw, Xtest_fft, Xtest_derv


# Add statistical features from interval data
def Addstatfea(X, n, t):
    # print("Running Addstatfea")
    """
    @brief Add statistical features (mean, std, min, max, median, IQR, slope) from interval data.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with additional statistical features.
    """
    X = Addmean(X, n, t)
    X = Addstd(X, n, t)
    X = Addmin(X, n, t)
    X = Addmax(X, n, t)
    X = Addmedian(X, n, t)
    X = AddIQR(X, n, t)
    X = Addslope(X, n, t)
    return X


# Mean feature
def Addmean(X, n, t):
    # print("Running Addmean")
    """
    @brief Add mean feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with mean features appended.
    """
    T = X.shape[1]
    Xmean = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmean[:,i] = np.mean(X[:,i*t:(i+1)*t],1)
       else:
           Xmean[:,i] = np.mean(X[:,i*t:T],1)
    X = np.concatenate((X, Xmean),1)
    return X


# Std feature
def Addstd(X, n, t):
    # print("Running Addstd")
    """
    @brief Add standard deviation feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with std features appended.
    """
    T = X.shape[1]
    Xstd = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xstd[:,i] = np.std(X[:,i*t:(i+1)*t],1)
       else:
           Xstd[:,i] = np.std(X[:,i*t:T],1)
    X = np.concatenate((X, Xstd),1)
    return X


# Min feature
def Addmin(X, n, t):
    # print("Running Addmin")
    """
    @brief Add minimum value feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with min features appended.
    """
    T = X.shape[1]
    Xmin = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmin[:,i] = np.min(X[:,i*t:(i+1)*t],1)
       else:
           Xmin[:,i] = np.min(X[:,i*t:T],1)
    X = np.concatenate((X, Xmin),1)
    return X


# Max feature
def Addmax(X, n, t):
    # print("Running Addmax")
    """
    @brief Add maximum value feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with max features appended.
    """
    T = X.shape[1]
    Xmax = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmax[:,i] = np.max(X[:,i*t:(i+1)*t],1)
       else:
           Xmax[:,i] = np.max(X[:,i*t:T],1)
    X = np.concatenate((X, Xmax),1)
    return X


# Median feature
def Addmedian(X, n, t):
    # print("Running Addmedian")
    """
    @brief Add median value feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with median features appended.
    """
    T = X.shape[1]
    Xmedian = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           Xmedian[:,i] = np.median(X[:,i*t:(i+1)*t],1)
       else:
           Xmedian[:,i] = np.median(X[:,i*t:T],1)
    X = np.concatenate((X, Xmedian),1)
    return X


# IQR feature
def AddIQR(X, n, t):
    # print("Running AddIQR")
    """
    @brief Add interquartile range (IQR) feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with IQR features appended.
    """
    T = X.shape[1]
    XIQR = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           xtem = np.percentile(X[:,i*t:(i+1)*t],[25,75], 1).T
           XIQR[:,i] = xtem[:,1] - xtem[:,0]        
       else:
           xtem = np.percentile(X[:,i*t:T],[25,75], 1).T
           XIQR[:,i] = xtem[:,1] - xtem[:,0]    
    X = np.concatenate((X, XIQR),1)
    return X


# Slope feature
def Addslope(X, n, t):
    # print("Running Addslope")
    """
    @brief Add slope (trend) feature for each interval.
    @param X: Input data array.
    @param n: Number of intervals.
    @param t: Length of each interval.
    @return Array with slope features appended.
    """
    T = X.shape[1]
    Xslope = np.zeros((X.shape[0],n))
    for i in range(n):
       if (i+1) * t <= T:
           intlen = t
           p = np.array(range(1,intlen+1)).reshape([1,-1])
           xtem = X[:,i*t:(i+1)*t]
           slopecur = (np.matmul(p,xtem.T)-np.sum(p)*(np.mean(xtem,1)))/\
               (np.matmul(p,p.T)-np.sum(p)*np.mean(p))
           Xslope[:,i] = np.arctan(slopecur)   
       else:
           intlen = T - i*t + 1
           p = np.array(range(1,intlen+1)).reshape([1,-1])
           xtem = X[:,i*t:T]
           slopecur = (p*xtem.T-sum(p)*(np.mean(xtem,1)))/(p*p.T-sum(p)*\
                                                           np.mean(p))
           Xslope[:,i] = np.arctan(slopecur)   
    X = np.concatenate((X, Xslope),1)
    return X


# Standardize data
def Stand_data(Xtrain, Xval, Xtest, val=False):
    # print("Running Stand_data")
    """
    @brief Standardize training, validation, and test datasets.
    @param Xtrain: Training data array.
    @param Xval: Validation data array.
    @param Xtest: Test data array.
    @param val: Whether to treat Xval as a separate validation set.
    @return Tuple of standardized (Xtrain, Xval, Xtest).
    """
    #fixed data leakage issue
    ss = StandardScaler()
    Xtrain = ss.fit_transform(Xtrain)
    
    if val:
        Xval = ss.transform(Xval)
        Xtest = ss.transform(Xtest)
    else:
        Xtest = ss.transform(Xtest)
        Xval = Xtest

    return Xtrain, Xval, Xtest


# Extract features from multi views
def Multi_view(Xtrain_raw, Xval_raw, Xtest_raw):
    # print("Running Multi_view")
    """
    @brief Extract features from multiple views of the input data.
    @param Xtrain_raw: Raw training data.
    @param Xval_raw: Raw validation data.
    @param Xtest_raw: Raw test data.
    @return Tuple of processed (Xtrain, Xval, Xtest).
    """
    N, T = calculate_dataset_metrics(Xtrain_raw)
    intvlen, nintv = Get_intinfo(T)
    
    Xtrain_raw, Xtrain_fft, Xtrain_derv = Splitview(Xtrain_raw, T)
    Xval_raw, Xval_fft, Xval_derv = Splitview(Xval_raw, T)
    Xtest_raw, Xtest_fft, Xtest_derv = Splitview(Xtest_raw, T)
    
    Xtrain_raw, Xtrain_fft, Xtrain_derv, Xval_raw, Xval_fft, Xval_derv, \
    Xtest_raw, Xtest_fft, Xtest_derv = Extract_intfea(Xtrain_raw, Xtrain_fft, \
    Xtrain_derv, Xval_raw, Xval_fft, Xval_derv, Xtest_raw, Xtest_fft, \
        Xtest_derv, nintv, intvlen)
    
    Xtrain = np.concatenate((Xtrain_raw, Xtrain_fft, Xtrain_derv), 1)
    Xval = np.concatenate((Xval_raw, Xval_fft, Xval_derv), 1)
    Xtest = np.concatenate((Xtest_raw, Xtest_fft, Xtest_derv), 1)
    
    Xtrain, Xval, Xtest = Stand_data(Xtrain, Xval, Xtest)

    return Xtrain, Xval, Xtest







