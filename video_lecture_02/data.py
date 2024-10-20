#!/usr/bin/env python3
import pandas as pd
import numpy as np

def get_adult(filename, train_size=0.8):
    df = pd.read_csv(filename)
    # Remove rows with missing values
    df = df.dropna()
    # Shuffle rows
    df = df.sample(frac=1, replace=False, random_state=1234)

    c_attrs = ['workclass','education','marital-status',
               'occupation','relationship','race','sex',
               'native-country']
    n_attrs = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    num_train = int(train_size*len(df))
    num_val = len(df)-num_train

    epsilon = 1e-7

    # Just for clarity, standardize each numerical attribute separately,
    # inside the dataframe
    for attr in n_attrs:
        mu = df.head(num_train)[attr].mean()
        sigma = df.head(num_train)[attr].std()
        df[attr]=(df[attr]-mu)/(np.sqrt(sigma*sigma + epsilon))
        # dfmin=df.head(num_train)[attr].min()
        # dfmax=df.head(num_train)[attr].max()
        # df[attr]=(df[attr]-dfmin)/(dfmax-dfmin)

    # Categorical attributes (aka multilevel variables in the statistics
    # jargon) should be one-hot encoded (so the subvectors are already in
    # [0,1]). We use pandas to create the encoding
    df = pd.get_dummies(df,columns=c_attrs, prefix=[c[:2] for c in c_attrs])

    y_train = df.head(num_train)['salary'].values=='<=50K'
    y_val = df.tail(num_val)['salary'].values=='<=50K'
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    del df['salary']  # this is the target!!

    # Take care of the # of bits for floating point. This affects
    # *significantly* memory and FlOps in the GPU.
    X_train=df.head(num_train).values.astype(np.float32)
    X_val=df.tail(num_val).values.astype(np.float32)
    return X_train, y_train, X_val, y_val
