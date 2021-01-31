import os
import shutil
import sys
import argparse
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Train hyperparams.')
    parser.add_argument('--input', type=str, help='input dataset')
    parser.add_argument('--n_components', type=int, help='number of pca components')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    for arg in vars(args):   
        # Log a parameter (key-value pair)
        log_param(arg, getattr(args, arg))

    # delete and remake the artifact dir
    shutil.rmtree('outputs')
    os.makedirs("outputs")

    df = pd.read_csv(args.input)
    y = df.pop('label')
    X = df

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean() + 1.5,
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
            edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    # plt.show()

    # Log a metric; metrics can be updated throughout the run
    # log_metric("foo", random())
    # log_metric("foo", random() + 1)
    # log_metric("foo", random() + 2)

    
    # save the plot as an artifact
    fig.savefig( "outputs/plot.png")
    
    # log the artifacts
    log_artifacts("outputs")
