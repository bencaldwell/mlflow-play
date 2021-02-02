import os
import shutil
import argparse
from mlflow import log_metric, log_param, log_artifacts
from numpy import testing

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Train hyperparams.')
    parser.add_argument('--input', type=str, help='input dataset')
    parser.add_argument('--max_depth', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # delete and remake the artifact dir
    shutil.rmtree('outputs')
    os.makedirs("outputs")

    df = pd.read_csv(args.input)
    y = df.pop('label')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Log a metric; metrics can be updated throughout the run
    log_metric("score", clf.score(X_test, y_test))
        
    # save the plot as an artifact
    # fig.savefig( "outputs/plot.png")
    
    # log the artifacts
    # log_artifacts("outputs")
