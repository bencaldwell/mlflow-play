import os
import shutil
import argparse
from mlflow import log_metric, log_param, log_artifacts
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import mlflow.sklearn

MODEL_NAME = 'sklearn-rf'

def parse_args():
    parser = argparse.ArgumentParser(description='Train hyperparams.')
    parser.add_argument('--input', type=str, help='input dataset')
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--test_size', type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # delete and remake the artifact dir
    shutil.rmtree('outputs')
    os.makedirs("outputs")

    df = pd.read_csv(args.input)
    y = df.pop('label').astype(float)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)
    print(y_train.dtypes)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Log a metric; metrics can be updated throughout the run
    log_metric("score", clf.score(X_test, y_test))

    model_dir = os.path.join('outputs', 'model')
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.save_model(clf, model_dir, signature=signature)

    # save a confusion matrix plot as an artifact    
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig("outputs/cfx.png")
    
    # log the artifacts
    log_artifacts("outputs")
