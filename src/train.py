import os
import sys
import argparse
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description='Train hyperparams.')
    parser.add_argument('--dummy', type=float, default=1.0, help='dummy variable')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    for arg in vars(args):   
        # Log a parameter (key-value pair)
        log_param(arg, getattr(args, arg))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")
