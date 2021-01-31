import os
from sklearn import datasets
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path

params = yaml.safe_load(open('params.yaml'))['get_raw']


def parse_args():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    iris = datasets.load_iris()
    df = pd.DataFrame(
        data=np.c_[(iris['target'], iris['data'])],
        columns=['label'] + iris['feature_names']
    )

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
