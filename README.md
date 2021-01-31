# MLFLOW PLAY

This is a sample project to work out a basic MLFlow to be used for local experimentation and serving.

## Conda Environment

Remember to keep the `conda.yaml` updated with your latest development environment.

There is a subtle bug in conda where the command `conda env export > conda.yaml` results in an unusable (by conda) UTF-16 LE file.
Instead use this to update `conda.yaml`

```
conda env export -f conda.yaml
```

## Preparing the data

Use dvc to prepare the data.

```
dvc repro
```

or if the data is prepared and in remote storage

```
dvc pull
```

## Running the experiment

Use the `MLproject` definition to keep things consistent by running:

```
mlflow run .
```
