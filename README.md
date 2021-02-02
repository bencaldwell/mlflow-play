# MLFLOW PLAY

This is a sample project to work out a basic workflow using DVC and MLFlow to be used for local experimentation and serving.

## TODO:

Using MLFLow:

* [X] Store a model file - done for a simple sklearn RF classifier
* [X] Serve a model file
* [ ] Serve a model file as a docker image
* [ ] Deploy a docker model on another onprem
* [ ] Deploy a docker model to sagemaker

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
mlflow run -e pca --experiment-name "my experiment" .
```

## Serving the model

Get the model folder from the MLFlow run and serve with e.g.

```
mlflow models serve -m ./mlruns/0/a46ecc1db20c4fd19165340b3ab408aa/artifacts/model
```

The model is then available as a rest endpoint and it can be tested using a query e.g.

```
invoke-webrequest -Uri "http://127.0.0.1:5000/invocations" -Headers @{'Content-Type'='application/json'} -Method POST -Body '{"data":[[4.9,3.0,1.4,0.2]]}'
```