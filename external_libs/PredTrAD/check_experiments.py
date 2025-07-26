#!/usr/bin/env python3
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()

# fetch all experiments (active + deleted)
experiments = client.search_experiments(view_type=ViewType.ALL)

print("Registered experiments:")
for exp in experiments:
    print(f"  • ID={exp.experiment_id!r}\tname={exp.name!r}")
