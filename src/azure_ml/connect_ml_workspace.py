# first install package azure-ai-ml: pip install azure-ai-ml

# 1. Connection to Azure ML Workspace
from xxlimited import new

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command  # for creating a job

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name",
)

# 2. If creating a job
job = command(
    code="./src/azure_ml",
    command="python train.py --arg1 value1 --arg2 value2",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
    compute="your-compute-target",
)

# submit job
returned_job = ml_client.create_or_update(job)


# scratch pad bellow
from azure.ai.ml.entities import AzureBlobDatastore, AccountKeyConfiguration

bs = AzureBlobDatastore(
    name="my_blob_datastore",
    account_name="your-storage-account-name",
    container_name="your-container-name",
    credentials=AccountKeyConfiguration(account_key="your-storage-account-key"),
)
ml_client.create_or_update(bs)

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data = Data(
    name="my_data_asset",
    path="azureml://datastores/my_blob_datastore/paths/my-data-path",
    type=AssetTypes.URI_FILE,
)
ml_client.data.create_or_update(data)

from azure.ai.ml.entities import ComputeInstance

cb = ComputeInstance(
    name="my_compute_instance",
    size="Standard_DS3_v2",
    resource_group="your-resource-group",
    workspace_name="your-workspace-name",
)
# ml_client.compute.create_or_update(cb)
ml_client.begin_create_or_update(cb).result()

from azure.ai.ml.entities import Command

job = Command(
    name="my_command_job",
    command="python train.py --arg1 value1 --arg2 value2",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
    compute="my_compute_instance",
)
ml_client.jobs.create_or_update(job)

envs = ml_client.environments.list()
for env in envs:
    print(env.name)

from azure.ai.ml.entities import Environment

env = Environment(
    name="my_custom_env",
    docker_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file="./environment.yml",
)
ml_client.environments.create_or_update(env)


from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input
from azure.ai.ml import automl

data_input = Input(
    type=AssetTypes.MLTABLE,
    path="azureml://datastores/my_blob_datastore/paths/my-data-path",
)
automl_job = automl.classification(
    name="my_automl_job",
    training_data=data_input,
    target_column_name="target",
    primary_metric="accuracy",
    compute="my_compute_instance",
    max_time_sec=3600,
    max_concurrent_iterations=4,
    featurization="auto",
    n_cross_validations=5,
)
ml_client.jobs.create_or_update(automl_job)
