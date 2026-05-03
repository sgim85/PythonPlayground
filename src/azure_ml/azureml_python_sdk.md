#### Resources/Labs: https://github.com/MicrosoftLearning/mslearn-azure-ml/tree/main/Labs

## Datastores
In AML, datastores are abstractions for cloud data sources. They encapsulate the information needed to connect to data sources.

The benefits of using datastores are:
* Provides easy-to-use URIs to your data storage.
* Facilitates data discovery within Azure Machine Learning.
* Securely stores connection information, without exposing secrets and keys to data scientists.

When you create a datastore with an existing storage account on Azure, you have the choice between two different authentication methods:
* **Credential-based:** Use a *service principal*, *shared access signature (SAS)* token or *account key* to authenticate access to your storage account.
	* **Service principals** provide secure, identity-based, long-term application authentication using Entra ID (best for backend services). 
	* **SAS tokens** grant time-limited, delegated access to specific resources (best for temporary access). 
	* **Account keys** offer unrestricted, full administrative access to storage accounts and should be avoided or tightly secured due to risk
* **Identity-based:** Use your Microsoft Entra identity or managed identity.

Create a datastore to connect to an Azure Blob Storage container using an account key to athenticate.
```python
from azure.ai.ml.entities import AzureBlobDatastore, AccountKeyConfiguration

blob_datastore = AzureBlobDatastore(
    name="blob_example",
    description="Datastore pointing to a blob container",
    account_name="mytestblobstore",
    container_name="data-container",
    credentials=AccountKeyConfiguration(account_key="XXXxxxXXXxXXXXxxXXX"),
)

ml_client.create_or_update(blob_datastore)
```

Create a datastore to connect to an Azure Blob Storage container using a SAS token to authenticate.
```python
from azure.ai.ml.entities import AzureBlobDatastore, SasTokenConfiguration

sas_datastore = AzureBlobDatastore(
    name="blob_sas_example",
    description="Datastore pointing to a blob container",
    account_name="mytestblobstore",
    container_name="data-container",
    credentials=SasTokenConfiguration(
        sas_token="?xx=XXXX-XX-XX&xx=xxxx&xxx=xxx&xx=xxxxxxxxxxx&xx=XXXX-XX-XXXXX:XX:XXX&xx=XXXX-XX-XXXXX:XX:XXX&xxx=xxxxx&xxx=XXxXXXxxxxxXXXXXXXxXxxxXXXXXxxXXXXXxXXXXxXXXxXXxXX"
    ),
)

ml_client.create_or_update(sas_datastore)
```
## Data Assets
In AML, data assets are references to where the data is stored, how to get access, and any other relevant metadata. You can create data assets to get access to data in datastores, Azure storage services, public URLs, or data stored on your local device. 

Data assets are most useful when executing machine learning tasks as Azure Machine Learning jobs. A data asset can be parsed as both an input or output of an Azure Machine Learning job.

The benefits of using data assets are:
* You can **share and reuse data** with other members of the team such that they don't need to remember file locations.
* You can **seamlessly access data** during model training (on any supported compute type) without worrying about connection strings or data paths.
* You can **version** the metadata of the data asset.

There are three main types of data assets you can use:
* **URI file:** Points to a specific file. The supported paths you can use when creating a URI file data asset are:
    * Local: *./<_path_>*
    * Azure Blob Storage: *wasbs://<account_name>.blob.core.windows.net/<container_name>/<_folder_>/<_file_>*
    * Azure Data Lake Storage (Gen 2): *abfss://<file_system>@<account_name>.dfs.core.windows.net/<_folder_>/<_file_>*
    * Datastore: *azureml://datastores/<datastore_name>/paths/<_folder_>/<_file_>*
* **URI folder:** Points to a folder.
* **MLTable:** Points to a folder or file, and includes a schema to read as tabular data.

### Create a URI file data asset
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = "<supported-path>"

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="<description>",
    name="<name>",
    version="<version>",
)
ml_client.data.create_or_update(my_data)
```

Reading the file data
```python
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head(10))

# if reading json file instead of csv
df_json = pd.read_json(args.input_data)
print(df_json.head(10))
```

### Create a URI folder data asset
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="<description>",
    name="<name>",
    version='<version>'
)

ml_client.data.create_or_update(my_data)
```

Read all files in the folder
```python
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

data_path = args.input_data
all_files = glob.glob(data_path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
```

### Create a MLTable data asset
A MLTable data asset allows you to point to tabular data. When you create a MLTable data asset, you specify the schema definition to read the data. As the schema is already defined and stored with the data asset, you don't have to specify how to read the data when you use it.

Therefore, you want to use a MLTable data asset when the schema of your data is complex or changes frequently. Instead of changing how to read the data in every script that uses the data, you only have to change it in the data asset itself.

When you define the schema when creating a MLTable data asset, you can also choose to only specify a subset of the data.

For certain features in Azure Machine Learning, like Automated Machine Learning, you need to use a MLTable data asset, as Azure Machine Learning needs to know how to read the data.

To define the schema, you can include a **MLTable file** in the same folder as the data you want to read. The MLTable file includes the path pointing to the data you want to read, and how to read the data:
```yml
type: mltable

paths:
  - pattern: ./*.txt
transformations:
  - read_delimited:
      delimiter: ','
      encoding: ascii
      header: all_files_same_headers
```

Create a MLTable data asset with the Python SDK
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<path-including-mltable-file>'

my_data = Data(
    path=my_path,
    type=AssetTypes.MLTABLE,
    description="<description>",
    name="<name>",
    version='<version>'
)

ml_client.data.create_or_update(my_data)
```

Reading the data from the MLTable data asset
```python
import argparse
import mltable
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

tbl = mltable.load(args.input_data)
df = tbl.to_pandas_dataframe()

print(df.head(10))
```

A common approach is to convert the tabular data to a Pandas data frame. However, you can also convert the data to a Spark data frame if that suits your workload better.

## Compute Types and Deployments
Azure Machine Learning supports multiple types of compute for experimentation, training, and deployment.
* **Compute Instance:** Ideal for experimentation and developement, such us working with Jupyter notebooks.
* **Computer Clusters:** On-demand and scalable.
* **Serverless Compute:** On-demand and scalable. 
* **Attached Compute:** Allows you to attach existing compute like Azure virtual machines or Azure Databricks clusters to your workspace.
* **Kubernetes Clusters**

For **batch predictions**, you can run a pipeline job in Azure Machine Learning. Compute targets like **compute clusters** and Azure Machine Learning's **serverless compute** are ideal for pipeline jobs as they're on-demand and scalable.

When you want real-time predictions, you need a type of compute that is running continuously. Real-time deployments therefore benefit from more lightweight (and thus more cost-efficient) compute. Containers are ideal for real-time deployments. When you deploy your model to a **managed online endpoint**, Azure Machine Learning creates and manages containers for you to run your model. Alternatively, you can attach **Kubernetes clusters** to manage the necessary compute to generate real-time predictions.

### Computer Instance
You can create a compute instance in the Azure Machine Learning studio, using the Azure command-line interface (CLI), or the Python software development kit (SDK).

If working with Jupyter notebooks, a compute instance can only be assigned to one user because it can't support parallel workloads. Since the instance is always-on, you can schedule to turn it off when not needed to save costs. You can run notebooks in AML studio. Alternatively a notebook can be run in VS Code by attaching the compute instance.

Creating a compute instance using the python SDK.

```python
from azure.ai.ml.entities import ComputeInstance

ci_basic_name = "basic-ci-12345"
ci_basic = ComputeInstance(
    name=ci_basic_name, 
    size="STANDARD_DS3_v2"
)
ml_client.begin_create_or_update(ci_basic).result()
```

### Computer Cluster
When you run code in production environments, it's better to use scripts instead of notebooks. Compute clusters are scalable and therefore the prefered compute type. You can create a compute cluster in the Azure Machine Learning studio, using the Azure command-line interface (CLI), or the Python software development kit (SDK).

There are three main scenarios in which you can use a compute cluster:
* Running a pipeline job you built in the Designer.
* Running an Automated Machine Learning job.
* Running a script as a job.

Creating a compute cluster using the python SDK.

```python
from azure.ai.ml.entities import AmlCompute

cluster_basic = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="STANDARD_DS3_v2",
    location="westus",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    tier="low_priority",
)
ml_client.begin_create_or_update(cluster_basic).result()
```

## Environments
Environments are runtime configurations of specific packages, libraries, and python version, needed to run experiments. To improve portability, you usually create environments in Docker containers that are in turn hosted on compute targets, such as your development computer, virtual machines, or clusters in the cloud.

**Curated environments** are prebuilt environments for the most common machine learning workloads, available in your workspace by default.

Most commonly, you use environments when you want to run a script as a (command) job.
```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-with-curated-environment",
    experiment_name="train-with-curated-environment"
)

# submit job
returned_job = ml_client.create_or_update(job)
```

List environments using the python SDK
```python
envs = ml_client.environments.list()
for env in envs:
    print(env.name)
```

Review details of an environment. E.g. when verifying what python packages it contains.
```python
env = ml_client.environments.get(name="my-environment", version="1")
print(env)
```

### Create and use custom environments
You can also create your own environments if the curated environments aren't suitable.

You can create an environment from a Docker image hosted in a public registry like DockerHub or a private one like Azure Container registry.

E.g. Creating an environment from a public registry image with PyTorch installed for deep learning model training.
```python
from azure.ai.ml.entities import Environment

env_docker_image = Environment(
    image="pytorch/pytorch:latest", 
    name="public-docker-image-example",
    description="Environment created from a public Docker image.",
)
ml_client.environments.create_or_update(env_docker_image)

# NOTE: You can also use AML base images such as those used by curated envs. E.g. "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
```

When you need to include other packages or libraries in your environment, you can add a **conda** specification file to a Docker image when creating the environment.

A conda specification file is a YAML file, which lists the packages that need to be installed using _conda_ or _pip_. Such a YAML file may look like:

```yaml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
```

Then specify the conda specification file when creating the environment.
```python
from azure.ai.ml.entities import Environment

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./conda-env.yml",
    name="docker-image-plus-conda-example",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)
```
#### using an environment
Most commonly, you use environments when you want to run a script as a (command) job.

To specify which environment you want to use to run your script, you reference an environment using the *\<curated-environment-name\>:\<version\>* or _\<curated-environment-name\>@latest_ syntax.

```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="docker-image-plus-conda-example:1",
    compute="aml-cluster",
    display_name="train-custom-env",
    experiment_name="train-custom-env"
)

# submit job
returned_job = ml_client.create_or_update(job)
```
The first time you use an environment, it can take 10-15 minutes to build the environment.

When Azure Machine Learning builds a new environment, it's added to the list of custom environments in the workspace. The image of the environment is hosted in the **Azure Container registry** associated to the workspace. Whenever you use the same environment for another job (and another script), the environment is ready to go and doesn't need to be built again.

## Automated Machine Learning (AutoML)
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/06/Classification%20with%20Automated%20Machine%20Learning.ipynb

AutoML allows you to try multiple **preprocessing transformations** and algorithms with your data to find the best machine learning model (e.g. a classification model). It's cumbersome doing this manually in AML studio.

By default, AutoML will perform featurization (preprocessing transformations) on your data. You can disable it if you don't want the data to be transformed.

**NOTE:** _Preprocessing transformations_ convert raw data into clean, structured formats suitable for machine learning, focusing on scaling, encoding, and cleaning. Key techniques include scaling (MinMax, Standard), handling missing data, encoding categories (One-Hot, Label), normalization, and reducing skewness with log transforms. Common in Python using scikit-learn or TFX.

To pass a dataset as an input to an automated machine learning job, the data must be in tabular form and include a target column. For the data to be interpreted as a tabular dataset, the input dataset must be a **MLTable**.
```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")
```

Configure an AutoML experiment
```python
from azure.ai.ml import automl

# configure the classification job
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)
```

The **primary metric** is the target performance metric for which the optimal model will be determined. To retrieve the list of metrics available when you want to train a classification model
```python
from azure.ai.ml.automl import ClassificationPrimaryMetrics
 
list(ClassificationPrimaryMetrics)
```

Training machine learning models will cost compute. To minimize costs and time spent on training, you can set limits to an AutoML experiment or job by using **set_limits()**.
```python
# set the limits (optional)
classification_job.set_limits(
    timeout_minutes=60, # Number of minutes after which the complete AutoML experiment is terminated.
    trial_timeout_minutes=20, #Maximum number of minutes one trial can take.
    max_trials=5, # Maximum number of trials, or models that will be trained.
    enable_early_termination=True, # Whether to end the experiment if the score isn't improving in the short term.
)
```

To save time, you can also run multiple trials in parallel. When you use a compute cluster, you can have as many parallel trials as you have nodes. You can set the maximum numnber of parallel trials using **max_concurrent_trials.**

AutoML will try various combinations of featurization and algorithms to train a machine learning model. If you already know that certain algorithms aren't well-suited for your data, you can exclude (or include) a subset of the available algorithms.

```python
# set the training properties (optional)
classification_job.set_training(
    blocked_training_algorithms=["LogisticRegression"], 
    enable_onnx_compatible_models=True
)
```

Submit AutoML job
```python
# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)
```

Get a direct link to the AutoML job by running the following code:
```python
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```

## MLflow
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/07/Track%20model%20training%20with%20MLflow.ipynb

MLflow is an open-source library for tracking and managing your machine learning experiments. It logs everything about the model you're training, such as parameters, metrics, and artifacts.

You can run notebooks in Azure Machine Learning or on a local device with MLflow enabled. If using AML, MLflow is auto configured for you and ready to use. If running on a local device, you will need to set the MLflow tracking url.

To group model training results, you'll use **experiments**. If you don't create an experiment, MLflow will assume the default experiment with name **Default**.

Creating MLflow experiment:
```python
import mlflow

mlflow.set_experiment(experiment_name="heart-condition-classifier")

# If running notebook on local device, set mlflore tracking url (copied from AML workspace)
mlflow.set_tracking_uri = "MLFLOW-TRACKING-URI"
```

To Log results with MLflow, you have two options to enable:
* Enable **autologging**: Enable this on the model framework you're using. E.g. if using xgboost, _mlflow.xgboost.autolog()_

    ```python
    from xgboost import XGBClassifier

    with mlflow.start_run():
        mlflow.xgboost.autolog() # Enable autolog

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    ```

* Use **custom logging**: Logs supplementary or custom information that isn't logged through autologging. Common functions used with custom logging are:
    * _mlflow.log_param()_
    * _mlflow.log_metric()_
    * _mlflow.log_artifact()_
    * _mlflow.log_model()_

    ```python
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    with mlflow.start_run():
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy) # log accuracy metric
    ```

Reasons you'd want to track models during training:
* To compare the results of models you train with different **hyperparameter** values.
    ```python
    # Job 1
    mlflow.log_param("regularization_rate", 0.1)

    # Job 2
    mlflow.log_param("regularization_rate", 0.01)
    ```
* Evaluating different estimators. E.g. If training a classificaiton model, you evaluate **logistic regression** estimator (LogisticRegression) vs the **decision tree classifier** estimator (DecisionTreeClassifier).
     ```python
    # Job 1
    mlflow.log_param("estimator", "LogisticRegression")

    # Job 2
    mlflow.log_param("estimator", "DecisionTreeClassifier")
    ```

Once autolog is enabled on a ML library, MLflow will automatically log any model trained with that library. To disable the autologging, run the following code (in this case disabling autolog on scikit-learn).
```python
# Disable autolog
mlflow.sklearn.autolog(disable=True)
```

### Retrieve metrics with MLflow in a notebook
When you run a training script as a job in Azure Machine Learning, and track your model training with MLflow, you can query the runs in a notebook by using MLflow. Using MLflow in a notebook gives you more control over which runs you want to retrieve to compare.

When using MLflow to query your runs, you'll refer to experiments and runs.

To get all active experiments:
```python
# to include archieved experiments, set the parameter "view_type=ViewType.ALL"
experiments = mlflow.search_experiments(max_results=2)
for exp in experiments:
    print(exp.name)
```

To retrieve a specific experiment:
```python
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)
```

MLflow allows you to search for runs inside of any experiment. You need either the experiment ID or the experiment name.
```python
mlflow.search_runs(exp.experiment_id)
```

You can search across all the experiments in the workspace. Can be useful in case you want to compare runs of the same model when it's being logged in different experiments (by different people or different project iterations).
```python
all_runs = mlflow.search_runs(search_all_experiments=True)
```

You can also look for a run with a specific combination in the hyperparameters.
```python
mlflow.search_runs(
    exp.experiment_id, filter_string="params.num_boost_round='100'", max_results=2
)
```


## Command Job
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/08/Run%20script%20as%20a%20command%20job.ipynb

Though notebooks are ideal for experimentation and development, scripts are a better fit for production workloads. In Azure Machine Learning, you can run a script as a command job. When you submit a command job, you can configure various parameters like the input data and the compute environment. 

```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src", # folder that includes the script to run.
    command="python train.py", # the file to run.

    # the necessary packages to be installed on the compute before running the command.
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",

    compute="aml-cluster", # compute to use to run the command.
    display_name="train-model", # name of the individual job.
    experiment_name="train-classification-model" # name of the experiment the job belongs to.
    )

# submit job
returned_job = ml_client.create_or_update(job)
```

You can monitor and review the job in the Azure Machine Learning studio. Jobs will be grouped by the experiment name.

### Use parameters in a command job
You can increase the flexibility or portability of your scripts by using parameters. To use parameters in a script, you must use a library such as _argparse_ to read arguments passed to the script and assign them to variables.

```python
# import libraries
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)
```

Passing arguments to a script:
```python
python train.py --training_data diabetes.csv
```

Passing parameters to a script you want to run as a command job.
```python
from azure.ai.ml import command

# configure job
job = command(
    command="python train.py --training_data diabetes.csv",
    #....more command options set here
    )
```

## Hyperparameters
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/09/Hyperparameter%20tuning.ipynb

In machine learning, models are trained to predict unknown labels for new data based on correlations between known labels and features found in the training data. Depending on the algorithm used, you may need to specify **hyperparameters** to configure how the model is trained.

**Hyperparameter tuning** is accomplished by training the multiple models, using the same algorithm and training data but different hyperparameter values. The resulting model from each training run is then evaluated to determine the performance **metric** for which you want to optimize (for example, _accuracy_), and the best-performing model is selected.

### Sweep Job
In Azure Machine Learning, you can tune hyperparameters by submitting a script as a **sweep job**. A sweep job will run a trial for each hyperparameter combination to be tested. Each trial uses a training script with parameterized hyperparameter values to train a model, and logs the target performance metric achieved by the trained model.

### Search Space
A **search space** is the set of hyperparameter values tried during hyperparameter tuning. You can define a search space for a discrete parameter using a Choice from:
    
* a list: **_Choice(values=[10,20,30])_**
* a range: **_Choice(values=range(1,10))_**
* an arbitrary set of comma-separated values: **_Choice(values=(30,50,100))_**

You can also select discrete values from any of the following discrete distributions:
* **_QUniform(min_value, max_value, q)_**: Returns a value like _round(Uniform(min_value, max_value) / q) * q_
* **_QLogUniform(min_value, max_value, q)_**: Returns a value like _round(exp(Uniform(min_value, max_value)) / q) * q_
* **_QNormal(mu, sigma, q)_**: Returns a value like _round(Normal(mu, sigma) / q) * q_
* **_QLogNormal(mu, sigma, q)_**: Returns a value like _round(exp(Normal(mu, sigma)) / q) * q_

### Continuous hyperparameters
Some hyperparameters are _continuous_ - in other words you can use any value along a scale, resulting in an infinite number of possibilities. To define a search space for these kinds of value, you can use any of the following distribution types:


* **_Uniform(min_value, max_value)_**: Returns a value uniformly distributed between _min_value_ and _max_value_
* **_LogUniform(min_value, max_value)_**: Returns a value drawn according to _exp(Uniform(min_value, max_value))_ so that the logarithm of the return value is uniformly distributed
* **_Normal(mu, sigma)_**: Returns a real value that's normally distributed with mean _mu_ and standard deviation _sigma_
* **_LogNormal(mu, sigma)_**: Returns a value drawn according to _exp(Normal(mu, sigma))_ so that the logarithm of the return value is normally distributed

### Defining a search space
To define a search space for hyperparameter tuning, create a dictionary with the appropriate parameter expression for each named hyperparameter.

For example, the following search space indicates that the _batch_size_ hyperparameter can have the value 16, 32, or 64, and the _learning_rate_ hyperparameter can have any value from a normal distribution with a mean of 10 and a standard deviation of 3.

```python
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),
)
```

### Configure a sampling method
The specific values used in a hyperparameter tuning run, or sweep job, depend on the type of sampling used.

There are three main sampling methods available in Azure Machine Learning:

* **Grid sampling**: Tries every possible combination. Can only be applied when all hyperparameters are discrete.
    ```python
    from azure.ai.ml.sweep import Choice

    command_job_for_sweep = command_job(
        batch_size=Choice(values=[16, 32, 64]),
        learning_rate=Choice(values=[0.01, 0.1, 1.0]),
    )

    sweep_job = command_job_for_sweep.sweep(
        sampling_algorithm = "grid",
        ...
    )
    ```

* **Random sampling**: Randomly chooses values from the search space. **Sobol** (a variation of _random sampling_) adds a seed to random sampling to make the results reproducible. The selected value for each hyperparameter can be a mix of discrete and continuous values.
    ```python
    from azure.ai.ml.sweep import Normal, Uniform

    command_job_for_sweep = command_job(
        batch_size=Choice(values=[16, 32, 64]),   
        learning_rate=Normal(mu=10, sigma=3),
    )

    sweep_job = command_job_for_sweep.sweep(
        sampling_algorithm = "random",
        ...
    )

    # If using Sobol to reproduce a random sampling sweep job by using a seed
    sweep_job = command_job_for_sweep.sweep(
        sampling_algorithm = RandomSamplingAlgorithm(seed=123, rule="sobol"),
        ...
    )
    ```

* **Bayesian sampling**: Chooses new values based on previous results. It's based on the _Bayesian optimization algorithm_, which tries to select parameter combinations that will result in improved performance from the previous selection
    ```python
    from azure.ai.ml.sweep import Uniform, Choice

    command_job_for_sweep = job(
        batch_size=Choice(values=[16, 32, 64]),    
        learning_rate=Uniform(min_value=0.05, max_value=0.1),
    )

    sweep_job = command_job_for_sweep.sweep(
        sampling_algorithm = "bayesian",
        ...
    )
    ```

### Early termination
Hyperparameter tuning helps you fine-tune your model and select the hyperparameter values that will make your model perform best.

For you to find the best model, however, can be a never-ending conquest. You always have to consider whether it's worth the time and expense of testing new hyperparameter values to find a model that may perform better.

An **early termination policy**, particularly for when working with continous hyperparameters, may make sense to configure with your sweep job. It may not be worthful in 
certain scenarios, however, such as when working with a _discrete_ search space using the _grid sampling_ method.

#### Configure an early termination policy
* **evaluation_interval**: Specifies at which interval you want the policy to be evaluated. Every time the primary metric is logged for a trial counts as an interval.
* **delay_evaluation**: Specifies when to start evaluating the policy. This parameter allows for at least a minimum of trials to complete without an early termination policy affecting them.

New models may continue to perform only slightly better than previous models. To determine the extent to which a model should perform better than previous trials, there are three options for early termination:

* **Bandit policy**: Uses a _slack_factor_ (relative) or _slack_amount_ (absolute). Any new model must perform within the slack range of the best performing model.

    For example, the following code applies a bandit policy with a delay of five trials, evaluates the policy at every interval, and allows an absolute slack amount of 0.2.
    ```python
    from azure.ai.ml.sweep import BanditPolicy

    sweep_job.early_termination = BanditPolicy(
        slack_amount = 0.2, 
        delay_evaluation = 5, 
        evaluation_interval = 1
    )
    ```
    Imagine the primary metric is the accuracy of the model. When after the first five trials, the best performing model has an accuracy of 0.9, any new model needs to perform better than (0.9-0.2) or 0.7. If the new model's accuracy is higher than 0.7, the sweep job will continue. If the new model has an accuracy score lower than 0.7, the policy will terminate the sweep job.

    You can also apply a bandit policy using a _slack factor_, which compares the performance metric as a ratio rather than an absolute value.

* **Median stopping policy**: Uses the median of the averages of the primary metric. Any new model must perform better than the median.

    For example, the following code applies a median stopping policy with a delay of five trials and evaluates the policy at every interval.

    ```python
    from azure.ai.ml.sweep import MedianStoppingPolicy

    sweep_job.early_termination = MedianStoppingPolicy(
        delay_evaluation = 5, 
        evaluation_interval = 1
    )
    ```

    Imagine the primary metric is the accuracy of the model. When the accuracy is logged for the sixth trial, the metric needs to be higher than the median of the accuracy scores so far. Suppose the median of the accuracy scores so far is 0.82. If the new model's accuracy is higher than 0.82, the sweep job will continue. If the new model has an accuracy score lower than 0.82, the policy will stop the sweep job, and no new models will be trained.

* **Truncation selection policy**: Uses a _truncation_percentage_, which is the percentage of lowest performing trials. Any new model must perform better than the lowest performing trials.

    For example, the following code applies a truncation selection policy with a delay of four trials, evaluates the policy at every interval, and uses a truncation percentage of 20%.

    ```python
    from azure.ai.ml.sweep import TruncationSelectionPolicy

    sweep_job.early_termination = TruncationSelectionPolicy(
        evaluation_interval=1, 
        truncation_percentage=20, 
        delay_evaluation=4 
    )
    ```

    Imagine the primary metric is the accuracy of the model. When the accuracy is logged for the fifth trial, the metric should not be in the worst 20% of the trials so far. In this case, 20% translates to one trial. In other words, if the fifth trial is not the worst performing model so far, the sweep job will continue. If the fifth trial has the lowest accuracy score of all trials so far, the sweep job will stop.


### Use a sweep job for hyperparameter tuning

To run a sweep job, you need to create a training script just the way you would do for any other training job, except that your script must:

* Include an argument for each hyperparameter you want to vary.
* Log the target performance metric with **MLflow**. A logged metric enables the sweep job to evaluate the performance of the trials it initiates, and identify the one that produces the best performing model.

For example, the following example script trains a logistic regression model using a _'--regularization'_ argument to set the _regularization rate_ hyperparameter, and logs the _accuracy_ metric with the name _'Accuracy'_:

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("data.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)
```

To prepare the sweep job, you must first create a base **command job** that specifies which script to run and defines the parameters used by the script:

```python
from azure.ai.ml import command

# configure command job as base
job = command(
    code="./src",
    command="python train.py --regularization ${{inputs.reg_rate}}",
    inputs={
        "reg_rate": 0.01,
    },
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    )
```

You can then override your input parameters with your search space:
```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = job(
    reg_rate=Choice(values=[0.01, 0.1, 1]),
)
```

Finally, call sweep() on your command job to sweep over your search space:
```python
from azure.ai.ml import MLClient

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="Accuracy",
    goal="Maximize",
)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-example"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=4, max_concurrent_trials=2, timeout=7200)

# submit the sweep
returned_sweep_job = ml_client.create_or_update(sweep_job)
```

You can monitor sweep jobs in Azure Machine Learning studio. The sweep job will initiate trials for each hyperparameter combination to be tried. For each trial, you can review all logged metrics.

Additionally, you can evaluate and compare models by visualizing the trials in the studio. You can adjust each chart to show and compare the hyperparameter values and metrics for each trial.

## Pipelines
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/09/Run%20a%20pipeline%20job.ipynb

In Azure Machine Learning, a **pipeline** is a workflow of machine learning tasks in which each task is defined as a **component**. Pipelines are ideal when you need to run your scripts at scale, i.e. in a prod env.

Components can be arranged sequentially or in parallel, enabling you to build sophisticated flow logic to orchestrate machine learning operations. Each component can be run on a specific compute target, making it possible to combine different types of processing as required to achieve an overall goal.

A pipeline can be executed as a process by running the pipeline as a pipeline job. Each component is executed as a **child job** as part of the overall **pipeline job**.

### Components
Components allow you to create reusable scripts that can easily be shared across users within the same Azure Machine Learning workspace. There are two main reasons why you'd use components:
* To build a pipeline.
* To share ready-to-go code.

For example, a component may consist of a Python script that normalizes your data, trains a machine learning model, or evaluates a model.

A component consists of three parts:
* **Metadata**: Includes the component's name, version, etc.
* **Interface**: Includes the expected input parameters (like a dataset or hyperparameter) and expected output (like metrics and artifacts).
* **Command, code and environment**: Specifies how to run the code.

To create a component, you need two files:
* A script that contains the workflow you want to execute.
* A YAML file to define the metadata, interface, and command, code, and environment of the component.

For example, you may have a Python script prep.py that prepares the data by removing missing values and normalizing the data:
```python
# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# setup arg parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("--input_data", dest='input_data',
                    type=str)
parser.add_argument("--output_data", dest='output_data',
                    type=str)

# parse args
args = parser.parse_args()

# read the data
df = pd.read_csv(args.input_data)

# remove missing values
df = df.dropna()

# normalize the data    
scaler = MinMaxScaler()
num_cols = ['feature1','feature2','feature3','feature4']
df[num_cols] = scaler.fit_transform(df[num_cols])

# save the data as a csv
output_df = df.to_csv(
    (Path(args.output_data) / "prepped-data.csv"), 
    index = False
)
```

To create a component for the prep.py script, you'll need a YAML file prep.yml:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_data
display_name: Prepare training data
version: 1
type: command
inputs:
  input_data: 
    type: uri_file
outputs:
  output_data:
    type: uri_file
code: ./src
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
command: >-
  python prep.py 
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}
```

You can load the component with the following code:
```python
from azure.ai.ml import load_component
parent_dir = ""

loaded_component_prep = load_component(source=parent_dir + "./prep.yml")
```

To make the components accessible to other users in the workspace, you can also register components to the Azure Machine Learning workspace.
```python
prep = ml_client.components.create_or_update(prepare_data_component)
```

### Create a pipeline
You can create the YAML file, or use the _@pipeline()_ function to create the YAML file.

For example, if you want to build a pipeline that first prepares the data, and then trains the model, you can use the following code:
```python
from azure.ai.ml.dsl import pipeline

@pipeline()
def pipeline_function_name(pipeline_job_input):
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }
```

To pass a registered data asset as the pipeline job input, you can call the function you created with the data asset as input:
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, 
    path="azureml:data:1"
))
```

You can retrieve the configuration of the pipeline job by printing the _pipeline_job_ object:
```python
print(pipeline_job)
```

```yaml
display_name: pipeline_function_name
type: pipeline
inputs:
  pipeline_job_input:
    type: uri_file
    path: azureml:data:1
outputs:
  pipeline_job_transformed_data: null
  pipeline_job_trained_model: null
jobs:
  prep_data:
    type: command
    inputs:
      input_data:
        path: ${{parent.inputs.pipeline_job_input}}
    outputs:
      output_data: ${{parent.outputs.pipeline_job_transformed_data}}
  train_model:
    type: command
    inputs:
      input_data:
        path: ${{parent.outputs.pipeline_job_transformed_data}}
    outputs:
      output_model: ${{parent.outputs.pipeline_job_trained_model}}
tags: {}
properties: {}
settings: {}
```

### Run a pipeline job
You can run the workflow as a **pipeline job**. Note: You can change any parameter of the pipeline job configuration by referring to the parameter and specifying the new value:
```python
# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"

# set pipeline level compute
pipeline_job.settings.default_compute = "aml-cluster"

# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"
```

To submit the pipeline job, run the following code:
```python
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)
```

After you submit a pipeline job, a new job will be created in the Azure Machine Learning workspace. A pipeline job also contains child jobs, which represent the execution of the individual components. The Azure Machine Learning studio creates a graphical representation of your pipeline.

### Schedule a pipeline job
A pipeline is ideal if you want to get your model ready for production. Pipelines are especially useful for automating the retraining of a machine learning model. To automate the retraining of a model, you can schedule a pipeline.

To create a schedule that fires every minute, run the following code:
```python
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute", # OR: hour, day, week, month
    interval=1,
)
```

To schedule a pipeline, you'll need pipeline_job to represent the pipeline you've built:
```python
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()
```

The display names of the jobs triggered by the schedule will be prefixed with the name of your schedule. 

To delete a schedule, you first need to disable it:
```python
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()
```

## Log models with MLflow
Jupyter notebook source: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/10/Log%20models%20with%20MLflow.ipynb

When you train a machine learning model with Azure Machine Learning, you can use MLflow to register your model. MLflow standardizes the packaging of models, which means that an MLflow model can easily be imported or exported across different workflows.

When you train and log a model, you store all relevant artifacts in a directory. When you register the model, an _MLmodel_ file is created in that directory. The _MLmodel_ file contains the model's metadata, which allows for model traceability.

### Use autologging to log a model
When you train a model, you can include _mlflow.autolog()_ to enable autologging. MLflow's autologging automatically logs parameters, metrics, artifacts, and the model you train. The model is logged when the _.fit()_ method is called. The framework you use to train your model is identified and included as the **flavor** of your model.

Optionally, you can specify which flavor you want your model to be identified as by using _mlflow.\<flavor>.autolog()_. E.g.
* Keras: _mlflow.keras.autolog()_
* Scikit-learn: _mlflow.sklearn.autolog()_
* LightGBM: _mlflow.lightgbm.autolog()_
* XGBoost: _mlflow.xgboost.autolog()_

When you use autologging, an output folder is created which includes all necessary model artifacts, including the **MLmodel** file that references these files and includes the model's metadata.

### Manually log a model
When you want to have more control over how the model is logged, you can use autolog (for your parameters, metrics, and other artifacts), and set _**log_models=False**_. When you set the _**log_models**_ parameter to false, MLflow doesn't automatically log the model, and you can add it manually.

Logging the model allows you to easily deploy the model. To specify how the model should behave at inference time, you can customize the model's expected inputs and outputs. The schemas of the expected inputs and outputs are defined as the signature in the **MLmodel** file.

The model signature defines the schema of the model's inputs and outputs. The signature is stored in JSON format in the **MLmodel** file, together with other metadata of the model.

The model signature can be inferred from datasets or created manually by hand.

To log a model with a signature that is inferred from your training dataset and model predictions, you can use _**infer_signature()**_. For example, the following example takes the training dataset to infer the schema of the inputs, and the model's predictions to infer the schema of the output:
```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

# Infer the signature from the training dataset and model's predictions
signature = infer_signature(iris_train, clf.predict(iris_train))

# Log the scikit-learn model with the custom signature
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)
```

Alternatively, you can create the signature manually:
```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define the schema for the input data
input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])

# Define the schema for the output data
output_schema = Schema([ColSpec("long")])

# Create the signature object
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

### Understand the MLflow model format
MLflow uses the MLModel format to store all relevant model assets in a folder or directory. One essential file in the directory is the **MLmodel** file. The **MLmodel** file is the single source of truth about how the model should be loaded and used.

The **MLmodel** file may include:
* **artifact_path**: During the training job, the model is logged to this path.
* **flavor**: The machine learning library with which the model was created.
* **model_uuid**: The unique identifier of the registered model.
* **run_id**: The unique identifier of job run during which the model was created.
* **signature**: Specifies the schema of the model's inputs and outputs:
    * **inputs**: Valid input to the model. For example, a subset of the training dataset.
    * **outputs**: Valid model output. For example, model predictions for the input dataset.

The most important things to set are the **flavor** and the **signature**.

An example of a MLmodel file created for a computer vision model trained with **fastai** may look like:

```python
artifact_path: classifier
flavors:
  fastai:
    data: model.fastai
    fastai_version: 2.4.1
  python_function:
    data: model.fastai
    env: conda.yaml
    loader_module: mlflow.fastai
    python_version: 3.8.12
model_uuid: e694c68eba484299976b06ab9058f636
run_id: e13da8ac-b1e6-45d4-a9b2-6a0a5cfac537
signature:
  inputs: '[{"type": "tensor",
             "tensor-spec": 
                 {"dtype": "uint8", "shape": [-1, 300, 300, 3]}
           }]'
  outputs: '[{"type": "tensor", 
              "tensor-spec": 
                 {"dtype": "float32", "shape": [-1,2]}
            }]'
```

**Python function** flavor is the default model interface for models created from an MLflow run. Any MLflow python model can be loaded as a _**python_function**_ model, which allows for workflows like deployment to work with any python model regardless of which framework was used to produce the model. This interoperability is immensely powerful as it reduces the time to operationalize in multiple environments.
_[Source: https://learn.microsoft.com/en-us/training/modules/register-mlflow-model-azure-machine-learning/3-understand-mlflow-model-format]_

Apart from **flavors**, the **MLmodel** file also contains **signatures** that serve as data contracts between the model and the server running your model.

There are two types of signatures:
* **Column-based**: used for tabular data with a _**pandas.Dataframe**_ as inputs.
* **Tensor-based**: used for n-dimensional arrays or tensors (often used for unstructured data like text or images), with _**numpy.ndarray**_ as inputs.

As the **MLmodel** file is created when you register the model, the signature also is created when you register the model. When you enable MLflow's autologging, the signature is inferred in the best effort way. If you want the signature to be different, you need to manually log the model.

The signature's inputs and outputs are important when deploying your model. When you use Azure Machine Learning's no-code deployment for MLflow models, the inputs and outputs set in the signature will be enforced. In other words, when you send data to a deployed MLflow model, the expected inputs and outputs need to match the schema as defined in the signature.

### Register an MLflow model
The model registry makes it easy to organize and keep track of your trained models. When you register a model, you store and version your model in the workspace.

Registered models are identified by name and version. Each time you register a model with the same name as an existing one, the registry increments the version. You can also add more metadata tags to more easily search for a specific model.

There are three types of models you can register:
* **MLflow**: Model trained and tracked with MLflow. Recommended for standard use cases.
* **Custom**: Model type with a custom standard not currently supported by Azure Machine Learning.
* **Triton:** Model type for deep learning workloads. Commonly used for TensorFlow and PyTorch model deployments.

To register an MLflow model, you can use the studio, the Azure CLI, or the Python SDK.

If using the Python SDK: To train the model, you can submit a training script as a command job by using the following code:
```python
from azure.ai.ml import command

# configure job

job = command(
    code="./src",
    command="python train-model-signature.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="diabetes-train-signature",
    experiment_name="diabetes-training"
    )

# submit job
returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```

Once the job is completed and the model is trained, use the job name to find the job run and register the model from its outputs.

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

job_name = returned_job.name

run_model = Model(
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
    name="mlflow-diabetes",
    description="Model created from run.",
    type=AssetTypes.MLFLOW_MODEL,
)
# Uncomment after adding required details above
ml_client.models.create_or_update(run_model)
```

All registered models are listed in the **Models** page of the Azure Machine Learning studio. The registered model includes the model's output directory. When you log and register an MLflow model, you can find the _MLmodel_ file in the artifacts of the registered model.

## Create a responsible AI dashboard to evaluate your models
Source: https://learn.microsoft.com/en-us/training/modules/manage-compare-models-azure-machine-learning/

Jupyter Notebook: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/10/Create%20Responsible%20AI%20dashboard.ipynb

## Deploy model to managed endpoint (real-time predictions)
Jupyter notebook: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/11/Deploy%20to%20online%20endpoint.ipynb

https://learn.microsoft.com/en-us/training/modules/deploy-model-managed-online-endpoint/

To get real-time predictions, you can deploy a model to an HTTPs endpoint. Any data you send to the endpoint will serve as the input for the scoring script hosted on the endpoint. The scoring script loads the trained model to predict the label for the new input data, which is also called **inferencing**. The label is then part of the output that's returned.

Within Azure Machine Learning, there are two types of online endpoints:
* **Managed online endpoints**: Azure Machine Learning manages all the underlying infrastructure.
* **Kubernetes online endpoints**: Users manage the Kubernetes cluster which provides the necessary infrastructure.

Example of creating a managed online endpoint:
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="endpoint-example", # Must be unique in the Azure region.
    description="Online endpoint",
    auth_mode="key", 
    # auth_mode: Use 'key' for key-based authentication. Use 'aml_token' for Azure Machine Learning token-based authentication.
)

ml_client.begin_create_or_update(endpoint).result()
```

### Deploy model
After you create an endpoint in the Azure Machine Learning workspace, you can deploy a model to that endpoint. Four things to specify:

* **Model assets** like the model pickle file, or a registered model in the Azure Machine Learning workspace.
* **Scoring script** that loads the model.
* **Environment** which lists all necessary packages that need to be installed on the compute of the endpoint.
* **Compute configuration** including the needed compute size and scale settings to ensure you can handle the amount of requests the endpoint will receive.


### Blue/green deployment
Blue/green deployment allows for multiple models to be deployed to an endpoint. You can decide how much traffic to forward to each deployed model. This way, you can switch to a new version of the model without interrupting service to the consumer.

### Deploy an MLflow model to an endpoint
The easiest way to deploy a model to an online endpoint is to use an MLflow model. Azure Machine Learning will automatically generate the scoring script and environment for MLflow models.

To deploy an MLflow model, you must have model files stored on a local path or with a registered model. You can log model files when training a model by using MLflow tracking.

In this example, we're taking the model files from a local path. The files are all stored in a local folder called _model_. The folder must include the _MLmodel_ file, which describes how the model can be loaded and used.

```python
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# create a blue deployment
model = Model(
    path="./model",
    type=AssetTypes.MLFLOW_MODEL,
    description="my sample mlflow model",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
```

To route traffic to a specific deployment, use the following code:
```python
# blue deployment takes 100% traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
```

To deploy a model without using the MLflow model format, you'll need to create the scoring script and define the environment necessary during inferencing.

### Create the scoring script
The scoring script needs to include two functions:
* **init()**: Called when the service is initialized.
* **run()**: Called when new data is submitted to the service.

The **init** function is called when the deployment is created or updated, to load and cache the model from the model registry. The **run** function is called for every time the endpoint is invoked, to generate predictions from the input data. The following example Python script shows this pattern:

```python
import json
import joblib
import numpy as np
import os

# called when the deployment is created or updated
def init():
    global model
    # get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    # get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # get a prediction from the model
    predictions = model.predict(data)
    # return the predictions as any JSON serializable format
    return predictions.tolist()
```

### Create environment
You can create an environment with a Docker image with Conda dependencies, or with a Dockerfile.

To create an environment using a base Docker image, you can define the Conda dependencies in a **conda.yml** file:
```yml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
```

Then, to create the environment, run the following code:
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
```

### Create the deployment
When you have your model files, scoring script, and environment, you can create the deployment.
```python
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

model = Model(path="./model",

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    environment="deployment-environment",
    code_configuration=CodeConfiguration(
        code="./src", scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
```

### Test managed online endpoints
You can test the endpoint either in AML studio or via python SDK.

E.g. Test data (json)
```json
{
  "data":[
      [0.1,2.3,4.1,2.0], // 1st case
      [0.2,1.8,3.9,2.1],  // 2nd case,
      ...
  ]
}
```

The response from the deployed model is a JSON collection with a prediction for each case that was submitted in the data.

```python
# test the blue deployment with some sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="sample-data.json",
)

if response[1]=='1':
    print("Yes")
else:
    print ("No")
```

## Deploy model to a batch endpoint
Jupyter: https://github.com/MicrosoftLearning/mslearn-azure-ml/blob/main/Labs/11/Deploy%20to%20batch%20endpoint.ipynb

https://learn.microsoft.com/en-us/training/modules/deploy-model-batch-endpoint/

In machine learning, **batch inferencing** is used to asynchronously apply a predictive model to multiple cases and write the results to a file or database.

To create a batch endpoint:
```python
# create a batch endpoint
endpoint = BatchEndpoint(
    name="endpoint-example",
    description="A batch endpoint",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint)
```

You can deploy multiple models to a batch endpoint. Whenever you call the batch endpoint, which triggers a batch scoring job, the **default deployment** will be used unless specified otherwise.

The ideal compute to use for batch deployments is the Azure Machine Learning compute cluster. If you want the batch scoring job to process the new data in parallel batches, you need to provision a compute cluster with more than one maximum instances.

```python
from azure.ai.ml.entities import AmlCompute

cpu_cluster = AmlCompute(
    name="aml-cluster",
    type="amlcompute",
    size="STANDARD_DS11_V2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
    tier="Dedicated",
)

cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)
```

### Deploy MLflow model to a batch endpoint
An easy way to deploy a model to a batch endpoint is to use an MLflow model. Azure Machine Learning will automatically generate the scoring script and environment for MLflow models.

To register an MLflow model with the Python SDK, you can use the following code:
```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_name = 'mlflow-model'
model = ml_client.models.create_or_update(
    Model(name=model_name, path='./model', type=AssetTypes.MLFLOW_MODEL)
)
```

To deploy an MLflow model to a batch endpoint, you can use the following code:
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    instance_count=2,
    max_concurrency_per_instance=2, # Maximum number of parallel scoring script runs per compute node.
    mini_batch_size=2, # Number of files passed per scoring script run.
    output_action=BatchDeploymentOutputAction.APPEND_ROW, # or SUMMARY_ONLY
    output_file_name="predictions.csv", # File to which predictions will be appended, if you choose 'append_row' for 'output_action'.
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)
```

### Deploy a custom model to a batch endpoint
If you want to deploy a model to a batch endpoint without using the MLflow model format, you need to create the scoring script and environment.

The scoring script must include two functions:
* **init()**: Called once at the beginning of the process, so use for any costly or common preparation like loading the model.
* **run()**: Called for each mini batch to perform the scoring. Should return a pandas DataFrame or an array/list.

```python
import os
import mlflow
import pandas as pd


def init():
    global model

    # get the path to the registered model file and load it
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.pyfunc.load(model_path)


def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        pred = model.predict(data)

        df = pd.DataFrame(pred, columns=["predictions"])
        df["file"] = os.path.basename(file_path)
        resultList.extend(df.values)

    return resultList
```

Your deployment requires an execution environment in which to run the scoring script. Any dependency your code requires should be included in the environment.

You can create an environment with a Docker image with Conda dependencies, or with a Dockerfile.

To create an environment using a base Docker image, you can define the Conda dependencies in a _conda.yaml_ file:
```yml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pandas
  - pip
  - pip:
      - azureml-core # required for batch deployments to work.
      - mlflow
```

Then, to create the environment, run the following code:
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda-env.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
```

Finally, you can configure and create the deployment with the BatchDeployment class.
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    code_path="./code",
    scoring_script="score.py",
    environment=env,
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size=2,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)
```

### Trigger the batch scoring job
To prepare data for batch predictions, you can register a folder as a data asset in the Azure Machine Learning workspace.

You can then use the registered data asset as input when invoking the batch endpoint with the Python SDK:
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

input = Input(type=AssetTypes.URI_FOLDER, path="azureml:new-data:1")

job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint.name, 
    input=input)
```

When you invoke a batch endpoint, you trigger an Azure Machine Learning _pipeline job_. 

You can monitor the run of the pipeline job in the Azure Machine Learning studio. All jobs that are triggered by invoking the batch endpoint will show in the Jobs tab of the batch endpoint.

The predictions will be stored in the default datastore.










