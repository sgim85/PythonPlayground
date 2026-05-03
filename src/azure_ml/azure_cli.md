Remove all versions of the cli extension (v1 & v2)
```
az extension remove -n azure-cli-ml
az extension remove -n ml
```

Install AML CLI extension v2
```
az extension add -n ml -y
```

Create a resource group
```
az group create --name "rg-dp100-labs" --location "eastus"
```

Create a workspace
```
az ml workspace create --name "mlw-dp100-labs60413220" -g "rg-dp100-labs"
```

Create a compute instance to enable training of a ml model.
```
az ml compute create --name "ci60413220" --size STANDARD_DS11_V2 --type ComputeInstance -w mlw-dp100-labs60413220 -g rg-dp100-labs
```

Create a compute cluster. Preferred over compute instance in prod when training ml models. Automatically resizes to meet job demands, then falls back to 0 when no longer needed to save costs.
```
az ml compute create --name "aml-cluster" --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute -w mlw-dp100-labs60413220 -g rg-dp100-labs
```

 NOTE: ml command options to create an asset or resource can be placed in a yaml file to simplify automation and readability. E.g.
 
 Yaml file:
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json 
name: aml-cluster
type: amlcompute
size: STANDARD_DS3_v2
min_instances: 0
max_instances: 5
```

Command referencing yaml file:

 ```
 az ml compute create --file compute.yml --resource-group my-resource-group --workspace-name my-workspace
 ```