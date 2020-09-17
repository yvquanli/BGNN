# RGNN
Code for "Residual graph learning with directed message passing for molecular properties prediction"


### Requirements 

```
PyTorch >= 1.4.0
torch-gemetric >= 1.3.2
rdkit >= '2019.03.4'
```

### Usage example

**For qm9 dataset**
```sh
git clone https://github.com/yvquanli/rgnn
# download and unzip dataset from https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip
python run.py $task $gpu

```

**For alchemy dataset**
```sh
git clone https://github.com/yvquanli/rgnn
# download and unzip dataset from https://alchemy.tencent.com/
python run.py $task $gpu

```

