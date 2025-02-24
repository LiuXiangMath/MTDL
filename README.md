Manifold Topological Deep Learning (MTDL)
====

    This manual is for the code implementation of paper "Manifold Topological Deep Learning for Biomedical Data"
    
****

# Prerequisites
- numpy 1.26.4
- h5py 3.11.0
- medmnist 3.0.2
- scipy 1.11.4
- python 3.11.7
- pytorch 2.4.1
- pytorch-cuda 11.8



# MTDL model architecture
![folder structure](picture/model.png) 


# How to Use
```linux
# prepare the decomposed image
python ./code/prepare_decomposition.py

# train model
python ./code/main.py
```
