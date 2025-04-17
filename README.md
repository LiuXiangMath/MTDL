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


# Reproduce our results
To reproduce our results, you need to first generate the decomposed images, and then use them to train and evaluate our model. The default parameters are configured for the RetinaMNIST dataset.
1. Run the **prepare_decomposition.py** script
```linux
python ./code/prepare_decomposition.py
```
This will create a folder named "retinamnist" that contains all the decomposed images for this dataset

2. Run the **main.py** script
```linux
python ./code/main.py
```
This will perform training and evaluation on the decomposed images

3. To test other datasets, you can modify the parameters in the above two scripts accordingly.
