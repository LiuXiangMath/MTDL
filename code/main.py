from train import train_model_2d, train_model_3d
import torch

device = torch.device('cuda')




data_2d = [ 'retinamnist','pneumoniamnist','dermamnist','bloodmnist','organamnist','organcmnist','organsmnist','pathmnist','octmnist','tissuemnist','chestmnist']
data_3d = [ 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']
dataname  = data_2d[0]
epoch     = 30
layer_num = 5





if dataname in data_2d:
    train_model_2d(dataname=dataname,epoch=epoch,layer_num=layer_num,device=device)
elif dataname in data_3d:
    train_model_3d(dataname=dataname,epoch=epoch,layer_num=layer_num,device=device)

