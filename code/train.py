import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import time
import numpy as np
from copy import deepcopy
import medmnist
from medmnist import INFO
from medmnist.evaluator import getACC, getAUC

from model import TransCNN2d, TransCNN3d
from data_2d import get_train_val_test_data_2d
from data_3d import get_train_val_test_data_3d

torch.set_default_dtype(torch.float32)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)







def train_and_test_model(model,train_loader,val_loader,test_loader,criterion,optimizer,scheduler,epoch,class_num,task,device,config):
    print('start training')
    if config['task'] == 'multi-label, binary-class':
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)
    for e in range(epoch):
        model.train()
        train_loss = 0.0
        train_true, train_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for batch_index, (data,target) in enumerate(train_loader):
            data = data.to(device)
            if config['task'] == 'multi-label, binary-class':
                target = target.to(torch.float32).to(device)
            else:
                target = torch.squeeze(target, 1).long().to(device)
            y_pred = model(data)
            optimizer.zero_grad()
            loss = criterion(y_pred,target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss = train_loss + loss.item()
            outputs = prediction(y_pred)
            train_true = torch.cat((train_true, deepcopy(target)), 0)
            train_pred = torch.cat((train_pred, deepcopy(outputs.detach())), 0)
        train_ACC = getACC(train_true.cpu().numpy(), train_pred.cpu().numpy(), task)
        train_AUC = getAUC(train_true.cpu().numpy(), train_pred.cpu().numpy(), task)

        
        val_true, val_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_index, (data,target) in enumerate(val_loader):
                data = data.to(device)
                if config['task'] == 'multi-label, binary-class':
                    target = target.to(torch.float32).to(device)
                else:
                    target = torch.squeeze(target, 1).long().to(device)
                y_pred = model(data)
                loss_now = criterion(y_pred,target)
                val_loss = val_loss + loss_now.item()
                outputs = prediction(y_pred)
                val_true = torch.cat((val_true, deepcopy(target)), 0)
                val_pred = torch.cat((val_pred, deepcopy(outputs)), 0)
        val_ACC = getACC(val_true.cpu().numpy(), val_pred.cpu().numpy(), task)
        val_AUC = getAUC(val_true.cpu().numpy(), val_pred.cpu().numpy(), task)
        
        print('epoch:',e+1,'train loss:',round(train_loss,2),'acc:',round(train_ACC*100,2),'auc:',round(train_AUC*100,2),'val acc:',round(val_ACC*100,2),'auc:',round(val_AUC*100,2))
        
        if e==epoch-1:
            test_true, test_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
            model.eval()
            with torch.no_grad():
                for batch_index, (data,target) in enumerate(test_loader):
                    data = data.to(device)
                    if config['task'] == 'multi-label, binary-class':
                        target = target.to(torch.float32).to(device)
                    else:
                        target = torch.squeeze(target, 1).long().to(device)
                    y_pred = model(data)
                    outputs = prediction(y_pred)
                    test_true = torch.cat((test_true, deepcopy(target)), 0)
                    test_pred = torch.cat((test_pred, deepcopy(outputs)), 0)
            test_ACC = getACC(test_true.cpu().numpy(), test_pred.cpu().numpy(), task)
            test_AUC = getAUC(test_true.cpu().numpy(), test_pred.cpu().numpy(), task)
            print('test ','acc:',round(test_ACC*100,2),'auc:',round(test_AUC*100,2))        
    
    





def train_model_2d(dataname='retina',img_size=224,batch=64,lr=1e-3,epoch=30,h_channel=72,head=4,layer_num=5,seed=42,device='cuda'):
    # random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    info = INFO[dataname]
    config = {
    'dataset':dataname, 'img_size':img_size, 'batch_size':batch, 'learning_rate':lr, 'epochs':epoch, 'h_channel':h_channel, 'head':head, 'layer_num':layer_num,
    'task': info['task'], 'class_num':len(info['label']), 'in_channel':info['n_channels']*2*3, 'seed':seed, 'group':info['n_channels'],
    }
    
    train_dataset,val_dataset,test_dataset = get_train_val_test_data_2d(config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=20, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=20, worker_init_fn=seed_worker, generator=g)
    
    
    
    
    model = TransCNN2d(in_channel=config['in_channel'],h_channel=config['h_channel'],head=config['head'],group=config['group'],layer_num=config['layer_num'],class_num=config['class_num'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], steps_per_epoch=len(train_loader), epochs=config['epochs'],pct_start=0.3)
    criterion = nn.CrossEntropyLoss()
    
    if config['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    
    print(config)
    print('train:',len(train_dataset),'test:',len(test_dataset),'val:',len(val_dataset))
    train_and_test_model(model,train_loader,val_loader,test_loader,criterion,optimizer,scheduler,epoch,config['class_num'],config['task'],device,config)
    
    
    


def train_model_3d(dataname='organmnist3d',img_size=64,batch=16,lr=1e-3,epoch=20,h_channel=64,head=4,layer_num=5,seed=42,device='cuda'):
    # random seed 42,23456789,98765432
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    info = INFO[dataname]
    config = {
    'dataset':dataname, 'img_size':img_size, 'batch_size':batch, 'learning_rate':lr, 'epochs':epoch, 'h_channel':h_channel, 'head':head, 'layer_num':layer_num,
    'task': info['task'], 'class_num':len(info['label']), 'in_channel':9, 'seed':seed, 'group':1,
    }
    
    train_dataset,val_dataset,test_dataset = get_train_val_test_data_3d(config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=20, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=20, worker_init_fn=seed_worker, generator=g)
    
    model = TransCNN3d(in_channel=config['in_channel'],h_channel=config['h_channel'],head=config['head'],group=config['group'],layer_num=config['layer_num'],class_num=config['class_num'])
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'], steps_per_epoch=len(train_loader), epochs=config['epochs'],pct_start=0.3)
    criterion = nn.CrossEntropyLoss()
    
    print(config)
    print('train:',len(train_dataset),'test:',len(test_dataset),'val:',len(val_dataset))
    train_and_test_model(model,train_loader,val_loader,test_loader,criterion,optimizer,scheduler,epoch,config['class_num'],config['task'],device,config)
    
    
    
    
    

    

