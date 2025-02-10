import numpy as np
import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
from medmnist.dataset import MedMNIST3D






       
def load_decomposition_img(dataname,typ,index,label):
    train_test_num = {'organmnist3d':[971,610],'nodulemnist3d':[1158,310],'adrenalmnist3d':[1188,298],'fracturemnist3d':[1027,240],'vesselmnist3d':[1335,382],
                      'synapsemnist3d':[1230,352] }
    train_num,test_num = train_test_num[dataname]
    
    
    rgb_list = [0]
    size = 64
    if typ=='train':
        now_index = index
    elif typ=='test':
        now_index = index+train_num
    elif typ=='val':
        now_index = index+train_num+test_num
        
    filename = './' + dataname + '/' + str(now_index) + '.h5'
    with h5py.File(filename, 'r') as file:
        label1 = file['label'][:]
        #assert label1==label
        
        
        img_high = []
        for k in rgb_list:
            for typ in ['cur','div','har']: 
                for v in [1.1]:
                    name = str(now_index) + '-' + str(v) + '-' + str(k) + '-' + typ  # all, cur, div, har
                    img_now1 = np.round(file[name][:],8)*2.0
                    img_high.append(img_now1[:,1:1+size,1:1+size,1:1+size])
        
        #img_high = np.vstack(img_high)
        img_high = np.concatenate(img_high,axis=0)
        
        return torch.tensor(img_high,dtype=torch.get_default_dtype())
       
        



class My_MedMNIST3D(MedMNIST3D):
    def __getitem__(self, index):
        
        """
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img / 255.0] * (3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img2 = load_decomposition_img(self.flag,self.split,index,self.labels[index])
        return img2, target
        

class OrganMNIST3D(My_MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(My_MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(My_MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(My_MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(My_MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(My_MedMNIST3D):
    flag = "synapsemnist3d"


    
def get_data_class(typ):
    if typ=='organmnist3d':
        return OrganMNIST3D
    elif typ=='nodulemnist3d':
        return NoduleMNIST3D
    elif typ=='adrenalmnist3d':
        return AdrenalMNIST3D
    elif typ=='fracturemnist3d':
        return FractureMNIST3D
    elif typ=='vesselmnist3d':
        return VesselMNIST3D
    elif typ=='synapsemnist3d':
        return SynapseMNIST3D


class To3DTensor:
    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32)

def get_train_val_test_data_3d(config):
    DataClass = get_data_class(config['dataset'])
    
    data_transform = transforms.Compose([
            To3DTensor(),
        ])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
    val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
    test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=False, size=config['img_size'])
    
    return train_dataset,val_dataset,test_dataset
    
    
    
