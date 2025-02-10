import numpy as np
import torch
from PIL import Image
from medmnist.dataset import MedMNIST2D,MedMNIST3D




class My_MedMNIST2D(MedMNIST2D):
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


class PathMNIST(My_MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(My_MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(My_MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(My_MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(My_MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(My_MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(My_MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(My_MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(My_MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(My_MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(My_MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(My_MedMNIST2D):
    flag = "organsmnist"
    
def get_2d_data_class(typ):
    if typ=='pneumoniamnist':
        return PneumoniaMNIST
    elif typ=='dermamnist':
        return DermaMNIST
    elif typ=='retinamnist':
        return RetinaMNIST
    elif typ=='breastmnist':
        return BreastMNIST
    elif typ=='bloodmnist':
        return BloodMNIST
    elif typ=='pathmnist':
        return PathMNIST
    elif typ=='octmnist':
        return OCTMNIST
    elif typ=='tissuemnist':
        return TissueMNIST
    elif typ=='organamnist':
        return OrganAMNIST
    elif typ=='organcmnist':
        return OrganCMNIST
    elif typ=='organsmnist':
        return OrganSMNIST
    elif typ=='chestmnist':
        return ChestMNIST




def get_2d_data(data_typ,size):
    DataClass = get_2d_data_class(data_typ)
    
    train_dataset = DataClass(split='train', download=True, size=size)
    val_dataset = DataClass(split='val', download=True, size=size)
    test_dataset = DataClass(split='test', download=True, size=size)
    
    train = [d for d in train_dataset]
    val   = [d for d in val_dataset]
    test  = [d for d in test_dataset]
    
    return train+test+val
    
    
    


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

        return img, target

    
        

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
    

def get_3d_data_class(typ):
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
    
    
def get_3d_data(data_typ,size):
    DataClass = get_3d_data_class(data_typ)
    
    train_dataset = DataClass(split='train', download=True, size=size)
    val_dataset = DataClass(split='val', download=True, size=size)
    test_dataset = DataClass(split='test', download=True, size=size)
    
    train = [d for d in train_dataset]
    val   = [d for d in val_dataset]
    test  = [d for d in test_dataset]
    
    return train+test+val












    
    
    
    