import torch
import torch.nn as nn


class MultiHeadConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,head):
        super(MultiHeadConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel*head,
                               groups=head,kernel_size=3,stride=1,padding=1,bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel*head, out_channels=out_channel, 
                               kernel_size=1,stride=1,bias=False)
    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        return x
        
class FeedForwardConv2d(nn.Module):
    def __init__(self,in_size,h_size,out_size,group):
        super(FeedForwardConv2d,self).__init__()
        self.lin = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=h_size, kernel_size=1,
                      groups=group,stride=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_size, out_channels=out_size, kernel_size=1,
                      groups=group,stride=1,bias=False),
            )
        
    def forward(self,x):
        return self.lin(x)
    

class TransConv2d(nn.Module):
    def __init__(self,channel,head,group):
        super(TransConv2d,self).__init__()
        self.mhc = MultiHeadConv2d(channel, channel, head)
        self.norm1 = nn.BatchNorm2d(channel)
        self.ff = FeedForwardConv2d(channel,channel*2,channel,group)
        self.norm2 = nn.BatchNorm2d(channel)
        
    def forward(self,x):
        x = x + self.mhc(x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x
    


class TransCNN2d(nn.Module):
    def __init__(self,in_channel,h_channel,head,group,layer_num,class_num):
        super(TransCNN2d,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, h_channel, kernel_size=3,
                      stride=1,padding=1,bias=False),
            nn.BatchNorm2d(h_channel),
            nn.ReLU(),
            )
        self.layers = nn.ModuleList([ nn.Sequential( TransConv2d(h_channel, head,group),nn.MaxPool2d(2) ) for i in range(layer_num) ])
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.final = nn.Sequential(
            nn.Linear(h_channel, h_channel*2),
            nn.ReLU(),
            nn.Linear(h_channel*2,h_channel),
            nn.ReLU(),
            nn.Linear(h_channel,class_num)
            
        )
    def forward(self,x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x
        



class MultiHeadConv3d(nn.Module):
    def __init__(self,in_channel,out_channel,head):
        super(MultiHeadConv3d,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel*head,
                               groups=head,kernel_size=3,stride=1,padding=1,bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channel*head, out_channels=out_channel, 
                               kernel_size=1,stride=1,bias=False)
    def forward(self,x):
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        return x
        
class FeedForwardConv3d(nn.Module):
    def __init__(self,in_size,h_size,out_size,group):
        super(FeedForwardConv3d,self).__init__()
        self.lin = nn.Sequential(
            nn.Conv3d(in_channels=in_size, out_channels=h_size, kernel_size=1,
                      groups=group,stride=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=h_size, out_channels=out_size, kernel_size=1,
                      groups=group,stride=1,bias=False),
            )
        
    def forward(self,x):
        return self.lin(x)
    

class TransConv3d(nn.Module):
    def __init__(self,channel,head,group):
        super(TransConv3d,self).__init__()
        self.mhc = MultiHeadConv3d(channel, channel, head)
        self.norm1 = nn.BatchNorm3d(channel)
        self.ff = FeedForwardConv3d(channel,channel*2,channel,group)
        self.norm2 = nn.BatchNorm3d(channel)
        
    def forward(self,x):
        x = x + self.mhc(x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x
    


class TransCNN3d(nn.Module):
    def __init__(self,in_channel,h_channel,head,group,layer_num,class_num):
        super(TransCNN3d,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, h_channel, kernel_size=3,
                      stride=1,padding=1,bias=False),
            nn.BatchNorm3d(h_channel),
            nn.ReLU(),
            )
        self.layers = nn.ModuleList([ nn.Sequential( TransConv3d(h_channel, head,group),nn.MaxPool3d(2) ) for i in range(layer_num) ])
        self.pool2 = nn.AdaptiveAvgPool3d((1,1,1))
        self.final = nn.Sequential(
            nn.Linear(h_channel, h_channel*2),
            nn.ReLU(),
            nn.Linear(h_channel*2,h_channel),
            nn.ReLU(),
            nn.Linear(h_channel,class_num)
            
        )
    def forward(self,x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x
        

