import torch
from torch import nn


## CNN
class ResidualBlock(nn.Module):
    
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
            
    def forward(self,x):
        
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

    
class ForkCNN(nn.Module):
    
    def __init__(self, nFeatures, BatchSize, GPUs=1):
        
        self.features = nFeatures
        self.batch = BatchSize
        self.GPUs = GPUs
        
        super(ForkCNN, self).__init__()
        
        ### ResNet34
        self.resnet34 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(3,2),
            ResidualBlock(64,64),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2),
            
            ResidualBlock(64,128),
            ResidualBlock(128,128),
            ResidualBlock(128,128),
            ResidualBlock(128,128,2),
            
            ResidualBlock(128,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256,2),
            
            ResidualBlock(256,512),
            ResidualBlock(512,512),
            ResidualBlock(512,512,2)
        )

        
        self.avgpool = nn.AvgPool2d(2)
        
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32,16,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        
        ### Fully-connected layers
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(1296, 512), # 1296 may vary with different image size
            nn.Linear(512, 128),
            nn.Linear(128, self.features),
        )

    
    def forward(self, x, y):
        
        x = self.resnet34(x)
        x = self.avgpool(x)
        
        #y = self.resnet18(y)
        y = self.cnn_layers(y)
        #y = self.avgpool(y)
        
        # Flatten
        x = x.view(int(self.batch/self.GPUs),-1)
        y = y.view(int(self.batch/self.GPUs),-1)
        # print(x.size())
        # print(y.size())
        
        # Concatenation
        z = torch.cat((x, y), -1)
        #z = z.view(-1)
        #print(z.size())
        z = self.fully_connected_layer(z)
        
        return z

    
## NN calibration
class CaliNN(nn.Module):
    
    def __init__(self):
        super(CaliNN, self).__init__()
        
        self.main_net = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(4,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,1),
        )

    
    def forward(self, x):
        
        x = self.main_net(x)
        return x
    
    


