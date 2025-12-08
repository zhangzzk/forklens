import torch
from torch import nn


## CNN
class ResidualBlock(nn.Module):
    """
    A residual block for ResNet architecture.
    """
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        """
        Initialize the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolution.
            kernel_size (int): Kernel size for the convolution.
            padding (int): Padding for the convolution.
            bias (bool): Whether to use bias in convolution.
        """
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
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

    
class ForkCNN(nn.Module):
    """
    ForkCNN architecture for galaxy shape measurement.
    
    This network takes two inputs (galaxy image and PSF image) and processes them
    through separate branches before combining them for the final prediction.
    """
    
    def __init__(self, nFeatures, BatchSize, GPUs=1, gal_shape=(128, 128), psf_shape=(48, 48)):
        """
        Initialize the ForkCNN.

        Args:
            nFeatures (int): Number of output features to predict.
            BatchSize (int): Batch size.
            GPUs (int): Number of GPUs to use.
        """
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

        # ResNet18
        self.resnet18 = nn.Sequential(
            # Initial Convolution 
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(3, 2),

            # Layer 1: 64 channels [2 Blocks]
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            
            # Layer 2: 128 channels [2 Blocks]
            # Standard ResNet typically downsamples at the START of the new layer
            ResidualBlock(64, 128, 2), 
            ResidualBlock(128, 128),
            
            # Layer 3: 256 channels [2 Blocks]
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256),
            
            # Layer 4: 512 channels [2 Blocks]
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512)
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
        
        # Calculate input size dynamically
        with torch.no_grad():
            # Use batch size 2 to avoid BatchNorm errors with small feature maps
            dummy_x = torch.zeros(2, 1, gal_shape[0], gal_shape[1])
            dummy_y = torch.zeros(2, 1, psf_shape[0], psf_shape[1])
            
            # Temporarily set to eval mode to avoid updating running stats
            self.resnet34.eval()
            self.cnn_layers.eval()
            
            out_x = self.resnet34(dummy_x)
            out_x = self.avgpool(out_x)
            
            out_y = self.cnn_layers(dummy_y)
            
            flat_size = out_x.view(2, -1).shape[1] + out_y.view(2, -1).shape[1]
            
            # Restore training mode
            self.resnet34.train()
            self.cnn_layers.train()

        ### Fully-connected layers
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(flat_size, 512), 
            nn.Linear(512, 128),
            nn.Linear(128, 64),
            nn.Linear(64, self.features),
        )

    
    def forward(self, x, y):
        """
        Forward pass of the ForkCNN.

        Args:
            x (torch.Tensor): Galaxy image tensor.
            y (torch.Tensor): PSF image tensor.

        Returns:
            torch.Tensor: Predicted features.
        """
        x = self.resnet34(x)
        x = self.avgpool(x)
        
        # y = self.resnet34(y)
        y = self.cnn_layers(y)
        # y = self.avgpool(y)
        
        # Flatten
        x = x.view(x.size(0),-1)
        y = y.view(y.size(0),-1)
        # print(x.size())
        # print(y.size())
        
        # Concatenation
        z = torch.cat((x, y), -1)
        # z = z.view(-1)
        # print(z.size())
        z = self.fully_connected_layer(z)
        
        return z

    
## NN calibration
class CaliNN(nn.Module):
    """
    Neural Network for calibration.
    
    A simple fully connected network to calibrate the predictions.
    """
    
    def __init__(self, in_features=4, out_features=1, hidden_features=32):
        """
        Initialize the CaliNN.

        Args:
            in_features (int): Number of input features. Default: 4
            out_features (int): Number of output features. Default: 1
            hidden_features (int): Number of hidden units. Default: 32
        """
        super(CaliNN, self).__init__()
        
        self.main_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.Linear(hidden_features, out_features),
        )

    
    def forward(self, x):
        """
        Forward pass of the CaliNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Calibrated output.
        """
        x = self.main_net(x)
        return x
    
    


