import torch.nn as nn
import torch

class CustomBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, stride=1, downsample=None):
        super().__init__()
        
        self.First_Conv = nn.Conv2d(inplanes, filters, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.Batch_Norm_1 = nn.BatchNorm2d(filters)
        self.Relu1 = nn.ReLU(inplace=True)
        self.Second_Conv = nn.Conv2d(filters, filters, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.Batch_Norm_2 = nn.BatchNorm2d(filters)
        
        self.downsample = downsample
        self.stride = stride
        
        # Normal addition is replaced with Skip addition for quantization purposes
        self.skip_addition = nn.quantized.FloatFunctional()
        
        self.Relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # 2 Convolutional layers are stacked together
        
        z = self.First_Conv(x)
        z = self.Batch_Norm_1(z)
        z = self.Relu1(z)

        z = self.Second_Conv(z)
        z = self.Batch_Norm_2(z)

        # Downsample the input if the size of the input and the output is different
        
        if self.downsample is not None:
            identity = self.downsample(x)

        #Skip connection (output = F(x) + x) 
        
        z = self.skip_addition.add(identity, z)
        z = self.Relu2(z)

        return z
    
class ResNetCustom(nn.Module):
    def __init__(self, block_type , layers_list, num_classes):
        
        super().__init__()

        self.inplanes = 64
        
        self.First_Conv = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.Batch_Norm_1 = nn.BatchNorm2d(self.inplanes)
        self.Relu = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Layer_1 = self.make_layers(block_type, 64, layers_list[0])
        self.Layer_2 = self.make_layers(block_type, 128, layers_list[1], stride=2)
        self.Layer_3 = self.make_layers(block_type, 256, layers_list[2], stride=2)
        self.Layer_4 = self.make_layers(block_type, 512, layers_list[3], stride=2)

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.Fully_Connected_1 = nn.Linear(512, num_classes)
        
    def make_layers(self, block_type, filters, num_blocks, stride =1):
        downsample = None
    
        if stride != 1 or self.inplanes != filters:
            downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, filters, 1, stride, bias=False),
            nn.BatchNorm2d(filters),
            )
        
        layers = []
        layers.append(block_type(self.inplanes,filters, stride, downsample))
        self.inplanes = filters
        for _ in range(1, num_blocks):
            layers.append(block_type(self.inplanes,filters))
            
        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        # First Conv Layer
        
        z = self.First_Conv(x)
        z = self.Batch_Norm_1(z)
        z = self.Relu(z)
        z = self.MaxPool(z)
        
        # Collection of Basic Blocks which implements skip connection and each basic block has 2 convolutional layers
        
        z = self.Layer_1(z)          
        z = self.Layer_2(z)         
        z = self.Layer_3(z)         
        z = self.Layer_4(z)
    
        # Here we have an average pooling layer + flattening + fully connected layer
        
        z = self.AvgPool(z)         
        z = torch.flatten(z, 1)     
        z = self.Fully_Connected_1(z)
        
        
        return z
    
    
def ResNet18():
        layers=[2, 2, 2, 2]
        return(ResNetCustom(CustomBasicBlock, layers, 10))
    
def ResNet22():
    layers=[2, 2, 4, 2]
    return(ResNetCustom(CustomBasicBlock, layers, 10))

def ResNet26():
    layers=[2, 4, 4, 2]
    return(ResNetCustom(CustomBasicBlock, layers, 10))

def ResNet30():
    layers=[3, 4, 4, 3]
    return(ResNetCustom(CustomBasicBlock, layers, 10))

def ResNet34():
    layers=[3, 4, 6, 3]
    return(ResNetCustom(CustomBasicBlock, layers, 10))

