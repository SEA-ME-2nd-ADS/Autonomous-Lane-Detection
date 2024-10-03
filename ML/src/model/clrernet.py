import torch
import torch.nn as nn
from torchvision.models import resnet

from .block import head, neck
from config import resnet18_neck


class CLRerNet(nn.Module):
    def __init__(self):
        super(CLRerNet, self).__init__()
        self.backbone = torch.compile(resnet.resnet18(weights= resnet.ResNet18_Weights.DEFAULT))
        self.neck = neck.CLRerNetFPN(in_channels = resnet18_neck['in_channels'], 
                                     out_channels = resnet18_neck['out_channels'], 
                                     num_outs = resnet18_neck['num_outs'])
        self.heads = head.CLRerHead()
        
        self.backbone_layers = [nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
                                , self.backbone.layer1
                                , self.backbone.layer2
                                , self.backbone.layer3]
    
    def get_lanes(self, output):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        fea = []
        for layer in self.backbone_layers:
            batch = layer(batch)
            fea.append(batch)
        
        fea = self.neck(fea)
        
        output = self.heads(fea)
        
        return output