import torch
from torch import nn
class SiameseCNN(nn.Module):
 def __init__(self,embedding=64):
  super().__init__(); self.encoder=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(48,embedding)); self.head=nn.Sequential(nn.Linear(embedding*2,64),nn.ReLU(),nn.Linear(64,6))
 def forward(self,a,b):
  za,zb=self.encoder(a),self.encoder(b); o=self.head(torch.cat([torch.abs(za-zb),za*zb],1)); return {"neighbor_logit":o[:,:1],"orientation_logits":o[:,1:5],"compatibility":torch.sigmoid(o[:,5:6])}

