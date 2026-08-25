import torch
from torch import nn
class MessageLayer(nn.Module):
 def __init__(self,d,e):super().__init__();self.msg=nn.Sequential(nn.Linear(2*d+e,d),nn.ReLU(),nn.Linear(d,d));self.upd=nn.GRUCell(d,d)
 def forward(self,x,edge_index,edge_attr):
  src,dst=edge_index; m=self.msg(torch.cat([x[src],x[dst],edge_attr],1)); agg=torch.zeros_like(x).index_add_(0,dst,m);return self.upd(agg,x)
class JigsawGNN(nn.Module):
 def __init__(self,node_dim=32,edge_dim=16,hidden=64,layers=3):
  super().__init__();self.node=nn.Linear(node_dim,hidden);self.layers=nn.ModuleList([MessageLayer(hidden,edge_dim) for _ in range(layers)]);self.edge_head=nn.Sequential(nn.Linear(hidden*2+edge_dim,hidden),nn.ReLU(),nn.Linear(hidden,6))
 def forward(self,nodes,edge_index,edge_attr):
  x=torch.relu(self.node(nodes));
  for layer in self.layers:x=layer(x,edge_index,edge_attr)
  src,dst=edge_index;o=self.edge_head(torch.cat([x[src],x[dst],edge_attr],1));return {"neighbor_logit":o[:,:1],"orientation_logits":o[:,1:5],"compatibility":torch.sigmoid(o[:,5:6]),"node_embeddings":x}
