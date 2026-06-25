# pyright: reportMissingImports=false

import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

NUM_OF_LAYERS = 2

class GCN(torch.nn.Module):
    def __init__(self,num_node_features,num_hexagons):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.convs = ModuleList()
        self.convs.append(GraphConv(num_node_features, 512))
        for _ in range(NUM_OF_LAYERS):
            self.convs.append(GraphConv(512, 512))

        self.linHexagon = Linear(512, 512)
        self.linHexagon2 = Linear(512, num_hexagons)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
        x = self.convs[-1](x, edge_index)
    
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 4. Apply a final classifier
        
        x = F.dropout(x, p=0.5, training=self.training)    
        hexagon = self.linHexagon(x)
        hexagon = F.relu(hexagon)
        hexagon = self.linHexagon2(hexagon)
        return hexagon

