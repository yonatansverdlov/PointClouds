import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from create_data import generate_quadratic_pc
from compute_ratio import compute_f_over_g_with_random_X_Y
class EGNNPosOnlyLayer(nn.Module):
    def __init__(self, hidden_feats=64, update_coords=True):
        super().__init__()
        self.update_coords = update_coords

        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_feats),
            nn.SiLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.SiLU()
        )

        self.coord_mlp = nn.Linear(hidden_feats, 1, bias=False)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.SiLU(),
            nn.Linear(hidden_feats, hidden_feats)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        pos_i, pos_j = pos[row], pos[col]
        rel_pos = pos_i - pos_j
        dist = rel_pos.norm(dim=-1, keepdim=True)  # Euclidean norm

        edge_feat = self.edge_mlp(dist)  # (E, hidden)

        # Message aggregation: sum edge messages to each node
        agg = torch.zeros(h.size(0), edge_feat.size(-1), device=h.device)
        agg.index_add_(0, row, edge_feat)

        if self.update_coords:
            coord_update = self.coord_mlp(edge_feat) * rel_pos / (dist + 1e-8)
            delta = torch.zeros_like(pos)
            delta.index_add_(0, row, coord_update)
            pos = pos + delta

        h = self.node_mlp(agg)
        return h, pos
class EGNN(nn.Module):
    def __init__(self, hidden_feats=64, out_feats=16, n_layers=3, update_coords=True, pooling='mean'):
        super().__init__()
        self.initial_h = nn.Parameter(torch.randn(1, hidden_feats))

        self.layers = nn.ModuleList([
            EGNNPosOnlyLayer(hidden_feats=hidden_feats, update_coords=update_coords)
            for _ in range(n_layers)
        ])

        self.readout = nn.Linear(hidden_feats, out_feats)
        self.pooling = pooling

    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index
        h = self.initial_h.expand(pos.size(0), -1)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        if self.pooling == 'mean':
            h = global_mean_pool(h, batch)
        elif self.pooling == 'sum':
            h = torch.zeros(batch.max().item() + 1, h.size(1), device=h.device)
            h.index_add_(0, batch, h)

        return self.readout(h)

device = 'cuda'

model = EGNN(hidden_feats=32, out_feats=8).to(device)
data_X, data_Y = generate_quadratic_pc(eps=0.01)

embedding_X = model(data_X)
embedding_Y = model(data_Y)
out_diff = torch.norm(embedding_X - embedding_Y)

metric1, metric2 = compute_f_over_g_with_random_X_Y(eps=1e-7) 
print(f'Diff 1: {out_diff/metric1}, Diff 2: {out_diff/metric2}')