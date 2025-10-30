
from torch import nn
import torch

# -------------------------------
# E_GCL: Equivariant Graph Conv Layer (EGCL)
# Implements one EGCL block from the paper.
# Performs updates on both node features h and coordinates x, using message passing.
# -------------------------------
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2  # Concatenate source and target features

        self.residual = residual  # Whether to add residual connection to node features
        self.attention = attention  # Whether to modulate edge messages with learned attention
        self.normalize = normalize  # Whether to normalize coordinate updates
        self.coords_agg = coords_agg  # 'mean' or 'sum' aggregation for coordinate updates
        self.tanh = tanh  # Whether to bound coordinate MLP outputs using tanh
        self.epsilon = 1e-8  # For numerical stability

        edge_coords_nf = 1  # One scalar: squared distance ||x_i - x_j||^2

        # φ_e: Edge MLP to compute message from edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # φ_h: Node MLP to update node features based on aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # φ_x: Coordinate update function (outputs scalar weight for radial vector)
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # Optional learned attention weight on edges
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    # Computes edge features from source, target node features and distance
    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    # Updates node features using aggregated edge features
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    # Updates coordinates using weighted directional vectors and coordinate MLP
    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg # Changed from coord += agg  Eitan 09/06
        return coord

    # Computes relative squared distances and coordinate differences
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    # Full EGCL forward: computes updated h and x
    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr

# -------------------------------
# EGNN: Full network with multiple stacked EGCL layers
# -------------------------------
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''
        EGNN network composed of n_layers EGCLs
        Performs equivariant message passing over coordinates and node features.

        :param in_node_nf: Number of input node features (dimension of h)
        :param hidden_nf: Dimension of hidden features in EGCLs
        :param out_node_nf: Number of output features after final EGCL
        :param in_edge_nf: Optional number of input edge features
        :param device: 'cpu' or 'cuda'
        :param act_fn: Activation function (e.g. SiLU)
        :param n_layers: Number of EGCL layers
        :param residual: Use residual connections for node features
        :param attention: Use attention in edge updates (optional)
        :param normalize: Normalize coordinate differences in updates (optional)
        :param tanh: Use tanh in coordinate MLP output (optional)
        '''
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Initial linear embedding of input node features
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        # Final projection to output node feature dimension
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # Stack n_layers of EGCLs
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    # Forward pass through stacked EGCLs
    def forward(self, h, x, edges, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x

# -------------------------------
# Utility functions for segment reductions (edge aggregation)
# -------------------------------
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

# -------------------------------
# Utility: Build fully connected edge index
# -------------------------------
def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges

def get_edges_batch(n_nodes, batch_size,device):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1,device=device)
    edges = [torch.tensor(edges[0],dtype=torch.long,device= device), torch.tensor(edges[1],dtype=torch.long,device=device)]
    #torch.tensor(edges, dtype=torch.long, device=device)
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Dummy data for testing
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    h = torch.ones(batch_size * n_nodes, n_feat)  # Node features
    x = torch.ones(batch_size * n_nodes, x_dim)  # Coordinates
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)  # Fully connected graph

    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    h, x = egnn(h, x, edges, edge_attr)  # Forward pass
