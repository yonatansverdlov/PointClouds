import torch
import numpy as np
from torch_geometric.data import Data
from itertools import combinations

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_quadratic_pc(eps):
    """
    Generate two independent PyTorch Geometric Data objects for two point clouds, X and Y.

    Args:
        eps (float): Perturbation value.

    Returns:
        tuple: Two PyTorch Geometric Data objects, one for X and one for Y.
    """
    # Define X
    X = torch.tensor([[2.5, 0], [-2, 0], [-1, 0], [0.5, 0]],dtype=torch.float32)
    num_nodes_X = X.shape[0]
    edge_index_X = torch.tensor(list(combinations(range(num_nodes_X), 2)), dtype=torch.long).t()
    edge_attr_X = torch.tensor([
        np.linalg.norm(X[i] - X[j]) for i, j in edge_index_X.t()]).view(-1, 1)
    data_X = Data(edge_index=edge_index_X, edge_attr=edge_attr_X, pos = X)

    # Define Y with perturbation
    Y = torch.tensor([[2.5, 0], [-2, 0], [-1, eps], [0.5, -eps]])
    num_nodes_Y = Y.shape[0]
    edge_index_Y = torch.tensor(list(combinations(range(num_nodes_Y), 2)), dtype=torch.long).t()
    edge_attr_Y = torch.tensor([
        np.linalg.norm(Y[i] - Y[j]) for i, j in edge_index_Y.t()
    ]).view(-1, 1)
    data_Y = Data(edge_index=edge_index_Y, edge_attr=edge_attr_Y, pos = Y)

    return data_X.to(device), data_Y.to(device)
