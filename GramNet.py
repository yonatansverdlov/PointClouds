import torch
import torch.nn as nn
import numpy as np

class embed_vec_sort(nn.Module):
   # Calculates a permutation-invariant embedding of n vectors in R^d, using Nadav
   # and Ravina's sort embedding.
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1 
    def __init__(self, d, n, d_out = None):
        super().__init__()
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
      
        if d_out is None:
            d_out = 2*d*n+1

        self.d = d
        self.n = n
        self.d_out = d_out

        self.A = nn.Parameter(torch.randn([d, d_out], device=self.device))
        self.w = nn.Parameter(torch.randn([1, n, d_out], device=self.device))
        self.init()
        
    def init(self):
        # Initialize the parameters
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.w)

    def forward(self, input):
        prod = tr12( torch.tensordot( tr12(input), self.A, [[2], [0]] ) ) 
        [prod_sort, inds_sort] = torch.sort(prod, dim=2)
        out = torch.sum( prod_sort * tr12(self.w), dim=2)

        return out


def tr12(input):
    return torch.transpose(input, 1, 2)


import torch

def rm_diag(A):
    return A[~torch.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


import torch
import torch.nn as nn
# from gramnet import embed_vec_sort

def rm_diag(A):
    return A[~torch.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

def tr12(input):
    return torch.transpose(input, 1, 2)

class BiLipGram(nn.Module):

    def __init__(self, d, n,dim_out, embd_type = 'equivariant'):
        super().__init__()
        if torch.cuda.is_available():
            self.device='cuda'
        else:
            self.device='cpu'
        self.embd_type = embd_type
        self.d = d
        self.d_in = d+1
        self.n = n-1
        self.d_out=2*(d+1)*(n-1)+1 if dim_out is None else dim_out

        self.eqv_embd_map = embed_vec_sort(d=self.d_in,n=self.n,d_out=self.d_out)
        self.inv_embd_map = embed_vec_sort(d=self.d_out+1,n=n)

    def forward(self,X):

        # Gram coefficients
        G = X@X.T
        X_perp = X[:,[1,0]]
        X_perp[:,0] = -X_perp[:,0]
        G_perp = X_perp@X.T

        # Extract norms
        N = X.shape[0]
        norms = torch.sqrt(torch.diag(G))
        max_norm = torch.max(norms)
        norms_noSelf = rm_diag(norms.repeat(N,1))

        # Extract inner products and normalize by max norm
        inner_prods = rm_diag(G)/(max_norm+1e-8)
        inner_prods_perp = rm_diag(G_perp)/(max_norm+1e-8)

        # Construct vectors multisets
        V = torch.stack([inner_prods,inner_prods_perp,norms_noSelf],dim=0)

        # Compute embedding
        V = V.permute(1,0,2) # 1st is i, 2nd dim are vec index, 3rd dim are coeefs
        eqv_embd = torch.cat([norms.unsqueeze(-1),self.eqv_embd_map(V)],dim=1)

        if self.embd_type == 'equivariant':
            return eqv_embd
        else:
            # embd_map = self.embed_vec_sort(d=d_out+1,n=N,d_out = 2*feature_dim*N+1)
            return self.inv_embd_map(eqv_embd.permute(1,0).unsqueeze(0))




#model = BiLipGram(d=2, n=5, d_out=64)
#x = torch.rand(2,10)
#print(model(x,'equivariant'))