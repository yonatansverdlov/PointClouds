
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils.EGNN as eg
from gmmPointCloudSim import genGmmPC
from sinkhorn_pointcloud import sinkhorn_loss
import time
import logging
from utils.GramNet import BiLipGram

logging.basicConfig(
    filename='egnnSinkhorn.log',
    filemode='w',  # or 'a' to append
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Dataset ===
class PointCloudMatchingDataset(Dataset):
    def __init__(self, dom,n_samples,nPts,perturb,nFactors = 1,device='cpu'):

        self.samples = []
        self.rng = np.random.default_rng(42)
        k = perturb[0]
        max_noise = perturb[1]

        for i in range(n_samples):
            # Compute current noise level
            stage = i // k
            num_stages = n_samples // k
            gamma = (stage + 1) / num_stages * max_noise

            # Generate GMM point cloud
            weights = np.ones(nFactors) / nFactors
            pc = genGmmPC(dom, nPts, weights, rng=self.rng)

            # Center and perturb
            X = pc.centerPC()
            Y, _ = X.addNoise(gamma, rng=self.rng)

            # Apply permutation to Y
            perm = torch.randperm(nPts)
            Y = Y.permutePT(perm)

            # Store tensors
            self.samples.append((
                torch.tensor(X.getPC(), dtype=torch.float32, device=device),
                torch.tensor(Y.getPC(), dtype=torch.float32, device=device),
                perm.to(device)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def randPT(self, dom, nFactors, nPts, device, perturb, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)

        # if isinstance(nFactors, tuple) or isinstance(nFactors, list):
        #     nFactors = rng.integers(nFactors[0], nFactors[1] + 1)

        weights = np.ones(nFactors) / nFactors
        pc = genGmmPC(dom, nPts, weights, rng)
        X = pc.getPC()
        X = X - np.mean(X, axis=0)

        if perturb[0] == 'pointwise':
            magnitude = perturb[1]
            if isinstance(magnitude, (tuple, list)):
                magnitude = rng.uniform(magnitude[0], magnitude[1])
            X_pert, _ = pc.addNoise(magnitude, rng)

        elif perturb[0] == 'scaling':
            scale = perturb[1]
            if isinstance(scale, (tuple, list)):
                scale = rng.uniform(scale[0], scale[1])
            X_pert, _ = pc.scalePC(scale, rng)

        elif perturb[0] == 'mixed':
            scale_rng = perturb[1]['scale']
            noise_mag = perturb[1]['noise']

            scale = rng.uniform(scale_rng[0], scale_rng[1]) if isinstance(scale_rng, (tuple, list)) else scale_rng
            noise = rng.uniform(0, noise_mag) if isinstance(noise_mag, (int, float)) else rng.uniform(*noise_mag)

            X_scaled, _ = pc.scalePC(scale, rng)
            # pc.pc = X_scaled
            # X_pert, _ = pc.addNoise(noise, rng)
            X_pert, _ = X_scaled.addNoise(noise, rng)
            # pc.pc = X_pert

        else:
            raise ValueError("Unknown perturbation type")

        # Rigid motion
        X_pert = X_pert.apply_random_rigid_motion(rng)

        # Permute
        perm = torch.randperm(X.shape[0])
        # X_pert = X_pert[perm.cpu().numpy()]
        X_pert = X_pert.permutePT(perm)
        X_pert = X_pert.centerPC()

        Xtnsr = torch.tensor(X.copy(), device=device, dtype=torch.float32)
        Xtnsr_pert = torch.tensor(X_pert.getPC(), device=device, dtype=torch.float32)

        return Xtnsr, Xtnsr_pert, perm

def permVec2Mat(perm,device):
    n = perm.size(0)
    eye = torch.eye(n).to(device=device)
    P = eye[perm].contiguous()  # (n, n)
    return (P/n).detach() #keops.LazyTensor(P,axis= 0)# P.view(n, 1, n, 1)

import torch
from torch.utils.data import DataLoader

def evaluate_model(model, test_loader,edges, edge_attr, device,sinkhorn_eps=0.01,nIter = 100):
    """
    Run model on test data and compute average accuracy.

    Args:
        model: trained EGNN+Sinkhorn model
        test_loader: DataLoader yielding (X, Y, perm)
        device: torch device ('cuda' or 'cpu')

    Returns:
        accuracy: mean matching accuracy over test set
    """
    model.eval()
    total_correct = 0
    total_points = 0

    with torch.no_grad():
        for X, Y, perm in test_loader:
            X, Y, perm = X.to(device), Y.to(device), perm.to(device)
            n_points = X.size(1)
            h_X = torch.ones((n_points, 1), device=device)
            h_Y = torch.ones((n_points, 1), device=device)
            # Forward pass: model outputs [N, N] soft permutation matrix
            z_X = model( X.squeeze())
            z_Y = model(Y.squeeze())
            P,_ = sinkhorn_loss(z_X, z_Y, sinkhorn_eps, n_points, nIter,device = device) # shape: [batch_size, N, N] or [N, N]

            # If batch size is 1, squeeze
            if P.dim() == 3:
                P = P.squeeze(0)

            # Predict permutation (row-wise argmax)
            pred_perm = P.T.argmax(dim=1)  # predicted: index in Y for each point in X

            # Compare to ground truth permutation
            correct = (pred_perm == perm).sum().item() #100*(pred_perm == perm).sum().item()/perm.size(1)
            total_correct += correct
            total_points += n_points #1

    accuracy = total_correct / total_points
    return accuracy


def test(model, config):
    # === load model === 
    # Reconstruct model using the saved config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    n_samples = config['dataConfig']['n_train']
    n_points = config['dataConfig']['n_points']
    k = config['k']
    maxPerturb = config['dataConfig']['maxPerturb']
    nFactors = config['dataConfig']['nFactors']
    # === Test data ===
    test_dataset_size = int(n_samples)
    test_dataset = PointCloudMatchingDataset(dom,n_samples = test_dataset_size,nPts = n_points,perturb=(k,maxPerturb),nFactors = nFactors, device=device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === Load model ===
    edges, edge_attr = eg.get_edges_batch(n_points, 1,device=device)

    # === Evaluate === 
    acc = evaluate_model(model, test_loader,edges, edge_attr, device)
    print(f"Test accuracy is: {acc:.4f}")
    
if __name__ == "__main__":

    # === Config ===
    dataConfig = {
        'n_points' : 90, # number of points in each point cloud
        'n_train' : 1000,# number of samples in traning set
        'n_test_ratio' : 1.0, # test set will have n_test_ratio*n_train samples
        'nFactors' : 3, # Number of components in each Gaussian mixture point cloud
        'dom' : [[-1,1],[-1,1]],
        'maxPerturb' :  0.1 # Maximal pointwise perturbation
    }

    config = {
        'model_type': 'EGNN+Sinkhorn',
        'hidden_nf': 64,
        'out_node_nf' : 1,
        'n_layers': 2,
        'normalize' : False,
        'batch_size' : 16, 
        'sinkhorn_iters': 100,
        'sinkhorn_eps_start' : 0.2,
        'sinkhorn_eps_end' : 0.05,
        'device': 'cuda',
        'nEpochs': 10,
        'lr' : 5e-3, # learning rate 
        'factor' : 0.7,
        'dataConfig' : dataConfig
    }


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_points = dataConfig['n_points']
    sc = torch.tensor(n_points**(-0.5), dtype=torch.float32)
    # sinkhorn_eps = config['sinkhorn_eps']
    lr = config['lr']
    n_epochs = config['nEpochs']
    dataset_size = dataConfig['n_train'] 
    nFactors = dataConfig['nFactors']

    # === EGNN hyperparameters ===
    batch_size = config['batch_size']
    n_batch = dataset_size/batch_size

    # === Generate sythetic data ===
    dom = dataConfig['dom']
    maxPerturb =  dataConfig['maxPerturb']
    k = dataset_size/16
    config['k'] = k
    dataset = PointCloudMatchingDataset(dom,n_samples = dataset_size,nPts = n_points,perturb=(k,maxPerturb),nFactors = nFactors, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Construct EGNN ===
    edges, edge_attr = eg.get_edges_batch(n_points, 1,device=device)

    # === Initialize EGNN ===

    model = BiLipGram(d=2,n=n_points,dim_out=64,embd_type='equivariant').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'])

    h_X = torch.ones((n_points, 1), device=device)
    h_Y = torch.ones((n_points, 1), device=device)

    start_time = time.time()
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        batch_idx = 0

        sinkhorn_eps = config['sinkhorn_eps_start'] + (
            (config['sinkhorn_eps_end'] - config['sinkhorn_eps_start']) * epoch / (n_epochs - 1)
        )

        for X_batch, Y_batch, perm_batch in dataloader:
            optimizer.zero_grad()
            losses = []
            accs = []
            typs = []

            for X, Y, perm in zip(X_batch, Y_batch, perm_batch):
                # model = model.eval()
                # Forward step
                z_X = model(X)
                z_Y = model(Y)

                # Sinkhorn step
                P,_ = sinkhorn_loss(z_X, z_Y, sinkhorn_eps, n_points, config['sinkhorn_iters'],device = device)
                log_P = torch.log(n_points*P.T + 1e-08)

                loss = -log_P[torch.arange(P.size(0)), perm].mean()
                losses.append(loss)

                # Accuracy and typical P - for debugging log
                with torch.no_grad():
                    pred = P.T.argmax(dim=1)
                    acc = (pred == perm).float().mean()
                    typ = torch.exp(-loss)

                accs.append(acc)
                typs.append(typ)
            
            batch_loss = torch.stack(losses).mean()
            batch_acc = torch.stack(accs).mean()
            batch_typ = torch.stack(typs).mean()

            # Backward and optimize over current batch
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            accs.append(batch_acc.item())
            typs.append(batch_typ.item())
            
            print(f"  [Batch {batch_idx+1}] Loss: {batch_loss.item():.4f} | Acc: {batch_acc.item():.4f} | TypP: {batch_typ.item():.4f}")
            batch_idx += 1

        typ = torch.exp(torch.tensor(-total_loss/n_batch))
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/batch_size:.4f} | Typical P: {typ}")
        logging.info(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/n_batch:.4f} | Typical P: {typ}")
        scheduler.step(total_loss/batch_size)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    test(model, config)
    # Save trained model and input

    
