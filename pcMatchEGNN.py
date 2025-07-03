
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import EGNN as eg
from gmmPointCloudSim import genGmmPC
from sinkhorn_pointcloud import sinkhorn_loss
import logging

logging.basicConfig(
    filename='egnnSinkhorn.log',
    filemode='w',  # or 'a' to append
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Dataset class for generating simulated 2D point clouds ===

class PointCloudMatchingDataset(Dataset):
    def __init__(self, dom,n_samples,nPts,perturb,nFactors = 1,device='cpu'):
        

        self.samples = []
        self.rng = np.random.default_rng(42)
        k = perturb[0] # Number of learning stages in curriculum learning
        max_noise = perturb[1] # Maximal noise level 

        for i in range(n_samples):
            # Compute current noise level - implmenets curriculum learning - noise grows on average with samples
            stage = i // k # Noise grows in k stages
            num_stages = n_samples // k
            gamma = (stage + 1) / num_stages * max_noise

            # Generate GMM point cloud
            weights = np.ones(nFactors) / nFactors # nFcators cetroids, all with equal marginal probabilites
            pc = genGmmPC(dom, nPts, weights, rng=self.rng) # Sample nPts points from Gaussian mixture defined by weights 

            # Center point cloud (X) and add independent pointwise perturbations (Y)
            X = pc.centerPC()
            X_pert, _ = X.addNoise(gamma, rng=self.rng)

            # Add rigid motion
            X_pert = X_pert.apply_random_rigid_motion(self.rng)

            # Apply random permutation 
            perm = torch.randperm(nPts)
            X_pert = X_pert.permutePT(perm)

            # Store tensors on GPU
            self.samples.append((
                torch.tensor(X.getPC(), dtype=torch.float32, device=device),
                torch.tensor(X_pert.getPC(), dtype=torch.float32, device=device),
                perm.to(device) # We do not differentiate w.r.t discrete permutations
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

import torch
from torch.utils.data import DataLoader


# Test generalization 
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

            # Forward pass through EGNN
            z_X, _ = model(h_X, X.squeeze(), edges,edge_attr)
            z_Y, _ = model(h_Y, Y.squeeze(), edges,edge_attr)
            P,_ = sinkhorn_loss(z_X, z_Y, sinkhorn_eps, n_points, nIter,device = device) # shape: [batch_size, N, N] or [N, N]

            # If batch size is 1, squeeze
            if P.dim() == 3:
                P = P.squeeze(0)

            # Predict permutation (row-wise argmax)
            pred_perm = P.argmax(dim=1)  # predicted: index in Y for each point in X

            # Compare to ground truth permutation
            correct = 100*(pred_perm == perm).sum().item()/perm.size(1)
            total_correct += correct
            total_points += 1

    accuracy = total_correct / total_points
    return accuracy

if __name__ == "__main__":

    # === Config ===
    dataConfig = {
        'n_points' : 90, # number of points in each point cloud
        'n_train' : 1024,# number of samples in traning set
        'n_test_ratio' : 0.25, # test set will have n_test_ratio*n_train samples
        'nFactors' : 3, # Number of components in each Gaussian mixture point cloud
        'dom' : [[-1,1],[-1,1]],
        'maxPerturb' :  0.1 # Maximal pointwise perturbation
    }

    config = {
        'model_type': 'EGNN+Sinkhorn',
        'hidden_nf': 128,
        'out_node_nf' : 1,
        'n_layers': 6,
        'normalize' : False,
        'batch_size' : 32, 
        'sinkhorn_iters': 100,
        'sinkhorn_eps_start' : 0.2,
        'sinkhorn_eps_end' : 0.05,
        'device': 'cuda',
        'nEpochs': 100,
        'lr' : 1e-3, # learning rate 
        'dataConfig' : dataConfig
    }

    # Optimization parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_points = dataConfig['n_points']
    sc = torch.tensor(n_points**(-0.5), dtype=torch.float32)
    lr = config['lr'] # learning rate for EGNN
    n_epochs = config['nEpochs']
    dataset_size = dataConfig['n_train'] 
    nFactors = dataConfig['nFactors'] # Number of Gausians in each sample mixture

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
    model = eg.EGNN(in_node_nf=1, hidden_nf=64, out_node_nf=64, in_edge_nf=1,n_layers=2,normalize=False,device=device)

    # === Initialize optimzer with learning scheduler  ===
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.7)

    h_X = torch.ones((n_points, 1), device=device)
    h_Y = torch.ones((n_points, 1), device=device)

    # === Run optimization ===
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        batch_idx = 0

        # Sinkhorn regularization paramter schedulr - grow linearly in epochs 
        sinkhorn_eps = config['sinkhorn_eps_start'] + (
            (config['sinkhorn_eps_end'] - config['sinkhorn_eps_start']) * epoch / (n_epochs - 1)
        )

        for X_batch, Y_batch, perm_batch in dataloader:
            optimizer.zero_grad()
            losses = []
            accs = []
            typs = []
            before = model.embedding_in.weight.clone().detach().cpu()

            for X, Y, perm in zip(X_batch, Y_batch, perm_batch):

                # Forward step
                z_X, _ = model(h_X, X, edges,edge_attr)
                z_Y, _ = model(h_Y, Y, edges,edge_attr)

                # Sinkhorn step
                P,_ = sinkhorn_loss(z_X, z_Y, sinkhorn_eps, n_points, config['sinkhorn_iters'],device = device)
                log_P = torch.log(n_points*P.T + 1e-08)

                loss = -log_P[torch.arange(P.size(0)), perm].mean()
                losses.append(loss)

                # === Accuracy and typical P - for debugging log ===
                with torch.no_grad():
                    pred = P.T.argmax(dim=1)
                    acc = (pred == perm).float().mean()
                    typ = torch.exp(-loss)

                accs.append(acc)
                typs.append(typ)
                batch_acc = torch.stack(accs).mean()
                batch_typ = torch.stack(typs).mean()

                # =============================================== #  
            
            # Backward and optimize over current batch
            batch_loss = torch.stack(losses).mean()
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

    # Save trained model and input
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'epoch': epoch,
        'optimizer_state': optimizer.state_dict()},
        f"checkpoint_{dataConfig['n_points']}_{dataConfig['nFactors']}_{dataConfig['maxPerturb']}"
    )
