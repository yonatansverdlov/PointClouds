
import torch as torch
import numpy as np
from scipy.stats import ortho_group
from math import cos, sin

# Class for perturbing point clouds
class PointCloud:

    def __init__(self,pc,component_ids,means,covars):
        self.pc = pc # Coordinates of points
        self.component_ids = component_ids # Which centorid generated each point 
        self.means = means
        self.covars = covars
        self.center = np.mean(pc,axis=0)

    # Generate point cloud from given Gaussian mixture parameters
    def addNoise(self,gamma,rng = None):

        if rng == None:
            rng = np.random.default_rng(42)
        N,d = self.pc.shape
        noise = rng.multivariate_normal(mean=np.zeros(d), cov=gamma*np.eye(d), size=N).astype(np.float32)
        return self.copy(self.pc+noise), rng
    
    def getPC(self):
        return self.pc
    
    def centerPC(self):
        return self.copy(self.pc - self.center)

    # Apply a random rigid motion to the cloud
    def apply_random_rigid_motion(self, rng):
        # Random 2D rotation
        theta = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
        X_rot = self.pc @ R.T

        # Random shift
        shift = rng.uniform(-1, 1, size=(1, 2))
        X_shifted = X_rot + shift
        return self.copy(X_shifted)
    
    def copy(self,pc_pert):
        pc = PointCloud(
            pc=pc_pert,
            component_ids=self.component_ids.copy(),
            means=self.means.copy(),
            covars=self.covars.copy(),
        )
        pc.center = self.center
        return pc
    
    # Permute point cloud coordinate vectors by perm
    def permutePT(self,perm):
        return self.copy(self.pc[perm.cpu().numpy()])

    # Perturb Gaussian mixture point cloud by randomly shifting centroids
    def shiftCentroids(self,maxShift,rng = None):

        if rng == None:
            rng = np.random.default_rng(42)
        nFactors = self.covars.shape[2]
        N,d = self.pc.shape
        perturbedPC = np.zeros(self.pc.shape)
        for k in range(nFactors):
            idx = self.component_ids == k
            shift = np.array(maxShift*rng.multivariate_normal(mean=np.zeros(d), cov=np.eye(d)))
            perturbedPC[idx,:] = self.pc[idx,:] + shift# shift # np.array([1,1]) #shift
        return perturbedPC,rng

    # Perturb Gaussian mixture point cloud by randomly rotating centroids
    def rotateCentroids(self,gamma = None,rng = None):

        if rng == None:
            rng = np.random.default_rng(42)
        nFactors = self.covars.shape[2]
        d = self.pc.shape[1]
        perturbedPC = np.zeros(self.pc.shape)
        for k in range(nFactors): 
            idx = self.component_ids == k
            if gamma != None and d == 2:
               theta = rng.uniform(low = 0, high = np.pi*gamma/180)
               R = np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
            else:
                R = ortho_group.rvs(d)
            perturbedPC[idx,:] = (self.pc[idx,:]-self.means[k,:])@R + self.means[k,:]
        return perturbedPC,rng

    # Scaling perturbation
    def scalePC(self,minScale,maxScale,rng= None):
        if rng == None:
            rng = np.random.default_rng(42)
        scale = rng.uniform(low = minScale, high = maxScale)
        return self.copy(scale*(self.pc - self.center)+self.center),rng


# Generate a random Gaussian mixture with nFactors centroids in domain dom, where dom is a list of d intervals forming a box
# in d-dimensional space.
def genGmmPC(dom,nSamples,weights,rng = None,minSig = 0.1):

    if rng == None:
        rng = np.random.default_rng(42)
    d = len(dom)
    nFactors = weights.shape[0]

    # Generate random centers for Gaussians
    means = np.zeros((nFactors,d))
    for i in range(d):
        low_i = -6
        high_i = 6 
        means[:,i] = rng.uniform(low = low_i,high = high_i,size = nFactors) # Place Gaussian centers in [-6,6]**d
    
    # Generate covariance matrices for Gaussians
    covars = np.zeros((d,d,nFactors))
    for k in range(nFactors):
        sig_k = np.zeros((d,1))
        max_sVal = 1
        for i in range(d):
            sig_k[i,0] = rng.uniform(minSig,max_sVal,1)
            max_sVal = sig_k[i]
        R = ortho_group.rvs(d)
        covars[:,:,k] = R.T@(sig_k*R)

    # Choose component indices according to the mixture weights
    component_ids = rng.choice(nFactors, size=nSamples, p=weights)

    # Allocate output
    X = np.zeros((nSamples,d),dtype=np.float32)
    for k in range(nFactors):
        idx = component_ids == k
        n_k = np.sum(idx)
        if n_k > 0:
           #X = np.concatenate((X,rng.multivariate_normal(mean=means[k,:], cov=covars[:,:,k], size=n_k)),axis=0)
           X[idx,:] = rng.multivariate_normal(mean=means[k,:], cov=covars[:,:,k], size=n_k).astype(np.float32)
    return PointCloud(X,component_ids,means,covars)

# Test sampling function
# nSamples = 500
# dom = [[-1,1],[-1,1]]
# d = len(dom[0])
# nFactor = 10
# weights = np.array([1/nFactor]*nFactor)
# pt = genGmmPC(dom,nSamples,weights)
# rng = np.random.default_rng(42)
# # noisyPT = pt.addNoise(0.5,rng)
# #noisyPT = pt.shiftCentroids(1,rng)
# #noisyPT = pt.rotateCentroids(30,rng)
# noisyPT = pt.scalePC(1.05)

# pc = pt.getPC()
# plt.scatter(pc[:,0], pc[:,1])
# plt.scatter(noisyPT[:,0], noisyPT[:,1])
# plt.show()
