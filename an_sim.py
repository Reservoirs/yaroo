from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
#from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils
import torch
import time
import pickle
import glob
import matplotlib.pyplot as plt
import need
import os

def invert(img3):
    h,w = img3.shape
    border = [img3[:5,:].ravel(), img3[:,:5].ravel(), img3[h-5:,:].ravel(), img3[:,w-5:].ravel()]
    flag=0;
    for b in border:
        tp = np. where(b==1)[0]
        if len(tp)/len(b)>0.9:
            flag+=1;
    if flag>=2:
        return 1-img3,flag
    else:
        return img3,flag
    

def calculate_entropy(probabilities):
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()

def bestent(emd):
    e = emd.size(1)
    length = emd.size(0) * emd.size(1)
    
    hist = []; entropy=[]
    for v in range(e):
        #hist.append(torch.histc(emd[:, v].view(-1), bins=30))
        entropy.append(calculate_entropy((0.000001 + torch.histc(emd[:, v].view(-1), bins=30)) / length))

    #entropy = [calculate_entropy((0.000001 + h) / length) for h in hist]
    
    entropy = torch.nan_to_num(torch.tensor(entropy), nan=5)
    return entropy

def bestentC(emd):
    e = emd.size(1)
    hist = []
    for v in range(e):
        hist.append(torch.histc(emd[:, v].view(-1), bins=30))
    
    length = emd.size(0) * emd.size(1)
    entropy = [calculate_entropy((0.000001 + h) / length) for h in hist]
    
    entropy = torch.nan_to_num(torch.tensor(entropy), nan=5)
    return entropy

'''

def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def bestent(emd,maxENT=5):
    
    #emd = emd.cpu().numpy()
    
    e = emd.size(1); n = emd.size(0); hist=[]
    for v in range(0,e):
        hist.append(np.histogram(emd[:,v].cpu().numpy().ravel(), bins=30)[0])
    
    #print(hist[0])
    length = n * e

    entropy = [np.nan_to_num(calculate_entropy(h/length), nan=maxENT) for h in hist]
    
    #entropy = np.nan_to_num(entropy, nan=5)
    #print(entropy)
    #most_discriminated_matrix_index = np.argmin(entropy)
    
    #print("The most discriminated matrix is matrix", most_discriminated_matrix_index)
    return entropy
'''
def do(feats,continues=True):
    
    start_time = time.time()

    #entropy = bestentC(feats)
    #ln = len(entropy) // 3
    #v = torch.argsort(entropy)[0:ln]
    #print('old',v)
        
    entropy = bestent(feats)
    ln = len(entropy) // 3
    v = np.argsort(entropy)[0:128]
    feats = feats[:, v]
    #print('new',v)
    
    #feats = feats[:, :128]
    #feats = torch.index_select(feats, dim=1, index=v)
    #feats = feats[:, v]
    #print(torch.sum(feats1-feats2))
    
    #feats = feats.narrow(1, 0, 128)
    
    feats = F.normalize(feats, p=2, dim=-1)

    eigenvalues=[];eigenvectors=[];
    
    if continues==True:
        ### Feature affinities 
        W_feat = (feats @ feats.T)
        W_feat = (W_feat * (W_feat > 0))
        W_feat = W_feat / W_feat.max()
        W_feat = W_feat.cpu().numpy()

        W_comb = W_feat #+ W_color * image_color_lambda  # combination
        D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check


        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)

        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

        # Sign ambiguity
        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
                eigenvectors[k] = 0 - eigenvectors[k]

        finish = time.time() - start_time
    
    return eigenvalues, eigenvectors,feats

import torch

def calculate_affinity_matrix(tensor):
    N, D = tensor.size()

    # Expand dimensions for broadcasting
    tensor_expanded = tensor.unsqueeze(1)

    # Calculate pair-wise differences
    differences = 1 - torch.abs(tensor_expanded - tensor_expanded.permute(1, 0, 2))

    #if differences!=0:
    #    differences = 1/differences
        
    # Calculate maximum distance along feature dimension (dim=2)
    distances = torch.abs(differences).max(dim=2).values
    #distances = differences.sum(dim=2)

    # Build affinity matrix using the calculated distances
    affinity_matrix = torch.max(differences,dim=2).values

    return affinity_matrix

import torch
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import sys


def otherx_affinty(original_matrix,typ):
    if typ!='normal':
        output_matrix = pairwise_distances(original_matrix, metric=typ)
        output_matrix = np.nan_to_num(output_matrix , nan=0.0, posinf=0.0, neginf=0.0)
        return output_matrix
    else:
        return original_matrix @ original_matrix.T
    
def other_affinty(original_matrix,typ):
    if typ!='normal':
        output_matrix = 1/(1+pairwise_distances(original_matrix, metric=typ))
        output_matrix = np.nan_to_num(output_matrix , nan=0.0, posinf=0.0, neginf=0.0)
        return output_matrix
    else:
        return original_matrix @ original_matrix.T

def do_sp_metric(W_feat,W_moment):
    # Calculate affinity matrix
    
    #feats_sep = F.normalize(feats_sep, p=2, dim=-1)
    
    #W_feat = calculate_affinity_matrix(feats_sep)
    
    typ='braycurtis'
    W_moment = other_affinty(W_moment,typ)
    
    #W_moment = (W_moment @ W_moment.T)
    W_moment = (W_moment * (W_moment > 0))
    W_moment = W_moment / W_moment.max()
    
    # Print the result
    #print("Affinity Matrix:")
    #print(W_feat.size())
    W_feat = other_affinty(W_feat.cpu().numpy(),typ)
    #W_feat = (W_feat @ W_feat.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()
    #W_feat = W_feat.cpu().numpy()

    W_comb = W_feat + W_moment  # combination
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    
    
    #K=50
    #try:
    #    eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
    #except:
    eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)

    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()


    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
            

    return eigenvalues, eigenvectors

def do_sp(W_feat,W_moment):
    # Calculate affinity matrix
    
    #feats_sep = F.normalize(feats_sep, p=2, dim=-1)
    
    #W_feat = calculate_affinity_matrix(feats_sep)
    
    
    W_moment = (W_moment @ W_moment.T)
    W_moment = (W_moment * (W_moment > 0))
    W_moment = W_moment / W_moment.max()
    
    # Print the result
    #print("Affinity Matrix:")
    #print(W_feat.size())
    W_feat = (W_feat @ W_feat.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()
    W_feat = W_feat.cpu().numpy()

    W_comb = W_feat + W_moment  # combination
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    
    
    #K=50
    try:
        eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
    except:
        eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)

    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()


    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]
            

    return eigenvalues, eigenvectors

def clustering(query,bg,gt_mask):
    tps = np.where(bg!=0)
    fps = np.where(bg==0)
    #pos = np.asarray([tps[0],tps[1]]).T
    #query = (ax*(1/np.std(bh0[i]))*bh0[i] + bx*(1/np.std(bh1[i]))*(bh1[i])) * bgs[i]
    #query = (query - np.min(query)) / (np.max(query) - np.mean(query))
    #query[tps] = query[tps] + 50
    pixels = query.reshape((-1, 1))
    #context = np.hstack((pixels, pos))
    cluster = KMeans(n_clusters=3,random_state=42).fit(pixels)
    labels = cluster.labels_
    clustered_image = labels.reshape(query.shape)
    
    real = find_most_frequent_number(clustered_image[fps])
    if real!=0:
        #print('carbon')
        zps = np.where(clustered_image==0)
        clustered_image[fps] = 0
        clustered_image[zps] = real
    
    prediction_masks = sep(clustered_image)
    mean_iou,precisionT,recallT,f1T,accuracyT = need.calculate_instance_segmentation_accuracy(gt_mask, prediction_masks)
    return mean_iou,f1T#,clustered_image

def do_sp_metric_pair(feats_sep,out):
    
    feats_sep = feats_sep.cpu().numpy()
    eigs={}
    
    distance_metric=['normal','chebyshev','braycurtis','correlation','cosine','l2','l1','mahalanobis']
    
    #distance_metric=['SUM_braycurtis_chebyshev','MUL_braycurtis_chebyshev','SUB_braycurtis_chebyshev','div1_braycurtis_chebyshev','div2_braycurtis_chebyshev','MAX_braycurtis_chebyshev','MIN_braycurtis_chebyshev']
    #distance_metric=['div1_braycurtis_chebyshev']
    #distance_metric=['correlation']

    u=0;
    K=5
    for distance in distance_metric:
        
        W_feat = other_affinty(feats_sep,distance)
        #W_featc = other_affinty(feats_sep,'chebyshev')
     
        
        #W_feat = W_featb  / (0.000000001+W_featc)

            
        W_feat = (W_feat * (W_feat > 0))
        W_feat = W_feat / W_feat.max()
        W_comb = W_feat
        D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check
        
        eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

        #eigs.update({distance:[eigenvalues, eigenvectors,eigenvaluesxx,eigenvectorsxx]})
        # Sign ambiguity
        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
                eigenvectors[k] = 0 - eigenvectors[k]
    
        eigs.update({distance:[eigenvalues, eigenvectors]})
        eigenvalues=[];eigenvectors=[];
        u+=1;
        

    with open(out, 'wb') as fp:
      pickle.dump(eigs, fp)



import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis,skew   
import sys


if __name__ == "__main__":
    import pickle
    #print(sys.argv[0])
    with open(sys.argv[1], 'rb') as fp:
        f = pickle.load(fp)
    [arg1, arg2] = f
    do_sp_metric_pair(arg1,arg2)
