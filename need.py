import numpy as np
from sklearn.cluster import KMeans
#from show_cluster import show as shc
import time
#from metric import do_unsupervis
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.optimize import linear_sum_assignment
#from pycocotools import mask as maskUtils
#from pycocotools.coco import COCO
import json
#from pycocotools import mask as masktools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import torch
import warnings
import cv2
import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

import numpy as np
    
import numpy as np
from sklearn.decomposition import PCA
    
def bestent(emd):
    
    e = emd.shape[2];hist=[]
    for v in range(0,e):
        hist.append(np.histogram(emd[:,:,v].ravel(), bins=30)[0])
    
    #print(hist[0])
    length = len(emd[:,:,v].ravel())

    def calculate_entropy(probabilities):
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    entropy = [calculate_entropy(h/length) for h in hist]
    
    entropy = np.nan_to_num(entropy, nan=5)
    #print(entropy)
    most_discriminated_matrix_index = np.argmin(entropy)
    
    #print("The most discriminated matrix is matrix", most_discriminated_matrix_index)
    return most_discriminated_matrix_index,entropy

def getpca(e,emd):
    ln = len(e)//10
    v = np.argsort(e)[0:ln]
    shalok = emd[:,:,v]

    c = shalok.shape[2]
    
    h = shalok.shape[0]
    w = shalok.shape[1]

    # Reshape the array into a 2D matrix
    data = np.reshape(shalok, (h * w, c))

    pca = PCA(n_components=4)
    pca.fit(data)

    # Get the principal components
    principal_components = pca.transform(data)

    # Reshape the principal components back to the original image shape
    principal_components = np.reshape(principal_components, (h, w, 4))

    return principal_components,bestent(principal_components)


import cv2
import numpy as np

def remove_small(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # Find the index of the contour with the largest area
    largest_contour_index = np.argmax(contour_areas)

    # Get the area of the largest contour
    largest_contour_area = contour_areas[largest_contour_index]
    
    try:
        # Calculate the ratio of each contour area to the area of the largest contour
        area_ratios = [area / largest_contour_area for area in contour_areas]

        # Create an empty image of the same size as the input image
        refined_image = np.zeros_like(image)

        # Iterate through the contours and their corresponding area ratios
        for contour, ratio in zip(contours, area_ratios):
            # If the ratio is greater than or equal to 0.1, draw the contour on the refined image
            if ratio >= 0.1:
                cv2.drawContours(refined_image, [contour], -1, 1, thickness=cv2.FILLED)

        return refined_image
    except:
        return image


def evaluate_segmentation(pred, gt):
    # Flatten the prediction and ground truth arrays
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()

    # Calculate precision, recall, and F1-score
    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat)

    # Calculate accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)

    return precision, recall, f1, accuracy

def calculate_iou(mask1, mask2):
   
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0:
        return 0
    else:
        return intersection / union

def calculate_instance_segmentation_accuracy(gt_instances, prediction_masks):

    iou_matrix = np.zeros((len(prediction_masks), len(gt_instances)))

    for i in range(len(prediction_masks)):
        for j in range(len(gt_instances)):
            
            sm = calculate_iou(prediction_masks[i], gt_instances[j])
            
            iou_matrix[i, j] = sm
            
          
    # Use the Hungarian algorithm to find the optimal assignment of predictions to ground truth instances
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
  
    mean_iou = 0
    mean_iou_w = 0
    num_matches = 0
    num_true_positives = 0
    #if iou_matrix
    score_w = 0;n7=0;

    precisionT=0; recallT=0; f1T=0; accuracyT=0;
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:

            mean_iou += iou_matrix[i, j]
           
            pred = prediction_masks[i]
            gt = gt_instances[j]
            
            '''
            kk = list(gt_instances.keys())
            
            
            annlist = [];pred=[]
            for frame in range(0,len(prediction_masks[i])):
                img = gt_instances[kk[j]][frame]
                annlist.append(img)
                predi = prediction_masks[i][frame]
                pred.append(predi)
            '''  
            precision, recall, f1, accuracy = evaluate_segmentation(pred,gt)

            precisionT+=precision
            recallT+=recall
            f1T+=f1
            accuracyT+=accuracy
                                
          
            num_matches += 1
            num_true_positives += 1

    if num_matches > 0:
        mean_iou /= len(gt_instances) #num_matches
        mean_iou_w /=num_matches
        
        precision = num_true_positives / len(prediction_masks)
        
        precisionT=precisionT/num_matches; recallT=recallT/num_matches;
        f1T=f1T/num_matches; accuracyT=accuracyT/num_matches;

    else:
        mean_iou = 0
        mean_iou_w = 0
        precision = 0


    return mean_iou,precisionT,recallT,f1T,accuracyT


from scipy.optimize import linear_sum_assignment

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_segmentation(pred, gt):
    # Flatten the prediction and ground truth arrays
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()

    # Calculate precision, recall, and F1-score
    precision = precision_score(gt_flat, pred_flat)
    recall = recall_score(gt_flat, pred_flat)
    f1 = f1_score(gt_flat, pred_flat)

    # Calculate accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)

    return precision, recall, f1, accuracy

def resultg(pred,gt):
    precision=0; recall=0; f1=0; accuracy=0;
    for i in range(len(gt)):
        p = pred[i]
        g = gt[i];g[g!=0]=1; 
        h=p.shape[0]; w=p.shape[1];
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
        
        try:
            pr, re, f, ac = evaluate_segmentation(g, p)
            precision = precision + pr;
            recall = recall + re;
            f1 = f1 + f;
            accuracy = accuracy + ac
        except:
            print('E')
    return precision/len(gt),recall/len(gt),f1/len(gt),accuracy/len(gt);

def calculate_iou(mask1, mask2):
   
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0:
        return 0
    else:
        return intersection / union
    
def calculate_instance_segmentation_accuracy_video(gt_instances, prediction_masks):

    # Calculate the IoU between each prediction and each ground truth instance
    iou_matrix = np.zeros((len(prediction_masks), len(gt_instances)))

    for i in range(len(prediction_masks)):
        for j, index in enumerate(list(gt_instances.keys())):
            gt_mask = gt_instances[index]

            sm  = 0
            for k in range(0,len(prediction_masks[i])):
                
                tmg = gt_mask[k]
                tmpx = prediction_masks[i][k]
                
                if tmpx.shape[0]!=tmg.shape[0] or tmpx.shape[1]!=tmg.shape[1]:
                    
                    ame = cv2.resize(tmpx, (tmg.shape[1], tmg.shape[0]), interpolation=cv2.INTER_NEAREST)
                    sm += calculate_iou(ame, gt_mask[k])
                else:
                    sm += calculate_iou(prediction_masks[i][k], gt_mask[k])

            sm = sm / len(prediction_masks[i])
            #print('pred_mask',prediction_masks[i]['mask'].shape)
            iou_matrix[i, j] = sm

    # Use the Hungarian algorithm to find the optimal assignment of predictions to ground truth instances
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    #print('row_indices, col_indices',row_indices, col_indices)

    # Calculate the mean IoU and precision for the matched predictions
    mean_iou = 0
    mean_iou_w = 0
    num_matches = 0
    num_true_positives = 0
    #if iou_matrix
    score_w = 0;n7=0;
    #if np.isnan(score_w):
    #  score_w=0;
    precisionT=0; recallT=0; f1T=0; accuracyT=0;
    for i, j in zip(row_indices, col_indices):
        if iou_matrix[i, j] > 0:

            mean_iou += iou_matrix[i, j]
            #mean_iou_w += prediction_masks[i]['score'] * iou_matrix[i, j]
            #score_w += prediction_masks[i]['score']
            #if iou_matrix[i, j]>=0.75:
            #  score_w += prediction_masks[i]['score'] * iou_matrix[i, j]
            #  n7+=1;

            kk = list(gt_instances.keys())
            
            
            annlist = [];pred=[]
            for frame in range(0,len(prediction_masks[i])):
                img = gt_instances[kk[j]][frame]
                annlist.append(img)
                predi = prediction_masks[i][frame]
                pred.append(predi)
                
            precision, recall, f1, accuracy = resultg(pred,annlist)

            precisionT+=precision
            recallT+=recall
            f1T+=f1
            accuracyT+=accuracy
                                
            '''
            print(i,j,iou_matrix[i, j])
            fig, axs = plt.subplots(1, len(prediction_masks[i]), figsize=(15, 9));axs = axs.flatten()
            for frame in range(0,len(prediction_masks[i])):
                img = prediction_masks[i][frame]
                axs[frame].imshow(img)
                axs[frame].axis('off')
            plt.show();fig.suptitle(str(iou_matrix[i, j]));plt.figure()
            fig, axs = plt.subplots(1, 8, figsize=(15, 9));axs = axs.flatten()
            for frame in range(0,8):
                img = gt_instances[kk[j]][frame]
                axs[frame].imshow(img)
                axs[frame].axis('off')
    
            plt.show();plt.figure()
            '''
        

                
            """
            print(i,j)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(str(iou_matrix[i, j]))
            ax1.imshow(prediction_masks[i][7])
            kk = list(gt_instances.keys())
            ax2.imshow(gt_instances[kk[j]][7])
            plt.figure()
            """

            num_matches += 1
            num_true_positives += 1

    if num_matches > 0:
        mean_iou /= num_matches
        mean_iou_w /=num_matches
        
        precision = num_true_positives / len(prediction_masks)
        
        precisionT=precisionT/num_matches; recallT=recallT/num_matches;
        f1T=f1T/num_matches; accuracyT=accuracyT/num_matches;

    else:
        mean_iou = 0
        mean_iou_w = 0
        precision = 0
    #if n7>0:
    #  score_w /=n7 


    return mean_iou,precisionT,recallT,f1T,accuracyT
