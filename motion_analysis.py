from argparse import Namespace
import json
from models.resnet3d.main import generate_model, get_inference_utils, resume_model
from models.resnet3d.model import generate_model, make_data_parallel
from pathlib import Path
import torch
from models.resnet3d import spatial_transforms
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Normalize, ToPILImage
import os
import re
from glob import glob
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.cluster import DBSCAN
from einops import rearrange
import csv
import pandas as pd
import func


#create model and data loader
ucf101dm = func.UCF101_data_model()
model = ucf101dm.model
model.to('cuda')
model.eval()
inference_loader = ucf101dm.inference_loader
class_names = ucf101dm.inference_class_names

class_labels = {}
for k in class_names.keys():
    cls_name = class_names[k]
    class_labels[cls_name.lower()] = k

'''
replace the whole video with just one frame repeated
and see how the prediction changes
'''
#**************************************************************************************
def motion_importance(video):
    # video = ucf101dm.load_jpg_ucf101(vid_path)
    # video = video.unsqueeze(0).permute(0,2,1,3,4)
    pred = model(video)
    pred = F.softmax(pred,dim=1)
    pred_cls = torch.argmax(pred,dim=1).item()

    logit_original = pred[:,pred_cls].item()
    logits = []
    preds = []
    for idx in range(video.size(2)):
        frame = video[:,:,idx,:].unsqueeze(2).repeat(1,1,16,1,1)
        pred_ = model(frame)
        pred_ = F.softmax(pred_,dim=1)
        pred_cls_ = torch.argmax(pred_,dim=1)
        preds.append(pred_cls_.item())
        logits.append(pred_[:,pred_cls].item())

    #is there at least one frame that correctly predicts the original class?
    any_correct = any([p==pred_cls for p in preds])

    max_logit = max(logits)
    perc_change = (logit_original - max_logit)/logit_original

    ret = {
        'pred_original_class': class_names[pred_cls],
        'pred_original_logit': logit_original,
        'all_logits': logits,
        'max_frame_logit': max_logit,
        'percent_change': perc_change,
        'any_frame_correct': any_correct
    }

    return ret

MAX_VID = 10
import random
import json

'''
per_change_sol: percentage of change of the predicted logit of gt_class
lower -> better prediction
'''
def pair_importance(video, gt_class, pred_logit, change_threshold, clustered_ids):
    uniqueval_indices = get_uniqueval_indices(clustered_ids)
    pairs = get_motion_pairs(clustered_ids)
    pairs.insert(0,(None,None))
    vals = list(uniqueval_indices.keys())
    vals.sort()

    pair_imp = []
    for pair in pairs:
        vals_ = vals.copy()
        nan_idx = [i for i, v in enumerate(vals_) if v not in pair]
        for i in nan_idx:
            vals_[i] = None
        numbers = [c for c in vals if c not in vals_]
        pairs_ = [p for p in pairs if p != pair]
        solutions = find_all_solutions(vals_, numbers, pairs_)

        #create video with the solutions
        k = min(MAX_VID,len(solutions))
        sample_sols = random.sample(solutions, k)

        sol_vids = torch.empty(0).to(video.device)
        for sol in sample_sols:
            v = torch.empty_like(video)
            cur_idx=0
            for s in sol:
                s_len = len(uniqueval_indices[s])
                v[:,cur_idx:cur_idx+s_len,:] = video[:,s].unsqueeze(1)
                cur_idx += s_len
            #evaluate the prediction for this create new video
            sol_vids = torch.concatenate([sol_vids,v.unsqueeze(0)])

        pred_sol = model(sol_vids).mean(dim=0)
        pred_sol = F.softmax(pred_sol.unsqueeze(0),dim=1)
        logit_sol = pred_sol[:,gt_class].item()
        per_change_sol = (pred_logit-logit_sol)/pred_logit
        pair_imp.append((pair, per_change_sol))

        #if the model does not look at the motion between frames
        if pair == (None,None) and per_change_sol<=change_threshold:
            # print('Model does not care about motion')
            break
    return pair_imp

def motion_importance_dataset():
    output_path = Path(r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json')
    change_threshold = 0.05
    anlysis_data = {}
    n_samples, n_correct = 0, 0
    for idx, batch in enumerate(inference_loader):
        file_analysis = {}
        
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        video = inputs[0].to('cuda')
        gt_class_name = targets[0][0].split('_')[1]
        file_name = targets[0][0]
        # if file_name != 'v_ApplyEyeMakeup_g03_c01':
        #     continue
        seq = targets[0][1]
        ret = motion_importance(video.unsqueeze(0))
        pred_logit = ret['pred_original_logit']
        pred_class = ret['pred_original_class']

        if pred_class.lower() != gt_class_name.lower(): # we are only interested in correctly predicted examples
            continue

        all_logits = ret['all_logits']
        percent_change = ret['percent_change']
        max_logit = ret['max_frame_logit']
        gt_class = class_labels[gt_class_name.lower()]
        seq = ','.join([str(s) for s in seq])
        ret['seq'] = seq
        ret['gt_class'] = gt_class_name 

        file_analysis['motion_importance'] = ret

        if percent_change > change_threshold: #there is no single frame that can exaplain the whole video
            file_analysis['single_frame_structure'] = False
            sorted_indices = [i for i, _ in sorted(enumerate(all_logits), key=lambda x: x[1], reverse=True)]
            file_analysis['sorted_importance_frame_idx'] = sorted_indices

            for i in range(2, len(sorted_indices)):
                valuable_indices = sorted_indices[0:i+1]
                clustered_ids = create_frame_cluster_idxs(valuable_indices)
                #update the video with only the valuable frames
                video_ = video_keep_given_frames(video, clustered_ids)

                pred_ = model(video_.unsqueeze(0))
                pred_ = F.softmax(pred_,dim=1)
                pred_cls_ = torch.argmax(pred_,dim=1)
                logit = pred_[:,gt_class].item()

                percent_change_ = (pred_logit - logit)/pred_logit

                if percent_change_ <= change_threshold: #the video can be explained well (enough) just with the given set of frames
                    pair_analysis = {}
                    pair_analysis['new_logit'] = logit
                    pair_analysis['percent_change_'] = percent_change_

                    #check if the motion among the frames are important for this prediction
                    pair_imp = pair_importance(video, gt_class, pred_logit, change_threshold, clustered_ids)
                    pair_analysis['pair_importance'] = pair_imp
                    file_analysis['pair_analysis'] = pair_analysis

                    #break if the first solution is found s.t. percent_change_ <= change_threshold
                    break
        else: # there is atleas a single frame that can explain the video classificaiton
            file_analysis['single_frame_structure'] = True

        #is there just one frame that can explain the whole video ?
        sfs = file_analysis['single_frame_structure']
        if sfs:
            continue
        else: # no
            pair_imp = file_analysis['pair_analysis']['pair_importance']
            if len(pair_imp)==1 and pair_imp[0][0]==(None,None): #motion does not matter at all for the prediction
                pass
            else: #motion is important for this video
                pair_imp = [p for p in pair_imp if p[0][0]!=None]
                #lets find which motions are important
                sort_idx = np.argsort(np.array([val[1] for val in pair_imp]))
                uniqueval_indices = get_uniqueval_indices(clustered_ids)
                new_cluster_ids = uniqueval_indices.copy()

                for s in range(0,len(sort_idx)-1):
                    #create video with given two motions unchanged
                    # solutions = find_all_solutions(vals_, numbers, pairs_)
                    pair1 = pair_imp[int(sort_idx[s])][0]
                    pair2 = pair_imp[int(sort_idx[s+1])][0]
                    
                    #are the two pairs adjecent?
                    if pair1[1]==pair2[0] or pair2[1]==pair1[0]: # yes
                        f_ = [int(i) for i in np.unique(np.array([pair1[0],pair1[1],pair2[0],pair2[1]]))]
                        sel_frames = []
                        for fval in f_:
                            sel_frames.extend(new_cluster_ids[fval])
                            del new_cluster_ids[fval]
                        new_key = max(new_cluster_ids.keys())+1
                        new_cluster_ids[new_key] = sel_frames
                    else:
                        for p in [pair1,pair2]:
                            sel_frames = []
                            sel_frames.extend(new_cluster_ids[p[0]])
                            sel_frames.extend(new_cluster_ids[p[1]])
                            sel_frames.sort()
                            del new_cluster_ids[p[0]] 
                            del new_cluster_ids[p[1]]
                            new_key = max(new_cluster_ids.keys())+1
                            new_cluster_ids[new_key] = sel_frames
                    pass









                    




        

                        
        #is the prediction correct
        pred = ret['pred_original_class']
        file_analysis['pred_original_class'] = ret['pred_original_class']
        anlysis_data[file_name] = file_analysis

        if pred == gt_class_name:
            n_correct += 1
        n_samples += 1

    accuracy = n_correct / n_samples
    print(f'Overall accuracy: {accuracy:.3f}')
    anlysis_data['accuracy'] = accuracy

    # vid_path = ucf101dm.construct_vid_path('ApplyLipstick', 1, 1)
    with open(output_path, "w") as f:
        json.dump(anlysis_data, f)

#**************************************************************************************

def video_keep_given_frames(video, frame_cluster_idxs):
    new_video = torch.zeros_like(video)
    for idx in range(len(frame_cluster_idxs)):
        new_video[:,idx,:] = video[:,frame_cluster_idxs[idx],:]
        # print(f'Frame {idx} taken from original frame {frame_cluster_idxs[idx]}')
    return new_video

def create_frame_cluster_idxs(idx_list, len_array=16):
    idx_list.sort()
    new_array = np.zeros((len_array,), dtype=int)

    assert len(idx_list) > 0 , "idx_list must contain at least one index."

    if len(idx_list) == 1:
        new_array[:] = idx_list[0]
        return new_array

    new_array[0:idx_list[1]] = idx_list[0]
    for idx in range(1, len(idx_list)-1):
        new_array[idx_list[idx]:idx_list[idx+1]] = idx_list[idx]
    new_array[idx_list[-1]:] = idx_list[-1]
    
    return new_array


def find_all_solutions(original_array, numbers, forbidden_pairs):
    #check inputs for validity
    orig_num = [n for n in original_array if n is not None]
    assert len([n for n in orig_num if n in numbers])==0, "Error. Original array contains numbers that are in the given number set."
    assert len(original_array) == len(orig_num)+len(numbers), "Error. The number of None entries in original_array must be equal to the number of given numbers + numer of non None elements in the origincal array"

    n = len(original_array)
    skip_indices = [i for i, val in enumerate(original_array) if val is not None]
    valid_indices = [i for i in range(n) if i not in skip_indices]
    valid_indices.sort()
    result = original_array.copy()

    #handle special cases seperately
    if len(original_array)==3 and len(skip_indices)==2:
        if valid_indices[0]==2:
            if (original_array[1],numbers[0]) in forbidden_pairs:
                ret_array = [numbers[0],original_array[0],original_array[1]]
            else:
                ret_array = [original_array[0],original_array[1],numbers[0]]
        if valid_indices[0]==0:
            if (numbers[0],original_array[1]) in forbidden_pairs:
                ret_array = [original_array[1],original_array[2],numbers[0]]
            else:
                ret_array = [numbers[0],original_array[1],original_array[2]]
        return [ret_array]

    used = set()
    all_solutions = []  # Store ALL solutions here

    # Convert forbidden_pairs to set for O(1) lookup
    forbidden_set = set(forbidden_pairs)
    
    def backtrack(pos):
        if pos == len(valid_indices):
            all_solutions.append(result.copy())  # Save solution
            return
        
        if len(all_solutions)>1000:
            return
        
        ind = valid_indices[pos]
        
        for num in numbers:
            if num in used:
                continue
            
            # Check constraints
            valid = True
            if ind==0:
                if (num, result[1]) in forbidden_set:
                    valid = False
            elif ind<valid_indices[-1]:
                if (result[ind-1], num) in forbidden_set or (num, result[ind+1]) in forbidden_set:
                    valid = False
            else:  # ind == valid_indices[-1]
                if (result[ind-1], num) in forbidden_set:
                    valid = False

            if valid:
                result[ind] = num
                used.add(num)
                
                backtrack(pos + 1)  # Always continue searching
                
                # Backtrack
                result[ind] = None
                used.remove(num)
    
    backtrack(0)
    return all_solutions  # Returns list of ALL valid solutions

def get_motion_pairs(ids):
    ids = np.array(ids)
    diffs = np.abs(np.diff(ids))

    start_idx = np.argwhere(diffs>0)

    pairs = []
    for idx in start_idx:
        pairs.append((ids[idx].item(), ids[idx+1].item()))

    return pairs

def get_uniqueval_indices(ids):
    unique_ids = np.unique(np.array(ids))
    args = [np.argwhere(ids==id) for id in unique_ids]
    cluster_ids = {}
    for i, id in enumerate(unique_ids):
        cluster_ids[id.item()] = [int(i) for i in args[i][:,0]]
    return cluster_ids


import json
def analyze_motion_imporance():
    path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json'
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')

        inputs, targets = batch
        video = inputs[0].to('cuda')
        gt_file_name = targets[0][0]
        if gt_file_name in data: # if this sample is in the data. i.e if the prediction for this class is correct
            d = data[gt_file_name]

            #is there just one frame that can explain the whole video ?
            sfs = d['single_frame_structure']
            if sfs:
                continue
            else: # no
                pair_imp = d['pair_analysis']['pair_importance']
                if len(pair_imp)==1 and pair_imp[0][0]==[None,None]: #motion does not matter at all for the prediction
                    pass
                else:
                    print('motion is important for this video!!')
                    pair_imp[4]


            




if __name__ == '__main__':
    motion_importance_dataset()
    # analyze_motion_imporance()






    pass






