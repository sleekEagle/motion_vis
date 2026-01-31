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
    lowest_perc_change = (logit_original - max_logit)/logit_original

    ret = {
        'pred_original_class': class_names[pred_cls],
        'pred_original_logit': logit_original,
        'all_logits': logits,
        'max_frame_logit': max_logit,
        'lowest_perc_change': lowest_perc_change,
        'any_frame_correct': any_correct
    }

    return ret

MAX_VID = 10
import random
import json

def get_avg_pred(video, clustered_ids, combinations):
    #create video with the solutions
    comb_vids = torch.empty(0).to(video.device)
    for comb in combinations:
        comb_vid = create_new_video(video, comb, clustered_ids)
        comb_vids = torch.concatenate([comb_vids,comb_vid.unsqueeze(0)])

    pred_comb = model(comb_vids).mean(dim=0)
    pred_comb = F.softmax(pred_comb.unsqueeze(0),dim=1)

    return pred_comb

'''
Monte-Carlo method to fill the array and avoid the forbidden pairs
'''
def sample_fill_array(
    original,
    numbers,
    pairs_to_avoid,
    max_solutions=10,
    max_trials=10_000,
    forbid_reverse=False,
    seed=None,
):
    
    orig_num = [n for n in original if n is not None]
    assert len([n for n in orig_num if n in numbers])==0, "Error. Original array contains numbers that are in the given number set."
    assert len(original) == len(orig_num)+len(numbers), "Error. The number of None entries in original_array must be equal to the number of given numbers + numer of non None elements in the origincal array"


    if seed is not None:
        random.seed(seed)

    forbidden = set(pairs_to_avoid)
    if forbid_reverse:
        forbidden |= {(b, a) for (a, b) in pairs_to_avoid}

    n = len(original)
    fixed = original[:]
    fixed_vals = {x for x in fixed if x is not None}
    free_positions = [i for i, x in enumerate(fixed) if x is None]
    available = [x for x in numbers if x not in fixed_vals]

    #handle special cases seperately
    skip_indices = [i for i, val in enumerate(original) if val is not None]
    valid_indices = [i for i in range(n) if i not in skip_indices]
    valid_indices.sort()
    if len(original)==3 and len(fixed_vals)==2:
        if valid_indices[0]==2:
            if (original[1],numbers[0]) in pairs_to_avoid:
                ret_array = [numbers[0],original[0],original[1]]
            else:
                ret_array = [original[0],original[1],numbers[0]]
        if valid_indices[0]==0:
            if (numbers[0],original[1]) in pairs_to_avoid:
                ret_array = [original[1],original[2],numbers[0]]
            else:
                ret_array = [numbers[0],original[1],original[2]]
        return [ret_array]


    solutions = []
    seen = set()

    def is_valid(arr):
        for i in range(n - 1):
            if (arr[i], arr[i + 1]) in forbidden:
                return False
        return True

    for _ in range(max_trials):
        if len(solutions) >= max_solutions:
            break

        random.shuffle(available)
        candidate = fixed[:]

        for pos, val in zip(free_positions, available):
            candidate[pos] = val

        if not is_valid(candidate):
            continue

        key = tuple(candidate)
        if key in seen:
            continue

        seen.add(key)
        solutions.append(candidate)

    return solutions


'''
usage:

original = [None, None, 3, 4, None, None, None]
numbers = [1, 2, 5, 10, 12]
pairs_to_avoid = [(1,2), (2,5), (10,12)]

solutions = sample_fill_array(
    original,
    numbers,
    pairs_to_avoid,
    max_solutions=5,
    max_trials=5000
)
'''

def get_motion_pairs(ids):
    pairs = []
    keys = list(ids.keys())
    for idx in range(len(keys)-1):
        pairs.append((keys[idx], keys[idx+1]))

    return pairs

def get_uniqueval_indices(ids):
    unique_ids = np.unique(list(ids.keys()))
    args = [np.argwhere(ids==id) for id in unique_ids]
    cluster_ids = {}
    for i, id in enumerate(unique_ids):
        cluster_ids[id.item()] = [int(i) for i in args[i][:,0]]
    return cluster_ids

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

        all_logits = ret['all_logits']
        lowest_perc_change = ret['lowest_perc_change']
        max_logit = ret['max_frame_logit']
        gt_class = class_labels[gt_class_name.lower()]
        seq = ','.join([str(s) for s in seq])
        ret['seq'] = seq
        ret['gt_class'] = gt_class_name 

        file_analysis['motion_importance'] = ret

        if lowest_perc_change > change_threshold: #there is no single frame that can exaplain the whole video
            file_analysis['single_frame_structure'] = False
            sorted_indices = [i for i, _ in sorted(enumerate(all_logits), key=lambda x: x[1], reverse=True)]
            file_analysis['sorted_importance_frame_idx'] = sorted_indices

            for i in range(2, len(sorted_indices)):
                valuable_indices = sorted_indices[0:i]
                clustered_ids = create_frame_cluster_idxs(valuable_indices)
                #update the video with only the valuable frames
                ordered_keys = list(dict(sorted(clustered_ids.items(), key=lambda x: x[1][0])).keys())
                video_ = create_new_video(video, ordered_keys, clustered_ids)

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
                    clustered_ids = dict(sorted(clustered_ids.items(), key=lambda x:x[1][0]))
                    pairs = get_motion_pairs(clustered_ids)
                    pairs.insert(0,(None,None))
                    vals = list(clustered_ids.keys())

                    pair_imp = []
                    for pair in pairs:
                        vals_ = vals.copy()
                        not_nan_idx = [i for i, v in enumerate(vals_) if v not in pair]
                        for i in not_nan_idx:
                            vals_[i] = None
                        fill_numbers = [c for c in vals if c not in vals_]
                        fobbiden_pairs_ = [p for p in pairs if p != pair]
                        combinations = sample_fill_array(vals_, fill_numbers, fobbiden_pairs_,
                                        max_solutions=20,
                                        max_trials=5000
                                    )
                        # check solutions to see if it has forbidden pairs
                        # for comb in combinations:
                        #     for i in range(len(comb)-1):
                        #         pair = (comb[i],comb[i+1])
                        #         if pair in fobbiden_pairs_:
                        #             print('forbidden pair detected')
                        #             break
                        # print('check complete')

                        avg_pred = get_avg_pred(video, clustered_ids, combinations)
                        comb_logit = avg_pred[:,gt_class].item()
                        comb_per_change = (pred_logit-comb_logit)/pred_logit
                        pair_imp.append((pair, comb_per_change))

                    pair_analysis['pair_importance'] = pair_imp
                    pair_analysis['clustered_ids'] = clustered_ids
                    file_analysis['pair_analysis'] = pair_analysis

                    #break if the first solution is found s.t. percent_change_ <= change_threshold
                    break
        else: # there is atleas a single frame that can explain the video classificaiton
            file_analysis['single_frame_structure'] = True

        anlysis_data[file_name] = file_analysis

        if pred_class.lower() == gt_class_name.lower():
            n_correct += 1

        n_samples += 1

    accuracy = n_correct / n_samples
    print(f'Overall accuracy: {accuracy:.3f}')
    anlysis_data['accuracy'] = accuracy
    anlysis_data['threshold'] = change_threshold
    
    # vid_path = ucf101dm.construct_vid_path('ApplyLipstick', 1, 1)
    with open(output_path, "w") as f:
        json.dump(anlysis_data, f)

#**************************************************************************************

def create_new_video(video, ordered_keys, frame_cluster_idxs):
    new_video = torch.zeros_like(video)
    cur_idx=0
    for k in ordered_keys:
        for f in frame_cluster_idxs[k]:
            new_video[:,cur_idx,:] = video[:,f,:]
            cur_idx += 1
    return new_video

def create_frame_cluster_idxs(idx_list, len_array=16):
    assert len(idx_list) > 0 , "idx_list must contain at least one index."
    max_idx = max(idx_list)
    assert max_idx < len_array, "max value in idx_list must not be larger than len_array"

    idx_list.sort()
    d = {}
    if len(idx_list) == 1:
        d[idx_list[0]] = [idx_list[0]]*len_array
        return d

    d[idx_list[0]] = [idx_list[0]]*(idx_list[1])
    for i in range(1, len(idx_list)-1):
        d[idx_list[i]] = [idx_list[i]]*(idx_list[i+1]-idx_list[i])
    d[idx_list[-1]] = [idx_list[-1]]*(len_array - idx_list[-1])
    
    return d


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


            
def print_clus_ids(c):
    for k in c.keys():
        print(f'{k} : {c[k]}')


if __name__ == '__main__':
    motion_importance_dataset()






