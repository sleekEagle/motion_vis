import func
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sam_ui
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

analysis_path = r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json'

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


def structure_metrics(video, d, gt_class_idx, pred_logit):
    metrics = {}
    if sfs:
        mimp = d['motion_importance']
        best_frame = int(np.argmax(np.array(mimp['all_logits'])))
        v = video[best_frame,:].unsqueeze(0).repeat(16,1,1,1)
        v = v.permute(1,0,2,3)
        n_frames = 1/video.size(0)    
        imp_frames = [best_frame]
    else:
        clustered_ids = d['pair_analysis']['clustered_ids']
        ordered_keys = list(dict(sorted(clustered_ids.items(), key=lambda x: x[1][0])).keys())
        imp_frames = [clustered_ids[k][0] for k in ordered_keys]
        v = func.create_new_video(video.permute(1,0,2,3), clustered_ids, ordered_keys)
        n_frames = len(ordered_keys)/video.size(0)

    pred_stats_clus = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
    per_change = max(1e-5, pred_stats_clus['per_change'])
    change_frames = per_change*n_frames

    metrics['per_change'] = per_change
    metrics['n_frames'] = n_frames
    metrics['change_frames'] = change_frames
    
    #remove unimportant frames one by one and see how the prediciton is affected
    all_frames = np.arange(0,video.size(0))
    unimp_frames = [int(f) for f in all_frames if f not in imp_frames]
    random.shuffle(unimp_frames)

    stats = func.get_pred_stats(model, video.permute(1,0,2,3), gt_class=gt_class_idx, orig_pred_logit=pred_logit)
    pred_logits = [stats['logit']]
    for i in range(0, len(unimp_frames)):
        rem_ar = unimp_frames[0:i+1]
        valid_idx = all_frames[~np.isin(all_frames, rem_ar)].tolist()
        clusters = func.temporal_freeze(idx_list=valid_idx)
        v = func.create_new_video(video.permute(1,0,2,3), clusters['cluster_ids'], clusters['ordered_keys'])
        stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
        pred_logits.append(stats['logit'])
    #calc AUC
    x = np.linspace(0, 1, len(pred_logits))
    AUC_remove = float(np.trapezoid(pred_logits, x))
    metrics['AUC_unimp_delete'] = AUC_remove

    #add unimportant frames one by one and see how the prediciton is affected
    #start with the selected video
    clustered_ids = d['pair_analysis']['clustered_ids']
    clustered_idx = [clustered_ids[k][0] for k in clustered_ids]
    ordered_keys = list(dict(sorted(clustered_ids.items(), key=lambda x: x[1][0])).keys())
    pred_logits = [stats['logit']]
    for i in range(0, len(unimp_frames)):
        add_ar = unimp_frames[0:i+1]
        clusters = func.temporal_freeze(idx_list=clustered_idx+add_ar)
        v = func.create_new_video(video.permute(1,0,2,3), clusters['cluster_ids'], clusters['ordered_keys'])
        stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
        pred_logits.append(stats['logit'])
    #calc AUC
    x = np.linspace(0, 1, len(pred_logits))
    AUC_remove = float(np.trapezoid(pred_logits, x))
    metrics['AUC_unimp_add'] = AUC_remove

    #add important frames one by one 
    clustered_idx = [clustered_ids[k][0] for k in clustered_ids]
    logits = d['motion_importance']['all_logits']
    cluster_logits = [logits[idx] for idx in clustered_idx]
    sort_args = np.argsort(-1*np.array(cluster_logits))
    sort_clus_idx = np.array(clustered_idx)[sort_args].tolist()

    v = torch.randn_like(video).permute(1,0,2,3)
    stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
    pred_logits = [stats['logit']]
    for i in range(0, len(sort_clus_idx)):
        sel_idx = sort_clus_idx[0:i+1]
        clusters = func.temporal_freeze(idx_list=sel_idx)
        v = func.create_new_video(video.permute(1,0,2,3), clusters['cluster_ids'], clusters['ordered_keys'])
        stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
        pred_logits.append(stats['logit'])
    x = np.linspace(0, 1, len(pred_logits))
    AUC_insert = float(np.trapezoid(pred_logits, x))
    metrics['AUC_imp_insert'] = AUC_insert

    #start with the selected cluster frames and remove important frames one by one
    pred_logits = [] 
    for i in range(0, len(sort_clus_idx)):
        clusters = func.temporal_freeze(idx_list=sort_clus_idx[i:])
        v = func.create_new_video(video.permute(1,0,2,3), clusters['cluster_ids'], clusters['ordered_keys'])
        stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
        pred_logits.append(stats['logit'])
    x = np.linspace(0, 1, len(pred_logits))
    AUC_remove = float(np.trapezoid(pred_logits, x))
    metrics['AUC_imp_remove'] = AUC_remove
    
    return metrics

def motion_metrics(video, d, gt_class_idx, pred_logit):
    metrics = {}
    clustered_ids = d['pair_analysis']['clustered_ids']
    dict_ = {}
    for k in clustered_ids:
        dict_[int(k)] = clustered_ids[k]
    clustered_ids = dict_

    pairs = func.get_motion_pairs(clustered_ids)
    nums = [int(k) for k in clustered_ids.keys()]
    original = [None]*len(nums)
    fobbiden_pairs = [(int(p[0]),int(p[1])) for p in pairs]
    combinations = func.sample_fill_array(original, nums, fobbiden_pairs,
                max_solutions=20,
                max_trials=5000
            )
    # test combinations for forbidden pairs
    for comb in combinations:
        for i in range(len(comb)-1):
            p = (comb[i],comb[i+1])
            assert p not in fobbiden_pairs, 'pair detected'

    pred = func.get_avg_pred(model, video.permute(1,0,2,3), clustered_ids, combinations)
    logit_nomotion = pred[:,gt_class_idx]

    p_imp = d['pair_analysis']['pair_importance']
    p_ = [p[0] for p in p_imp]
    p_imp_= [p[1] for p in p_imp]
    if p_[0]==[None,None]:
        p_ = p_[1:]
        p_imp_ = p_imp_[1:]

    sort_args = np.argsort(-1*np.array(p_imp_))
    pairs_sort = np.array(p_)[sort_args].tolist()

    vals = list(clustered_ids.keys())

    for i in range(0,len(pairs_sort)):
        vals_ = vals.copy()
        sel_pairs = pairs_sort[:i+1]
        forbidden_pairs = pairs_sort[i+1:]

        vals_to_use = []
        for v in vals_:
            present = False
            for s in forbidden_pairs:
                if v in s:
                    present = True
            if not present:
                vals_to_use.append(v)
        
        for val in vals_to_use:
            idx = int(np.argwhere(np.array(vals)==val)[0,0])
            vals_[idx] = None

        for i in not_nan_idx:
            vals_[i] = None
        fill_numbers = [c for c in vals if c not in vals_]
        fobbiden_pairs_ = [p for p in pairs if p != pair]
        combinations = func.sample_fill_array(vals_, fill_numbers, fobbiden_pairs_,
                max_solutions=20,
                max_trials=5000
            )

def sample_fill_array(numbers, existing, forbidden_pairs, seed=None, max_solutions=10, max_trials=10000):

    flat = [item for sublist in existing for item in sublist]
    out_len = len(flat) + len(numbers)
    assert len([n for n in numbers if n in flat])==0, 'any number in numbers must not be present in existing'

    if seed is not None:
        random.seed(seed)
    
    solutions = []
    seen = set()

    new_list = existing.copy()
    i=0
    while i < len(new_list):
        p = new_list[i]
        rest = [item for item in new_list if item!=p]
        found=False
        for r in rest:
            assert not(r[0]==p[-1] and r[-1]==p[0]), 'invalid combinations'
            if r[0]==p[-1]:
                new_comb = list(set(r + p))
                rest_rest = [item for item in rest if item!=r]
                new_list = [new_comb]+rest_rest
                found=True
                break
            if r[-1]==p[0]:
                new_comb = list(set(r + p))
                rest_rest = [item for item in rest if item!=r]
                new_list = [new_comb]+rest_rest
                found=True
                break
        if found:
            continue
        i+=1
    existing = new_list

    def is_valid(arr):
        for i in range(len(arr)-1):
            if (arr[i], arr[i + 1]) in forbidden_pairs:
                return False
        return True
    
    for _ in range(max_trials):
        if len(solutions) >= max_solutions:
            break
        #place existing pairs in the array
        array = [None]*out_len
        idx_arr = list(range(out_len))
        for idx in range(len(existing)):
            avail = False
            while not avail:
                l = len(existing[idx])
                i = random.sample(idx_arr[:-l+1],1)[0]
                avail = bool(np.array([idx_ in idx_arr for idx_ in range(i,i+l+1)]).all())

            for e_idx, e in enumerate(existing[idx]):
                array[i+e_idx] = existing[idx][e_idx]
                idx_arr.remove(i+e_idx)

        #fill the rest of the array with the given numbers

    pass

import random
def sample_fill_array_(
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
    

if __name__ == '__main__':
    forbidden_pairs = [[1,3],[5,8],[10,12]]
    numbers = [0,1,2,3,6,8,9,10,11,12,15]
    existing = [[4,5],[13,14],[5,7]]
    sample_fill_array(numbers, existing, forbidden_pairs)
    # with open(analysis_path, 'r', encoding='utf-8') as file:
    #     data_dict = json.load(file)
    
    # for i, k in enumerate(data_dict):
    #     print(f'Processing sample {i+1}/{len(data_dict)}: {k}', end='\r')
    #     d = data_dict[k]
    #     gt_class = d['motion_importance']['gt_class']
    #     print(f'Class: {gt_class}')
    #     gt_class_idx = class_labels[gt_class.lower()]
    #     pred_logit = d['motion_importance']['pred_original_logit']
    #     pred_class = d['motion_importance']['pred_original_class']

    #     #we consider only correctly lassified sampless
    #     if gt_class.lower() != pred_class.lower():
    #         continue
    #     sfs = d['single_frame_structure']

    #     g = k.split('_')[2][1:]
    #     c = k.split('_')[3][1:]
    #     cls_name = d['motion_importance']['gt_class']
    #     vid_path = ucf101dm.construct_vid_path(cls_name,g,c)
    #     video = ucf101dm.load_jpg_ucf101(vid_path,n=0).to(device)
    #     sm = structure_metrics(video, d, gt_class_idx, pred_logit)
    #     mm = motion_metrics(video, d, gt_class_idx, pred_logit)
