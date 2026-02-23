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
    video = video.permute(1,0,2,3)
    v = func.create_new_video(video, clustered_ids)
    stats = func.get_pred_stats(model, v, gt_class_idx, pred_logit)

    l = stats['logit']

    #sort pairs according to importance
    p_imp = d['pair_analysis']['pair_importance']
    p_ = [p[0] for p in p_imp]
    p_imp_= [p[1] for p in p_imp]
    if p_[0]==[None,None]:
        p_ = p_[1:]
        p_imp_ = p_imp_[1:]

    sort_args = np.argsort(-1*np.array(p_imp_))
    pairs_sort = np.array(p_)[sort_args].tolist()

    #insertion test. Start with no motion, add one motion pair one by one
    numbers = list(clustered_ids.keys())
    forbidden_pairs = pairs_sort
    solutions = func.sample_fill_array(numbers, [], forbidden_pairs)
    with torch.no_grad():
        pred = func.get_avg_pred(model, video, clustered_ids, solutions)
    logit_rand = [pred[:,gt_class_idx].item()]

    logits = []
    for i in range(len(pairs_sort)):
        keep_pairs = pairs_sort[:i+1]
        forbidden_pairs = pairs_sort[i+1:]
        keep_pairs_flat = [p_ for p in keep_pairs for p_ in p]
        num = [n for n in numbers if n not in keep_pairs_flat]
        solutions = func.sample_fill_array(num, keep_pairs, forbidden_pairs)
        solutions = [list(s) for s in solutions]
        p = func.get_avg_pred(model, video, clustered_ids, solutions)
        l = p[:,gt_class_idx]
        logits.append(l.item())

    logits = logit_rand + logits
    x = np.linspace(0, 1, len(logits))
    AUC_insert = float(np.trapezoid(logits, x))
    metrics['AUC_insert'] = AUC_insert

    #deletion test. Start with no motion, add one motion pair one by one
    pass

    




        







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
    

if __name__ == '__main__':
    with open(analysis_path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    
    for i, k in enumerate(data_dict):
        print(f'Processing sample {i+1}/{len(data_dict)}: {k}', end='\r')
        d = data_dict[k]
        gt_class = d['motion_importance']['gt_class']
        print(f'Class: {gt_class}')
        gt_class_idx = class_labels[gt_class.lower()]
        pred_logit = d['motion_importance']['pred_original_logit']
        pred_class = d['motion_importance']['pred_original_class']

        #we consider only correctly lassified sampless
        if gt_class.lower() != pred_class.lower():
            continue
        sfs = d['single_frame_structure']

        g = k.split('_')[2][1:]
        c = k.split('_')[3][1:]
        cls_name = d['motion_importance']['gt_class']
        vid_path = ucf101dm.construct_vid_path(cls_name,g,c)
        video = ucf101dm.load_jpg_ucf101(vid_path,n=0).to(device)
        sm = structure_metrics(video, d, gt_class_idx, pred_logit)
        mm = motion_metrics(video, d, gt_class_idx, pred_logit)
