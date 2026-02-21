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

    pred_stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
    per_change = max(1e-5, pred_stats['per_change'])
    change_frames = per_change*n_frames

    metrics['per_change'] = per_change
    metrics['n_frames'] = n_frames
    metrics['change_frames'] = change_frames
    
    #remove unimportant frames one by one and see how the prediciton is affected
    all_frames = np.arange(0,video.size(0))
    unimp_frames = [int(f) for f in all_frames if f not in imp_frames]
    random.shuffle(unimp_frames)

    pred_logits = []
    per_changes = []
    for i in range(0, len(unimp_frames)):
        rem_ar = unimp_frames[0:i+1]
        valid_idx = all_frames[~np.isin(all_frames, rem_ar)].tolist()
        clusters = func.temporal_freeze(idx_list=valid_idx)
        v = func.create_new_video(video.permute(1,0,2,3), clusters['cluster_ids'], clusters['ordered_keys'])
        stats = func.get_pred_stats(model, v, gt_class=gt_class_idx, orig_pred_logit=pred_logit)
        pred_logits.append(stats['logit'])
        per_changes.append(max(1e-5, stats['per_change']))
    #calc AUC
    AUC_remove = sum(per_changes)
    metrics['AUC_remove'] = AUC_remove

    #add important frames one by one and see how the prediciton is affected
    

    return metrics







    pass

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

        #we consider only correctly lassified samples
        if gt_class.lower() != pred_class.lower():
            continue
        sfs = d['single_frame_structure']

        g = k.split('_')[2][1:]
        c = k.split('_')[3][1:]
        cls_name = d['motion_importance']['gt_class']
        vid_path = ucf101dm.construct_vid_path(cls_name,g,c)
        video = ucf101dm.load_jpg_ucf101(vid_path,n=0).to(device)
        structure_metrics(video, d, gt_class_idx, pred_logit)









        pass
