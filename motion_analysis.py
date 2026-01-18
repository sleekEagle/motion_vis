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
    class_labels[cls_name] = k

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

def motion_importance_dataset():
    output_path = Path(r'C:\Users\lahir\Downloads\UCF101\analysis\motion_importance.json')
    change_threshold = 0.05
    anlysis_data = {}
    n_samples, n_correct = 0, 0
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        video = inputs[0].to('cuda')
        gt_class_name = targets[0][0].split('_')[1]
        seq = targets[0][1]
        ret = motion_importance(video.unsqueeze(0))
        pred_logit = ret['pred_original_logit']
        all_logits = ret['all_logits']
        percent_change = ret['percent_change']
        max_logit = ret['max_frame_logit']
        gt_class = class_labels[gt_class_name]
        if percent_change > change_threshold:
            sorted_indices = [i for i, _ in sorted(enumerate(all_logits), key=lambda x: x[1], reverse=True)]
            for i in range(2, len(sorted_indices)):
                valuable_indices = sorted_indices[0:i]
                clustered_ids = create_frame_cluster_idxs(valuable_indices)
                #update the video with only the valuable frames
                video_ = video_keep_given_frames(video, clustered_ids)
                pred_ = model(video_.unsqueeze(0))
                pred_ = F.softmax(pred_,dim=1)
                pred_cls_ = torch.argmax(pred_,dim=1)
                logit = pred_[:,gt_class].item()

                percent_change_ = (pred_logit - logit)/pred_logit
                if percent_change_ <= change_threshold:
                    #check if the motion among the frames are important for this prediction
                    pass

                pass




            pass










        seq = ','.join([str(s) for s in seq])
        ret['seq'] = seq
        ret['gt_class'] = gt_class_name 
        anlysis_data[idx] = ret
        #is the prediction correct
        pred = ret['pred_original_class']
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
        print(f'Frame {idx} taken from original frame {frame_cluster_idxs[idx]}')
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


if __name__ == '__main__':
    motion_importance_dataset()





