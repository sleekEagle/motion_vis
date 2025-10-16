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
import matplotlib.pyplot as plt

def read_video(path):
    files = [str(p) for p in Path(path).rglob('*') if p.is_file()]
    files.sort()
    video = np.empty((len(files),240,320,3),dtype=np.uint8)
    for f_idx, f in enumerate(files):
        img = Image.open(f).convert('RGB')
        img = np.array(img,dtype=np.uint32)
        video[f_idx,:,:,:] = img
        # func.play_tensor_video_opencv(video, fps=1)
    return video

def show_gray_img(img):
    plt.imshow(img, cmap='gray')
    plt.title(os.path.basename(p))
    plt.axis('off')
    plt.show()

def img_recolor(img):
    arr = img.reshape(-1,3)
    unique = np.unique(arr,axis=0)
    gray = np.zeros(arr.shape[0],dtype=np.uint8)

    for idx, color in enumerate(unique):
        mask = np.all(arr == color, axis=1)
        gray[mask] = idx

    gray = gray.reshape(img.shape[0],img.shape[1])
    return gray

def video_recolor(video):
    arr = video.reshape(-1,3)
    unique = np.unique(arr,axis=0)
    gray = np.zeros(arr.shape[0],dtype=np.uint8)
    for idx, color in enumerate(unique):
        mask = np.all(arr == color, axis=1)
        gray[mask] = idx
    gray = gray.reshape(video.shape[0],video.shape[1],video.shape[2])
    # func.play_tensor_video_opencv(torch.tensor(gray[:,:,:,None]).repeat(1,1,1,3), fps=1)
    return gray



#read model options
opt_path = "models/r3d/ucf101.json"
with open(opt_path, "r") as f:
    model_opt = json.load(f)
model_opt = Namespace(**model_opt)

model = generate_model(model_opt)
model = resume_model(model_opt.resume_path, model_opt.arch, model)
model.eval()

model_opt.inference_batch_size = 1
for attribute in dir(model_opt):
    if "path" in str(attribute) and getattr(model_opt, str(attribute)) != None:
        setattr(model_opt, str(attribute), Path(getattr(model_opt, str(attribute))))
inference_loader, inference_class_names = get_inference_utils(model_opt)
class_labels_map = {v.lower(): k for k, v in inference_class_names.items()}
transform = inference_loader.dataset.spatial_transform


def importance():
    THR = 1e-3
    frmo_imp = r'C:\Users\lahir\Downloads\UCF101\frame_motion_importance.csv'
    df = pd.read_csv(frmo_imp)
    for idx, batch in enumerate(inference_loader):
        inputs, targets = batch
        target = targets[0][0]
        print(f'{target} {idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        cls = class_labels_map[targets[0][0].split('_')[1].lower()]
        if not target in df['vid_name'].values:
            continue
        
        #get the important frames in terms of motion
        row = df[df['vid_name']==target].iloc[0]
        l_orig = row['orig_l']
        l_list = row['l_list']
        l_list = np.array([float(val) for val in l_list.replace('[','').replace(']','').split(',')])
        l_red = np.maximum((l_orig - l_list)/l_orig,0)
        valid_idx = np.argwhere(l_red > THR)[:,0]
        if len(valid_idx) == 0:
            continue

        pass
    video = inputs[0]

def mock_importance():
    video = read_video(r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01')
    video = video[:12]
    seg_video = read_video(r'C:\Users\lahir\Downloads\test_out\35')
    seg_video = video_recolor(seg_video)

    # func.play_tensor_video_opencv(torch.tensor(seg_video), fps=1)
    flow_dir = r'C:\Users\lahir\Downloads\UCF101\raft_flow\ApplyEyeMakeup\v_ApplyEyeMakeup_g01_c01'
    flow_files = sorted([str(p) for p in Path(flow_dir).rglob('*') if p.is_file()])
    flow = np.empty((len(flow_files),240,320,2),dtype=np.float32)
    for idx, f in enumerate(flow_files):
        flow[idx,:] = func.read_flow_yaml(f)

    # func.play_tensor_video_opencv(torch.tensor(flow_mag), fps=1)



    pass


# dir_path = r'C:\Users\lahir\Downloads\test_out\35'
# video = read_video(dir_path)
# gray = video_recolor(video)


# flow = func.read_flow_yaml(r'C:\Users\lahir\Downloads\UCF101\raft_flow\Bowling\v_Bowling_g06_c06\flow_3.txt')
# flow_mag = np.sqrt(flow[:,:,:,0]**2 + flow[:,:,:,1]**2)
# show_gray_img(flow_mag)


if __name__ == '__main__':
    mock_importance()

