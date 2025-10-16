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

path = r'C:\Users\lahir\Downloads\test_out\40'
# list all files under `path` (recursively) and print their full paths


p = os.path.join(path, '00005.png')
img = Image.open(p).convert('RGB')
img = np.array(img,dtype=np.uint32)

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



dir_path = r'C:\Users\lahir\Downloads\test_out\35'
video = read_video(dir_path)
gray = video_recolor(video)


flow = func.read_flow_yaml(r'C:\Users\lahir\Downloads\UCF101\raft_flow\Bowling\v_Bowling_g06_c06\flow_3.txt')
flow_mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
show_gray_img(flow_mag)


pass
