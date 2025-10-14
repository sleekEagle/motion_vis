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

opt_path = "models/r3d/ucf101.json"
with open(opt_path, "r") as f:
    model_opt = json.load(f)
model_opt = Namespace(**model_opt)

for attribute in dir(model_opt):
    if "path" in str(attribute) and getattr(model_opt, str(attribute)) != None:
        setattr(model_opt, str(attribute), Path(getattr(model_opt, str(attribute))))
inference_loader, inference_class_names = get_inference_utils(model_opt)
class_labels_map = {v.lower(): k for k, v in inference_class_names.items()}
transform = inference_loader.dataset.spatial_transform


raftof = func.RAFT_OF()

def main():
    out_path = r'C:\Users\lahir\Downloads\UCF101\raft_flow'
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        t = targets[0]
        cls = targets[0][0].split('_')[1]
        vid  = inputs[0]
        vid = ((vid - vid.min()) / (vid.max() - vid.min())*255).to(torch.uint8)

        flows = raftof.predict_flow_video(vid.permute(1,0,2,3))
        f_re = F.interpolate(
            flows, 
            size=(240, 320), 
            mode='bilinear', 
            align_corners=False 
        )

        # import matplotlib.pyplot as plt
        # f_re = f_re.sum(dim=1)
        # f_re = (f_re - f_re.min()) / (f_re.max() - f_re.min())
        # plt.imshow(f_re[0,:,:].cpu(), cmap='gray')
        # plt.show()
        dir_path = os.path.join(out_path, cls, t[0])
        os.makedirs(dir_path, exist_ok=True)

        for f_idx in range(f_re.size(0)):
            f_ = f_re[f_idx,:,:,:].permute(1,2,0).cpu().numpy()
            func.write_flow_yaml(f_, os.path.join(dir_path ,f'flow_{f_idx}.txt'))


if __name__ == "__main__":
    main()