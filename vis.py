import os
import sys
import numpy as np
import pandas as pd
import torch
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict


root = r'C:\Users\lahir\data\kinetics400\val\val_256'
device = "cpu"

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)
with open(json_filename, "r") as f:
    kinetics_classnames_ = json.load(f)
kinetics_classnames = {}
for k, v in kinetics_classnames_.items():
    s = str(k).replace('"', "")
    s = s.replace(' ', '_')
    if 'passing' in s:
        pass
    kinetics_classnames[s] = v

# List all directories in the root path
data_list = []
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
for d in dirs:
    label = kinetics_classnames[d]
    dir_path = os.path.join(root, d)
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    for f in files:
        data_list.append({'video_path': os.path.join(dir_path, f), 'label': label, 'class': d})




# Pick a pretrained model and load the pretrained weights
model_name = "x3d_s"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()


# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

####################
# SlowFast transform
####################
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# Get transform parameters based on model
transform_params = model_transform_params[model_name]

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
# The duration of the input clip is also specific to the model.
start_sec = 0
end_sec = start_sec + clip_duration

def get_input(i):
    path = data_list[i]['video_path']
    class_name = data_list[i]['class']
    class_id = data_list[i]['label']
    video = EncodedVideo.from_path(path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = transform(video_data)
    inputs = video_data["video"][None, ...]
    return inputs, class_id, class_name


'''
Accuracy: 0.6269
'''

def test_model():
    gt_labels = []
    pred_labels = []
    n_missing = 0
    for i in range(len(data_list)):
        print(f'Processing video {i+1}/{len(data_list)}',end='\r')
        try:
            inputs, class_id, class_name = get_input(i)
        except Exception as e:
            print(f'Error processing video {i}: {e}')
            n_missing += 1

        preds = model(inputs)

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices[0]
        gt_labels.extend([class_id])
        pred_labels.extend([pred_classes[0].item()])

        # # Map the predicted classes to the label names
        # pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
        # print(f'GT label: {class_name}')
        # print("Predicted labels: %s" % ", ".join(pred_class_names))
        # print('***********************')

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    accuracy = (gt_labels == pred_labels).sum()/len(gt_labels)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Missing videos: {n_missing} outpf {len(data_list)} ')


def vis_model():
    gt_labels = []
    pred_labels = []
    for i in range(len(cls_df)):
        print(f'Processing video {i+1}/{len(cls_df)}')
        inputs, class_id, class_name = get_input(i)

        preds = model(inputs)

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices
        gt_labels.extend([class_id])
        pred_labels.extend([pred_classes[0][0].item()])

        # # Map the predicted classes to the label names
        # pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
        # print(f'GT label: {class_name}')
        # print("Predicted labels: %s" % ", ".join(pred_class_names))
        # print('***********************')

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    accuracy = (gt_labels == pred_labels).sum()/len(gt_labels)
    print(f'Accuracy: {accuracy:.4f}')

def get_KLDiv(p, q):
    import torch.nn.functional as F
    log_p = torch.log(p)
    log_q = torch.log(q)
    kl_divergence = F.kl_div(log_q, p, reduction='batchmean') 
    return kl_divergence


def motion_importance():
    import torch.nn.functional as F

    NPASS = 10
    for i in range(len(cls_df)):
        inputs, class_id, class_name = get_input(i)
        preds = model(inputs)
        preds_per_list = []
        for n in range(NPASS):
            idx = torch.randperm(inputs.shape[2])
            inputs_per = inputs.index_select(2, idx)
            preds_per = model(inputs_per)
            preds_per_list.append(preds_per)

        kldivs = [get_KLDiv(preds,pred_per).item() for pred_per in preds_per_list]
        kldivs_mean = sum(kldivs)/len(kldivs)
        
        pass

if __name__ == "__main__":
    test_model()