import os
import sys
import numpy as np
import pandas as pd
import torch
import json
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


root = r'C:\Users\lahir\data\k400'
device = "cpu"

csv_path = os.path.join(root, 'val.csv')
df = pd.read_csv(csv_path)
cls_df = df
# df[df['label']=='archery']


# Pick a pretrained model and load the pretrained weights
model_name = "x3d_s"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()


with open("dataloaders/kinetics_classnames.json", "r") as f:
    kinetics_classnames_ = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
kinetics_classnames = {}
for k, v in kinetics_classnames_.items():
    kinetics_id_to_classname[v] = str(k).strip('"\'')
    kinetics_classnames[str(k).strip('\'"')] = v



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
    yt_id = cls_df.iloc[i]['youtube_id']
    class_name = cls_df.iloc[i]['label']
    class_id = kinetics_classnames[class_name]
    time_start = int(cls_df.iloc[i]['time_start'])
    time_end = int(cls_df.iloc[i]['time_end'])
    video_path = os.path.join(root, f'{yt_id}_{time_start:06d}_{time_end:06d}.mp4')
    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = transform(video_data)
    inputs = video_data["video"][None, ...]
    return inputs, class_id, class_name

def test_model():
    gt_labels = []
    pred_labels = []
    for i in range(len(cls_df)):
        print(f'Processing video {i+1}/{len(cls_df)}',end='\r')
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