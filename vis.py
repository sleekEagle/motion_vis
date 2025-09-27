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
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch
import cv2
import av
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, VideoMAEModel
from transformers import BatchFeature


root = r'C:\Users\lahir\data\kinetics400\val\val_256'
device = "cpu"


def read_video_pyav(container,clip_len=16):
    frames = []
    container.seek(0)
    
    for i, frame in enumerate(container.decode(video=0)):
        frames.append(frame)

    start_idx = np.random.randint(0, max(1, len(frames) - clip_len))
    end_idx = start_idx + clip_len - 1
    frames = frames[start_idx:end_idx + 1]

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


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
# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# List all directories in the root path
data_list = []
# dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d=='abseiling']
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
for d in dirs:
    label = kinetics_classnames[d]
    dir_path = os.path.join(root, d)
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    for f in files:
        data_list.append({'video_path': os.path.join(dir_path, f), 'label': label, 'class': d})

'''
model from : 
https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics
'''
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")



def get_input(i):
    path = data_list[i]['video_path']
    class_name = data_list[i]['class']
    class_id = data_list[i]['label']

    try:
        container = av.open(path)
        video = read_video_pyav(container,clip_len=16)
    except Exception as e:
        print(f'Error processing video {i}: {e}')

    inputs = image_processor(list(video), return_tensors="pt")

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

        #pytorch model
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f'GT class: {class_name} Predicted class: {model.config.id2label[predicted_class_idx]}')

        gt_labels.extend([class_id])
        pred_labels.extend([predicted_class_idx])

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

def copy_paste_frame(batch_feature, src_idx, dst_idx):
    frames = batch_feature['pixel_values'].clone()
    frames[:,dst_idx,:] = frames[:,src_idx,:]

    batch_feature = BatchFeature(data={
        'pixel_values': frames
    })

    return batch_feature

def replace_video(batch_feature, src_idx):
    frames = batch_feature['pixel_values'].clone()
    frames[:,:,:] = frames[:,src_idx,:]

    batch_feature = BatchFeature(data={
        'pixel_values': frames
    })

    return batch_feature


def play_tensor_video_opencv(tensor, fps=30, window_name="Tensor Video"):
    # Convert tensor to numpy and ensure correct format
    if isinstance(tensor, torch.Tensor):
        frames = tensor.detach().cpu().numpy()
    else:
        frames = tensor
    
    # Ensure shape: [T, C, H, W] -> [T, H, W, C] for OpenCV
    if frames.shape[1] == 3:  # [T, C, H, W]
        frames = frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
    
    # Convert RGB to BGR for OpenCV
    frames = frames[..., ::-1]  # RGB -> BGR
    
    # Normalize to 0-255 if needed
    frames =  (frames - frames.min())/(frames.max() - frames.min() + 1e-5)
    frames = frames * 255.0
    frames = frames.astype(np.uint8)
    
    # Play video
    for frame in frames:
        cv2.imshow(window_name, frame)
        
        # Wait for key press or delay
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()

def get_video_frame_motion_importance(inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    l = logits[0,predicted_class_idx]

    # inputs = copy_paste_frame(inputs, 0, 1)
    n_frames = inputs['pixel_values'].size(1)
    logits_frame = []
    for n in range(1, n_frames):
        inputs = copy_paste_frame(inputs, n-1, n)
        outputs = model(**inputs)
        logits_frame.append(outputs.logits[0,predicted_class_idx].item())

    out = {
        'predicted_class': model.config.id2label[predicted_class_idx],
        'predicted_class_id': predicted_class_idx,
        'original_logit': l.item(),
        'logits_frame': logits_frame
    }
    return out

def frozen_motion_importance(inputs,idx):
    n_frames = inputs['pixel_values'].size(1)
    logits_list = []
    for n in range(0, n_frames):
        inputs_frozen = replace_video(inputs, n)
        # play_tensor_video_opencv(inputs_frozen['pixel_values'][0], fps=2)
        with torch.no_grad():
            outputs = model(**inputs_frozen)
            logits = outputs.logits
            l = logits[0,idx].item()
            logits_list.append(l)
    return logits_list
    


def frame_motion_importance():
    import csv

    last_cls = None
    n_samples = 0
    skip = False
    out_path = r'C:\Users\lahir\data\kinetics400\val\tmp_imp.csv'
    for i in range(len(data_list)):
        print(f'Processing video {i+1}/{len(data_list)}',end='\r')
        cls = data_list[i]['class']
        if not last_cls:
            last_cls = cls
        if skip and cls != last_cls:
            skip = False
        if skip: continue
        inputs, class_id, class_name = get_input(i)
        #check if the prediction is correct
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        if predicted_class_idx != class_id:
            print(f'Skipping video {i} as prediction is incorrect')
            continue

        imp = get_video_frame_motion_importance(inputs)
        frozen_logits = frozen_motion_importance(inputs,class_id)
        path = data_list[i]['video_path']
        n_samples += 1
        last_cls = cls
        if n_samples==5:
            n_samples = 0
            skip = True
        
        #write to file
        imp['path'] = path
        imp['frozen_logits'] = frozen_logits
        imp['orig_class'] = class_id
        with open(out_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=imp.keys())
            if os.path.getsize(out_path) == 0:
                writer.writeheader()
            # No writeheader() when appending
            writer.writerows([imp])

if __name__ == "__main__":
    frame_motion_importance()