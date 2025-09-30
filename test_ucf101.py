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


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_jpg_ucf101(l, g, c, n, inference_class_names, transform):
    name = inference_class_names[l]
    dir = os.path.join(
        "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs", name, "v_{}_g{}_c{}".format(name, str(g).zfill(2), str(c).zfill(2))
    )
    path = sorted(glob(dir + "/*"), key=numericalSort)

    target_path = path[n * 16 : (n + 1) * 16]
    if len(target_path) < 16:
        print("not exist")
        return False

    video = []
    for _p in target_path:
        video.append(transform(Image.open(_p)))

    return torch.stack(video)

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


def main():
    # inputs, targets = iter(inference_loader).__next__()
    # video_size = inputs[[0]].shape
    transform = inference_loader.dataset.spatial_transform

    # _transforms = transform.transforms
    # idx = [type(i).__name__ for i in _transforms].index('Normalize')
    # normalize = _transforms[idx]
    # mean = torch.tensor(normalize.mean)
    # std = torch.tensor(normalize.std)

    # unnormalize = transforms.Compose(
    #     [
    #         Normalize((-mean / std).tolist(), (1 / std).tolist()),
    #         ToPILImage(),
    #     ]
    # )

    # spatial_crop_size = 16
    # spatial_stride = 8
    # temporal_stride = 2

    l = 21
    g = 1  # > 0
    c = 1  # > 0
    n = 1

    video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)
    target = l
    with torch.inference_mode():
        pred = model(video.unsqueeze(0)).cpu().numpy().argmax()

    pass
    # video_orgimg = []
    # for i in range(video_size[2]):
    #     img = video.squeeze().transpose(0, 1)[i]
    #     video_orgimg.append(np.array(unnormalize(img)))
    # video_orgimg = np.array(video_orgimg)

class GradcamModel(nn.Module):
    def __init__(self, model):
        super(GradcamModel, self).__init__()
        self.model = model

        self.model.layer4[2].conv3.register_forward_hook(self.save_activations)
        self.model.layer4[2].conv3.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        # grad_out[0][0,1:,:].min(), grad_out[0][0,1:,:].max()s
        self.gradients = grad_out

    def forward(self, x):
        out = self.model(x)
        return out
    
gmodel = GradcamModel(model)

def gradcam():
    root = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"
    transform = inference_loader.dataset.spatial_transform

    for k in inference_class_names.keys():
        print(f'Inferring class {k} out of {len(list(inference_class_names.keys()))}')
        class_path = os.path.join(root,inference_class_names[k])
        dirs = [item.name for item in Path(class_path).iterdir() if item.is_dir()]

        for d in dirs:
            l = k
            g = int(d.split('_')[2][1:])
            c = int(d.split('_')[3][1:])
            n=0
            video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)

            pred = gmodel(video.unsqueeze(0))
            lpred = pred.argmax()
            if int(lpred) == k:
                gmodel.zero_grad()
                pred[0,lpred].backward()

                act = gmodel.activations
                grad = gmodel.gradients[0]
                grad = grad.mean(dim=(2,3,4),keepdim=True)

                cam = act * grad
                cam = F.relu(cam.sum(dim=1,keepdim=True))
                cam_int = F.interpolate(cam,
                            size=(16,112,112),           # Target size
                            mode='trilinear',         # 'nearest' | 'bilinear' | 'bicubic'
                            align_corners=False      # Set True for some modes
                        ).squeeze(dim=1)
                cam_int = (cam_int - cam_int.min())/(cam_int.max() - cam_int.min() + 1e-5)
                
                video = video.permute(1,2,3,0)
                video = np.uint8((video - video.min())/(video.max() - video.min() + 1e-5)*255)

                transformed = np.uint8(cam_int.detach().numpy()*255)
                h_col = np.concatenate([cv2.applyColorMap(img, cv2.COLORMAP_JET)[None,:] for img in list(transformed[0])],axis=0)

                final_img = cv2.addWeighted(video, 0.6, h_col, 0.4, 0)
                final_img = torch.tensor(final_img).permute(0,3,1,2)
                
                play_tensor_video_opencv(final_img, fps=2)

'''
Acuracy : 0.9516516516516517
'''

def test():
    root = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"
    transform = inference_loader.dataset.spatial_transform
    n_samples = 0
    n_correct = 0
    for k in inference_class_names.keys():
        print(f'Inferring class {k} out of {len(list(inference_class_names.keys()))}')
        class_path = os.path.join(root,inference_class_names[k])
        dirs = [item.name for item in Path(class_path).iterdir() if item.is_dir()]

        for d in dirs:
            l = k
            g = int(d.split('_')[2][1:])
            c = int(d.split('_')[3][1:])
            n=0
            video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)
            with torch.inference_mode():
                n_samples += 1
                pred = model(video.unsqueeze(0)).cpu().numpy().argmax()
                if int(pred) == k:
                    n_correct += 1
    print(f'Acuracy : {n_correct/n_samples}')

if __name__ == '__main__':
    gradcam()


