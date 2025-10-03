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
transform = inference_loader.dataset.spatial_transform

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
    
#video shape : 3, t , h , w
#where t is the number of frames
def calc_flow(video):
    flow = torch.empty(0)
    for i in range(0, video.size(1)-1):
        f0_ = video[:,i+1,:,:].cpu().numpy().transpose(1, 2, 0)
        f0_ = cv2.cvtColor(f0_, cv2.COLOR_BGR2GRAY)
        f0_= cv2.normalize(f0_, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if f0_.shape[0]!=112:
            f0_ = cv2.resize(f0_, (112, 112), interpolation=cv2.INTER_LINEAR)
        f1_ = video[:,i,:,:].cpu().numpy().transpose(1, 2, 0)
        f1_ = cv2.cvtColor(f1_, cv2.COLOR_BGR2GRAY)
        f1_= cv2.normalize(f1_, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if f1_.shape[0]!=112:
            f1_ = cv2.resize(f1_, (112, 112), interpolation=cv2.INTER_LINEAR)
        flow_ = cv2.calcOpticalFlowFarneback(
        f0_, f1_, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
        )
        flow = torch.cat((flow,torch.tensor(flow_).unsqueeze(0)),dim=0)
    return flow

def warp_video(video, flow):
    B, C, H, W = video.shape
    flow = flow.permute(0,3,1,2)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=video.device),
        torch.arange(0, W, device=video.device),
        indexing='ij'
    )
    # Normalize coordinates to [-1, 1]
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    # Expand to batch size
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    flow_normalized = torch.stack([
        2.0 * flow[:, 0] / (W - 1),  # dx normalized
        2.0 * flow[:, 1] / (H - 1)   # dy normalized
    ], dim=-1).permute(0, 3, 1, 2)

    new_grid = grid + flow_normalized.permute(0, 2, 3, 1)
    warped = F.grid_sample(
        video, 
        new_grid, 
        mode='bilinear', 
        padding_mode='zeros',  # or 'border', 'reflection'
        align_corners=True
    )

    return warped

def input_flow_grad(video):
    delta = 0.01 # ratio of the original flow to change
    flow = calc_flow(video)
    # warped = warp_video(video[:,1:,:].permute(1,0,2,3),flow)
    # play_tensor_video_opencv(warped,fps=2)
    # play_tensor_video_opencv(video[:,1:,:].permute(1,0,2,3),fps=2)
    f_delta = torch.ones_like(flow) * delta

    delta_x = f_delta.clone()
    delta_x[:,:,:,1] = 0
    delta_y = f_delta.clone()
    delta_y[:,:,:,0] = 0

    v = video[:,:-1,:].permute(1,0,2,3)
    warped_x = warp_video(v, flow+delta_x)
    warped_y = warp_video(v, flow+delta_y)

    v1 = video[:,1:,:].permute(1,0,2,3)
    dI_x = (warped_x - v1) / delta_x[:,:,:,0][:,None,:,:]
    dI_y = (warped_y - v1) / delta_y[:,:,:,1][:,None,:,:]

    d = torch.concat([dI_x[:,:,:,:,None],dI_y[:,:,:,:,None]],dim=-1)
    d = torch.max(d,dim=1)[0]
    # d = (d-d.min())/(d.max()-d.min()+1e-5)
    # flow = (flow-flow.min())/(flow.max()-flow.min())
    
    # h = flow*d
    # d = torch.mean(d**2,dim=-1)

    # d = (dI_x**2+dI_y**2)**0.5
    # d = (d-d.min())/(d.max()-d.min()+1e-5)
    # play_tensor_video_opencv(d,fps=1)

    # import matplotlib.pyplot as plt
    # d_ = cv2.cvtColor(d[6,:].permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
    # plt.imshow(d_, cmap='gray')
    # plt.show()

    return d,flow

def gradcam_flow():
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels_map[t[0].split('_')[1].lower()] for t in targets]
        dI_dF, flow = input_flow_grad(inputs[0])


        # dI_dF = torch.mean(dI_dF,dim=1)
        # dI_dF = (dI_dF-dI_dF.min())/(dI_dF.max()-dI_dF.min()+1e-5)

        #calculate saliency map
        inputs.requires_grad = True
        pred = model(inputs)
        pred_cls = torch.argmax(pred,dim=1)
        
        i=0
        if cls[i]==pred_cls[i]:
            model.zero_grad()
            score = pred[i, pred_cls[i]]
            score.backward()
            slc,_ = torch.max(torch.abs(inputs.grad),dim=1)
            slc = slc[i,1:,:]
            slc = slc[:,:,:,None].repeat(1,1,1,2)
            dPred_dF = slc * dI_dF
            # dPred_dF = torch.mean(dPred_dF,dim=(1,2),keepdim=True)
            flowcam = dPred_dF
            # flowcam = dPred_dF * flow
            flowcam =  F.relu(torch.sum(flowcam,dim=3))
            # flowcam = 1-flowcam
            flowcam = (flowcam-flowcam.min())/(flowcam.max()-flowcam.min())

            video = inputs[0][:,1:,:]
            video = video.permute(1,2,3,0).detach().numpy()
            video = np.uint8((video - video.min())/(video.max() - video.min() + 1e-5)*255)

            transformed = np.uint8(flowcam[None,:].detach().numpy()*255)
            h_col = np.concatenate([cv2.applyColorMap(img, cv2.COLORMAP_JET)[None,:] for img in list(transformed[0])],axis=0)
            final_img = cv2.addWeighted(video, 0.6, h_col, 0.4, 0)
            final_img = torch.tensor(final_img).permute(0,3,1,2)
            # play_tensor_video_opencv(final_img,fps=2)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(os.path.join('C:\\Users\\lahir\\Downloads\\UCF101\\gradflow_salient\\',f'{idx}.mp4'), fourcc, 2, (112, 112))
            
            # Write each frame
            for i in range(final_img.size(0)):
                frame = final_img[i].permute(1, 2, 0).numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            video_writer.release()


        # flowcam = F.relu(torch.sum(dPred_dF * flow,dim=-1))
        # flowcam = torch.sum(flowcam**2,dim=-1)

        # v = (v-v.min())/(v.max()-v.min())
        # flowcam = (flowcam-flowcam.min())/(flowcam.max()-flowcam.min())
        # display_vid = torch.concat([inputs[0].permute(1,0,2,3)[1:],flowcam[:,None,:].repeat(1,3,1,1)],dim=-2)
        # play_tensor_video_opencv(display_vid,fps=2)
        # play_tensor_video_opencv(flowcam[:,None,:].repeat(1,3,1,1),fps=2)
    
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
Acuracy : 0.8575338233022137
'''

def test():
    n_samples = 0
    n_correct = 0
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels_map[t[0].split('_')[1].lower()] for t in targets]
        with torch.inference_mode():
            pred = model(inputs)
            pred_cls = torch.argmax(pred,dim=1)
            n_samples += len(pred_cls)
            n_correct += ((pred_cls == torch.tensor(cls)).sum()).item()

    # root = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"
    # for k in inference_class_names.keys():
    #     print(f'Inferring class {k} out of {len(list(inference_class_names.keys()))}')
    #     class_path = os.path.join(root,inference_class_names[k])
    #     dirs = [item.name for item in Path(class_path).iterdir() if item.is_dir()]

    #     for d in dirs:
    #         l = k
    #         g = int(d.split('_')[2][1:])
    #         c = int(d.split('_')[3][1:])
    #         n=0
    #         video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)
    #         with torch.inference_mode():
    #             n_samples += 1
    #             pred = model(video.unsqueeze(0)).cpu().numpy().argmax()
    #             if int(pred) == k:
    #                 n_correct += 1
    print(f'Acuracy : {n_correct/n_samples}')

def copy_paste_frame(frames_in, src_idx, dst_idx):
    frames = frames_in.clone()
    frames[:,dst_idx,:] = frames[:,src_idx,:]
    return frames

def get_video_frame_motion_importance(video):
    pred = model(video.unsqueeze(0))
    pred_cls = torch.argmax(pred)
    pred_l = pred[0,pred_cls]
    n_frames = video.size(1)
    logits_frame = []
    for n in range(1, n_frames):
        inputs = copy_paste_frame(video, n-1, n)
        pred_ = model(inputs.unsqueeze(0))
        pred_cls_ = torch.argmax(pred_)
        pred_l_ = pred_[0,pred_cls_]
        logits_frame.append(pred_l_.item())

    out = {
        'original_logit': pred_l.item(),
        'logits_frame': logits_frame
    }
    return out

def replace_video(input, src_idx):
    frames = input.clone()
    return frames[:,src_idx,:,:][:,None,:].repeat(1,input.size(1),1,1)

def frozen_motion_importance(inputs,idx):
    n_frames = inputs.size(1)
    logits_list = []
    for n in range(0, n_frames):
        inputs_frozen = replace_video(inputs, n)
        # play_tensor_video_opencv(inputs_frozen['pixel_values'][0], fps=2)
        with torch.no_grad():
            pred = model(inputs_frozen.unsqueeze(0))
            pred_cls = torch.argmax(pred)
            pred_l = pred[0,pred_cls]
            logits_list.append(pred_l.item())
    return logits_list

def frame_motion_importance():
    import csv

    out_path = r'C:\Users\lahir\Downloads\UCF101\tmp_imp.csv'
    root = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"
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
                pred = model(video.unsqueeze(0)).cpu().numpy().argmax()
                if int(pred) == k:
                    imp = get_video_frame_motion_importance(video)
                    frozen_logits = frozen_motion_importance(video,0)
                    p = os.path.join(root,d)
                    imp['path'] = p
                    imp['frozen_logits'] = frozen_logits
                    imp['orig_class'] = k
                    with open(out_path, 'a', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=imp.keys())
                        if os.path.getsize(out_path) == 0:
                            writer.writeheader()
                        writer.writerows([imp])

if __name__ == '__main__':
    gradcam_flow()


