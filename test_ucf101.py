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

        #feature extraction model for video clustering
        # weights = ResNet50_Weights.IMAGENET1K_V1
        # resnetmodel = resnet50(weights=weights)
        # self.feature_model = create_feature_extractor(resnetmodel, return_nodes={'avgpool':'f_layer'})
        # self.feature_tr = weights.transforms()

        self.model.layer4[2].conv3.register_forward_hook(self.save_activations)
        self.model.layer4[2].conv3.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        # grad_out[0][0,1:,:].min(), grad_out[0][0,1:,:].max()
        self.gradients = grad_out

    def forward(self, x):
        out = self.model(x)
        return out
    
    def replace_video(self, input, src_idx):
        frames = input.clone()
        return frames[:,src_idx,:,:][:,None,:].repeat(1,input.size(1),1,1)
    
    def copy_paste_frame(self, frames_in, src_idx, dst_idx):
        frames = frames_in.clone()
        frames[:,dst_idx,:] = frames[:,src_idx,:]
        return frames
        
    def cluster_frames(self, x):
        x = x.permute(1,0,2,3)
        x = self.feature_tr(x)
        f = self.feature_model(x)['f_layer'].squeeze()
        f_norm = ((f-f.min())/(f.max()-f.min()+1e-5)).detach().numpy()
        clustering = DBSCAN(eps=10, min_samples=2)
        print(clustering.fit_predict(f_norm))

        play_tensor_video_opencv(x,fps=5)
    
    def motion_importance(self, x):
        pred = self.forward(x[None,:])
        pred = F.softmax(pred,dim=1)
        pred_cls_orig = torch.argmax(pred,dim=1).item()
        pred_l_orig = pred[0,pred_cls_orig].item()
        l_list = []
        for i in range(x.size(1)):
            x_rep = self.replace_video(x,i)
            # play_tensor_video_opencv(x_rep.permute(1,0,2,3),fps=2)
            pred = self.forward(x_rep[None,:])
            pred = F.softmax(pred,dim=1)
            pred_cls = torch.argmax(pred,dim=1).item()
            pred_l = pred[0,pred_cls].item()
            l_list.append(pred_l)

        out = {
            'orig_pred_cls': pred_cls_orig,
            'orig_l': pred_l_orig,
            'l_list': l_list
        }
        return out
    
    def frame_motion_importance(self, x):
        pred = self.forward(x[None,:])
        pred = F.softmax(pred,dim=1)
        pred_cls_orig = torch.argmax(pred,dim=1).item()
        pred_l_orig = pred[0,pred_cls_orig].item()

        l_list = []
        for i in range(x.size(1)-1):
            x_cp = self.copy_paste_frame(x, i, i+1)
            pred = self.forward(x_cp[None,:])
            pred = F.softmax(pred,dim=1)
            pred_cls = torch.argmax(pred,dim=1).item()
            pred_l = pred[0,pred_cls].item()
            l_list.append(pred_l)
        
        out = {
            'orig_pred_cls': pred_cls_orig,
            'orig_l': pred_l_orig,
            'l_list': l_list
        }
        return out
    
    def calc_gradcam(self, x):
        act = self.activations
        grad = self.gradients[0]
        grad = grad.mean(dim=(2,3,4),keepdim=True)
        cam = act * grad
        cam = F.relu(cam.sum(dim=1,keepdim=True))
        cam_int = F.interpolate(cam,
                            size=(16,112,112),           # Target size
                            mode='trilinear',         # 'nearest' | 'bilinear' | 'bicubic'
                            align_corners=False      # Set True for some modes
                        ).squeeze(dim=1)
        cam_int = (cam_int - cam_int.min())/(cam_int.max() - cam_int.min() + 1e-5)
        return cam_int
    
    def calc_flow_saliency(self, x, i=0):
        slc,_ = torch.max(x.grad ,dim=1)
        slc = slc[i,1:,:]
        slc = slc[:,:,:,None].repeat(1,1,1,2)

        x = x.detach()
        x = x[i,:]
        dI_dF, _ = func.input_flow_grad(x)
        dPred_dF = slc * dI_dF
        dPred_dF[:,:5,:] = 0
        dPred_dF[:,-5:,:] = 0
        dPred_dF[:,:,:5] = 0
        dPred_dF[:,:,-5:] = 0

        # from matplotlib import pyplot as plt
        # plt.imshow(dPred_dF[0,:,:,0].cpu().numpy())
        # plt.show()

        dPred_dF =  F.relu(torch.sum(dPred_dF,dim=3))
        dPred_dF = (dPred_dF-dPred_dF.min())/(dPred_dF.max()-dPred_dF.min())
        return dPred_dF

gmodel = GradcamModel(model)

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



def motion_importance():
    import csv
    out_path = r'C:\Users\lahir\Downloads\UCF101\motion_importance.csv'
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = class_labels_map[targets[0][0].split('_')[1].lower()]
        # gmodel.cluster_frames(inputs[0])
        out = gmodel.motion_importance(inputs[0])
        out['vid_name'] = targets[0][0]
        out['gt_cls'] = cls

        with open(out_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=out.keys())
            if os.path.getsize(out_path) == 0:
                writer.writeheader()
            writer.writerows([out])

def process_motion_importance():
    import pandas as pd
    df_path = r'C:\Users\lahir\Downloads\UCF101\motion_importance.csv'
    df_out_path = r'C:\Users\lahir\Downloads\UCF101\motion_importance_processed.csv'

    df = pd.read_csv(df_path)
    def getmax(x):
        x = x.replace('[','').replace(']','').split(',')
        x = np.array([float(x_) for x_ in x]).max()
        return x
    df['l_max'] = df['l_list'].apply(lambda x: getmax(x))
    df['reduction_ratio'] = (df['orig_l'] - df['l_max'])/df['orig_l']
    df.to_csv(df_out_path)

def show_videos():
    #examined 140 videos
    for idx, batch in enumerate(inference_loader):
        print(f'{idx} of {len(inference_loader)}')
        inputs, targets = batch
        video = rearrange(inputs, 'b c t h w -> (b t) c h w')
        play_tensor_video_opencv(video)
        pass


def frame_motion_importance():
    THR = 0.4

    out_path = r'C:\Users\lahir\Downloads\UCF101\frame_motion_importance.csv'
    df_path_motion_imp = r'C:\Users\lahir\Downloads\UCF101\motion_importance_processed.csv'
    df = pd.read_csv(df_path_motion_imp)

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        target = targets[0][0]
        cls = class_labels_map[targets[0][0].split('_')[1].lower()]
        #check if motion is important
        if float(df[df['vid_name']==target]['reduction_ratio'].values[0]) < THR:
            continue

        out = gmodel.frame_motion_importance(inputs[0])
        out['vid_name'] = target
        with open(out_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=out.keys())
            if os.path.getsize(out_path) == 0:
                writer.writeheader()
            writer.writerows([out])

def motion_saliency():
    THR = 0.004
    out_path = r'C:\Users\lahir\Downloads\UCF101\flow_saliency_imgs1'
    frmo_imp = r'C:\Users\lahir\Downloads\UCF101\frame_motion_importance.csv'
    df = pd.read_csv(frmo_imp)

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        target = targets[0][0]
        cls = class_labels_map[targets[0][0].split('_')[1].lower()]
        if not target in df['vid_name'].values:
            continue
        pass
        
        #get the important frames in terms of motion
        row = df[df['vid_name']==target].iloc[0]
        l_orig = row['orig_l']
        l_list = row['l_list']
        l_list = np.array([float(val) for val in l_list.replace('[','').replace(']','').split(',')])
        l_red = np.maximum((l_orig - l_list)/l_orig,0)
        valid_idx = np.argwhere(l_red > THR)[:,0]
        if len(valid_idx) == 0:
            continue
        from matplotlib import pyplot as plt
        # plt.plot(l_red)
        # plt.show()

        #calculate gracam
        gmodel.zero_grad()
        inputs.requires_grad = True
        pred = gmodel(inputs[0][None,:])
        pred_idx = torch.argmax(pred,dim=1)
        pred[0,pred_idx].backward()
        gcam = gmodel.calc_gradcam(inputs[0])
        dPred_dF = gmodel.calc_flow_saliency(inputs,0)

        dPred_dF_cam = gcam[0,1:,:] * dPred_dF

        #save images
        vid_dir = os.path.join(out_path, target)
        os.makedirs(vid_dir,exist_ok=True)
        for idx in valid_idx:
            img = inputs[0][:,idx,:].squeeze().permute(1,2,0).detach().numpy()
            img = np.uint8((img - img.min())/(img.max() - img.min() + 1e-5)*255)
            cv2.imwrite(os.path.join(vid_dir,f'img_{idx}.png'), img)

            gcam_ = gcam[0,idx,:].unsqueeze(-1).repeat(1,1,3)
            gcam_ = (gcam_ - gcam_.min())/(gcam_.max() - gcam_.min() + 1e-5)
            gcam_ = np.uint8(gcam_.detach().numpy()*255)
            cv2.imwrite(os.path.join(vid_dir,f'gradcam_{idx}.png'), gcam_)

            dPred_dF_ = dPred_dF[idx,:].unsqueeze(-1).repeat(1,1,3)
            dPred_dF_ = (dPred_dF_ - dPred_dF_.min())/(dPred_dF_.max() - dPred_dF_.min() + 1e-5)
            dPred_dF_ = np.uint8(dPred_dF_.detach().numpy()*255)
            cv2.imwrite(os.path.join(vid_dir,f'motion_sal_{idx}.png'), dPred_dF_)

def gradcam_sal():
    img_path = r'C:\Users\lahir\Downloads\UCF101\flow_saliency_imgs'
    dirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
    for d in dirs:
        print(f'Processing {d} out of {len(dirs)} dirs.', end='\r')
        p = os.path.join(img_path, d)
        img_files = [f for f in os.listdir(p) if (os.path.isfile(os.path.join(p,f)) and f.startswith('img'))]
        nums = [int(f.split('_')[1].split('.')[0]) for f in img_files]
        out_dir = os.path.join(p, 'gcam_sal')
        os.makedirs(out_dir, exist_ok=True)
        for n in nums:
            img = cv2.imread(os.path.join(p,f'img_{n}.png'))
            gcam = cv2.imread(os.path.join(p,f'gradcam_{n}.png'))
            gcam = (gcam-gcam.min())/(gcam.max()-gcam.min()+1e-5)
            msal = cv2.imread(os.path.join(p,f'motion_sal_{n}.png'))
            msal = (msal-msal.min())/(msal.max()-msal.min()+1e-5)

            msal_gcam = gcam * msal
            msal_gcam = np.uint8((msal_gcam-msal_gcam.min())/(msal_gcam.max()-msal_gcam.min())*255)
            h_col = cv2.applyColorMap(msal_gcam, cv2.COLORMAP_JET)
            final_img = cv2.addWeighted(img, 0.6, h_col, 0.4, 0)

            cv2.imwrite(os.path.join(out_dir,f'{n}.png'), final_img)

if __name__ == '__main__':
    motion_saliency()

