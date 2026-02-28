import torch
import cv2
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import DBSCAN

'''
input is a tensor
input img shape: [H, W] if grayscale
or [H, W, 3] if RGB
'''
def show_img(img):
    import matplotlib.pyplot as plt
    img = img.detach().cpu().numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

#input img shape: [H, W], np array
def show_gray_image(gray,block=False):
    import matplotlib.pyplot as plt
    gray =  (gray - gray.min())/(gray.max() - gray.min() + 1e-5)
    gray = gray * 255.0
    gray = gray.astype(np.uint8)
    plt.imshow(gray, cmap='gray')
    plt.axis("off")
    plt.show(block=block)

#input img shape: [H, W, 3], np array
def show_rgb_image(img):
    import matplotlib.pyplot as plt
    img =  (img - img.min())/(img.max() - img.min() + 1e-5)
    img = img * 255.0
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=True)

'''
inp_np: 224,224,3
mask_np: 224,224
'''
def overlay_mask(img_np, mask_np):
    plt.figure(figsize=(8, 4))
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    # Mask only
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(mask_np, cmap='jet', alpha=0.5)  # Overlay with transparency
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)

'''
break a tensor into sliding windows
input t: B,C,H,W
'''
def get_windows(t, window_w):
    B,C,H,W = t.size()
    stride = window_w
    windows = F.unfold(t.float(), kernel_size=(window_w, window_w), stride=stride)
    windows = windows.view(B, C, window_w, window_w, int(H/window_w), int(H/window_w))
    return windows

def fold_windows(t, w, H):
    C = t.size(1)
    t = t.view(1,C, w * w,-1)
    t = F.fold(t[0,:], (H,H) , (w,w), stride=w)
    return t

def flow_to_rgb(flow):
    """
    Convert optical flow (2, H, W) to RGB image (H, W, 3)
    where hue represents direction and value represents magnitude
    
    Args:
        flow: numpy array of shape (2, H, W)
    
    Returns:
        rgb: numpy array of shape (H, W, 3) in range [0, 255]
    """
    # Extract flow components
    u = flow[0]  # horizontal flow (x-direction)
    v = flow[1]  # vertical flow (y-direction)
    
    # Calculate flow magnitude and angle
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # in radians, range [-π, π]
    
    # Normalize magnitude to [0, 1]
    max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1
    magnitude_norm = magnitude / max_mag
    
    # Convert angle from [-π, π] to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)  # range [0, 1]
    
    # Create HSV image
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = hue  # Hue
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = magnitude_norm  # Value from magnitude
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    
    # Scale to 0-255 for image display
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def show_images_seq(img1, img2):
    images = [img1, img2]
    titles = ['Image 1', 'Image 2']
    display_times = [3, 3]  # seconds

    for img, title, display_time in zip(images, titles, display_times):
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show(block=False)  # Non-blocking display
        plt.pause(display_time)  # Wait for specified time
        plt.close()  # Close current figure

def normalize_to_neg1_pos1(tensor):
    """
    Normalize ANY tensor to [-1, 1] range using min-max scaling
    Works for any value range
    """
    # Get min and max values
    t_min = tensor.min()
    t_max = tensor.max()
    
    # Avoid division by zero
    if t_max - t_min == 0:
        return torch.zeros_like(tensor)
    
    # Normalize to [0, 1] first, then to [-1, 1]
    normalized = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    
    return normalized

'''
input video should be tensor of shape [T, 3, H, W]
'''
def play_tensor_video_opencv(tensor, fps=30, window_name="Video", titles=None):
    # Convert tensor to numpy and ensure correct format
    if isinstance(tensor, torch.Tensor):
        frames = tensor.detach().cpu().numpy()
    else:
        frames = tensor

    if titles is not None and len(titles) == frames.shape[0]:
        border_size = 10
        frames = np.pad(
            frames,
            pad_width=((0, 0), (border_size, border_size), (border_size, border_size), (0, 0)),
            mode='constant',
            constant_values=0
        )
    
    # Ensure shape: [T, C, H, W] -> [T, H, W, C] for OpenCV
    if frames.shape[1] == 3:  # [T, C, H, W]
        frames = frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
    
    # Convert RGB to BGR for OpenCV
    frames = frames[..., ::-1]  # RGB -> BGR
    
    # Normalize to 0-255 if needed
    if frames.dtype != np.uint8 or frames.max() > 1.0:
        frames =  (frames - frames.min())/(frames.max() - frames.min() + 1e-5)
        frames = frames * 255.0
        frames = frames.astype(np.uint8)
    
    # Play video
    for i, frame in enumerate(frames):
        if titles is not None:
            print(titles[i])
            cv2.putText(frame, str(titles[i]), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, frame)
        
        # Wait for key press or delay
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()


'''
prediction related
'''
def get_pred_stats(model, v, gt_class=None, orig_pred_logit=None):
    ret = {}
    with torch.no_grad():
        pred = model(v.unsqueeze(0))
    pred = F.softmax(pred,dim=1)
    pred_cls = torch.argmax(pred,dim=1).item()
    pred_logit = pred[:,pred_cls].item()
    ret['pred_logit'] = pred_logit

    if gt_class is not None:
        logit = pred[:,gt_class].item()
        per_change = (orig_pred_logit - logit)/orig_pred_logit
        ret['per_change'] = per_change
        ret['logit'] = logit

    return ret

'''
shuffle the given video based on the clustered_ids and the given combinations and get the avg predions
'''
def get_avg_pred(model, video, clustered_ids, combinations):
    #create video with the solutions
    comb_vids = torch.empty(0).to(video.device)
    for comb in combinations:
        comb_vid = create_new_video(video, clustered_ids, comb)
        comb_vids = torch.concatenate([comb_vids,comb_vid.unsqueeze(0)])

    with torch.no_grad():
        pred_comb = model(comb_vids).mean(dim=0)
    pred_comb = F.softmax(pred_comb.unsqueeze(0),dim=1)

    return pred_comb

#video shape : 3, t , h , w
#where t is the number of frames
from matplotlib import pyplot as plt
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

        # plt.imshow(f0_, cmap='gray')
        # plt.show()
        # plt.imshow(f1_, cmap='gray')
        # plt.show()

        # v = torch.tensor(np.concat([f0_[None,:],f1_[None,:]],axis=0))
        # play_tensor_video_opencv(v[:,None,:].repeat(1,3,1,1), fps=1)

        # f = (flow_**2).sum(axis=-1)**0.5
        # plt.imshow(f, cmap='gray')
        # plt.show()


        pass
    return flow


def warp_batch(imgs, flow):
    """
    imgs: (B, C, H, W)
    flow: (B, 2, H, W)  dx, dy in pixels
    return: warped imgs (B, C, H, W)
    """

    B, C, H, W = imgs.shape

    # base grid (H, W)
    y, x = torch.meshgrid(
        torch.arange(H, device=imgs.device),
        torch.arange(W, device=imgs.device),
        indexing="ij"
    )

    grid = torch.stack((x, y), dim=0).float()   # (2, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

    # add flow
    vgrid = grid + flow

    # normalize to [-1, 1]
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / (W - 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / (H - 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # (B, H, W, 2)

    # backward warp
    warped = F.grid_sample(
        imgs,
        vgrid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    # v = torch.concat([imgs,warped])
    # fmag = (flow**2).sum(dim=1)**0.5
    # show_gray_image(fmag[0].cpu().numpy())
    # play_tensor_video_opencv(v,fps=1)
    # show_rgb_image(imgs[0].permute(1,2,0).cpu().numpy())
    

    return warped

def input_flow_grad(video):
    delta = 1.0
    flow = calc_flow(video)

    # flow_mag = torch.sum(flow**2,dim=-1)
    # flow_mag = flow[:,:,:,0]
    # from matplotlib import pyplot as plt
    # plt.imshow(flow_mag[0,:].cpu().numpy())
    # plt.show()
    # play_tensor_video_opencv(flow_mag[:,None,:].repeat(1,3,1,1), fps=2)
    # play_tensor_video_opencv(video.permute(1,0,2,3), fps=2)

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

    return d,flow

def dI_df(img0, img1, flow):
    delta = 2.0

    f_delta = torch.ones_like(flow) * delta
    delta_x = f_delta.clone()
    delta_x[:,:,1] = 0
    delta_y = f_delta.clone()
    delta_y[:,:,0] = 0

    warped_x = warp_batch(img0, (flow+delta_x).permute(2,0,1).unsqueeze(0))
    warped_y = warp_batch(img0, (flow+delta_y).permute(2,0,1).unsqueeze(0))
    dI_x = (warped_x - img1) / delta_x[:,:,0][None,None,:,:]
    dI_y = (warped_y - img1) / delta_y[:,:,1][None,None,:,:]
    d = torch.concat([dI_x[:,:,:,:,None],dI_y[:,:,:,:,None]],dim=-1)
    d = torch.max(d, dim=1)[0]

    return d

def input_flow_grad_mag(video):
    flow = calc_flow(video)
    v = video[:,:-1,:].permute(1,0,2,3)
    warped_v = warp_video(v, flow+torch.ones_like(flow))
    v1 = video[:,1:,:].permute(1,0,2,3)
    dI_dF = (warped_v - v1) 
    dI_dF = torch.max(dI_dF,dim=1)[0]

    # from matplotlib import pyplot as plt
    # plt.imshow(dI_dF[5,:].cpu().numpy())
    # plt.show()

    return dI_dF, flow



''''
function to modify video 
'''

#video: 3, t , h , w
def create_new_video(video, frame_cluster_idxs, ordered_keys=None):
    new_video = torch.zeros_like(video)
    cur_idx=0
    if ordered_keys == None:
        ordered_keys = frame_cluster_idxs.keys()
    for k in ordered_keys:
        for f in frame_cluster_idxs[k]:
            new_video[:,cur_idx,:] = video[:,f,:]
            cur_idx += 1
    return new_video

def replace_frame(video, ordered_keys, frame_cluster_idxs, key, img):
    new_video = video.clone()
    cur_idx=0
    for k in ordered_keys:
        for f in frame_cluster_idxs[k]:
            if str(f)==str(key):
                new_video[:,cur_idx,:] = img    
            cur_idx += 1
    return new_video

def get_motion_pairs(ids):
    pairs = []
    keys = list(ids.keys())
    for idx in range(len(keys)-1):
        pairs.append((keys[idx], keys[idx+1]))

    return pairs

def get_cluster_frameids(cluster_ids, order_ids):
    frame_ids = {}
    idx=0
    for i in order_ids:
        idx_ar = []
        for val in cluster_ids[i]:
            idx_ar.append(idx)
            idx+=1
        frame_ids[i] = idx_ar
    return frame_ids

#if the json file was written the whole dict at once
def read_json_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            data_dict = json.load(file)
    else:
        data_dict = {}
    return data_dict
#if the json file was written line by line
def read_json_line(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

'''
generate new cluster id given idx list
this uses temporal freezing. 
The first occurance of each frame is not changed
A frame is repeated until the next viable frames is found
'''
def temporal_freeze(idx_list, len_array=16):
    assert len(idx_list) > 0 , "idx_list must contain at least one index."
    max_idx = max(idx_list)
    assert max_idx < len_array, "max value in idx_list must not be larger than len_array"

    idx_list.sort()
    d = {}
    if len(idx_list) == 1:
        d[idx_list[0]] = [idx_list[0]]*len_array
    else:
        d[idx_list[0]] = [idx_list[0]]*(idx_list[1])
        for i in range(1, len(idx_list)-1):
            d[idx_list[i]] = [idx_list[i]]*(idx_list[i+1]-idx_list[i])
        d[idx_list[-1]] = [idx_list[-1]]*(len_array - idx_list[-1])
    ordered_keys = list(dict(sorted(d.items(), key=lambda x: x[1][0])).keys())

    clusters = {}
    clusters['cluster_ids'] = d
    clusters['ordered_keys'] = ordered_keys
    
    return clusters

'''
Monte-Carlo method to fill the array and avoid the forbidden pairs
give fixed existing pairs with already initialized array
'''
import random
def sample_fill_array_fixed_pos(
    original,
    numbers,
    pairs_to_avoid,
    max_solutions=10,
    max_trials=10_000,
    forbid_reverse=False,
    seed=None,
):
    
    orig_num = [n for n in original if n is not None]
    assert len([n for n in orig_num if n in numbers])==0, "Error. Original array contains numbers that are in the given number set."
    assert len(original) == len(orig_num)+len(numbers), "Error. The number of None entries in original_array must be equal to the number of given numbers + numer of non None elements in the origincal array"


    if seed is not None:
        random.seed(seed)

    forbidden = set(pairs_to_avoid)
    if forbid_reverse:
        forbidden |= {(b, a) for (a, b) in pairs_to_avoid}

    n = len(original)
    fixed = original[:]
    fixed_vals = {x for x in fixed if x is not None}
    free_positions = [i for i, x in enumerate(fixed) if x is None]
    available = [x for x in numbers if x not in fixed_vals]

    #handle special cases seperately
    skip_indices = [i for i, val in enumerate(original) if val is not None]
    valid_indices = [i for i in range(n) if i not in skip_indices]
    valid_indices.sort()
    if len(original)==3 and len(fixed_vals)==2:
        if valid_indices[0]==2:
            if (original[1],numbers[0]) in pairs_to_avoid:
                ret_array = [numbers[0],original[0],original[1]]
            else:
                ret_array = [original[0],original[1],numbers[0]]
        if valid_indices[0]==0:
            if (numbers[0],original[1]) in pairs_to_avoid:
                ret_array = [original[1],original[2],numbers[0]]
            else:
                ret_array = [numbers[0],original[1],original[2]]
        return [ret_array]


    solutions = []
    seen = set()

    def is_valid(arr):
        for i in range(n - 1):
            if (arr[i], arr[i + 1]) in forbidden:
                return False
        return True

    for _ in range(max_trials):
        if len(solutions) >= max_solutions:
            break

        random.shuffle(available)
        candidate = fixed[:]

        for pos, val in zip(free_positions, available):
            candidate[pos] = val

        if not is_valid(candidate):
            continue

        key = tuple(candidate)
        if key in seen:
            continue

        seen.add(key)
        solutions.append(candidate)

    return solutions

def is_valid(arr, forbidden_pairs):
    for i in range(len(arr) - 1):
        if [arr[i], arr[i + 1]] in forbidden_pairs:
            return False
    return True

def contains_sublist(lst, sublist):
    n = len(sublist)
    return any(sublist == lst[i:i+n] for i in range(len(lst) - n + 1))

'''
Monte-Carlo method to fill the array and avoid the forbidden pairs
pairs can occupy any position. not fixed
'''
def sample_fill_array(numbers, existing, forbidden_pairs, seed=None, max_solutions=100, max_trials=10000):

    for ex in existing:
        assert ex not in forbidden_pairs, 'Error: pairs in the existing cannot be there in the forbidden_pairs'


    flat = [item for sublist in existing for item in sublist]
    assert len([n for n in numbers if n in flat])==0, 'any number in numbers must not be present in existing'

    if seed is not None:
        random.seed(seed)
    
    solutions = set()

    #merge suitable items from the existing list
    new_existing = existing.copy()
    i=0
    while i < len(new_existing):
        p = new_existing[i]
        rest = [item for item in new_existing if item!=p]
        found=False
        for r in rest:
            assert not(r[0]==p[-1] and r[-1]==p[0]), 'invalid combinations'
            if r[0]==p[-1]:
                new_comb = list(dict.fromkeys(p + r))
                rest_rest = [item for item in rest if item!=r]
                new_existing = [new_comb]+rest_rest
                found=True
                break
            if r[-1]==p[0]:
                new_comb = list(dict.fromkeys(r + p))
                rest_rest = [item for item in rest if item!=r]
                new_existing = [new_comb]+rest_rest
                found=True
                break
        if found:
            continue
        i+=1
    
    for _ in range(max_trials):
        if len(solutions) >= max_solutions:
            break
        #calculate output length
        e_ = [e_ for e in new_existing for e_ in e]
        l_existing = len(e_)
        assert len([e for e in e_ if e in numbers])==0, 'Error: there is an overlap between existing and the new numbers to fill.'
        l_out = l_existing + len(numbers)


        #fill existing numbers
        #*******
        array = [None]*l_out
        filled = len([a for a in array if a != None])

        # random.shuffle(new_existing)
        while filled < l_existing:
            array = [None]*l_out
            not_found=False
            for i_ex in range(len(new_existing)):
                l = len(new_existing[i_ex])

                #get available indices to fill
                idx_none = [i for i,a in enumerate(array) if a == None]
                random.shuffle(idx_none)
                idx_free = []
                for idx in idx_none:
                    avail = True
                    for add in range(l):
                        if (idx+add) not in idx_none:
                            avail = False
                            break
                    if avail: idx_free.append(idx)
                if len(idx_free) == 0: # no solution found : retry
                    not_found = True
                    break
                if not not_found:
                    idx_sample = random.sample(idx_free,1)[0]
                    for idx_e_, e in enumerate(new_existing[i_ex]):
                        array[idx_sample+idx_e_] = new_existing[i_ex][idx_e_]
            if not not_found:
                filled = len([a for a in array if a != None])

        assert filled == l_existing, 'Error, Cannot fill all the existing numbers!'
        #*******
        
        #fill the rest of the array with the given numbers
        avail_idx = [i for i,val in enumerate(array) if val is None]
        random.shuffle(avail_idx)
        for idx, ai in enumerate(avail_idx):
            array[ai] = numbers[idx]
        
        #check if there are forbidden pairs in the solution

        
        if not is_valid(array, forbidden_pairs):
            continue
        
        key = tuple(array)
        if key in solutions:
            continue
        solutions.add(key)
    
    #check solution
    for s in solutions:
        assert is_valid(s, forbidden_pairs), 'Error: A solution with a fobidden pair found!'
    for s in solutions:
        for ex in existing:
            assert contains_sublist(list(s), ex), f'Items in the existing list is not present in a solution {s}'
    
    solutions = [list(s) for s in solutions]
    return solutions


# sol = sample_fill_array([0,2,3,4,6,7,8,10,11,12], [[1,5],[9,15]] ,[[5,8],[0,2]])
# pass

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvis_F
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from pathlib import Path
from urllib.request import urlretrieve
import tempfile


class RAFT_OF:
    def __init__(self):
        self.model = raft_large(pretrained=True, progress=False).to('cuda')
        self.model = self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def resize_flow_interpolate(self, flow, target_size=(112, 112), mode='bilinear'):
        """
        Resize optical flow using interpolation
        
        Args:
            flow: Tensor of shape [B, 2, H, W] or [B, C, H, W]
            target_size: (height, width) tuple
            mode: 'bilinear' (smooth) or 'nearest' (preserve edges)
        """
        # Resize flow
        flow_resized = F.interpolate(
            flow, 
            size=target_size, 
            mode=mode, 
            align_corners=False
        )
        
        # IMPORTANT: Scale flow values to match new spatial dimensions
        # Flow is displacement in pixels, so we need to scale it
        scale_h = target_size[0] / flow.shape[2]  # height scale
        scale_w = target_size[1] / flow.shape[3]  # width scale
        
        # Scale flow vectors (x-component * scale_w, y-component * scale_h)
        flow_resized[:, 0, :, :] *= scale_w  # x component
        flow_resized[:, 1, :, :] *= scale_h  # y component
        
        return flow_resized

    def preprocess(self, batch):
        batch = (batch-batch.min())/(batch.max()-batch.min()+1e-5)
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(224, 224)),
            ]
        )
        batch = transforms(batch)
        return batch

    def predict_flow_batch(self,batch1,batch2):
        img1_batch = self.preprocess(batch1)
        img2_batch = self.preprocess(batch2)

        with torch.no_grad():
            flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        return flows[-1]
    
    def predict_flow_video(self, video, out_size=(224,224)):
        img1_batch = video[:-1,:]
        img2_batch = video[1:,:]
        flow = self.predict_flow_batch(img1_batch,img2_batch)

        #resize flow if needed
        if flow.size(2)!=out_size[0] or flow.size(3)!=out_size[1]:
            flow = self.resize_flow_interpolate(flow, target_size=(out_size[0], out_size[1]), mode='bilinear')
        return flow

    def visualize(self, img_batch, predicted_flows):
        flow_imgs = flow_to_image(predicted_flows)
        img_batch = (img_batch - img_batch.min())/(img_batch.max() - img_batch.min()+1e-5)
        grid = [[img1, flow_img] for (img1, flow_img) in zip(img_batch, flow_imgs)]
        self.plot(grid)

    def plot(self, imgs, **imshow_kwargs):
        plt.rcParams["savefig.bbox"] = "tight"
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = tvis_F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

'''
Example usage of RAFT_OF
'''

# from torchvision.io import read_video

# video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
# video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
# _ = urlretrieve(video_url, video_path)
# frames, _, _ = read_video(str(video_path), output_format="TCHW")
# N,C,H,W = frames.shape
# top = (H - 224) // 2
# left = (W - 224) // 2
# frames = frames[:,:,top:top+224, left:left+224]

# raftof = RAFT_OF()

# img1_batch = frames[310][None,:]
# img2_batch = frames[311][None,:]
# flows = raftof.predict_flow_batch(img2_batch, img1_batch)

# warped = warp_batch(img1_batch.float().detach(), flows.detach().cpu())

# play_tensor_video_opencv(torch.stack([img1_batch[0],warped[0,:]]),fps=1)

# play_tensor_video_opencv(torch.stack([img1_batch[0],img2_batch[0,:]]),fps=1)

# show_rgb_image(warped[0,:].permute(1,2,0).cpu().numpy())
# show_rgb_image(img2_batch[0].permute(1,2,0).cpu().numpy())


def write_flow_yaml(flow, filename):
    """
    Write optical flow to OpenCV YAML format
    
    Args:
        flow: numpy array of shape (height, width, 2) 
        filename: output YAML filename
    """
    height, width = flow.shape[:2]
    
    # Flatten the flow data in row-major order
    # OpenCV stores data as [u1, v1, u2, v2, u3, v3, ...]
    flattened_data = flow.reshape(-1, 2).flatten()
    
    with open(filename, 'w') as f:
        # Write YAML header
        f.write("%YAML:1.0\n")
        
        # Write matrix metadata
        f.write("mat: !!opencv-matrix\n")
        f.write(f"   rows: {height}\n")
        f.write(f"   cols: {width}\n")
        f.write('   dt: "2f"\n')  # 2 channels of float
        
        # Write data section
        f.write("   data: [ ")
        
        # Format numbers in scientific notation like OpenCV
        data_lines = []
        current_line = []
        
        for i, value in enumerate(flattened_data):
            # Format in scientific notation with proper precision
            if value >= 0:
                formatted = f"{value:.8e}"
            else:
                formatted = f"{value:.8e}".replace('e-0', 'e-').replace('e+0', 'e+')
            
            current_line.append(formatted)
            
            # Break into lines of 4 elements each (like OpenCV does)
            if len(current_line) == 4:
                data_lines.append(", ".join(current_line))
                current_line = []
        
        # Add any remaining elements
        if current_line:
            data_lines.append(", ".join(current_line))
        
        # Write data with proper indentation and line breaks
        f.write(data_lines[0])
        for line in data_lines[1:]:
            f.write(",\n        " + line)
        
        f.write(" ]\n")

# from torchvision.utils import save_image
# import os
# import torch.nn.functional as F

# out_path = r'C:\Users\lahir\Downloads\dir'
# for i in range(flows[-1].size(0)):
#     img = frames[i,:]
#     resized = F.interpolate(
#         img[None,:,:,:], 
#         size=(520, 960), 
#         mode='bilinear', 
#         align_corners=False 
#     )
#     img = resized[0,:,:,:]
#     f = flows[-1][i,:]
#     save_image(img/255, os.path.join(out_path,'imgs',f'img_{i}.png'))
#     f = f.permute(1,2,0).cpu().numpy()
#     write_flow_yaml(f, os.path.join(out_path,'flow',f'flow_{i}.txt'))
# f = flows[-1].permute(0,2,3,1).cpu().numpy()
# write_flow_yaml(f[0], r'C:\Users\lahir\Downloads\test.flo')

def read_flow_yaml(filepath):
    flow = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    flow_matrix = flow.getNode('mat').mat()
    return flow_matrix


# raft calculated of
# min, max = -0.0966113 0.14445452

# of from repo
# min, max = -8.420414 4.1761527

# path = r'C:\Users\lahir\Downloads\frame_0001.flo.txt'
# f = read_flow_yaml(path)
# print(f.min(),f.max())



# img from repo
# min, max = 0 63

# img from dataset
# 0, 255

# path = r'C:\Users\lahir\Downloads\UCF101\jpgs\WritingOnBoard\v_WritingOnBoard_g25_c03\image_00047.jpg'
# img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# img.min(), img.max()


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class GradcamModel(nn.Module):
    def __init__(self, model, of_method='raft'):
        super(GradcamModel, self).__init__()
        self.model = model
        self.of_method = of_method
        if of_method == 'raft':
            self.raftof = RAFT_OF()

        #feature extraction model for video clustering
        # weights = ResNet50_Weights.IMAGENET1K_V1
        # resnetmodel = resnet50(weights=weights)
        # self.feature_model = create_feature_extractor(resnetmodel, return_nodes={'avgpool':'f_layer'})
        # self.feature_tr = weights.transforms()

        self.model.layer4[2].conv3.register_forward_hook(self.save_activations)
        self.model.layer4[2].conv3.register_full_backward_hook(self.save_gradients)

        # add noise to input
        #noise parameters from https://github.com/pkmr06/pytorch-smoothgrad
        self.noise_tr = transforms.Compose([
            AddGaussianNoise(mean=0., std=0.01)
        ])

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
        self.zero_grad()
        x.requires_grad = True
        pred = self(x)
        pred_idx = torch.argmax(pred,dim=1)
        pred[0,pred_idx].backward()

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
    
    def calc_raft_of(self, x, cluster_dict, frame_pairs):

        order_ids = cluster_dict['order']
        cluster_ids = cluster_dict['clusters']
        frame_ids = cluster_dict['frame_ids']

        img1_batch, img2_batch = torch.empty(0), torch.empty(0)
        img1_batch, img2_batch = img1_batch.to('cuda'), img2_batch.to('cuda')

        for f in frame_pairs:
            idx0 = frame_ids[str(f[0])][0]
            idx1 = frame_ids[str(f[1])][0]

            img1_ = x[:,idx0,:][None,:]
            img2_ = x[:,idx1,:][None,:]
            img1_batch = torch.concat([img1_batch,img1_],dim=0)
            img2_batch = torch.concat([img2_batch,img2_],dim=0)
        
        flows = self.raftof.predict_flow_batch(img2_batch, img1_batch)

        return flows


    '''
    grad_method: what method is used to find dPred_dI
                sal: saliency (raw gradient)
                gradcam: using gradcam
    '''
    def calc_flow_saliency(self, x, cluster_dict, frame_pairs, gt_class_idx, grad_method='sal'):
        x.requires_grad_(True)
        pred = self.forward(x[None,:])
        pred_logit = pred[:,gt_class_idx]
        pred_logit.backward()
        x.requires_grad_(False)

        if self.of_method == 'raft':
            flow = self.calc_raft_of(x, cluster_dict, frame_pairs)
        else:
            #calculate OF in the classical method
            pass

        frame_ids = cluster_dict['frame_ids']
        ret = {}
        for idx, p in enumerate(frame_pairs):
            f = flow[idx,:].permute(1,2,0)
            d={}
            idx0 = frame_ids[str(p[0])][0]
            idx1 = frame_ids[str(p[1])][0]

            # mag = torch.sum(f**2, dim=-1)**0.5
            # plt.imshow(mag.detach().cpu().numpy(), cmap='hot', alpha=0.5)
            # plt.show(block=True)
            img0 = x[:,idx0,:][None,:]
            img1 = x[:,idx1,:][None,:]
            if flow.size(2) != img0.size(2):
                img0 = F.interpolate(img0, size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)
                img1 = F.interpolate(img1, size=(flow.size(2), flow.size(2)), mode='bilinear', align_corners=False)
            
            dI_dF = dI_df(img0, img1, f)[0,:]

            if grad_method=='sal':
                slc,_ = torch.max(x.grad ,dim=0)
                slc = slc[idx1,:]
                if slc.size(1) != f.size(0):
                    slc = F.interpolate(slc[None,None,:], size=(f.size(0), f.size(0)), mode='bilinear', align_corners=False)
                    slc = slc[0,0,:]
                grad = slc
                d['slc'] = slc

            elif grad_method=='gradcam':
                gcam = self.calc_gradcam(x)[:,p[1],:]
                if gcam.size(1) != f.size(0):
                    gcam = F.interpolate(gcam[None,:], size=(f.size(0), f.size(0)), mode='bilinear', align_corners=False)[:,0,:]
                gcam = gcam[0,:,:,None]
                grad = gcam
                d['gcam'] = gcam

            dPred_dF = grad[:,:,None] * dI_dF
            dPred_dF =  F.relu(dPred_dF)
            dPred_dF = torch.sum(dPred_dF**2, dim=2)**0.5
            dPred_dF[:5,:] = 0
            dPred_dF[-5:,:] = 0
            dPred_dF[:,:5] = 0
            dPred_dF[:,-5:] = 0

            f_mean = f.mean(dim=(0,1))[None,None,:]
            f_res = f - f_mean
            f_mag = torch.sum(f_res**2,dim=2)**0.5

            tau = np.percentile(f_mag.cpu().numpy(), 90) 
            #remove noise (very small values)
            f_mag[f_mag<tau] = 0

            f_mag = (f_mag-f_mag.min())/(f_mag.max()-f_mag.min())
            # show_gray_image(f_mag.cpu().numpy())

            d = {
                'pair': p,
                'dPred_dF': dPred_dF.cpu().numpy(),
                'dPred_dF*flow': (dPred_dF*f_mag).cpu().numpy(),
                'flow': f.cpu().numpy(),
                'flow_mag': f_mag.cpu().numpy(),
                'img0': img0[0,:].cpu().numpy(),
                'img1': img1[0,:].cpu().numpy()
            }

            ret[idx] = d
        
        return ret
    
    def get_gradcam_from_video(self, video, correct_class):
        pred = self(video.unsqueeze(0))
        lpred = pred.argmax()
        if int(lpred) == correct_class:
            self.zero_grad()
            pred[0,lpred].backward()

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
            
            video = video.permute(1,2,3,0)
            video = np.uint8((video - video.min())/(video.max() - video.min() + 1e-5)*255)

            transformed = np.uint8(cam_int.detach().numpy()*255)
            h_col = np.concatenate([cv2.applyColorMap(img, cv2.COLORMAP_JET)[None,:] for img in list(transformed[0])],axis=0)

            final_img = cv2.addWeighted(video, 0.6, h_col, 0.4, 0)
            final_img = torch.tensor(final_img).permute(0,3,1,2)

            return cam_int, final_img
        



'''
acess the UCF101 model and inference loader
'''
import json
from models.resnet3d.main import generate_model, get_inference_utils, resume_model
from models.resnet3d.model import generate_model, make_data_parallel
from models.resnet3d.spatial_transforms import Compose
from argparse import Namespace
import re
import os
from PIL import Image
from glob import glob

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class UCF101_data_model:
    def __init__(self):
        #read model options
        opt_path = "models/r3d/ucf101.json"
        with open(opt_path, "r") as f:
            model_opt = json.load(f)
        model_opt = Namespace(**model_opt)

        #create model
        self.model = generate_model(model_opt)
        self.model = resume_model(model_opt.resume_path, model_opt.arch, self.model)
        self.model.eval()

        model_opt.inference_batch_size = 1
        for attribute in dir(model_opt):
            if "path" in str(attribute) and getattr(model_opt, str(attribute)) != None:
                setattr(model_opt, str(attribute), Path(getattr(model_opt, str(attribute))))
        self.inference_loader, self.inference_class_names = get_inference_utils(model_opt)
        self.class_labels_map = {v.lower(): k for k, v in self.inference_class_names.items()}
        self.transform = self.inference_loader.dataset.spatial_transform
        self.mask_transforms = Compose([
            self.transform.transforms[0],  # Resize
            self.transform.transforms[1],  # CenterCrop
            self.transform.transforms[2]   # ToTensor
        ])

        self.mask_dir = r'C:\Users\lahir\Downloads\UCF101\analysis\masks'
        self.data_path = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"

    def construct_vid_path(self, cls_name, g, c):
        dir = os.path.join(
            self.data_path , cls_name, "v_{}_g{}_c{}".format(cls_name, str(g).zfill(2), str(c).zfill(2))
        )
        return dir
    
    def construct_vid_path_from_full(self, path):
        g = path.split('_')[2][1:]
        c = path.split('_')[3][1:]
        cls_name = path.split('_')[1]
        path = self.construct_vid_path(cls_name, g, c)
        return path

    def load_jpg_ucf101_param(self, l, g, c, n):
        name = self.inference_class_names[l]
        dir = self.construct_vid_path(name, g, c)
        path = sorted(glob(dir + "/*"), key=numericalSort)

        target_path = path[n * 16 : (n + 1) * 16]
        if len(target_path) < 16:
            print("not exist")
            return False

        video = []
        for _p in target_path:
            video.append(self.transform(Image.open(_p)))

        return torch.stack(video)
    
    def load_jpg_ucf101(self, path, n=0):

        path = sorted(glob(path + "/*"), key=numericalSort)

        target_path = path[n * 16 : (n + 1) * 16]
        if len(target_path) < 16:
            print("not exist")
            return False

        video = []
        for _p in target_path:
            video.append(self.transform(Image.open(_p)))

        return torch.stack(video)
    
    def load_masks(self, filename, n=0):
        cls_name = filename.split('_')[1]
        dir_path = os.path.join(self.mask_dir, cls_name, filename)
        frame_idx = sorted(glob(dir_path + "/*"), key=numericalSort)
        if len(frame_idx)==0:
            return -1
        frame_idx = frame_idx[n * 16 : (n + 1) * 16]

        masks = {}
        for idx in frame_idx:
            obj_ids = sorted(glob(idx + "/*"), key=numericalSort)
            f_ = os.path.basename(idx)
            o_masks = {}
            for o in obj_ids:
                o_ = os.path.basename(o).split('.')[0]
                mask = self.mask_transforms(Image.fromarray(np.repeat(np.array(Image.open(o))[...,None], repeats=3, axis=-1)))[0,:]
                mask = mask > 0.5
                o_masks[int(o_)] = mask
            masks[int(f_)] = o_masks
        
        return masks

    

