import torch
import cv2
import torch.nn.functional as F
import numpy as np

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



import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from pathlib import Path
from urllib.request import urlretrieve
import tempfile


class RAFT_OF:
    def __init__(self):
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    def preprocess(self, img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
        img1_batch , img2_batch = self.transforms(img1_batch, img2_batch)
        self.img1_batch = img1_batch
        return img1_batch , img2_batch

    def predict_flow_batch(self,batch1,batch2):
        img1_batch, img2_batch = self.preprocess(batch1, batch2)
        with torch.no_grad():
            flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))
        return flows
    
    def predict_flow_video(self, video):
        flows = torch.empty(0).to(self.device)
        for i in range(video.size(0)-1):
            img1_batch = video[i,:][None,:]
            img2_batch = video[i+1,:][None,:]
            f = self.predict_flow_batch(img1_batch,img2_batch)[-1]
            flows = torch.cat((flows,f),dim=0)
        return flows

    def visualize(self, img_batch, predicted_flows):
        flow_imgs = flow_to_image(predicted_flows[-1])
        img_batch = [(img1 + 1) / 2 for img1 in self.img1_batch]
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
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

'''
Example usage of RAFT_OF
'''

# raftof = RAFT_OF()

# video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
# video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
# _ = urlretrieve(video_url, video_path)


# from torchvision.io import read_video
# frames, _, _ = read_video(str(video_path), output_format="TCHW")
# flows = raftof.predict_flow_video(frames[0:10,:])
# raftof.visualize(frames[0:9,:],flows)

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


