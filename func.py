import torch
import cv2
import torch.nn.functional as F

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