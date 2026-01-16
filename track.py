import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import func

def plot_with_pytorch(video_tensor, points_tensor, visibility_tensor=None,
                     output_path="tracked_video_pytorch.mp4"):
    """
    Pure PyTorch visualization (no OpenCV dependency)
    """
    import torch
    import torchvision
    from torchvision.io import write_video
    import numpy as np
    
    # Make copies to avoid modifying originals
    frames = video_tensor.clone()
    points = points_tensor.clone()
    
    if visibility_tensor is not None:
        visibility = visibility_tensor.clone()
    else:
        visibility = torch.ones_like(points[..., 0], dtype=torch.bool)
    
    # Ensure frames are in [0, 255] uint8 range
    if frames.dtype != torch.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).to(torch.uint8)
        else:
            frames = frames.to(torch.uint8)
    
    T, C, H, W = frames.shape
    N = points.shape[1]
    
    # Create color palette
    colors = torch.zeros(N, 3, dtype=torch.uint8)
    for i in range(N):
        # Generate distinct colors using HSV-like pattern
        hue = i * 360 // N
        # Simple HSV to RGB conversion (approximate)
        if hue < 60:
            colors[i] = torch.tensor([255, hue * 255 // 60, 0])
        elif hue < 120:
            colors[i] = torch.tensor([(120 - hue) * 255 // 60, 255, 0])
        elif hue < 180:
            colors[i] = torch.tensor([0, 255, (hue - 120) * 255 // 60])
        elif hue < 240:
            colors[i] = torch.tensor([0, (240 - hue) * 255 // 60, 255])
        elif hue < 300:
            colors[i] = torch.tensor([(hue - 240) * 255 // 60, 0, 255])
        else:
            colors[i] = torch.tensor([255, 0, (360 - hue) * 255 // 60])
    
    # Process each frame
    output_frames = []
    
    for t in range(T):
        # Clone the frame to draw on
        frame = frames[t].clone()  # (C, H, W)
        
        # Get current points and visibility
        current_points = points[t]  # (N, 2)
        current_visibility = visibility[t]  # (N,)
        
        # Draw points
        for i in range(N):
            if not current_visibility[i]:
                continue
                
            x, y = current_points[i]
            x_int = int(round(x.item()))
            y_int = int(round(y.item()))
            
            # Check bounds
            if 0 <= x_int < W and 0 <= y_int < H:
                # Draw a simple cross
                size = 3
                for dx in range(-size, size+1):
                    for dy in range(-size, size+1):
                        if abs(dx) + abs(dy) <= size:  # Diamond shape
                            px = min(max(x_int + dx, 0), W-1)
                            py = min(max(y_int + dy, 0), H-1)
                            frame[:, py, px] = colors[i]
        
        # Convert to (H, W, C) for video writing
        frame = frame.permute(1, 2, 0)  # (H, W, C)
        output_frames.append(frame)
    
    # Stack frames
    output_frames = torch.stack(output_frames)  # (T, H, W, C)
    
    # Write video
    write_video(output_path, output_frames, fps=30)
    
    print(f"Saved PyTorch video to {output_path}")
    return output_path

class CoTracker:
    def __init__(self, device='cuda', grid_size=10):
        self.device = device
        self.grid_size = grid_size
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    def load_video_from_frames(self, video_dir, start_idx=0, n_imgs=50):
        transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        
        img_files = sorted(os.listdir(video_dir))
        img_files = img_files[start_idx:start_idx+n_imgs]
        
        video = torch.concatenate([transform(Image.open(os.path.join(video_dir, f)).convert('RGB')) * 255 for f in img_files], dim=0).unsqueeze(0)
        return video.float().to(self.device)
    
    import matplotlib.pyplot as plt

    def get_points_to_track(self, video, mask):
        #define grid
        H, W = video.size(2), video.size(3)
        stride = 1  # 4–8 is good

        ys, xs = torch.meshgrid(
            torch.arange(0, H, stride),
            torch.arange(0, W, stride),
            indexing="ij"
        )
        points = torch.stack([xs.flatten(), ys.flatten()], dim=-1)

        in_mask = mask[0,points[:,1], points[:,0]]
        masked_points = points[in_mask>0]

        return masked_points

    import matplotlib.pyplot as plt

    def track_video_area(self, video, mask, start_idx=0):
        points = self.get_points_to_track(video, mask)
        frame_idx = torch.zeros(points.size(0),1)+start_idx
        queries = torch.concatenate([frame_idx, points], dim=1)
        # queries = queries[:1000,:]


        # gray = video[0,:].permute(1,2,0).numpy()
        # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        # gray =  (gray - gray.min())/(gray.max() - gray.min() + 1e-5)
        # gray = gray * 255.0
        # gray = gray.astype(np.uint8)
        # plt.imshow(gray, cmap='gray')
        # plt.scatter(points[:, 0], points[:, 1], s=10, c="red")
        # plt.axis("off")
        # plt.show()

        # queries = torch.tensor([
        # [0., 100., 150.],  # point tracked from the first frame
        # [10., 200., 78.], # frame number 10/
        # [20., 150., 65.], # ...
        # [30., 90., 200.]
        # ])
        if torch.cuda.is_available():
            queries = queries.cuda()
        # return self.cotracker(video, grid_size=self.grid_size)
        # ret = self.cotracker(video.unsqueeze(0).cuda(), queries=queries[None])
        return self.cotracker(video.unsqueeze(0).cuda(), queries=queries[None])
    
    def track_from_path(self, path, start_idx=0, n_imgs=50):
        video = self.load_video_from_frames(path, start_idx, n_imgs)
        pred_tracks, pred_visibility = self.track_video(video)
        return pred_tracks, pred_visibility


#how to use
# tracker = CoTracker(grid_size=20)
# video = tracker.load_video_from_frames(r'C:\Users\lahir\Downloads\UCF101\jpgs\TennisSwing\v_TennisSwing_g25_c05',
#                                        start_idx=0,
#                                        n_imgs=100)
# print('tracking....')
# pred_tracks, pred_visibility = tracker.track_video(video)
# print('saving....')
# plot_with_pytorch(video[0,:], pred_tracks[0,:], pred_visibility[0,:],
#                      output_path="tracked_video_pytorch.mp4")

pass











