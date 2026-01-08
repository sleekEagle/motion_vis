import torch
import os
from PIL import Image
import torchvision.transforms as T

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
    
    def track_video(self, video):
        return self.cotracker(video, grid_size=self.grid_size)
    
    def track_from_path(self, path, start_idx=0, n_imgs=50):
        video = self.load_video_from_frames(path, start_idx, n_imgs)
        pred_tracks, pred_visibility = self.track_video(video)
        return pred_tracks, pred_visibility


#how to use
tracker = CoTracker()
video = tracker.load_video_from_frames(r'C:\Users\lahir\Downloads\UCF101\jpgs\Typing\v_Typing_g25_c04',
                                       start_idx=0,
                                       n_imgs=37)
pred_tracks, pred_visibility = tracker.track_video(video)











