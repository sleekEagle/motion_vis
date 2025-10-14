import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image


class RAFT_OF:
    def __init__(self):
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    def _download_video(self):
        video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
        urlretrieve(self.video_url, video_path)
        return video_path

    def _read_frames(self):
        frames, _, _ = read_video(str(self.video_path), output_format="TCHW")
        return frames

    def _get_frame_batches(self):
        img1_batch = torch.stack([self.frames[i] for i in self.frame_indices])
        img2_batch = torch.stack([self.frames[i + 1] for i in self.frame_indices])
        return img1_batch, img2_batch

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
        img1_batch = video[0:-1,:]
        img2_batch = video[1:,:]
        flows = self.predict_flow_batch(img1_batch,img2_batch)
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
Example usage
'''

import tempfile
from pathlib import Path
from urllib.request import urlretrieve

raftof = RAFT_OF()

video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)


from torchvision.io import read_video
frames, _, _ = read_video(str(video_path), output_format="TCHW")
flows = raftof.predict_flow_video(frames[0:10,:])
raftof.visualize(frames[0:9,:],flows)







