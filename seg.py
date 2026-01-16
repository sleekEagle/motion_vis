from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np
import func
import torch
import cv2
import torch.nn.functional as F
import hdbscan


class DinoFeatureExtractor:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
    
    def extract_features_from_path(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
    
    #img: np array
    def extract_features_from_img(self, img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        image_pil = Image.fromarray(img.astype(np.uint8))

        inputs = self.processor(images=image_pil, return_tensors="pt")
        outputs = self.model(**inputs)
        feat = outputs.last_hidden_state[:,1:,:]
        h = int(feat.size(1)**0.5)
        feat_map = feat.reshape(1, h, h, -1).permute(0, 3, 1, 2)
        return feat_map
    
from sklearn.decomposition import PCA

class Segment:
    def __init__(self, s_samples=16):
        self.dino = DinoFeatureExtractor()
        self.raft = func.RAFT_OF()
        self.s_samples = s_samples
        self.ucf101dm = func.UCF101_data_model()

    def seg_video_path(self, vid_path, n):
        video = self.ucf101dm.load_jpg_ucf101(vid_path, n=n)
        # video = (video-video.min())/(video.max() - video.min()) * 255.0
        # video = video.type(torch.uint8)
        flow = self.raft.predict_flow_video(video[n:n+self.s_samples,:])

        #bilateral filtering of flow
        # flow_np = flow.cpu().numpy()
        # u,v = [],[]
        # for i in range(flow_np.shape[0]):
        #     u_ = cv2.bilateralFilter(flow[i,0,:].cpu().numpy(), 5, 25, 25)[None,:]
        #     v_ = cv2.bilateralFilter(flow[i,1,:].cpu().numpy(), 5, 25, 25)[None,:]
        #     u.append(u_)
        #     v.append(v_)
        # u = np.concatenate(u, axis=0)[None,:]
        # v = np.concatenate(v, axis=0)[None,:]
        # flow_filtered = np.concatenate([u,v], axis=0)
        # flow_filtered = torch.tensor(flow_filtered).float().to(flow.device)
        # flow_filtered = flow_filtered.permute(1,0,2,3)
        # self.raft.visualize(video[:4],flow[:4])
        
        #create appearance features with dino

        # video_features = [self.dino.extract_features_from_img(video[i,:].permute(1,2,0).numpy()) for i in range(video.size(0)) ]
        # video_features = torch.concatenate(video_features, dim=0)
        # video_features = F.interpolate(
        #     video_features, 
        #     size=(video.size(2), video.size(3)), 
        #     mode='bilinear', 
        #     align_corners=False
        # )

        # features = video_features[0,:,:,]
        # features = torch.nn.functional.normalize(features, dim=0)
        # features = features.detach().cpu().numpy()
        # X = features.reshape(768, -1).T
        # X_pca = PCA(n_components=32).fit_transform(X)

        # ys, xs = np.meshgrid(np.arange(video.size(2)), np.arange(video.size(3)), indexing="ij")
        # coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        # coords = coords / 112.0  # normalize

        # X_joint = np.concatenate([X_pca, 0.5 * coords], axis=1)
        # X_joint = X

        # clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=75,
        #     metric='euclidean'
        # )
        # labels = clusterer.fit_predict(X_joint)
        # seg = labels.reshape(112,112)

        # img = video[0,:].numpy()
        # img = (img-img.min())/(img.max() - img.min())
        # if img.dtype != np.uint8:
        #     img = (255 * img).astype(np.uint8)
        # else:
        #     img = img.copy()

        # labels = seg.copy()
        # unique_labels = np.unique(labels)

        # # ignore noise
        # unique_labels = unique_labels[unique_labels != -1]

        # rng = np.random.default_rng(42)
        # colors = {
        #     lbl: rng.integers(0, 255, size=3, dtype=np.uint8)
        #     for lbl in unique_labels
        # }

        # img = np.transpose(img, (1,2,0))
        # color_mask = np.zeros_like(img)

        # for lbl, color in colors.items():
        #     color_mask[labels == lbl] = color
        # color_mask[labels == -1] = [0, 0, 0]

        # alpha = 0.5  # 0 = original image, 1 = pure segmentation
        # overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        # import matplotlib.pyplot as plt

        # plt.imshow(overlay)
        # plt.axis("off")
        # plt.show()

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=75,
            metric='euclidean'
        )
        X = flow[0,:].reshape(2,-1).T
        labels = clusterer.fit_predict(X.detach().cpu().numpy())
        seg = labels.reshape(112,112)

        img = video[0,:].numpy()
        img = (img-img.min())/(img.max() - img.min())
        if img.dtype != np.uint8:
            img = (255 * img).astype(np.uint8)
        else:
            img = img.copy()

        labels = seg.copy()
        unique_labels = np.unique(labels)

        # ignore noise
        unique_labels = unique_labels[unique_labels != -1]

        rng = np.random.default_rng(42)
        colors = {
            lbl: rng.integers(0, 255, size=3, dtype=np.uint8)
            for lbl in unique_labels
        }

        img = np.transpose(img, (1,2,0))
        color_mask = np.zeros_like(img)

        for lbl, color in colors.items():
            color_mask[labels == lbl] = color
        color_mask[labels == -1] = [0, 0, 0]

        alpha = 0.5  # 0 = original image, 1 = pure segmentation
        overlay = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

        import matplotlib.pyplot as plt

        plt.imshow(overlay)
        plt.axis("off")
        plt.show()


        pass

        



# import func
# ucf101dm = func.UCF101_data_model()
# m = DinoFeatureExtractor()
# pass

# vid_path = r'C:\Users\lahir\Downloads\UCF101\jpgs\Archery\v_Archery_g25_c05'
# video = ucf101dm.load_jpg_ucf101(vid_path)
# img = video[0,:]

# feat = m.extract_features_from_img(img.permute(1,2,0).numpy())


# from torchvision.io import read_video
# import func
# raftof = func.RAFT_OF()

# path = r'C:\Users\lahir\Downloads\UCF101\UCF-101\SoccerPenalty\v_SoccerPenalty_g12_c06.avi'
# frames, _, _ = read_video(str(path), output_format="TCHW")
# flows = raftof.predict_flow_video(frames[0:10,:])
# raftof.visualize(frames[0:9,:],flows)


# seg = Segment()
# seg.seg_video_path(r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyLipstick\v_ApplyLipstick_g25_c03', n=0)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def onclick(event):
    """Handle mouse click events"""
    if event.xdata is not None and event.ydata is not None:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        print(f"You clicked at pixel coordinates: ({x}, {y})")
        
        # Store in a variable (or process as needed)
        global clicked_pixel
        clicked_pixel = (x, y)
        
        # If you want to show a marker at the clicked location
        plt.plot(x, y, 'ro', markersize=5)  # Red circle marker
        plt.draw()

image_path = r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyLipstick\v_ApplyLipstick_g25_c03\image_00009.jpg'  # Change to your image path
img = mpimg.imread(image_path)

fig, ax = plt.subplots()
ax.imshow(img)

# Connect the click event to the handler
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.title("Click on the image to get pixel coordinates")
plt.show()

# The clicked_pixel variable will contain the last clicked coordinates
print(f"Last clicked pixel: {clicked_pixel}")


