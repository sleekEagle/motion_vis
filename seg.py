from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np
import func
import torch
import cv2
import torch.nn.functional as F
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA

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
    



class Propagate:
    def __init__(self, s_samples=16):
        self.raft = func.RAFT_OF()
        self.s_samples = s_samples
        self.ucf101dm = func.UCF101_data_model()

    
    def onclick(self, event):
        """Handle mouse click events"""
        if event.xdata is not None and event.ydata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            print(f"You clicked at pixel coordinates: ({x}, {y})")
            
            # Store in a variable (or process as needed)
            global clicked_pixel
            clicked_pixel = (x, y)
            
            # If you want to show a marker at the clicked location
            plt.plot(x, y, 'ro', markersize=3)  # Red circle marker
            plt.draw()

            flow = self.flow_img
            # flow_rgb = func.flow_to_image(flow)
            # func.show_img(flow_rgb.permute(1,2,0))
            fxy = flow[:,y,x].unsqueeze(1).unsqueeze(2)  #shape (2,1,1)

            #define square around the point
            square_size = 10
            x_start = max(x - square_size, 0)
            x_end = min(x + square_size, flow.size(2)-1)
            y_start = max(y - square_size, 0)
            y_end = min(y + square_size, flow.size(1)-1)

            f_area = flow[:,y_start:y_end, x_start:x_end]
            diff = torch.sum((f_area - fxy)**2,dim=0)**0.5

            threshold = 0.01
            diff_mask = (diff < threshold).float()

            mask = torch.zeros(flow.size(1), flow.size(2))
            mask[y_start:y_end, x_start:x_end] = diff_mask
            mask = mask.unsqueeze(0).repeat(3,1,1)

            plt.imshow(mask[0,:].cpu().numpy(), alpha=0.5)

    def prop_video_path(self, vid_path, n):
        video = self.ucf101dm.load_jpg_ucf101(vid_path, n=n)
        flow = self.raft.predict_flow_video(video[n:n+self.s_samples,:])
        self.flow_img = flow[2,:]

        img = video[2].permute(1,2,0).numpy()
        img = (img - img.min())/(img.max() - img.min())*255.0
        img = img.astype(np.uint8)
        fig, ax = plt.subplots()
        ax.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()



        



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


prop = Propagate()
prop.prop_video_path(r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyLipstick\v_ApplyLipstick_g25_c03', n=0)






# image_path = r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyLipstick\v_ApplyLipstick_g25_c03\image_00009.jpg'  # Change to your image path
# img = mpimg.imread(image_path)

# fig, ax = plt.subplots()
# ax.imshow(img)

# # Connect the click event to the handler
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.title("Click on the image to get pixel coordinates")
# plt.show()

# # The clicked_pixel variable will contain the last clicked coordinates
# print(f"Last clicked pixel: {clicked_pixel}")


