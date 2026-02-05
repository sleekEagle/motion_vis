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


# prop = Propagate()
# prop.prop_video_path(r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyLipstick\v_ApplyLipstick_g25_c03', n=0)






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

def modify_flow(img1, img2, flow, x1,y1, x2,y2):
    FLOW_RATIO = 0.9
    #remove the original regions from img2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(y1, y2, device=img1.device),
        torch.arange(x1, x2, device=img1.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).to(flow.device)
    grid_flow = flow[0,:,y1:y2,x1:x2].permute(1,2,0)

    orig_grid = grid + grid_flow
    mod_grid = grid + grid_flow*FLOW_RATIO
    
    raw_ycords = grid[:,:,0].flatten().long().to('cpu')
    raw_xcords = grid[:,:,1].flatten().long().to('cpu')
    
    orig_ycords = orig_grid[:,:,0].flatten().long().to('cpu')
    orig_xcords = orig_grid[:,:,1].flatten().long().to('cpu')

    mod_ycords = mod_grid[:,:,0].flatten().long().to('cpu')
    mod_xcords = mod_grid[:,:,1].flatten().long().to('cpu')

    img2_mod = img2.clone().to('cpu')
    img2_mod[:,:,orig_xcords,orig_ycords] = 0
    img2_mod[:,:,mod_xcords,mod_ycords] = img1[:,:,raw_xcords,raw_ycords]

    img2_orig = img2.clone().to('cpu')
    img2_orig[:,:,orig_xcords,orig_ycords] = 0
    img2_orig[:,:,orig_xcords,orig_ycords] = img1[:,:,raw_xcords,raw_ycords]


    imgs = [img2, img2_orig, img2_mod]
    imgs = [img[0,:].permute(1,2,0) for img in imgs]
    titles = [f'Image {i+1}' for i in range(3)]

    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for idx, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')  # Turn off axes

    # Hide unused subplots
    for idx in range(len(imgs), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()



    plt.imshow(img2_new[0,:].permute(1,2,0).detach().cpu().numpy())
    plt.imshow(fmag[0,:].detach().cpu().numpy(), cmap='hot', alpha=0.5)
    plt.show(block=True)



from torchvision.io import read_image

if __name__ == '__main__':
    raftof = func.RAFT_OF()

    img1_path = r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyEyeMakeup\v_ApplyEyeMakeup_g25_c07\image_00003.jpg'
    img2_path = r'C:\Users\lahir\Downloads\UCF101\jpgs\ApplyEyeMakeup\v_ApplyEyeMakeup_g25_c07\image_00004.jpg'
    img1 = read_image(img1_path)[None,:]
    img2 = read_image(img2_path)[None,:]
    img1 = F.interpolate(img1, size=(224,224), mode='bilinear', align_corners=False)
    img2 = F.interpolate(img2, size=(224,224), mode='bilinear', align_corners=False)

    flow = raftof.predict_flow_batch(img1, img2)

    modify_flow(img1, img2, flow, 100,141, 113,160)
    
    fmag = torch.sum(flow**2,dim=1)**0.5
    plt.imshow(img1[0,:].permute(1,2,0).detach().cpu().numpy())
    plt.imshow(fmag[0,:].detach().cpu().numpy(), cmap='hot', alpha=0.5)
    plt.show(block=True)




