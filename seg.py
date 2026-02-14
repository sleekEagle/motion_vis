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
from transformers import Mask2FormerConfig, Mask2FormerModel
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import httpx
from io import BytesIO
import torch

import cv2
import torch
import numpy as np
from transformers import Sam2VideoModel, Sam2VideoProcessor

from transformers.models.sam2_video.processing_sam2_video import Sam2VideoProcessor

'''
for future: use EntitySAM: Segment Everything in Video
problem: very hard to install it (detectron 2 required for it in windows)
'''

class Seg:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-panoptic"
        )
    
    def get_segmentation_path(self, img_path):
        image = Image.open(img_path)
        inputs = self.image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.image_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["segmentation"]

        mask_ids = torch.unique(result)
        object_masks = torch.empty(0)
        for id in mask_ids:
            m = torch.zeros_like(result)
            m[result==id] = 1
            object_masks = torch.concat([object_masks, m.unsqueeze(0)], dim=0)

        return object_masks
    
    def get_segmentation(self, image):
        inputs = self.image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.image_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["segmentation"]

        mask_ids = torch.unique(result)
        object_masks = torch.empty(0)
        for id in mask_ids:
            m = torch.zeros_like(result)
            m[result==id] = 1
            object_masks = torch.concat([object_masks, m.unsqueeze(0)], dim=0)

        return object_masks



if __name__ == '__main__':

    seg = Seg()

    ucf101dm = func.UCF101_data_model()
    video = ucf101dm.load_jpg_ucf101(r'C:\Users\lahir\Downloads\UCF101\jpgs\PullUps\v_PullUps_g25_c02',n=0)
    video = video.permute(0,2,3,1)
    video = (video-video.min())/(video.max()-video.min())*255
    video = video.int()

    # seg.get_segmentation_path(r'C:\Users\lahir\Downloads\UCF101\jpgs\SoccerJuggling\v_SoccerJuggling_g01_c02\image_00392.jpg')

    img = Image.fromarray(video[0,:].to(torch.uint8).numpy())
    masks = seg.get_segmentation(img)
    # mask = masks.float()
    # masks = (masks > 0.5)
    masks = masks.unsqueeze(1)
    # masks = masks.to(device="cuda", dtype=torch.bfloat16)

    # func.show_gray_image(object_masks[3])
    # func.show_gray_image(result.detach().cpu().numpy())

#     model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to('cuda', dtype=torch.bfloat16)
#     processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
    
#     from transformers.video_utils import load_video
#     video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
#     video, _ = load_video(video_url)



#     inference_session = processor.init_video_session(
#         video=video,
#         inference_device='cuda',
#         dtype=torch.bfloat16,
#     )
#     # inference_session.reset_inference_session()

#     processor.add_inputs_to_inference_session(
#         inference_session=inference_session,
#         frame_idx=0,
#         obj_ids = [1,2,3,4],
#         input_masks=masks,
#     )

#     video_segments = {}
#     for sam2_video_output in model.propagate_in_video_iterator(inference_session):
#         video_res_masks = processor.post_process_masks(
#             [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
#         )[0]
#         video_segments[sam2_video_output.frame_idx] = video_res_masks




# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image

# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )


# np.random.seed(3)

# def show_anns(anns, borders=True):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.5]])
#         img[m] = color_mask 
#         if borders:
#             import cv2
#             contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#             # Try to smooth contours
#             contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#             cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

#     ax.imshow(img)

# image = Image.open(r'C:\Users\lahir\Downloads\cars.png')
# image = np.array(image.convert("RGB"))



# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()


# import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# checkpoint = r"C:\Users\lahir\code\segment-anything-2\checkpointssam2.1_hiera_large.pt"
# model_cfg = r"C:\Users\lahir\code\segment-anything-2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))









