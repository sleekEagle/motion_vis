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
    video = ucf101dm.load_jpg_ucf101(r'C:\Users\lahir\Downloads\UCF101\jpgs\SoccerJuggling\v_SoccerJuggling_g01_c02',n=0)
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

    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to('cuda', dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
    
    from transformers.video_utils import load_video
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video, _ = load_video(video_url)



    inference_session = processor.init_video_session(
        video=video,
        inference_device='cuda',
        dtype=torch.bfloat16,
    )
    # inference_session.reset_inference_session()

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids = [1,2,3,4],
        input_masks=masks,
    )

    video_segments = {}
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
        )[0]
        video_segments[sam2_video_output.frame_idx] = video_res_masks




