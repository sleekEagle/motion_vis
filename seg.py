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


if __name__ == '__main__':
    image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-cityscapes-panoptic"
    )
    # url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
    # with httpx.stream("GET", url) as response:
    #     image = Image.open(BytesIO(response.read()))
    image = Image.open(r'C:\Users\lahir\Downloads\UCF101\jpgs\SoccerJuggling\v_SoccerJuggling_g01_c02\image_00150.jpg')
    image.show()


    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # Perform post-processing to get panoptic segmentation map
    result = image_processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["segmentation"]
    #generate masks
    mask_ids = np.unique(result)
    object_masks = []
    for id in mask_ids:
        m = np.zeros_like(result)
        m[result==id] = 1
        object_masks.append(m)

    print(result.shape)

    # func.show_gray_image(object_masks[3])
    # func.show_gray_image(result.detach().cpu().numpy())
    def get_mask_grid(mask, stride=3):
        H,W = mask.shape
        stride = 3
        grid = torch.zeros((H,W))
        grid[::stride,::stride] = 1
        grid[::stride,::stride] = 1
        return grid*mask


    g = get_mask_grid(object_masks[3])
    
    func.show_gray_image(g.detach().cpu().numpy())


    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to('cuda', dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
    
    from transformers.video_utils import load_video
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video_frames, _ = load_video(video_url)

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device='cuda',
        dtype=torch.bfloat16,
    )
    inference_session.reset_inference_session()

    # Add multiple objects on the first frame
    ann_frame_idx = 0
    obj_ids = [2, 3]
    input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
    input_labels = [[[1], [1]]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_points=input_points,
        input_labels=input_labels,
    )

    # Get masks for both objects on first frame
    outputs = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
    )



