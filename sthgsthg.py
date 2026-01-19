# https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2


from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch

'''
dtype = np.uint8
range = [0, 255]
shape = (H, W, 3)
'''
video = np.random.randint(
    0, 256, size=(16, 224, 224, 3), dtype=np.uint8
)
video = list(video)  # list of frames

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")

inputs = processor(video, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
