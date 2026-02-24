import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()
        # Load model and video preprocessor
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").to(device)
        label2id_ = self.model.config.label2id
        self.label2id = {}
        for k in label2id_:
            new_k = k.replace('[','').replace(']','').replace('\'','')
            self.label2id[new_k] = label2id_[k]

    #input shape of x: 1,3,16,224,224
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        output = self.model(x)
        logits = output.logits
        return logits

    def sample_frames(self, video_path, num_frames=16):
        decoder = VideoDecoder(video_path)
        total = len(decoder)
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = decoder.get_frames_at(indices=indices)  # FrameBatch
        frames_np = frames.data.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
        return list(frames_np)

    def video_from_path(self, path):
        frames = self.sample_frames(str(path), num_frames=16)
        inputs = self.processor(frames, return_tensors="pt").to(device)
        return inputs
    
    def predict_from_path(self, path):
        inputs = self.video_from_path(path)
        outputs = self.model(**inputs)
        pred = outputs.logits.argmax(-1).item()
        return pred