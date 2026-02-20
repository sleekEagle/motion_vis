import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pathlib import Path
import imageio.v3 as iio

device = "cuda" if torch.cuda.is_available() else "cpu"

class VJEPA2:
    def __init__(self):
        # Load model and video preprocessor

        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2").to(device)
        label2id_ = self.model.config.label2id
        self.label2id = {}
        for k in label2id_:
            new_k = k.replace('[','').replace(']','').replace('\'','')
            self.label2id[new_k] = label2id_[k]

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
    
'''
Accuracy = 69.18 % 
'''
def test_s2s():
    model = VJEPA2()
    model.model.eval()

    cls_names = list(model.label2id.keys())
    
    path = Path(r'C:\Users\lahir\Downloads\s2s_test')
    dirs = [p.name for p in path.iterdir() if p.is_dir()]

    #make sure all the class names are present in the list of dirs
    for c in cls_names:
        assert c in dirs , f'{c} is not in the list of dirs'
        pass

    dirs = [p for p in path.iterdir() if p.is_dir()]
    n_files = len([p for p in path.rglob("*") if p.is_file()])

    n_correct = 0
    n_samples = 0
    for dir in dirs:
        d_name = dir.name
        gt_idx = model.label2id[d_name]
        files = [p for p in dir.iterdir() if p.is_file()]

        for file in files:
            print(f'{n_samples} or {n_files} is done.', end='\r')
            n_samples += 1
            with torch.no_grad():
                preds = model.predict_from_path(file)
                if preds==gt_idx:
                    n_correct += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')


if __name__ == '__main__':
    test_s2s()


