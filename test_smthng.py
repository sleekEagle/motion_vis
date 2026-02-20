import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModelForVideoClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

class VJEPA2:
    def __init__(self):
        # Load model and video preprocessor

        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")

        # self.model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
        # self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

        id2label = self.model.config.id2label
        label2id={}
        for k in id2label:
            text = id2label[k]
            text_mod = text.replace('[','').replace(']','').replace('\'','')
            label2id[text_mod] = k
        self.label2id = label2id
        self.id2label = self.model.config.id2label

    def video_from_path(self, path):
        vr = VideoDecoder(path)
        frame_idx = np.arange(0, self.model.config.frames_per_clip, 8) # you can define more complex sampling strategy
        video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width
        inputs = self.processor(video, return_tensors="pt").to(self.model.device)
        return inputs
    
    def predict_from_path(self, path):
        inputs = self.video_from_path(path)
        outputs = self.model(**inputs)
        logits = outputs.logits
        top5_indices = logits.topk(5).indices[0]
        top5_probs = torch.softmax(logits, dim=-1).topk(5).values[0]

        preds = {}
        for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
            text_label = self.id2label[idx.item()]
            preds[i] = {'idx':idx.item(), 'text_label': text_label, 'prob': prob}
        return preds
    
def test_s2s():
    model = VJEPA2()
    model.model.eval()

    model.model.config

    path = Path(r'C:\Users\lahir\Downloads\s2s_test')
    n_files = len([p for p in path.rglob("*") if p.is_file()])
    dirs = [p for p in path.iterdir() if p.is_dir()]

    n_correct = 0
    n_samples = 0
    for dir in dirs:
        d_name = dir.name
        cls_idx = model.label2id[d_name]
        files = [p for p in dir.iterdir() if p.is_file()]
        for file in files:
            print(f'{n_samples} or {n_files} is done.', end='\r')
            n_samples += 1
            with torch.no_grad():
                preds = model.predict_from_path(file)
                pred_cls_idx = preds[0]['idx']
                if cls_idx==pred_cls_idx:
                    n_correct += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')


if __name__ == '__main__':
    test_s2s()
