import torch
import numpy as np
from pathlib import Path
from models.ssv2 import VJEPA2
from dataloaders import ssv2
import torch.nn.functional as F

'''
Accuracy = 69.18 % 
'''

def make_inference(model, video, class_names):
    # video = ucf101dm.load_jpg_ucf101(vid_path)
    # video = video.unsqueeze(0).permute(0,2,1,3,4)
    pred = model(video)
    pred = F.softmax(pred,dim=1)
    pred_cls = torch.argmax(pred,dim=1).item()
    ret = {
        'pred_original_class': class_names[pred_cls],
        'pred_original_idx': pred_cls,
    }

    return ret

def test_s2s():
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)
    
    #make sure all the class names are present in the list of dirs
    for c in class_names:
        assert c in d_names , f'{c} is not in the list of dirs'
        pass

    n_correct = 0
    n_samples = 0

    for idx, p in enumerate(paths):
        print(f'{idx} of {n_files} is done.', end='\r')
        gt_idx = model.label2id[d_names[idx]]
        with torch.no_grad():
            # preds = model.predict_from_path(p)
            v = model.video_from_path(p)['pixel_values'][0,:].permute(1,0,2,3)
            ret = make_inference(model, v.unsqueeze(0), class_names)
            
            # preds = model(v[None,:])
            # pred_idx = torch.argmax(preds,dim=1).item()
            if ret['pred_original_idx']==gt_idx:
                n_correct += 1
            else:
                with open(r'C:\Users\lahir\Downloads\UCF101\analysis\ssv2_incorrect.txt', 'w') as file:
                    file.write(str(p))
        n_samples += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')

if __name__ == '__main__':
    test_s2s()


