import torch
import numpy as np
from pathlib import Path
from models.ssv2 import VJEPA2
from dataloaders import ssv2

'''
Accuracy = 69.18 % 
'''
def test_s2s():
    model = VJEPA2()
    model.eval()

    cls_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)
    
    #make sure all the class names are present in the list of dirs
    for c in cls_names:
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
            preds = model(v[None,:])
            pred_idx = torch.argmax(preds,dim=1).item()
            if pred_idx==gt_idx:
                n_correct += 1
        n_samples += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')

if __name__ == '__main__':
    test_s2s()


