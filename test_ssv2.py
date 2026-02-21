import torch
import numpy as np
from pathlib import Path
from dataloaders.ssv2 import VJEPA2
    
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


