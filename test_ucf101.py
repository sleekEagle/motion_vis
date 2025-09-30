from argparse import Namespace
import json
from models.resnet3d.main import generate_model, get_inference_utils, resume_model
from models.resnet3d.model import generate_model, make_data_parallel
from pathlib import Path
import torch
from models.resnet3d import spatial_transforms
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Normalize, ToPILImage
import os
import re
from glob import glob
from PIL import Image
import numpy as np

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_jpg_ucf101(l, g, c, n, inference_class_names, transform):
    name = inference_class_names[l]
    dir = os.path.join(
        "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs", name, "v_{}_g{}_c{}".format(name, str(g).zfill(2), str(c).zfill(2))
    )
    path = sorted(glob(dir + "/*"), key=numericalSort)

    target_path = path[n * 16 : (n + 1) * 16]
    if len(target_path) < 16:
        print("not exist")
        return False

    video = []
    for _p in target_path:
        video.append(transform(Image.open(_p)))

    return torch.stack(video)

#read model options
opt_path = "models/r3d/ucf101.json"
with open(opt_path, "r") as f:
    model_opt = json.load(f)
model_opt = Namespace(**model_opt)

model = generate_model(model_opt)
model = resume_model(model_opt.resume_path, model_opt.arch, model)
model.eval()

model_opt.inference_batch_size = 1
for attribute in dir(model_opt):
    if "path" in str(attribute) and getattr(model_opt, str(attribute)) != None:
        setattr(model_opt, str(attribute), Path(getattr(model_opt, str(attribute))))
inference_loader, inference_class_names = get_inference_utils(model_opt)
class_labels_map = {v.lower(): k for k, v in inference_class_names.items()}


def main():
    # inputs, targets = iter(inference_loader).__next__()
    # video_size = inputs[[0]].shape
    transform = inference_loader.dataset.spatial_transform

    # _transforms = transform.transforms
    # idx = [type(i).__name__ for i in _transforms].index('Normalize')
    # normalize = _transforms[idx]
    # mean = torch.tensor(normalize.mean)
    # std = torch.tensor(normalize.std)

    # unnormalize = transforms.Compose(
    #     [
    #         Normalize((-mean / std).tolist(), (1 / std).tolist()),
    #         ToPILImage(),
    #     ]
    # )

    # spatial_crop_size = 16
    # spatial_stride = 8
    # temporal_stride = 2

    l = 21
    g = 1  # > 0
    c = 1  # > 0
    n = 1

    video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)
    target = l
    with torch.inference_mode():
        pred = model(video.unsqueeze(0)).cpu().numpy().argmax()

    pass
    # video_orgimg = []
    # for i in range(video_size[2]):
    #     img = video.squeeze().transpose(0, 1)[i]
    #     video_orgimg.append(np.array(unnormalize(img)))
    # video_orgimg = np.array(video_orgimg)


def test():
    root = "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs"
    transform = inference_loader.dataset.spatial_transform
    n_samples = 0
    n_correct = 0
    for k in inference_class_names.keys():
        print(f'Inferring class {inference_class_names[k]}')
        class_path = os.path.join(root,inference_class_names[k])
        dirs = [item.name for item in Path(class_path).iterdir() if item.is_dir()]
        for d in dirs:
            l = k
            g = int(d.split('_')[2][1:])
            c = int(d.split('_')[3][1:])
            n=1
            video = load_jpg_ucf101(l, g, c, n, inference_class_names, transform).transpose(0, 1)
            with torch.inference_mode():
                n_samples += 1
                pred = model(video.unsqueeze(0)).cpu().numpy().argmax()
                if int(pred) == k:
                    n_correct += 1
    print(f'Acuracy : {n_correct/n_samples}')

if __name__ == '__main__':
    test()


