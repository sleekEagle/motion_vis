from argparse import Namespace
import json
from models.resnet3d.main import generate_model, get_inference_utils, resume_model
from models.resnet3d.model import generate_model, make_data_parallel
from pathlib import Path
import torch
import models.resnet3d

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

inputs, targets = iter(inference_loader).__next__()
video_size = inputs[[0]].shape
transform = inference_loader.dataset.spatial_transform

_transforms = transform.transforms
idx = [type(i) for i in _transforms].index(resnet3d.spatial_transforms.Normalize)
normalize = _transforms[idx]
mean = torch.tensor(normalize.mean)
std = torch.tensor(normalize.std)

unnormalize = transforms.Compose(
    [
        Normalize((-mean / std).tolist(), (1 / std).tolist()),
        ToPILImage(),
    ]
)

pass


