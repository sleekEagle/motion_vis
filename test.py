import torch
from pytorchvideo.models.hub import slow_r50
from pytorchvideo.data import Kinetics
from torch.utils.data import DataLoader

# Load pretrained model
model = slow_r50(pretrained=True)
model.eval()

# Download and load Kinetics-400 validation set
val_dataset = Kinetics(
    data_path="./kinetics",
    clip_sampler="uniform",
    video_sampler="uniform",
    decode_audio=False,
    transform=None
)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

pass