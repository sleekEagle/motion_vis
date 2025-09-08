import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "kinetics-400",
    split="validation",
    max_samples=10,
    shuffle=True,
)
session = fo.launch_app(dataset)

dataset = foz.load_zoo_dataset(
    "kinetics-400",
    split="validation",
    classes=["springboard diving", "surfing water"],
    max_samples=10,
)