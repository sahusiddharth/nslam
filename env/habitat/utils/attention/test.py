
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
from .gradient import SmoothGradCAMpp

model = resnet18(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model)
