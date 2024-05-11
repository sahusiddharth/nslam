import numpy as np
import torch
import os
from vision_transformers.vision_transformers import VitGenerator
from vision_transformers.preprocess import Loader,visualize_predict
device=torch.device("cpu")
# set some variables
name_model = 'vit_small'
patch_size = 8

model = VitGenerator(name_model, patch_size,
                     device, evaluate=True, random=False, verbose=True)

torch.save(model.vit_model.state_dict(), "model.pth")