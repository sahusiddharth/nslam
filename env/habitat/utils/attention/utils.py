from typing import cast
import PIL
import numpy as np
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
from PIL.Image import Image, fromarray

from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


def locate_candidate_layer(mod: nn.Module, input_shape: Tuple[int, ...] = (3, 224, 224)) -> Optional[str]:
    """Attempts to find a candidate layer to use for CAM extraction

    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch dimension

    Returns:
        str: the candidate layer for CAM
    """
    # Set module in eval mode
    module_mode = mod.training
    mod.eval()

    output_shapes: List[Tuple[Optional[str], Tuple[int, ...]]] = []

    def _record_output_shape(_: nn.Module, _input: Tensor, output: Tensor, name: Optional[str] = None) -> None:
        """Activation hook."""
        output_shapes.append((name, output.shape))

    hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    # forward hook on all layers
    for n, m in mod.named_modules():
        hook_handles.append(m.register_forward_hook(partial(_record_output_shape, name=n)))

    # forward empty
    with torch.no_grad():
        _ = mod(torch.zeros((1, *input_shape), device=next(mod.parameters()).data.device))

    # Remove all temporary hooks
    for handle in hook_handles:
        handle.remove()

    # Put back the model in the corresponding mode
    mod.training = module_mode

    # Check output shapes
    candidate_layer = None
    for layer_name, output_shape in reversed(output_shapes):
        # Stop before flattening or global pooling
        if len(output_shape) == (len(input_shape) + 1) and any(v != 1 for v in output_shape[2:]):
            candidate_layer = layer_name
            break

    return candidate_layer

def locate_linear_layer(mod: nn.Module) -> Optional[str]:
    """Attempts to find a fully connecter layer to use for CAM extraction

    Args:
        mod: the module to inspect

    Returns:
        str: the candidate layer
    """
    candidate_layer = None
    for layer_name, m in mod.named_modules():
        if isinstance(m, nn.Linear):
            candidate_layer = layer_name
            break

    return candidate_layer

def overlay_mask(img: Image, mask: Image, colormap: str = "jet", alpha: float = 0.7) -> Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """
    if not isinstance(img, Image) or not isinstance(mask, Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = plt.cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=PIL.Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    return fromarray((alpha * np.asarray(img) + (1 - alpha) * cast(np.ndarray, overlay)).astype(np.uint8))