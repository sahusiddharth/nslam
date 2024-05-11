import logging
import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .core import _CAM
from .utils import locate_linear_layer

__all__ = ["CAM", "ISCAM", "SSCAM", "ScoreCAM"]


class CAM(_CAM):
    r"""Implements a class activation map extractor as described in `"Learning Deep Features for Discriminative
    Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for unit :math:`k` in the fully
    connected layer..

    >>> from torchvision.models import resnet18
    >>> from torchcam.methods import CAM
    >>> model = resnet18(pretrained=True).eval()
    >>> cam = CAM(model, 'layer4', 'fc')
    >>> with torch.no_grad(): out = model(input_tensor)
    >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        fc_layer: either the fully connected layer itself or its name
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        fc_layer: Optional[Union[nn.Module, str]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        if isinstance(target_layer, list) and len(target_layer) > 1:
            raise ValueError("base CAM does not support multiple target layers")

        super().__init__(model, target_layer, input_shape, **kwargs)

        if isinstance(fc_layer, str):
            fc_name = fc_layer
        # Find the location of the module
        elif isinstance(fc_layer, nn.Module):
            fc_name = self._resolve_layer_name(fc_layer)
        # If the layer is not specified, try automatic resolution
        elif fc_layer is None:
            fc_name = locate_linear_layer(model)  # type: ignore[assignment]
            # Warn the user of the choice
            if isinstance(fc_name, str):
                logging.warning(f"no value was provided for `fc_layer`, thus set to '{fc_name}'.")
            else:
                raise ValueError("unable to resolve `fc_layer` automatically, please specify its value.")
        else:
            raise TypeError("invalid argument type for `fc_layer`")
        # Softmax weight
        self._fc_weights = self.submodule_dict[fc_name].weight.data
        # squeeze to accomodate replacement by Conv1x1
        if self._fc_weights.ndim > 2:
            self._fc_weights = self._fc_weights.view(*self._fc_weights.shape[:2])

    @torch.no_grad()
    def _get_weights(
        self,
        class_idx: Union[int, List[int]],
        *_: Any,
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        # Take the FC weights of the target class
        if isinstance(class_idx, int):
            return [self._fc_weights[class_idx, :].unsqueeze(0)]
        else:
            return [self._fc_weights[class_idx, :]]


class ScoreCAM(_CAM):
    r"""Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax\Big(Y^{(c)}(M_k) - Y^{(c)}(X_b)\Big)_k

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \frac{U(A_k) - \min\limits_m U(A_m)}{\max\limits_m  U(A_m) - \min\limits_m  U(A_m)})
        \odot X_b

    where :math:`\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    >>> from torchvision.models import resnet18
    >>> from torchcam.methods import ScoreCAM
    >>> model = resnet18(pretrained=True).eval()
    >>> cam = ScoreCAM(model, 'layer4')
    >>> with torch.no_grad(): out = model(input_tensor)
    >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        batch_size: batch size used to forward masked inputs
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        batch_size: int = 32,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))  # type: ignore[arg-type]
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, _: nn.Module, _input: Tensor) -> None:
        """Store model input tensor."""
        if self._hooks_enabled:
            self._input = _input[0].data.clone()

    @torch.no_grad()
    def _get_score_weights(self, activations: List[Tensor], class_idx: Union[int, List[int]]) -> List[Tensor]:
        b, c = activations[0].shape[:2]
        # (N * C, I, H, W)
        scored_inputs = [
            (act.unsqueeze(2) * self._input.unsqueeze(1)).view(b * c, *self._input.shape[1:]) for act in activations
        ]

        # Initialize weights
        # (N * C)
        weights = [torch.zeros(b * c, dtype=t.dtype).to(device=t.device) for t in activations]

        # (N, M)
        logits = self.model(self._input)
        idcs = torch.arange(b).repeat_interleave(c)

        for idx, scored_input in enumerate(scored_inputs):
            # Process by chunk (GPU RAM limitation)
            for _idx in range(math.ceil(weights[idx].numel() / self.bs)):
                _slice = slice(_idx * self.bs, min((_idx + 1) * self.bs, weights[idx].numel()))
                # Get the softmax probabilities of the target class
                # (*, M)
                cic = self.model(scored_input[_slice]) - logits[idcs[_slice]]
                if isinstance(class_idx, int):
                    weights[idx][_slice] = cic[:, class_idx]
                else:
                    _target = torch.tensor(class_idx, device=cic.device)[idcs[_slice]]
                    weights[idx][_slice] = cic.gather(1, _target.view(-1, 1)).squeeze(1)

        # Reshape the weights (N, C)
        return [torch.softmax(w.view(b, c), -1) for w in weights]

    @torch.no_grad()
    def _get_weights(
        self,
        class_idx: Union[int, List[int]],
        *_: Any,
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        self.hook_a: List[Tensor]  # type: ignore[assignment]

        # Normalize the activation
        # (N, C, H', W')
        upsampled_a = [self._normalize(act.clone(), act.ndim - 2) for act in self.hook_a]

        # Upsample it to input_size
        # (N, C, H, W)
        spatial_dims = self._input.ndim - 2
        interpolation_mode = "bilinear" if spatial_dims == 2 else "trilinear" if spatial_dims == 3 else "nearest"
        upsampled_a = [
            F.interpolate(
                up_a,
                self._input.shape[2:],
                mode=interpolation_mode,
                align_corners=False,
            )
            for up_a in upsampled_a
        ]

        # Disable hook updates
        self._hooks_enabled = False
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()

        weights: List[Tensor] = self._get_score_weights(upsampled_a, class_idx)

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs})"