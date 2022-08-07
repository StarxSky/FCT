import os
import torch

from torch import nn
from typing import Union


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing: float = 0.0):
        """Construct LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Logits tensor.
        :param target: Ground truth target classes.
        :return: Loss tensor.
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FeatureExtractor(nn.Module):
    """A wrapper class to return only features (no logits)."""

    def __init__(self,
                 model: Union[nn.Module, torch.jit.ScriptModule]) -> None:
        """Construct FeatureExtractor module.

        :param model: A model that outputs both logits and features.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Input data.
        :return: Feature tensor computed for x.
        """
        _, feature = self.model(x)
        return feature


class TransformedOldModel(nn.Module):
    """A wrapper class to return transformed features."""

    def __init__(self,
                 old_model: Union[nn.Module, torch.jit.ScriptModule],
                 side_model: Union[nn.Module, torch.jit.ScriptModule],
                 transformation: Union[
                     nn.Module, torch.jit.ScriptModule]) -> None:
        """Construct TransformedOldModel module.

        :param old_model: Old model.
        :param side_model: Side information model.
        :param transformation: Transformation model.
        """
        super().__init__()
        self.old_model = old_model
        self.transformation = transformation
        self.side_info_model = side_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param x: Input data
        :return: Transformed old feature.
        """
        old_feature = self.old_model(x)
        side_info = self.side_info_model(x)
        recycled_feature = self.transformation(old_feature, side_info)
        return recycled_feature


class BasicBlock(nn.Module):
    """Resnet basic block module."""

    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim = None,
    ) -> None:
        """Construct a BasicBlock module.

        :param inplanes: Number of input channels.
        :param planes: Number of output channels.
        :param stride: Stride size.
        :param downsample: Down-sampling for residual path.
        :param base_width: Base width of the block.
        :param nonlin: Whether to apply non-linearity before output.
        :param embedding_dim: Size of the output embedding dimension.
        """
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")
        if embedding_dim is not None:
            planes = embedding_dim
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.nonlin:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Resnet bottleneck block module."""

    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim = None,
    ) -> None:
        """Construct a Bottleneck module.

        :param inplanes: Number of input channels.
        :param planes: Number of output channels.
        :param stride: Stride size.
        :param downsample: Down-sampling for residual path.
        :param base_width: Base width of the block.
        :param nonlin: Whether to apply non-linearity before output.
        :param embedding_dim: Size of the output embedding dimension.
        """
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        if embedding_dim is not None:
            out_dim = embedding_dim
        else:
            out_dim = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_dim, kernel_size=1, stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.nonlin:
            out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    """Convenience convolution module."""

    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 normalizer = nn.BatchNorm2d,
                 activation = nn.ReLU) -> None:
        """Construct a ConvBlock module.

        :param channels_in: Number of input channels.
        :param channels_out: Number of output channels.
        :param kernel_size: Size of the kernel.
        :param stride: Size of the convolution stride.
        :param normalizer: Optional normalization to use.
        :param activation: Optional activation module to use.
        """
        super().__init__()

        self.conv = nn.Conv2d(channels_in, channels_out,
                              kernel_size=kernel_size, stride=stride,
                              bias=normalizer is None,
                              padding=kernel_size // 2)
        if normalizer is not None:
            self.normalizer = normalizer(channels_out)
        else:
            self.normalizer = None
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        x = self.conv(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x



"""""
以下是实现模型有关的功能的代码
"""""
def prepare_model_for_export(
        model: Union[nn.Module, torch.jit.ScriptModule]
) -> Union[nn.Module, torch.jit.ScriptModule]:
    """Prepare a model to be exported as torchscript."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    model.cpu()
    return model


def backbone_to_torchscript(model: Union[nn.Module, torch.jit.ScriptModule],
                            output_model_path: str) -> None:
    """Convert a backbone model to torchscript.

    :param model: A backbone model to be converted to torch script.
    :param output_model_path: Path to save torch script.
    """
    model = prepare_model_for_export(model)
    f = FeatureExtractor(model)
    model_script = torch.jit.script(f)
    torch.jit.save(model_script, output_model_path)


def transformation_to_torchscripts(
        old_model: Union[nn.Module, torch.jit.ScriptModule],
        side_model: Union[nn.Module, torch.jit.ScriptModule],
        transformation: Union[nn.Module, torch.jit.ScriptModule],
        output_transformation_path: str,
        output_transformed_old_model_path: str) -> None:
    """Convert a transformation model to torchscript.

    :param old_model: Old model.
    :param side_model: Side information model.
    :param transformation: Transformation model.
    :param output_transformation_path: Path to store transformation torch
        script.
    :param output_transformed_old_model_path: Path to store combined old and
        transformation models' torch script.
    """
    transformation = prepare_model_for_export(transformation)
    old_model = prepare_model_for_export(old_model)
    side_model = prepare_model_for_export(side_model)

    model_script = torch.jit.script(transformation)
    torch.jit.save(model_script, output_transformation_path)

    f = TransformedOldModel(old_model, side_model, transformation)
    model_script = torch.jit.script(f)
    torch.jit.save(model_script, output_transformed_old_model_path)
