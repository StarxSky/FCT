import torch
import torch.nn.functional as F

from torch import nn
from Core.Layers import Bottleneck



class ResNet(nn.Module):
    """Resnet module."""

    def __init__(
            self,
            block,
            layers,
            num_classes: int = 1000,
            base_width: int = 64,
            embedding_dim = None,
            last_nonlin: bool = True,
            norm_feature: bool = False,
    ) -> None:
        """Construct a ResNet module.

        :param block: Block module to use in Resnet architecture.
        :param layers: List of number of blocks per layer.
        :param num_classes: Number of classes in the dataset. It is used to
            form linear classifier weights.
        :param base_width: Base width of the blocks.
        :param embedding_dim: Size of the output embedding dimension.
        :param last_nonlin: Whether to apply non-linearity before output.
        :param norm_feature: Whether to normalized output embeddings.
        """
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.OUTPUT_SHAPE = [embedding_dim, 1, 1]
        self.is_normalized = norm_feature
        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if embedding_dim is not None:
            print("Using given embedding dimension = {}".format(embedding_dim))
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], embedding_dim=64 * block.expansion
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            embedding_dim=128 * block.expansion,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            embedding_dim=256 * block.expansion,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            nonlin=last_nonlin,
            embedding_dim=self.embedding_dim,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1,
                            stride=1, bias=False)

    def _make_layer(
            self,
            block: nn.Module,
            planes: int,
            blocks: int,
            embedding_dim: int,
            stride: int = 1,
            nonlin: bool = True
    ):
        """Make a layer of resnet architecture.

        :param block: Block module to use in this layer.
        :param planes: Number of output channels.
        :param blocks: Number of blocks in this layer.
        :param embedding_dim: Size of the output embedding dimension.
        :param stride: Stride size.
        :param nonlin: Whether to apply non-linearity before output.
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1,
                              stride=stride, bias=False)
            dbn = nn.BatchNorm2d(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        last_downsample = None

        layers = []
        if blocks == 1:  # If this layer has only one-block
            if stride != 1 or self.inplanes != embedding_dim:
                dconv = nn.Conv2d(self.inplanes, embedding_dim, kernel_size=1,
                                  stride=stride, bias=False)
                dbn = nn.BatchNorm2d(embedding_dim)
                if dbn is not None:
                    last_downsample = nn.Sequential(dconv, dbn)
                else:
                    last_downsample = dconv
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    last_downsample,
                    base_width=self.base_width,
                    nonlin=nonlin,
                    embedding_dim=embedding_dim,
                )
            )
            return nn.Sequential(*layers)
        else:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    base_width=self.base_width,
                )
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(
                block(self.inplanes, planes,
                      base_width=self.base_width)
            )

        if self.inplanes != embedding_dim:
            dconv = nn.Conv2d(self.inplanes, embedding_dim, stride=1,
                              kernel_size=1,
                              bias=False)
            dbn = nn.BatchNorm2d(embedding_dim)
            if dbn is not None:
                last_downsample = nn.Sequential(dconv, dbn)
            else:
                last_downsample = dconv
        layers.append(
            block(
                self.inplanes,
                planes,
                downsample=last_downsample,
                base_width=self.base_width,
                nonlin=nonlin,
                embedding_dim=embedding_dim,
            )
        )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Apply forward pass.

        :param x: input to the model with shape (N, C, H, W).
        :return: Tuple of (logits, embedding)
        """
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.avgpool(x)
        if self.is_normalized:
            feature = F.normalize(feature)

        x = self.fc(feature)
        x = x.view(x.size(0), -1)

        return x, feature



def ResNet50(num_classes: int,
             embedding_dim: int,
             last_nonlin: bool = True,
             **kwargs) -> nn.Module:
    """Get a ResNet50 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )

