import torch

from torch import nn
from Core.Layers import ConvBlock


class MLP_BN_SIDE_PROJECTION(nn.Module):
    """FCT transformation module."""

    def __init__(self,
                 old_embedding_dim: int,
                 new_embedding_dim: int,
                 side_info_dim: int,
                 inner_dim: int = 2048
                 ) -> None:
        """Construct MLP_BN_SIDE_PROJECTION module.

        :param old_embedding_dim: Size of the old embeddings.
        :param new_embedding_dim: Size of the new embeddings.
        :param side_info_dim: Size of the side-information.
        :param inner_dim: Dimension of transformation MLP inner layer.
        """
        super(MLP_BN_SIDE_PROJECTION, self).__init__()

        self.inner_dim = inner_dim
        self.p1 = nn.Sequential(
            ConvBlock(old_embedding_dim, 2 * old_embedding_dim),
            ConvBlock(2 * old_embedding_dim, 2 * new_embedding_dim),
        )

        self.p2 = nn.Sequential(
            ConvBlock(side_info_dim, 2 * side_info_dim),
            ConvBlock(2 * side_info_dim, 2 * new_embedding_dim),
        )

        self.mixer = nn.Sequential(
            ConvBlock(4 * new_embedding_dim, self.inner_dim),
            ConvBlock(self.inner_dim, self.inner_dim),
            ConvBlock(self.inner_dim, new_embedding_dim, normalizer=None,
                      activation=None)
        )

    def forward(self,
                old_feature: torch.Tensor,
                side_info: torch.Tensor) -> torch.Tensor:
        """Apply forward pass.

        :param old_feature: Old embedding.
        :param side_info: Side-information.
        :return: Recycled old embedding compatible with new embeddings.
        """
        x1 = self.p1(old_feature)
        x2 = self.p2(side_info)
        return self.mixer(torch.cat([x1, x2], dim=1))



