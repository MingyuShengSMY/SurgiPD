import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TensorEdgeAug:
    def __init__(self, img_in_dim=3, *args, **kwargs):
        self.img_in_dim = img_in_dim
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(img_in_dim, 1, 1, 1)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(img_in_dim, 1, 1, 1)

    def edge_aug(self, x):

        B, C, H, W = x.shape

        edge_x = F.conv2d(
            x,
            self.sobel_x,
            padding=1,
            groups=self.img_in_dim,
        )

        edge_y = F.conv2d(
            x,
            self.sobel_y,
            padding=1,
            groups=self.img_in_dim,
        )

        # 边缘强度
        edge = torch.sqrt(edge_x.square() + edge_y.square())

        # 边缘增强（原图 + 边缘）
        out = x + edge

        return out, edge



