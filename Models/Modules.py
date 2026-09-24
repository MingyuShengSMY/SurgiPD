import torch.nn as nn
import torch.nn.functional as F

from utils.utils import weight_init


class Encoder32(nn.Module):
    def __init__(self, in_dim, img_size=256):
        super().__init__()

        img_size1 = (img_size - 3) // 2 + 1  # //2

        img_size1 = img_size // 2  # //2
        img_size2 = img_size1 // 2  # //4
        img_size3 = img_size2 // 2  # //8

        fla_dim = 32 * img_size3 * img_size3
        self.out_dim = fla_dim
        self.feat_size = img_size3

        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

        self.neck = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, 1),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x -= 0.5
        x *= 2.0

        x = x.flatten(1, 2)

        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        y = self.neck(f3)

        return y


class CURLEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=32, mid_dim=32, img_size=64):
        super().__init__()

        # 64
        img_size1 = (img_size - 3) // 2 + 1  # 31
        img_size2 = (img_size1 - 3) // 1 + 1  # 29
        img_size3 = (img_size2 - 3) // 1 + 1  # 27
        img_size4 = (img_size3 - 3) // 1 + 1  # 25

        self.fla_dim = out_dim * img_size4 * img_size4
        # self.out_dim = 128
        self.out_dim = 50

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, 3, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, 3, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, out_dim, 3, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.out_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.fla_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x = x.flatten(1, 2)

        feat_map = self.net(x)
        y = self.out_net(feat_map)

        return y, feat_map


class CURLEncoder256(nn.Module):
    def __init__(self, in_dim, out_dim=32, mid_dim=32, img_size=64):
        super().__init__()

        # 64
        img_size1 = (img_size - 3) // 2 + 1  # 31
        img_size2 = (img_size1 - 3) // 2 + 1  # 29
        img_size3 = (img_size2 - 3) // 1 + 1  # 27
        img_size4 = (img_size3 - 3) // 1 + 1  # 25

        self.fla_dim = out_dim * img_size4 * img_size4
        # self.out_dim = 128
        self.out_dim = 50

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, 3, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, 3, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, out_dim, 3, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

        self.out_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.fla_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x = x.flatten(1, 2)

        feat_map = self.net(x)
        y = self.out_net(feat_map)

        return y, feat_map


class DrQEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=32, mid_dim=32, img_size=64, kernel_size=3):
        super().__init__()

        # 64
        img_size1 = (img_size - kernel_size) // 2 + 1  # 31
        img_size2 = (img_size1 - kernel_size) // 1 + 1  # 29
        img_size3 = (img_size2 - kernel_size) // 1 + 1  # 27
        img_size4 = (img_size3 - kernel_size) // 1 + 1  # 25

        fla_dim = out_dim * img_size4 * img_size4
        self.out_dim = fla_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, out_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x -= 0.5
        x *= 2.0

        x = x.flatten(1, 2)

        feat_map = self.net(x)

        return feat_map


class DrQEncoder256(nn.Module):
    def __init__(self, in_dim, out_dim=32, mid_dim=32, img_size=64, kernel_size=3):
        super().__init__()

        # 64
        img_size1 = (img_size - kernel_size) // 2 + 1  # 127
        img_size2 = (img_size1 - kernel_size) // 2 + 1  # 63
        img_size3 = (img_size2 - kernel_size) // 1 + 1  # 31
        img_size4 = (img_size3 - kernel_size) // 1 + 1  # 29

        fla_dim = out_dim * img_size4 * img_size4
        self.out_dim = fla_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, out_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x -= 0.5
        x *= 2.0

        x = x.flatten(1, 2)

        feat_map = self.net(x)

        return feat_map


class DrQEncoder512(nn.Module):
    def __init__(self, in_dim, out_dim=32, mid_dim=32, img_size=64, kernel_size=3):
        super().__init__()

        # 64
        img_size1 = (img_size - kernel_size) // 2 + 1  # 127
        img_size2 = (img_size1 - kernel_size) // 2 + 1  # 63
        img_size3 = (img_size2 - kernel_size) // 2 + 1  # 31
        img_size4 = (img_size3 - kernel_size) // 1 + 1  # 29

        fla_dim = out_dim * img_size4 * img_size4
        self.out_dim = fla_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, mid_dim, kernel_size, 2),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_dim, out_dim, kernel_size, 1),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.ReLU(inplace=True),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, T, C, H, W]

        x -= 0.5
        x *= 2.0

        x = x.flatten(1, 2)

        feat_map = self.net(x)

        return feat_map


class Decoder32(nn.Module):
    def __init__(self, seg_class_num, in_dim=32, feat_size=32):
        super().__init__()

        # 64
        feat_size1 = feat_size * 2  # //2
        feat_size2 = feat_size1 * 2  # //4
        feat_size3 = feat_size2 * 2  # //8

        self.seg_class_num = seg_class_num

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 32, 2, 2),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, 2),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, 2),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(16, seg_class_num, 1, 1),
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, 1, 1),
        )

        # self.apply(weight_init)

    def forward(self, x):
        # [B, D, H, W]

        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        feat = self.refine(f3)

        seg_pre = self.seg_head(feat)
        depth_pre = self.depth_head(feat)

        return seg_pre, depth_pre



