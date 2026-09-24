import os

import cv2
import numpy as np
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
# import torchvision.transforms.functional as TF
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as tvt


def cv2_imshow(img, name="show", wait=0):
    cv2.imshow(name, img.astype(np.uint8))
    cv2.waitKey(wait)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_bce(pre, gt):
    # Same shape
    loss = - gt * torch.log(pre + 1e-6) - (1 - gt) * torch.log(1 - pre + 1e-6)
    return loss.mean()


def rgb_add_noise(x: np.ndarray, p=1.0):
    while x.ndim <= 3:
        x = np.expand_dims(x, 0)
    if not np.random.choice([True, False], p=[p, 1 - p]).item():
        return x
    else:
        x = torch.from_numpy(x)

        x = TF.rotate_image(x, angle=np.random.randint(0, 16).item())
        x = TF.adjust_brightness(x, 1.0 + np.random.uniform(-0.2, 0.2))
        x = TF.adjust_contrast(x, 1.0 + np.random.uniform(-0.2, 0.2))
        x = TF.adjust_saturation(x, 1.0 + np.random.uniform(-0.2, 0.2))

        x = TF.gaussian_noise(x)
        x = TF.erase()


def sample_farthest_colors(n, seed_color=(0, 0, 0)):

    grid_size = 51
    xs = np.linspace(0, 255, grid_size)

    g = np.array(np.meshgrid(xs, xs, xs, indexing="ij"))

    candidates = g.reshape(3, -1).T

    selected = [np.array(seed_color, dtype=float)]

    dists = np.linalg.norm(candidates - selected[0], axis=1)

    for _ in range(1, n):
        idx = np.argmax(dists)
        next_color = candidates[idx]
        selected.append(next_color)

        dists = np.minimum(dists, np.linalg.norm(candidates - next_color, axis=1))

    return np.array(selected).astype(np.uint8)


def de_tanh_np(x):
    y = 0.5 * (np.log(1 + x) - np.log(1 - x))
    return y


def de_tanh_tn(x):
    y = 0.5 * (torch.log(1 + x) - torch.log(1 - x))
    return y


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def focal_loss_multiclass(logits, target, gamma=2):
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)

    target_logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    target_p = p.gather(1, target.unsqueeze(1)).squeeze(1)

    loss = -((1 - target_p) ** gamma) * target_logp
    return loss.mean()


def multiclass_dice_loss(logits, targets, num_classes, eps=1e-6):
    # logits: [B, C, H, W]
    # targets: [B, H, W]

    probs = torch.softmax(logits, dim=1)

    targets_onehot = F.one_hot(targets, num_classes)  # [B,H,W,C]
    targets_onehot = targets_onehot.permute(0,3,1,2).float()

    dims = (2,3)

    intersection = (probs * targets_onehot).sum(dims)
    union = probs.sum(dims) + targets_onehot.sum(dims)

    dice = ((2 * intersection + eps) / (union + eps)).mean(-1)
    dice_loss = 1 - dice
    dice_loss_mean = dice_loss.mean()
    return dice_loss_mean


def symmetric_kl_gaussian_fast(mu_a, std_a, mu_b, std_b, eps=1e-8):
    std_a = std_a.clamp(min=eps)
    std_b = std_b.clamp(min=eps)

    var_a = std_a ** 2
    var_b = std_b ** 2

    d_mu = (mu_a - mu_b) ** 2

    term1 = var_a / var_b
    term2 = var_b / var_a
    term3 = d_mu / var_a
    term4 = d_mu / var_b

    sym_kl = 0.5 * (term1 + term2 + term3 + term4 - 2.0)

    return sym_kl


def forward_kl_gaussian(
    mu_student,
    std_student,
    mu_teacher,
    std_teacher,
    eps=1e-8,
):
    std_teacher = std_teacher.clamp(min=eps)
    std_student = std_student.clamp(min=eps)

    var_teacher = std_teacher ** 2
    var_student = std_student ** 2

    d_mu = (mu_teacher - mu_student) ** 2

    kl = (
        torch.log(std_student / std_teacher)
        + (var_teacher + d_mu) / (2.0 * var_student)
        - 0.5
    )

    return kl.mean()


def remove_event_log_file(log_dir):
    file_list = os.listdir(f"{log_dir}")
    for file_i in file_list:
        if os.path.isfile(f"{log_dir}/{file_i}") and "events" in file_i:
            os.remove(f"{log_dir}/{file_i}")
        elif os.path.isdir(f"{log_dir}/{file_i}"):
            remove_event_log_file(f"{log_dir}/{file_i}")


def random_crop(rgb: torch.Tensor, img_size=84):
    # [B, T, C, H, W]

    B, T, C, H, W = rgb.shape

    pad = img_size // 20
    rgb_pad = F.pad(rgb.flatten(1, 2), (pad, pad, pad, pad), mode="replicate").unflatten(1, (T, C))

    img_size_arange = torch.arange(img_size)

    dx = torch.randint(0, 2 * pad + 1, size=[B, 1]) + img_size_arange
    dy = torch.randint(0, 2 * pad + 1, size=[B, 1]) + img_size_arange

    dx = dx[:, None, None, :, None]
    dy = dy[:, None, None, None, :]

    b_idx = torch.arange(B, device=rgb.device)[:, None, None, None, None]
    t_idx = torch.arange(T, device=rgb.device)[None, :, None, None, None]
    c_idx = torch.arange(C, device=rgb.device)[None, None, :, None, None]

    rgb_crop = rgb_pad[b_idx, t_idx, c_idx, dx, dy]

    return rgb_crop


def img_modal_corrupt_func(img, modal_corrupt, img_size=84):
    if modal_corrupt == "random_crop":
        img_mc = random_crop(img, img_size)
    elif "gaussian_noise" in modal_corrupt:
        sigma = float(modal_corrupt.split("_")[-1])
        img_mc = img + torch.randn(img.shape) * sigma
        img_mc.clamp_(0.0, 1.0)
    else:
        raise ValueError(f"Unknown modal_corrupt: [{modal_corrupt}")

    return img_mc


def main():
    color_list = sample_farthest_colors(100)
    color_dict = {i: c.tolist() for i, c in enumerate(color_list)}

    pass


if __name__ == '__main__':
    main()




