# from collections import defaultdict
# import gymnasium as gym
import os.path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms.functional as tvtf

from Models.BaseModel import BaseModel, MyNormal, TruncatedNormal, DDPGNormal, CURLNormal
from Models.Modules import Encoder32, Decoder32, DrQEncoder, DrQEncoder256
from utils.utils import weight_init, de_tanh_tn, focal_loss_multiclass, multiclass_dice_loss, \
    symmetric_kl_gaussian_fast, cv2_imshow, hellinger_gaussian, forward_kl_gaussian, bhattacharyya_gaussian, \
    img_modal_corrupt_func

sys.path.append("..")


class CriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, out_dim=1, mid_dim=1024):
        super().__init__()

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self.cat_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, out_dim),
        )

        # self.apply(weight_init)

    def forward(self, state, act):
        # [B, D1]
        # [B, D2]
        # [B, D]

        # state = state.to(self.cat_net.device)
        # img = img.to(self.cat_net.device)

        state_act = torch.cat([state, act], dim=-1)
        output = self.cat_net(state_act)

        return output


class StatePolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, mid_dim=1024):
        super().__init__()

        self.act_dim = act_dim

        self.cat_net = nn.Sequential(
            nn.Linear(obs_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, act_dim * 2),
        )

        # self.apply(weight_init)

    def forward(self, state, img=None):
        # [B, D1]
        # [B, D2]
        # [B, D]

        output = self.cat_net(state)

        return output


class RgbPolicyNet(nn.Module):
    def __init__(self, img_in_dim, act_dim, img_feat_dim=512, mid_dim=1024, bot_state_dim=0):
        super().__init__()

        self.act_dim = act_dim

        self.bot_state_dim = bot_state_dim
        self.bot_state_mark = bot_state_dim > 0

        self.img_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(img_in_dim, img_feat_dim + img_feat_dim * (1 - self.bot_state_mark)),
            nn.LayerNorm(img_feat_dim),
            # nn.Tanh(),
        )

        if self.bot_state_mark > 0:
            self.state_net = nn.Sequential(
                nn.Linear(bot_state_dim, img_feat_dim),
            )
        else:
            self.state_net = nn.Identity()

        self.cat_net = nn.Sequential(
            nn.Linear(img_feat_dim + img_feat_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, act_dim * 2),
        )

        # self.apply(weight_init)

    def forward(self, state, img):
        # [B, D1]
        # [B, D2]
        # [B, D]

        img_feat = self.img_net(img)
        if self.bot_state_mark:
            state_feat = self.state_net(state[..., :self.bot_state_dim])
            feat = torch.cat([img_feat, state_feat], dim=-1)
        else:
            feat = img_feat

        output = self.cat_net(feat)

        return output


class RgbPolicyNetSmall(nn.Module):
    def __init__(self, img_in_dim, act_dim, img_feat_dim=512, mid_dim=1024, bot_state_dim=0):
        super().__init__()

        self.act_dim = act_dim

        self.bot_state_dim = bot_state_dim
        self.bot_state_mark = bot_state_dim > 0

        self.img_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(img_in_dim, 50),
            nn.LayerNorm(50),
            # nn.Tanh(),
        )

        self.cat_net = nn.Sequential(
            nn.Linear(50 + bot_state_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True),

            nn.Linear(mid_dim, act_dim * 2),
        )

        # self.apply(weight_init)

    def forward(self, state, img):
        # [B, D1]
        # [B, D2]
        # [B, D]
        img_feat = self.img_net(img)
        feat = torch.cat([img_feat, state[..., :self.bot_state_dim]], dim=-1)

        output = self.cat_net(feat)

        return output


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ATGRL(BaseModel):
    lr_scheduler_a: torch.optim.lr_scheduler.LRScheduler
    lr_scheduler_c: torch.optim.lr_scheduler.LRScheduler

    def __init__(
            self,
            obs_dim,
            act_dim,
            mid_dim=64,
            gamma=0.99,
            gae_lambda=0.95,
            eps_clip=0.2,
            tau=0.005,
            continue_action=False,
            alpha=1.0,
            img_size=None,
            img_c_dim=3,
            frame_k=1,
            bot_state_dim=0,
            seg_class_num=7,

            ema_state_a_net_mark=False,
            rollout_method="state",  # state, rgb
            gaussian_noise_aug=False,

            distillation_loss="skl",  # skl, fkl, rkl, was, hel, bha

            sac_weights_path=None,  # None or str, None will jointly train teacher

    ):
        super().__init__()

        # self.gamma = 0
        # self.gamma_raise_rate = 1e-3

        self.continuous_action = continue_action

        if img_size is None:
            img_size = 224
        self.img_size = img_size
        self.img_c_dim = img_c_dim

        self.obs_dim = obs_dim
        self.mid_dim = mid_dim
        self.act_dim = act_dim
        self.bot_state_dim = bot_state_dim
        self.obs_state_dim = obs_dim - bot_state_dim

        self.frame_k = frame_k

        self.seg_class_num = seg_class_num

        rollout_method_dict = {
            "rgb": 1,
            "state": 2,
        }
        distillation_loss_dict = {
            "t_o": -1,  # Only train teacher
            "skl": 0,  # Jeffreys
            "fkl": 1,  # forward kl
            "rkl": 2,  # reverse kl
            "was": 3,  # Wasserstein
        }

        self.rollout_method = rollout_method_dict[rollout_method]
        self.distillation_option = 1
        self.distillation_loss = distillation_loss_dict[distillation_loss]
        self.gaussian_noise_aug = gaussian_noise_aug
        self.sac_only = False
        if self.distillation_loss == 0:
            self.dist_loss_func = self.skl_loss
        elif self.distillation_loss == 1:
            self.dist_loss_func = self.fkl_loss
        elif self.distillation_loss == 2:
            self.dist_loss_func = self.rkl_loss
        elif self.distillation_loss == 3:
            self.dist_loss_func = self.mse_loss
        elif self.distillation_loss == -1:
            self.sac_only = True
        else:
            raise ValueError

        self.ema_state_a_net_mark = ema_state_a_net_mark and not self.sac_only

        self.dist_lambda = dist_lambda

        self.skl_avg = 1.0

        rgb_a_net = RgbPolicyNetSmall
        # rgb_a_net = RgbPolicyNet
        state_a_net = StatePolicyNet
        q_net = CriticNet

        # self.rgb_normalizer = Normalizer(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        # self.rgb_normalizer = Normalizer(
        #     mean=[0.5] * 3,
        #     std=[0.5] * 3,
        # )

        if self.img_size >= 224:
            encoder = DrQEncoder256
        else:
            encoder = DrQEncoder
        # decoder = Decoder32

        encoder_kz = 3
        # encoder_kz = 5

        self.img_encoder = encoder(img_c_dim * self.frame_k, img_size=self.img_size, kernel_size=encoder_kz)
        self.img_encoder_t = encoder(img_c_dim * self.frame_k, img_size=self.img_size, kernel_size=encoder_kz)
        self.img_encoder_t.load_state_dict(self.img_encoder.state_dict())

        self.img_feat_dim = self.img_encoder.out_dim

        # self.img_decoder = decoder(seg_class_num, feat_size=self.img_encoder.feat_size)

        # self.a_net = Net(obs_dim, mid_dim, act_dim * (2 if self.continuous_action else 1))
        self.rgb_a_net = rgb_a_net(self.img_feat_dim, act_dim, img_feat_dim=self.mid_dim // 2, mid_dim=self.mid_dim, bot_state_dim=self.bot_state_dim)
        self.state_a_net = state_a_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
        self.q1_net = q_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
        self.q2_net = q_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
        self.qt1_net = q_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
        self.qt2_net = q_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
        self.qt1_net.load_state_dict(self.q1_net.state_dict())
        self.qt2_net.load_state_dict(self.q2_net.state_dict())
        if self.ema_state_a_net_mark:
            self.state_at_net = state_a_net(self.obs_dim, act_dim, mid_dim=self.mid_dim)
            self.state_at_net.load_state_dict(self.state_a_net.state_dict())
        else:
            self.state_at_net = nn.Identity()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.tau = tau
        self.imi_beta = 0.0

        self.target_entropy = -self.act_dim
        self.state_log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True).add(alpha).log())
        self.rgb_log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True).add(alpha).log())

        self.frozen_params.extend(self.qt1_net.parameters())
        self.frozen_params.extend(self.qt2_net.parameters())
        self.frozen_params.extend(self.img_encoder_t.parameters())
        if self.ema_state_a_net_mark:
            self.frozen_params.extend(self.state_at_net.parameters())

        self.hot_params.extend(self.state_a_net.parameters())
        self.hot_params.extend(self.q1_net.parameters())
        self.hot_params.extend(self.q2_net.parameters())
        self.hot_params.extend([self.state_log_alpha])
        if not self.sac_only:
            self.hot_params.extend(self.rgb_a_net.parameters())
            self.hot_params.extend([self.rgb_log_alpha])
            self.hot_params.extend(self.img_encoder.parameters())

        self.log_std_bound = [-4, 2]
        # self.log_std_bound = [-20, 2]
        # self.log_std_bound = [np.log(0.2), 2]

        # self.q_target_bound = [-150, 10]
        # self.q_target_bound = [-100, 20]
        # self.q_target_bound = [-100, 50]
        # self.q_target_bound = [-100, 10]  # best
        self.q_target_bound = [-100, 0]
        # self.q_target_bound = [-200, 0]
        # self.q_target_bound = [-100, 100]
        # self.q_target_bound = [-200, 200]
        # self.q_target_bound = [-500, 500]
        # self.q_target_bound = [-30, 30]
        # self.q_target_bound = [-30, 10]
        # self.q_target_bound = [-torch.inf, torch.inf]
        # self.q_target_bound = [-100, 3]

        self.get_action = self.get_action_

        self.frozen_params.requires_grad_(False)
        self.frozen_params.eval()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.grad_clip = torch.inf
        # self.grad_clip = 1.0  # best
        # self.grad_clip = 10.0

        self.train_sac = True
        self.sac_weights_path = sac_weights_path

        if sac_weights_path is not None and os.path.exists(sac_weights_path) and not self.sac_only:
            print("Loading Pretrained SAC...")
            self.train_sac = False
            a_net_state = torch.load(sac_weights_path, weights_only=False)
            self.state_a_net.load_state_dict(a_net_state)
            if self.ema_state_a_net_mark:
                self.state_at_net.load_state_dict(a_net_state)

            self.frozen_params.extend(self.state_a_net.parameters())
            self.frozen_params.append(self.state_log_alpha)

            if self.distillation_option == 1:
                self.frozen_params.extend(self.q1_net.parameters())
                self.frozen_params.extend(self.q2_net.parameters())

        if self.sac_only:
            self.rgb_a_net = nn.Identity()
            self.img_encoder = nn.Identity()

    # def get_optimizer(self, lr, *args, **kwargs):
    def get_optimizer(self, opt_config):

        lr_a = opt_config.opt_actor.lr
        lr_c = opt_config.opt_critic.lr
        lr_alpha = opt_config.opt_other.lr

        opt_alpha = torch.optim.Adam([self.state_log_alpha, self.rgb_log_alpha], lr=lr_alpha, betas=(0.9, 0.999))

        critic_param = nn.ParameterList()
        critic_param.extend(self.q1_net.parameters())
        critic_param.extend(self.q2_net.parameters())

        r_actor_param = nn.ParameterList()
        s_actor_param = nn.ParameterList()

        r_actor_param.extend(self.rgb_a_net.parameters())
        r_actor_param.extend(self.img_encoder.parameters())

        s_actor_param.extend(self.state_a_net.parameters())

        if not self.sac_only:
            opt_a_r = torch.optim.AdamW(r_actor_param, lr=lr_a)
        else:
            opt_a_r = None

        opt_a_s = torch.optim.AdamW(s_actor_param, lr=lr_a)
        opt_c = torch.optim.AdamW(critic_param, lr=lr_c)

        self.opt_list = [opt_a_r, opt_a_s, opt_c, opt_alpha]

        return self.opt_list

    def forward(self, state, img, feat_rgb_input_mark=False, state_a_net_mark=False, state_at_net_mark=False):
        # [B, D]
        # [B, T, C, H, W]

        if self.sac_only:
            state_a_net_mark = True
            feat_rgb_input_mark = True

        state = state.to(self.device)

        if img is not None:
            img = img.to(self.device)

        if feat_rgb_input_mark:
            img_feat_map = img
        else:
            img_feat_map = self.img_encoder(img)
            # img_feat = self.frame_fusion(img)

        if state_a_net_mark:
            if state_at_net_mark:
                a_net = self.state_at_net
            else:
                a_net = self.state_a_net
        else:
            a_net = self.rgb_a_net

        output = a_net.forward(state, img_feat_map)

        if self.continuous_action:
            output = output.reshape(-1, self.act_dim, 2)

            mu, log_std = output[..., 0:1], output[..., 1:2]
            log_std = log_std.tanh()
            log_std = self.log_std_bound[0] + 0.5 * (self.log_std_bound[1] - self.log_std_bound[0]) * (log_std + 1)
            # log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            output = torch.cat([mu, std], dim=-1)
        else:
            output = output.softmax(dim=-1)

        return output, img_feat_map

    def act(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def critic(self, state, img, *args, **kwargs):
        pass

    def all_zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()

    def get_vis_dict(self, img0, feat_map, high_res=False):
        # [H, W, D]
        # [D, H, W]

        if high_res:
            feat_map_resize = tvtf.resize(feat_map, size=img0.shape[:2], interpolation=tvtf.InterpolationMode.BICUBIC)  # [H, W, D]
        else:
            feat_map_resize = feat_map

        feat_map_resize = feat_map_resize.permute(1, 2, 0)  # [H, W, D]
        feat_map_norm = feat_map_resize.norm(dim=-1)  # [H, W]

        if high_res:
            # feat_map_norm = feat_map_norm.log1p()
            # feat_map_norm = feat_map_norm / feat_map_norm.max()
            # feat_map_norm = tvtf.gaussian_blur(feat_map_norm.unsqueeze(0), kernel_size=[31, 31], sigma=[9.0, 9.0]).squeeze(0)
            feat_map_norm = tvtf.gaussian_blur(feat_map_norm.unsqueeze(0), kernel_size=[51, 51], sigma=[15.0, 15.0]).squeeze(0)

        feat_map_norm = feat_map_norm.detach().cpu().numpy()

        feat_map_norm_min_max = cv2.normalize(
            feat_map_norm,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)

        if not high_res:
            feat_map_norm_min_max = cv2.resize(feat_map_norm_min_max, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            img0 = cv2.resize(img0, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

        heatmap = cv2.applyColorMap(feat_map_norm_min_max, cv2.COLORMAP_JET)[..., ::-1]

        # img_heat_blend = np.where(feat_map_normed[..., None] > 0, img0 * 0.5 + heatmap * 0.5, img0).astype(np.uint8)
        img_heat_blend = (img0 * 0.5 + heatmap * 0.5).astype(np.uint8)

        vis_dict = {
            "heat_feat_map": heatmap,
            "img_heat_blend": img_heat_blend,
            # "ecm": img0,
        }

        if high_res:
            vis_dict["ecm"] = img0

        return vis_dict

    def get_action_(self, eps, action_n, obs, img, greedy=False, random_sample=False, rsample=False, tensor_output=False, feat_rgb_input_mark=False, *args, **kwargs):
        b = len(obs)

        # s1 = time.time()

        vis_mark = kwargs.get("vis")

        state_a_net_mark = kwargs.get("state_a_net_mark")
        train_rollout_mark = kwargs.get("train_rollout_mark")
        state_at_net_mark = kwargs.get("state_at_net_mark")
        high_res = kwargs.get("high_res")
        modal_corrupt = kwargs.get("modal_corrupt")

        if self.sac_only:
            state_a_net_mark = True
            feat_rgb_input_mark = True

        use_state_a_net = state_a_net_mark
        if not use_state_a_net and train_rollout_mark:
            if self.rollout_method == 2:
                use_state_a_net = True
            elif self.rollout_method == 1:
                use_state_a_net = False
            else:
                raise ValueError(f"Unknown rollout_method: [{self.rollout_method}]")

        vis_dict = {}
        vis_dict_ = {}
        sample_dict = {}

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        if random_sample:
            act = (torch.rand(size=(b, action_n), dtype=torch.float32) - 0.5) * 2
            log_p = torch.zeros(size=(b, 1), device=act.device) + 0.1
            raw_act = de_tanh_tn(act)
            dist = MyNormal(loc=torch.zeros_like(act, device=act.device), scale=torch.ones_like(act, device=act.device))
            mean_std = act.mean()
            entropy = 1.0
        else:

            img_mc = img
            if modal_corrupt is not None:
                if self.sac_only:
                    sigma = float(modal_corrupt.split("_")[-1])
                    obs[self.bot_state_dim:] += torch.randn(obs[self.bot_state_dim:].shape) * sigma
                else:
                    img_mc = img_modal_corrupt_func(img, modal_corrupt, self.img_size)

            if high_res:
                b, t, _, _, _ = img.shape
                img_resized = tvtf.resize(img_mc.flatten(0, 1), [self.img_size, self.img_size], interpolation=tvtf.InterpolationMode.BICUBIC)
                img_resized = img_resized.unflatten(0, (b, t))
            else:
                img_resized = img_mc

            act_p, img_feat_map = self.forward(obs, img_resized, feat_rgb_input_mark, use_state_a_net, state_at_net_mark)  # [B, A] (p) or [B, A, 2]  (mu, std)

            sample_dict["mu_pre"] = act_p[..., 0]
            sample_dict["std_pre"] = act_p[..., 1]

            if vis_mark and not feat_rgb_input_mark:
                img_feat_map0 = img_feat_map[0]  # [D, H, W]
                img0 = img_mc[0, -1, ...].permute(1, 2, 0).mul(255.0).byte().cpu().numpy()  # [H, W, C]

                vis_dict_i = self.get_vis_dict(img0, img_feat_map0, high_res=high_res)
                vis_dict_.update(vis_dict_i)

            if self.continuous_action:
                dist = CURLNormal(loc=act_p[..., 0], scale=act_p[..., 1])
                mean_std = act_p[..., 1].mean()
            else:
                dist = Categorical(act_p)
                std = act_p.std(dim=-1)
                mean_std = std.mean()

            if greedy:
                if self.continuous_action:
                    act = act_p[..., 0]
                else:
                    act = torch.argmax(act_p, dim=-1)
                log_p_base = dist.log_prob(act).sum(dim=-1, keepdim=True)
            else:
                if rsample:
                    act, eps_gaussian = dist.rsample()
                else:
                    act, eps_gaussian = dist.sample()
                log_p_base = dist.log_prob_eps(eps_gaussian).sum(dim=-1, keepdim=True)

            entropy = dist.entropy.mean().detach()

            if self.continuous_action:
                raw_act = act
                act = act.tanh()
            else:
                raw_act = act

            if self.continuous_action:
                log_p = log_p_base - (2 * (np.log(2) - raw_act - F.softplus(-2.0 * raw_act))).sum(dim=-1, keepdim=True)
            else:
                log_p = log_p_base

        if not tensor_output:
            log_p = log_p.detach().cpu().numpy()
            act = act.detach().cpu().numpy()
            raw_act = raw_act.detach().cpu().numpy()

        vis_dict.update(vis_dict_)

        sample_dict["act"] = act
        sample_dict["log_p"] = log_p
        sample_dict["raw_act"] = raw_act
        sample_dict["dist"] = dist
        sample_dict["std"] = mean_std
        sample_dict["entropy"] = entropy
        sample_dict["vis"] = vis_dict

        return sample_dict

    @property
    def state_alpha(self):
        return self.state_log_alpha.exp()

    @property
    def rgb_alpha(self):
        return self.rgb_log_alpha.exp()

    def update_critic_and_decoder(self, info_dict, next_rgb_feat, obs, next_obs, act, reward, terminated, mask_gt, depth_gt, *args, **kwargs):
        with torch.no_grad():
            next_sample_dict = self.get_action(1e-8, self.act_dim, next_obs, next_rgb_feat, tensor_output=True,
                                                                 rsample=True, feat_rgb_input_mark=True, state_a_net_mark=True)
            next_act_pre, next_log_prob_pre = next_sample_dict["act"], next_sample_dict["log_p"]

            next_qt1: torch.Tensor = self.qt1_net(next_obs, next_act_pre)
            next_qt2: torch.Tensor = self.qt2_net(next_obs, next_act_pre)

            next_qt1 = torch.clamp(next_qt1, min=self.q_target_bound[0], max=self.q_target_bound[1])
            next_qt2 = torch.clamp(next_qt2, min=self.q_target_bound[0], max=self.q_target_bound[1])

            next_min_qt = torch.min(next_qt1, next_qt2) - self.state_alpha * next_log_prob_pre

            next_reward_gt = reward + (1 - terminated) * self.gamma * next_min_qt

        q1 = self.q1_net(obs, act)
        q2 = self.q2_net(obs, act)

        q1_loss = F.mse_loss(q1, next_reward_gt)
        q2_loss = F.mse_loss(q2, next_reward_gt)
        q_loss = q1_loss + q2_loss

        self.opt_list[2].zero_grad()

        q_en_loss = q_loss
        q_en_loss.backward()

        q1_grad_norm = clip_grad_norm_(self.q1_net.parameters(), self.grad_clip)
        q2_grad_norm = clip_grad_norm_(self.q2_net.parameters(), self.grad_clip)

        q_grad = (q1_grad_norm + q2_grad_norm) / 2

        self.opt_list[2].step()
        self.opt_list[2].zero_grad()

        info_dict["q_l"] = q_loss.detach()
        info_dict["q_g"] = q_grad

        q12 = torch.cat([q1, q2], dim=-1)

        info_dict["q_u"] = q12.mean().detach()
        info_dict["q_min"] = q12.min().detach()
        info_dict["q_max"] = q12.max().detach()

    def skl_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = symmetric_kl_gaussian_fast(rgb_mu, rgb_std, state_t_mu, state_t_std).mean(-1, keepdim=True)
        return loss * 0.5

    def mse_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = (F.mse_loss(rgb_mu, state_t_mu, reduction="none") + F.mse_loss(rgb_std, state_t_std, reduction="none")).mean(-1, keepdim=True)
        return loss

    def hel_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = hellinger_gaussian(rgb_mu, rgb_std, state_t_mu, state_t_std).mean(-1, keepdim=True)
        return loss

    def fkl_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = forward_kl_gaussian(rgb_mu, rgb_std, state_t_mu, state_t_std).mean(-1, keepdim=True)
        return loss

    def rkl_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = forward_kl_gaussian(state_t_mu, state_t_std, rgb_mu, rgb_std).mean(-1, keepdim=True)
        return loss

    def bha_loss(self, rgb_mu, rgb_std, state_t_mu, state_t_std):
        loss = bhattacharyya_gaussian(state_t_mu, state_t_std, rgb_mu, rgb_std).mean(-1, keepdim=True)
        return loss

    def update_actor_and_alpha(self, info_dict, rgb_feat, obs, *args, **kwargs):
        ##################### state_a_net #####################
        state_sample_dict = self.get_action(1e-8, self.act_dim, obs, rgb_feat, tensor_output=True, rsample=True,
                                      feat_rgb_input_mark=True, state_a_net_mark=True)
        state_act_pre, state_log_prob_pre = state_sample_dict["act"], state_sample_dict["log_p"]
        state_entropy = state_sample_dict["entropy"]
        state_mu = state_sample_dict["mu_pre"]
        state_std = state_sample_dict["std_pre"]
        info_dict["sstd"] = state_std.mean().detach()

        q1_ = self.q1_net(obs, state_act_pre)
        q2_ = self.q2_net(obs, state_act_pre)
        min_s_q = torch.min(q1_, q2_)

        state_act_loss = ((self.state_alpha.detach() * state_log_prob_pre) - min_s_q).mean()
        state_alpha_loss = (self.state_alpha * (- state_log_prob_pre.detach() - self.target_entropy)).mean()
        info_dict["sa_l"] = state_act_loss.detach()
        ##################### state_a_net #####################

        ##################### state_at_net #####################
        if self.ema_state_a_net_mark:
            with torch.no_grad():
                state_t_sample_dict = self.get_action(1e-8, self.act_dim, obs, rgb_feat, tensor_output=True, rsample=True,
                                                    feat_rgb_input_mark=True, state_a_net_mark=True, state_at_net_mark=True)
                state_t_act_pre = state_sample_dict["act"]
                q1_ = self.q1_net(obs, state_t_act_pre)
                q2_ = self.q2_net(obs, state_t_act_pre)
                min_st_q = torch.min(q1_, q2_)
            state_t_mu = state_t_sample_dict["mu_pre"]
            state_t_std = state_t_sample_dict["std_pre"]
        else:
            state_t_mu = state_mu.detach()
            state_t_std = state_std.detach()
            min_st_q = min_s_q.detach()
        ##################### state_at_net #####################

        if not self.sac_only:
        ##################### rgb_a_net #####################
            rgb_sample_dict = self.get_action(1e-8, self.act_dim, obs, rgb_feat, tensor_output=True, rsample=True,
                                          feat_rgb_input_mark=True)
            rgb_act_pre, rgb_log_prob_pre = rgb_sample_dict["act"], rgb_sample_dict["log_p"]
            mean_entropy = rgb_sample_dict["entropy"]
            rgb_mu = rgb_sample_dict["mu_pre"]
            rgb_std = rgb_sample_dict["std_pre"]
            info_dict["rstd"] = rgb_std.mean().detach()

            q1_ = self.q1_net(obs, rgb_act_pre)
            q2_ = self.q2_net(obs, rgb_act_pre)
            min_r_q = torch.min(q1_, q2_)

            rgb_act_loss = (self.rgb_alpha.detach() * rgb_log_prob_pre) - min_r_q
            # info_dict["ra_l"] = rgb_act_loss.mean().detach()
            # dist_loss = F.mse_loss(rgb_mu, state_mu.detach()) + F.mse_loss(rgb_std, state_std.detach())
            # dist_loss = symmetric_kl_gaussian_fast(rgb_mu, rgb_std, state_t_mu.detach(), state_t_std.detach())
            # dist_loss = F.mse_loss(rgb_mu, state_t_mu.detach())
            dist_loss = self.dist_loss_func(rgb_mu, rgb_std, state_t_mu.detach(), state_t_std.detach()).mean(-1, keepdim=True)
            info_dict["ds_l"] = dist_loss.mean().detach()

            rgb_alpha_loss = (self.rgb_alpha * (- rgb_log_prob_pre.detach() - self.target_entropy)).mean()

            with torch.no_grad():
                mu_dist = F.mse_loss(rgb_mu, state_t_mu)
                info_dict["mud"] = mu_dist

            if self.distillation_option == 1:
                rgb_loss = dist_loss.mean()
                rgb_alpha_loss *= 0.0
            else:
                raise ValueError(f"Unknown distillation_option: [{self.distillation_option}]")
        else:
            rgb_loss = 0.0
            rgb_alpha_loss = 0.0

        if not self.sac_only:
            self.opt_list[0].zero_grad()
        self.opt_list[1].zero_grad()

        if self.train_sac:
            act_loss = state_act_loss + rgb_loss
        else:
            act_loss = rgb_loss

        act_loss.backward()

        state_a_grad_norm = clip_grad_norm_(self.state_a_net.parameters(), self.grad_clip)
        rgb_a_grad_norm = clip_grad_norm_(self.rgb_a_net.parameters(), self.grad_clip)
        bb_grad_norm = clip_grad_norm_(self.img_encoder.parameters(), self.grad_clip)

        info_dict["sa_g"] = state_a_grad_norm
        info_dict["ra_g"] = rgb_a_grad_norm
        info_dict["bb_g"] = bb_grad_norm

        if not self.sac_only:
            self.opt_list[0].step()
        self.opt_list[1].step()

        if not self.sac_only:
            self.opt_list[0].zero_grad()
        self.opt_list[1].zero_grad()


        #############

        self.opt_list[3].zero_grad()

        if self.train_sac:
            alpha_loss = state_alpha_loss + rgb_alpha_loss
        else:
            alpha_loss = rgb_alpha_loss

        alpha_loss.backward()

        # state_alpha_grad_norm = clip_grad_norm_([self.state_log_alpha], self.grad_clip)
        # if not self.sac_only:
        #     rgb_alpha_grad_norm = clip_grad_norm_([self.rgb_log_alpha], self.grad_clip)

        self.opt_list[3].step()
        self.opt_list[3].zero_grad()

        info_dict["salp"] = self.state_alpha.detach()
        # if not self.sac_only:
        #     info_dict["ralp"] = self.rgb_alpha.detach()

    def get_error(self, batch_dict, *args, **kwargs):
        # s = time.time()

        info_dict = {}

        update_mark = kwargs.get("update_mark")
        imitating = kwargs.get("imitating")

        non_blocking = False

        obs = batch_dict['obs'].to(self.device, non_blocking=non_blocking).float()
        next_obs = batch_dict['next_obs'].to(self.device, non_blocking=non_blocking).float()
        # old_prob_log = batch_dict['prob'].to(self.device)
        act = batch_dict['act'].to(self.device, non_blocking=non_blocking).float()  # [B, 1]
        reward = batch_dict['rwd'].to(self.device, non_blocking=non_blocking).float().unsqueeze(-1)  # [B, 1]
        terminated = batch_dict['trm'].to(self.device, non_blocking=non_blocking).float().unsqueeze(-1)  # [B, 1]

        if not self.sac_only:
            rgb: torch.Tensor = batch_dict['rgb'].to(self.device, non_blocking=non_blocking) / 255.0
            next_rgb = batch_dict['next_rgb'].to(self.device, non_blocking=non_blocking) / 255.0
            if self.gaussian_noise_aug:
                rgb += torch.randn(size=rgb.shape, dtype=rgb.dtype,
                                   device=rgb.device) * 0.02  # Dilute the artifact rendered by simulator
                rgb.clamp_(0, 1)
        else:
            rgb = next_rgb = None

        mask, depth = None, None

        b = len(obs)

        if self.sac_only:
            rgb_feat = next_rgb_feat = rgb
        else:
            rgb_feat = self.img_encoder(rgb)
            # with torch.no_grad():
            #     next_rgb_feat = self.img_encoder_t(next_rgb)
            next_rgb_feat = rgb_feat.detach()

        if self.distillation_loss == -1 or self.distillation_option != 1:
            self.update_critic_and_decoder(info_dict, next_rgb_feat, obs, next_obs, act, reward, terminated, mask, depth, *args, **kwargs)

        self.update_actor_and_alpha(info_dict, rgb_feat, obs, *args, **kwargs)

        return 0.0, info_dict

    def update(self, batch_dict, loss: torch.Tensor, *args, **kwargs):

        pass

    def save(self, path, *args, **kwargs):
        best_mark = kwargs.get("best_mark")

        torch.save(self.state_dict(), path)

        if self.sac_only and best_mark is not None:
            torch.save(self.state_a_net.state_dict(), self.sac_weights_path)

    def after_iter(self, iter_i, *args, **kwargs):

        self.soft_update(self.q1_net, self.qt1_net, self.tau)
        self.soft_update(self.q2_net, self.qt2_net, self.tau)
        self.soft_update(self.img_encoder, self.img_encoder_t, self.tau)
        if self.ema_state_a_net_mark and self.train_sac and not self.sac_only:
            self.soft_update(self.state_a_net, self.state_at_net, self.tau)

        return
