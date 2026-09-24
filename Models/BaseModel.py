import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.distributions.utils import _standard_normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import distributions as pyd


class BaseModel(nn.Module):
    frozen_params = nn.ParameterList()
    hot_params = nn.ParameterList()
    device: torch.device
    opt_list: list[torch.optim.Optimizer]
    continuous_action: bool

    def __init__(self):
        super(BaseModel, self).__init__()

    def get_optimizer(self, opt_config):
        if len(list(self.hot_params.parameters())):
            self.opt_list = [torch.optim.AdamW(self.hot_params.parameters(), lr=opt_config.opt_actor.lr)]
        else:
            self.opt_list = []
        return self.opt_list

    def update(self, batch_dict, loss, *args, **kwargs):
        pass

    def get_error(self, batch_dict, *args, **kwargs):
        pass

    def save(self, path, *args, **kwargs):
        pass

    def before_train(self, *args, **kwargs):
        pass

    def after_iter(self, iter_i, *args, **kwargs):
        pass

    def critic(self, *args, **kwargs):
        pass

    def get_action_aq(self, eps, action_n, obs, ecm_img=None, greedy=False, random_sample=False, rsample=False,
                      tensor_output=False):
        # [B, D]

        b = len(obs)

        if np.random.random() < eps and not greedy or random_sample:
            act = np.random.randint(action_n, size=(b,))
            p = np.zeros_like(act) + 1 / action_n

        else:
            obs_tensor = torch.from_numpy(obs).float()

            act_q = self.forward(obs_tensor)  # [B, A]

            if isinstance(act_q, torch.Tensor):
                act_q_np = act_q.detach().cpu().numpy()
            else:
                act_q_np = act_q
            act = np.argmax(act_q_np, axis=-1)

            p = act_q_np / act_q_np.sum(axis=-1, keepdims=True)
            p = p[:, act][:, 0]

        p = np.log(p + 1e-5)

        return act, p

    def get_action_ap(self, eps, action_n, obs, ecm_img=None, greedy=False, random_sample=False, rsample=False,
                      tensor_output=False):
        # [B, D]

        b = len(obs)

        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.from_numpy(obs).float()
        else:
            obs_tensor = obs

        if random_sample:
            p = torch.zeros([b, action_n], device=obs_tensor.device)
            if self.continuous_action:
                dist = Normal(loc=p, scale=p + 1)
                act = dist.sample()
                log_p_base = dist.log_prob(act)
                raw_act = act
                act = act.tanh()
                log_p = log_p_base - torch.log(1 - act.square() + 1e-7)
            else:
                act = np.random.randint(action_n, size=(b, action_n))
                p += 1 / action_n
                log_p = np.log(p + 1e-8)
                raw_act = act
            log_p = log_p.sum(dim=-1, keepdim=True)
        else:
            act_p = self.forward(obs_tensor)  # [B, A] (p) or [B, A, 2]  (mu, std)

            if not isinstance(act_p, torch.Tensor):
                act_q_tf = torch.from_numpy(act_p).to(self.device)
            else:
                act_q_tf = act_p

            # act_q_tf = act_q_tf.detach()

            if self.continuous_action:
                dist = Normal(loc=act_q_tf[..., 0], scale=act_q_tf[..., 1])
            else:
                dist = Categorical(act_q_tf)

            if greedy:
                if self.continuous_action:
                    act = act_q_tf[..., 0]
                else:
                    act = torch.argmax(act_q_tf, dim=-1)
            else:
                if rsample:
                    act = dist.rsample()
                else:
                    act = dist.sample()

            log_p_base = dist.log_prob(act).sum(dim=-1, keepdim=True)

            if self.continuous_action:
                raw_act = act
                act = act.tanh()
            else:
                raw_act = act

            if self.continuous_action:
                # log_p_c = log_p_base - torch.log(1 - act.square() + 1e-6).sum(dim=-1, keepdim=True)
                log_p = log_p_base - (2 * (np.log(2) - raw_act - F.softplus(-2.0 * raw_act))).sum(dim=-1, keepdim=True)
            else:
                log_p = log_p_base

            # log_p = log_p.cpu().numpy()
            #
            # act = act.cpu().numpy()

        if not tensor_output:
            log_p = log_p.detach().cpu().numpy()
            act = act.detach().cpu().numpy()
            raw_act = raw_act.detach().cpu().numpy()

        return act, log_p, raw_act

    def calc_grad_norm(self, m: nn.Module):
        grad_norm = 0
        for p in m.parameters():
            if p.grad is not None:
                grad_norm_i = p.grad.data.norm()
                grad_norm += grad_norm_i ** 2

        grad_norm = grad_norm ** 0.5
        return grad_norm

    def get_action(self, obs, action_n, eps, greedy=False):
        return self.get_action_aq(eps, action_n, obs, greedy=greedy)

    def copy(self, model: nn.Module):
        for param, target_param in zip(model.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, from_model: torch.nn.Module, to_model: torch.nn.Module, tau=0.05):
        # from_net_params = parameters_to_vector(from_model.parameters())
        # to_net_params = parameters_to_vector(to_model.parameters())
        # to_net_params.mul_(1 - tau).add_(from_net_params, alpha=tau)
        # vector_to_parameters(to_net_params, to_model.parameters())
        for param, target_param in zip(from_model.parameters(), to_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class CURLNormal:
    def __init__(self, loc: torch.Tensor = None, scale: torch.Tensor=None):
        self.loc = loc if loc is not None else torch.tensor([0.0])
        self.scale = scale if scale is not None else torch.tensor([1.0])
        self.scale_log = self.scale.log()

        self.const = math.log(math.sqrt(2 * math.pi))

        self.entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + self.scale_log).sum(-1)

    def sample(self, *args, **kwargs):
        return self.rsample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        eps = torch.randn_like(self.loc, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale, eps

    def log_prob_eps(self, eps: torch.Tensor):
        # compute the variance
        residual = (-0.5 * eps.square() - self.scale_log) - self.const
        # residual = residual.sum(-1, keepdim=True)
        return residual

    def log_prob(self, value):
        # compute the variance
        var = self.scale ** 2
        log_scale = (
            self.scale.log()
        )
        return (
                -((value - self.loc) ** 2) / (2 * var)
                - log_scale
                - self.const
        )






