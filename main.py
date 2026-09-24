import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import multiprocessing.connection
import datetime
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from constant import *

from torch.utils.tensorboard import SummaryWriter

# import Config
# from MemoryBuffer import MemoryBuffer
# from MemoryBuffer_mp import MemoryBuffer as MemoryBuffer_mp
from MemoryBuffer import MemoryBuffer as MemoryBuffer_mp
# from MemoryBuffer import MemoryBuffer as MemoryBuffer_mp
from Models.DDPG import DDPG
from Models.DQN import DQN
from Models.DrQ import DrQ
from Models.PPO import PPO
from Models.Q_Learning import Q_Learning
from Config import *
import tqdm
import imageio

from utils.env_utils import *
# from utils.env_utils import action_denormalizer, obs_normalizer, img_normalizer, my_reward, action_normalizer

from Models.SAC_cv import SAC_CV
from utils.Config import load_config, Config
from utils.model_selection import model_selection, game_selection
from utils.utils import cv2_imshow, remove_event_log_file

sys.path.append("SurRoL")

from copy import deepcopy

from Models.REINFORCE import REINFORCE
from Models.SAC import SAC
from Reward import (
    Reward,
    NeedleReachReward,
    NeedlePickReward, NeedleRegraspReward,
)
from utils.utils import seed_everything

import torch.multiprocessing as t_mp
import multiprocessing as mp

from constant import *

# os.environ["PYBULLET_EGL_DEVICE"] = "0"
# os.environ["PYBULLET_SUPPRESS_OUTPUT"] = "0"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

CONN_IMG_NUM = 32


def draft():
    # game_name_full = "NeedleReach"
    # game_name_full = "NeedlePick"
    # game_name_full = "NeedleRegrasp"
    game_name_full = "PegTransfer"
    # game_name_full = "Reacher-v5"

    surrol_mark = game_name_full not in ["Reacher-v5"]

    game_name = game_name_full.split("#")[0]
    mode_dict = {}
    if len(game_name_full.split("#")) > 1:
        game_mode = game_name_full.split("#")[1]

        mode_dict = {game_mode: True}

    # render_size = 84
    # render_size = 128
    # render_size = 224
    # render_size = 256
    render_size = 512

    env = make_env(game_name=game_name, render_mode="human", surrol_mark=surrol_mark, **mode_dict)
    # env = make_env(game_name=game_name, render_mode="rgb_array", surrol_mark=surrol_mark, **mode_dict)
    # env = make_env(game_name=game_name, render_mode="rgb_array", **mode_dict)
    # env = make_env(game_name=game_name, render_mode=None, **mode_dict)

    # Reset the environment to generate the first observation
    # observation, info = env.reset(seed=RANDOM_SEED)
    observation, obs_dict, img = reset_env(env, game_name=game_name, surrol_mark=surrol_mark, render=True)
    n = 1
    done = False
    # for _ in range(1000):
    cum_reward = 0
    while not done:
        # this is where you would insert your policy
        # action = env.action_space.sample()
        action = env.get_oracle_action(obs_dict)
        # action = 1

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        # observation, reward, img, terminated, truncated, info, obs_dict = step_env(env, action, n, 100000, render=False)
        observation, reward, img_dict, terminated, truncated, win, info, obs_dict = step_env(env, action, game_name=game_name, surrol_mark=surrol_mark, step_n=n, max_step=100, render=True, h=render_size, w=render_size)

        # print(len(env.p.getContactPoints(env.psm1.body, env.tray_pad_id)))

        # img = img_dict["rgb"]
        img = env.render().astype(np.uint8)

        seg_coilor = SEG_COLOR_NP[img_dict["mask"]]

        # depth = (np.log(img_dict["depth"] + 1e-10) * 255.0).astype(np.uint8)
        # depth = (np.power(img_dict["depth"], 150) * 255.0).astype(np.uint8)
        depth = img_dict["depth"]

        # img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        # img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        #
        # img = (img / (img.max(axis=(1, 2), keepdims=True) + 1e-6) * 255).astype(np.uint8)
        # # img = cv2.resize(img[:, :, ::-1], (224, 224), interpolation=cv2.INTER_NEAREST)
        # # img = cv2.resize(img[:, :, ::-1], (128, 128), interpolation=cv2.INTER_NEAREST)
        # img = cv2.resize(img[:, :, ::-1], (84, 84), interpolation=cv2.INTER_NEAREST)

        # blur = cv2.GaussianBlur(img, (9, 9), sigmaX=1.0, sigmaY=1.0)
        # sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        # big = cv2.resize(sharp, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        # big = cv2.resize(big, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        seg_coilor = cv2.resize(seg_coilor, (512, 512), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_NEAREST)

        # cv2.imshow("big", big[..., ::-1])
        cv2.imshow("img", img[..., ::-1])
        # cv2.imshow("seg_coilor", seg_coilor[..., ::-1])
        # cv2.imshow("depth", depth)
        cv2.waitKey()

        cum_reward += reward

        # If the episode has ended then we can reset to start a new episode
        n += 1

        done = terminated or truncated or win

        if done:
            observation, obs_dict, img_dict = reset_env(env, game_name=game_name, surrol_mark=surrol_mark)
            print(n, cum_reward)
            n = 0
            done = False

    env.close()


def mp_train(config, child_conn: multiprocessing.connection.Connection, model, conn_cache_dir, p_idx, random_seed_start, random_seed):

    seed_everything(random_seed)

    env = make_env(game_name=config.game_name, render_mode="human" if config.GUI_mark else "rgb_array", max_step=config.max_game_step, surrol_mark=config.surrol_mark,
                   **config.game_mode_dict)

    custom_reward_obj = config.custom_reward_class()

    env_reset_counter = p_idx + random_seed_start

    p_num = config.mp_num

    pid = os.getpid()

    game_done = True

    os.makedirs(f"{conn_cache_dir}/rgb", exist_ok=True)
    os.makedirs(f"{conn_cache_dir}/mask", exist_ok=True)
    os.makedirs(f"{conn_cache_dir}/depth", exist_ok=True)

    while True:
        recv_info = child_conn.recv()

        game_step_counter = 0

        iter_i, required_step_num, expert_step_mark, rollout_mark, break_mark = recv_info

        # if rollout_mark:
        rollout_for_bar = tqdm.tqdm(
            total=int(required_step_num),
            desc=f"rollout-{pid}",
            position=2 + p_idx,
            leave=False,
            disable=required_step_num < 50,
        )
        # else:
        #     rollout_for_bar = None

        mem_item_list = []
        mem_dict_list = []

        cum_reward_list = []

        while game_step_counter < required_step_num:

            if game_done:
                obs, obs_dict, obs_img_dict = reset_env(env, game_name=config.game_name, surrol_mark=config.surrol_mark,
                                                        seed=env_reset_counter, render=config.vision_model,
                                                        h=config.method.img_size, w=config.method.img_size)
                ecm_img = obs_img_dict["rgb"]
                ecm_mask = obs_img_dict["mask"]
                ecm_depth = obs_img_dict["depth"]

                env_reset_counter += p_num

                game_done = False

                cum_reward = 0.0

                one_game_step_counter = 0

                custom_reward_obj.reset()

                img_frame_k_array = [ecm_img] * config.method.frame_k

            obs_normed = obs_normalizer(obs.reshape(1, -1))
            img_frame_k_array.pop(0)
            img_frame_k_array.append(ecm_img)
            img_frame_k_array_normed = img_normalizer(img_frame_k_array)
            if expert_step_mark and config.game_teacher_mark:
                action = env.get_oracle_action(obs_dict)
                normed_action = action_normalizer(action, config.game_name)
                prob = 0.1
                normed_action = np.clip(normed_action, -1 + 1e-5, 1 - 1e-5)
                # raw_action = 0.5 * (np.log(1 + normed_action) - np.log(1 - normed_action))
            else:
                act_sample_dict = model.get_action(1e-6,
                                                   config.action_n,
                                                   obs_normed,
                                                   img_frame_k_array_normed,
                                                   random_sample=rollout_mark,
                                                   # state_a_net_mark=True,
                                                   train_rollout_mark=True,
                                                   )  # [1,]
                normed_action = act_sample_dict["act"]
                prob = act_sample_dict["log_p"]
                # raw_action = act_sample_dict["raw_act"]
                act_dist = act_sample_dict["dist"]
                normed_action = normed_action[0]
                prob = prob[0].item()
                # raw_action = raw_action[0]
            action = action_denormalizer(normed_action, config.game_name)

            # next_obs, reward, terminated, truncated, info = env.step(normed_action)
            next_obs, reward, next_obs_img_dict, terminated, truncated, win, info, next_obs_dict = step_env(env, action,
                                                                                                  step_n=one_game_step_counter,
                                                                                                  max_step=config.max_game_step,
                                                                                                  game_name=config.game_name,
                                                                                                  surrol_mark=config.surrol_mark,
                                                                                                  render=config.vision_model,
                                                                                                  h=config.method.img_size,
                                                                                                  w=config.method.img_size
                                                                                                  )
            next_ecm_img = next_obs_img_dict["rgb"]
            next_ecm_mask = next_obs_img_dict["mask"]
            next_ecm_depth = next_obs_img_dict["depth"]

            game_done = terminated or truncated or win
            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array([next_obs])
            next_obs_normed = obs_normalizer(next_obs.reshape(1, -1))

            reward = my_reward(env, custom_reward_obj, obs, next_obs, reward, action, one_game_step_counter,
                               terminated, truncated)
            cum_reward += reward

            if game_done and not expert_step_mark:
                cum_reward_list.append(cum_reward)

            if config.method.on_policy:
                state_value = model.critic(obs_normed, img_frame_k_array_normed).squeeze().item()
            else:
                state_value = 0.0

            # memory_item = (obs_normed[0], raw_action, reward, cum_reward, next_obs_normed[0], game_done, prob,
            #                    state_value)
            # memory_item = (obs_normed[0], raw_action, reward, next_obs_normed[0], terminated, truncated)
            memory_item = (obs_normed[0], normed_action, reward, next_obs_normed[0], terminated, truncated)
            memory_dict = {
                "cum_rwd": cum_reward,
                "prob": prob,
                "value": state_value,
            }
            mem_item_list.append(memory_item)
            mem_dict_list.append(memory_dict)

            if required_step_num <= CONN_IMG_NUM:
                mem_dict_list[-1]["rgb"] = ecm_img
                mem_dict_list[-1]["next_rgb"] = next_ecm_img
                mem_dict_list[-1]["mask"] = ecm_mask
                mem_dict_list[-1]["depth"] = ecm_depth

            if config.vision_model:
                # Image.fromarray(ecm_img).save(f"{conn_cache_dir}/{game_step_counter:08d}.jpg")
                if required_step_num > CONN_IMG_NUM:
                    cv2.imwrite(f"{conn_cache_dir}/rgb/{game_step_counter:08d}.jpg",
                                cv2.cvtColor(ecm_img, cv2.COLOR_RGB2BGR))
                    if ecm_mask is not None:
                        np.save(f"{conn_cache_dir}/mask/{game_step_counter:08d}.npy",
                                    ecm_mask)
                    if ecm_depth is not None:
                        np.save(f"{conn_cache_dir}/depth/{game_step_counter:08d}.npy",
                                    ecm_depth)

            obs = next_obs
            obs_dict = next_obs_dict
            ecm_img = next_ecm_img
            ecm_mask = next_ecm_mask
            ecm_depth = next_ecm_depth

            one_game_step_counter += 1
            game_step_counter += 1
            # if rollout_mark:
            rollout_for_bar.update(1)

        if game_step_counter >= required_step_num:
            # if rollout_mark:
            rollout_for_bar.close()

            if config.vision_model and required_step_num:
                # Image.fromarray(ecm_img).save(f"{conn_cache_dir}/{game_step_counter:08d}.jpg")
                if required_step_num > CONN_IMG_NUM:
                    cv2.imwrite(f"{conn_cache_dir}/rgb/{game_step_counter:08d}.jpg",
                                cv2.cvtColor(ecm_img, cv2.COLOR_RGB2BGR))
                    if ecm_mask is not None:
                        np.save(f"{conn_cache_dir}/mask/{game_step_counter:08d}.npy",
                                    ecm_mask)
                    if ecm_depth is not None:
                        np.save(f"{conn_cache_dir}/depth/{game_step_counter:08d}.npy",
                                    ecm_depth)

            send_info = (game_step_counter, mem_item_list, mem_dict_list, cum_reward_list)

            child_conn.send(send_info)

            cum_reward_list = []

        # process = psutil.Process(pid)

        # memory_info = process.memory_info()

        # rollout_for_bar.write(f"rollout-{pid}: {memory_info.rss:.06f}B")

        if break_mark:
            break

    env.close()
    child_conn.close()


class Project:
    def __init__(self, config):

        # self.run_check = True
        self.run_check = False

        # self.eval_debug = True
        self.eval_debug = False

        # self.one_sample_check = True
        self.one_sample_check = False

        # self.load_mid_ckp = True
        self.load_mid_ckp = False

        self.config = config
        self.rwd_bias = 0

        self.on_policy = self.config.method.on_policy
        self.on_policy_str = "on_policy" if self.config.method.on_policy else "off_policy"

        self.config_unique_path = self.config.config_file_path[:-5]
        self.config.unique_path = self.config_unique_path

        self.model_load_dir = f"{self.config.dirs.model_load_dir}/{self.config.unique_path}"
        self.model_save_dir = f"{self.config.dirs.model_save_dir}/{self.config.unique_path}"
        self.memory_dir = f"{self.config.dirs.memory_dir}/{self.config.unique_path}"
        self.vis_dir = f"{self.config.dirs.vis_dir}/{self.config.unique_path}"
        self.log_dir = f"{self.config.dirs.log_dir}/{self.config.unique_path}"
        # self.rollout_memory_dir = f"{self.config.dirs.rollout_memory_dir}/{self.config.game.name}/{self.on_policy_str}"
        self.rollout_memory_dir = f"{self.config.dirs.rollout_memory_dir}/{self.config.unique_path}"
        self.conn_cache_dir = f"{self.config.dirs.conn_cache_dir}/{self.config.unique_path}"

        self.saved_models_ckp_dir = f"{self.model_save_dir}/ckp"
        os.makedirs(self.saved_models_ckp_dir, exist_ok=True)

        self.ckp_path = f"{self.saved_models_ckp_dir}/ckp"

        os.makedirs(self.model_load_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.rollout_memory_dir, exist_ok=True)
        os.makedirs(self.conn_cache_dir, exist_ok=True)

        self.method_name = self.config.method.name

        self.bc_mark = False
        self.config.train.bc_mark = False
        if self.method_name in ["BC", "ALOHA"]:
            self.bc_mark = True
            self.config.train.bc_mark = True

        game_name_full = self.config.game.name

        self.surrol_mark = game_name_full not in NON_SURROL_GAME
        self.config.surrol_mark = self.surrol_mark

        game_name_mode = game_name_full.split("#")

        self.game_name = game_name_mode[0]
        self.game_mode_dict = {}
        if len(game_name_mode) > 1:
            game_mode = game_name_mode[1]
            self.game_mode_dict = {game_mode: True}

        env = make_env(game_name=self.game_name, surrol_mark=self.surrol_mark, **self.game_mode_dict)

        self.print_title_mark = False

        self.eval_num = 10
        self.test_num = 100
        self.config.eval_num = self.eval_num
        self.config.test_num = self.test_num

        self.mid_dim = self.config.method.mid_dim
        self.img_size = self.config.method.img_size

        self.img_c_dim = 3

        self.rwd_bias, self.rwd_scale, self.max_game_step, self.extra_obs, self.custom_reward_class, self.game_teacher_mark, self.bot_state_dim, self.mask_class_num = game_selection(self.game_name)

        self.gpu_mark = torch.cuda.is_available() and self.config.gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.config.gpu else "cpu")

        self.epsilon = self.config.method.args.initial_epsilon
        self.epsilon_decay_rate = self.config.method.args.epsilon_decay_rate
        self.final_epsilon = self.config.method.args.final_epsilon

        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            self.action_n = env.action_space.n
            self.act_dim = 1
            self.continuous_act = False
            assert "DDPG" not in self.method_name
        else:
            self.action_n = env.action_space.shape[0]
            self.act_dim = self.action_n
            self.continuous_act = True

        if self.surrol_mark:
            self.obs_dim = env.observation_space.spaces["observation"].shape[0] + self.extra_obs
        else:
            self.obs_dim = env.observation_space.shape[0] + self.extra_obs

        self.obs_dim = self.obs_dim
        self.continuous_obs = True
        self.config.continuous_obs = self.continuous_obs
        self.config.continuous_act = self.continuous_act

        self.model, self.mem_extra_keys = model_selection(self.method_name, self.obs_dim, self.act_dim, self.mid_dim, self.config, img_size=self.img_size, img_c_dim=3, bot_dim=self.bot_state_dim, mask_class_num=self.mask_class_num)

        if self.load_mid_ckp:
            # load_path = f"{self.saved_models_ckp_dir}/ckp"
            load_path = f"{self.model_save_dir}/best"
            self.model.load_state_dict(torch.load(load_path, weights_only=False))

        if self.gpu_mark:
            self.model_cpu = deepcopy(self.model).to(torch.device("cpu"))
            self.model_cpu.device = torch.device("cpu")
        else:
            self.model_cpu = self.model

        self.model_cpu.share_memory()

        self.model.device = self.device
        self.model = self.model.to(self.device)
        # self.model.continuous_act = self.continuous_act

        self.model.requires_grad_(False)
        self.model.eval()
        self.model_cpu.requires_grad_(False)
        self.model_cpu.eval()

        self.model_update_k_epoch = self.config.train.model_update_k_epoch
        self.model_update_k_iter = self.config.train.model_update_k_iter

        if self.on_policy:
            self.train_thresh = self.config.train.memory_push_per_rollout
        else:
            self.train_thresh = self.config.train.start_rollout_size

        self.config.game_name = self.game_name
        self.config.max_game_step = self.max_game_step
        self.config.game_mode_dict = self.game_mode_dict
        self.config.custom_reward_class = self.custom_reward_class
        self.config.action_n = self.action_n
        self.config.rwd_bias = self.rwd_bias
        self.config.rwd_scale = self.rwd_scale
        self.config.img_size = self.img_size
        self.config.img_c_dim = self.img_c_dim
        self.config.game_teacher_mark = self.game_teacher_mark

        if self.run_check:
            self.train_thresh = 1

            if self.eval_debug:
                self.config.train.n_iter = 0
                self.config.train.log_per_iter = 1
            else:
                self.config.train.n_iter = 10
                self.config.train.log_per_iter = 10
            self.config.train.model_update_k_epoch = 1
            self.config.eval_num = 1
            self.config.train.model_update_k_iter = 1
            self.config.train.memory_push_per_rollout = 1
            self.config.train.memory_buffer_size = 50
            self.config.test_num = 1
            self.test_num = 1

        elif self.one_sample_check:
            self.train_thresh = 1
            self.config.train.n_iter = 100000000000000
            self.config.train.log_per_iter = 1
            self.config.train.model_update_k_epoch = 1
            self.config.eval_num = 1
            self.config.train.model_update_k_iter = 1
            self.config.train.memory_push_per_rollout = 1
            self.config.train.memory_buffer_size = 50

        env.close()

    def epsilon_decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay_rate)

    def model_update(self, memory_obj: MemoryBuffer_mp, iter_i=0):
        if self.config.method.on_policy:
            memory_obj.calculate_gae()

            iter_num = -1
        else:
            iter_num = self.config.train.model_update_k_iter

        error_avg = 0.0

        if self.config.train.model_update_k_epoch == 1:
            epoch_bar = range(self.config.train.model_update_k_epoch)
        else:
            epoch_bar = tqdm.tqdm(
                range(self.config.train.model_update_k_epoch),
                leave=False,
                position=1,
                desc="Model update (epoch)",
            )

        error_counter = 0

        info_dict = {}
        for _ in epoch_bar:
            batch_index_list = memory_obj.get_iter_index_set(self.config.train.batch_size, iter_num=iter_num, mp_pre_load_mark=iter_i==0)

            iter_num = self.config.train.model_update_k_iter if not self.config.method.on_policy else len(batch_index_list)

            if self.config.train.model_update_k_iter == 1:
                iter_bar = range(self.config.train.model_update_k_iter)
            else:
                iter_bar = tqdm.tqdm(
                    range(iter_num),
                    leave=False,
                    position=1,
                    desc="Model update (iter)",
                    disable= iter_num < 500,
                )

            for i in iter_bar:
                batch_index = batch_index_list[i]
                index_list, batch_dict = memory_obj.slice(batch_index)

                self.model.train()
                self.model.requires_grad_(True)
                self.model.frozen_params.eval()
                self.model.frozen_params.requires_grad_(False)

                # error, info_dict_i = self.model.get_error(batch_dict, update_mark=True)
                if self.eval_debug:
                    error, info_dict_i = 0.0, {}
                else:
                    error, info_dict_i = self.model.get_error(batch_dict)

                self.model.eval()
                self.model.requires_grad_(False)

                self.model.after_iter(iter_i)
                self.epsilon_decay()

                error_avg += error
                error_counter += 1

                for info_k in info_dict_i:
                    if info_dict.get(info_k) is not None:
                        info_dict[info_k] += info_dict_i[info_k]
                    else:
                        info_dict[info_k] = info_dict_i[info_k]

                pass

            # iter_bar.close()

        error_avg /= self.model_update_k_epoch
        error_avg /= error_counter
        for info_k in info_dict:
            info_dict[info_k] /= error_counter

        return error_avg, info_dict, error_counter

    def model_update_fun(self, rl_memory_obj, iter_i):
        memory_obj = rl_memory_obj

        error_i, info_dict_i, update_counter = self.model_update(memory_obj, iter_i=iter_i)

        if self.config.method.on_policy:
            memory_obj.clear()

        return error_i, info_dict_i, update_counter

    def vis_env(self, env, vis_dir, random_seed=None, vis_counter=0, high_res=False, modal_corrupt=None, save_gif=True):
        cum_rwd, vis_dict, step_n, win = self.play_one_round(env, greedy=True, vis=save_gif, random_seed=random_seed, high_res=high_res, modal_corrupt=modal_corrupt)

        # env.close()

        if save_gif:
            n_frames = len(vis_dict["rgb"])
            fps = 25
            raw_duration = n_frames / fps
            duration = min(5.0, raw_duration)
            ratio = raw_duration / duration
            if ratio > 1.0:
                step = int(np.ceil(ratio).item())
            else:
                step = 1

            for k in vis_dict:
                vis_list = vis_dict[k]
                vis_dir_k = f"{vis_dir}/{k}"
                os.makedirs(vis_dir_k, exist_ok=True)

                vis_path = f"{vis_dir_k}/{vis_counter:06d}.gif"

                vis_list = vis_list[::step]

                imageio.mimsave(vis_path, vis_list, duration=duration, loop=0)

        return cum_rwd, vis_dict, step_n, win

    def play_one_round(self, env, greedy=False, vis=False, random_seed=None, high_res=False, modal_corrupt=None):
        cum_reward = 0.0

        render_mark = vis or self.config.vision_model

        # vis_list = []
        vis_dict = {"rgb": [], "env": []}

        if high_res:
            img_size = 1024
        else:
            img_size = self.img_size

        obs, obs_dict, obs_img_dict = reset_env(env, game_name=self.game_name, surrol_mark=self.surrol_mark, seed=random_seed, render=render_mark, h=img_size, w=img_size)

        ecm_img = obs_img_dict["rgb"]

        # if vis:
        #     img = env.render()
        #     # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        #     vis_list.append(img)

        done = False

        step_n = 0

        win = False

        custom_reward_obj = self.custom_reward_class()

        img_frame_k_array = [ecm_img] * self.config.method.frame_k

        vis_count = 5

        while not done and vis_count > 0:
            obs_normed = obs_normalizer(obs.reshape(1, -1))
            img_frame_k_array.pop(0)
            img_frame_k_array.append(ecm_img)
            img_frame_k_array_normed = img_normalizer(img_frame_k_array)
            act_sample_dict = self.model.get_action(self.epsilon, self.action_n, obs_normed, img_frame_k_array_normed,
                                                    greedy=greedy, vis=self.config.vision_model, high_res=high_res, modal_corrupt=modal_corrupt,
                                                    # state_a_net_mark=True,
                                                    )  # [1,]
            action, prob = act_sample_dict["act"], act_sample_dict["log_p"]

            action = action[0]
            prob = prob[0].item()
            normed_action = action_denormalizer(action, self.game_name)

            if done:
                normed_action[:-1] *= 0
                vis_count -= 1

            # next_obs, reward, terminated, truncated, info = env.step(normed_action)
            next_obs, reward, next_obs_img_dict, terminated, truncated, win, info, next_obs_dict = step_env(env, normed_action, game_name=self.game_name, surrol_mark=self.surrol_mark, step_n=step_n,
                                                                                             max_step=self.max_game_step,
                                                                                             render=render_mark, h=img_size, w=img_size)
            # debug_action = np.load("debug_action.npy")
            # debug_img = np.load("debug_img.npy")
            #
            # print(np.max(debug_img - img_frame_k_array_normed), np.min(debug_img - img_frame_k_array_normed))
            # print(np.max(debug_action - normed_action), np.min(debug_action - normed_action))
            #
            # np.save("debug_action", normed_action)
            # np.save("debug_img", img_frame_k_array_normed)

            next_ecm_img = next_obs_img_dict["rgb"]

            done = terminated or truncated or win
            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array([next_obs])
            next_obs_normed = obs_normalizer(next_obs.reshape(1, -1))

            # if terminated:  # and reward > self.game_over_reward:
            #     win = True

            reward = my_reward(env, custom_reward_obj, obs, next_obs, reward, action, step_n,
                               terminated, truncated)

            reward = (reward + self.rwd_bias) * self.rwd_scale

            cum_reward += reward

            obs = next_obs
            ecm_img = next_ecm_img

            if vis:
                img = env.render().astype(np.uint8)
                # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                h, w, _ = img.shape
                cv2.putText(img, f"{cum_reward:.3f}", (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)
                cv2.putText(img, f"{step_n / self.max_game_step * 100:.02f}%", (w // 2, h // 2 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)
                vis_dict["env"].append(img)
                vis_dict["rgb"].append(ecm_img)

                if act_sample_dict.get("vis") is not None:
                    for k in act_sample_dict["vis"]:
                        if vis_dict.get(k) is None:
                            vis_dict[k] = [act_sample_dict["vis"][k]]
                        else:
                            vis_dict[k].append(act_sample_dict["vis"][k])

            step_n += 1

        return cum_reward, vis_dict, step_n, win

    def train(self):
        shutil.rmtree(self.config.dirs.rollout_memory_dir)

        for_bar = tqdm.tqdm(
            total=self.config.train.n_iter,
            desc="Training",
            position=0,
            leave=False,
        )

        rollout_memory_dir = f"{self.rollout_memory_dir}"
        os.makedirs(rollout_memory_dir, exist_ok=True)

        remove_event_log_file(f"{self.log_dir}")

        log_txt_path = f"{self.log_dir}/log.txt"

        writer = SummaryWriter(log_dir=f"{self.log_dir}")
        log_txt_file = open(log_txt_path, "w")
        log_txt_file.close()

        # memory_buffer_class = MemoryBuffer if not self.config.vision_model else MemoryBuffer_mp
        memory_buffer_class = MemoryBuffer_mp
        # memory_buffer_class = MemoryBuffer_zarr

        # rgb_in_disk = not self.on_policy and self.img_size > 84
        # rgb_in_disk = not self.on_policy or self.bc_mark
        # rgb_in_disk = not self.on_policy and not self.bc_mark
        rgb_in_disk = not self.on_policy

        rl_memory_obj = memory_buffer_class(
            memory_size=self.config.train.memory_buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            continuous_act=self.continuous_act,
            continuous_obs=self.continuous_obs,
            rwd_u=self.rwd_bias,
            rwd_v=self.rwd_scale,
            root_dir=rollout_memory_dir,
            memory_extra_keys=self.mem_extra_keys,
            # memory_extra_keys=["rgb"] if self.config.vision_model else None,
            num_workers=self.config.num_workers,
            frame_k=self.config.method.frame_k,
            img_size=self.img_size,
            rgb_in_disk=rgb_in_disk,
            img_c_dim=self.img_c_dim,
            data_aug=self.config.method.data_aug,
            rgb_pos=self.config.method.rgb_pos,
            modal_corrupt=self.config.modal_corrupt,
        )

        model_update_counter = 0

        error_avg = 0.0
        best_reward = -np.inf

        info_dict = {}

        iter_i = 0

        rollout_counter = self.train_thresh
        if self.config.train.load_rollout_memory:
            rl_memory_obj.load()
        rollout_counter -= rl_memory_obj.size()

        conn_list = []
        child_p_list = []
        conn_cache_dir_list = []
        random_seed_start = self.config.random_seed + self.config.eval_num + self.config.test_num
        for p_idx in range(self.config.mp_num):
            conn_cache_dir_i = f"{self.conn_cache_dir}/{p_idx}"
            os.makedirs(conn_cache_dir_i, exist_ok=True)
            main_conn, child_conn = t_mp.Pipe(duplex=True)
            child_p = mp.Process(target=mp_train, args=(self.config, child_conn, self.model_cpu, conn_cache_dir_i, p_idx, random_seed_start, self.config.random_seed))
            conn_list.append((main_conn, child_conn))
            child_p_list.append(child_p)
            conn_cache_dir_list.append(conn_cache_dir_i)

        for child_p in child_p_list:
            child_p.start()

        eval_counter = 0

        vis_counter = 0

        # env = make_env(game_name=self.game_name, surrol_mark=self.surrol_mark, render_mode="human" if self.config.GUI_mark else "rgb_array", max_step=self.max_game_step,
        #                **self.game_mode_dict)
        env = make_env(game_name=self.game_name, surrol_mark=self.surrol_mark, render_mode="rgb_array", max_step=self.max_game_step,
                       **self.game_mode_dict)

        env_eval = env

        train_cum_reward_list = []

        while True:

            if rollout_counter <= 0:
                required_step_num = self.config.train.memory_push_per_rollout
                expert_step_mark = False
                rollout_mark = False
                if self.config.train.bc_mark:
                    required_step_num = 0
            else:
                required_step_num = rollout_counter
                expert_step_mark = self.config.train.start_expert_rollout
                rollout_mark = True
                if self.config.train.bc_mark:
                    rollout_mark = True
                    expert_step_mark = True

            break_mark = iter_i >= self.config.train.n_iter
            required_step_num //= self.config.mp_num

            if not self.bc_mark or rollout_counter > 0:
                send_info = (iter_i, required_step_num, expert_step_mark, rollout_mark, break_mark)
                # self.model_cpu.copy(self.model)
                self.model_cpu.load_state_dict(self.model.state_dict())
                for main_conn, child_conn in conn_list:
                    main_conn.send(send_info)

            if rollout_counter <= 0 and not break_mark:
                error_i, _error_info_i, update_counter = self.model_update_fun(rl_memory_obj, model_update_counter)
                model_update_counter += 1
                # error_avg += error_i
                for info_k in _error_info_i:
                    if info_dict.get(info_k) is None:
                        info_dict[info_k] = _error_info_i[info_k]
                    else:
                        info_dict[info_k] += _error_info_i[info_k]

                # log_str_ = (
                #     f"Training: q_u = {_error_info_i['q_u']:.04f} | " +
                #     (f"alpha = {_error_info_i['alpha']:.04f} |" if _error_info_i.get("alpha") else "") +
                #     f"q_grad = {_error_info_i['q_grad']:.04f} |" +
                #     (f"log_pre = {_error_info_i['log_pre']:.04f} |" if _error_info_i.get("log_pre") else "")
                # )
                #
                # for_bar.set_description(log_str_)

            if rollout_counter <= 0:
                if self.config.method.on_policy:
                    iter_num = self.config.train.memory_push_per_rollout
                else:
                    iter_num = self.config.train.model_update_k_epoch * self.config.train.model_update_k_iter
                iter_i += iter_num
                eval_counter += iter_num
                for_bar.update(iter_num)

                if iter_i % (10000 // iter_num * iter_num) == 0:
                    self.model.save(self.ckp_path)
                    rl_memory_obj.save()

            eval_mark = False
            if eval_counter >= self.config.train.log_per_iter:
                eval_mark = True
                eval_counter -= self.config.train.log_per_iter
            eval_mark |= break_mark

            # if iter_i % self.demo.yaml.log_per_iter == 0 and collection_expert_iter <= 0 and rollout_counter <= 0:
            if eval_mark:

                if len(train_cum_reward_list):
                    train_reward_avg = np.mean(train_cum_reward_list)
                else:
                    train_reward_avg = 0.0
                train_cum_reward_list = []

                reward_avg = 0.0
                success_rate = 0

                eval_bar = tqdm.tqdm(
                    range(self.config.eval_num),
                    leave=False,
                    position=1,
                    desc="Evaluation",
                )

                for eval_i in eval_bar:
                    cum_reward, vis_list, step_n, win = self.play_one_round(env_eval, greedy=True, vis=False,
                                                                            random_seed=self.config.random_seed + eval_i + self.config.test_num)
                    reward_avg += cum_reward / self.config.eval_num
                    success_rate += win
                eval_bar.close()
                success_rate = success_rate / self.config.eval_num

                # self.model.save(self.ckp_path)

                if model_update_counter:
                    # error_avg /= model_update_counter
                    # if isinstance(error_avg, torch.Tensor):
                    #     error_avg = error_avg.item()
                    for info_k in info_dict:

                        info_dict[info_k] /= model_update_counter
                        if isinstance(info_dict[info_k], torch.Tensor):
                            info_dict[info_k] = info_dict[info_k].item()

                # saved_models_ckp_dir = f"{self.model_save_dir}/ckp"
                # os.makedirs(saved_models_ckp_dir, exist_ok=True)
                #
                # ckp_path = f"{saved_models_ckp_dir}/ckp"
                # self.model.save(self.ckp_path)

                save_mark = False
                if best_reward < reward_avg:
                    ckp_path = f"{self.model_save_dir}/best"
                    self.model.save(ckp_path, best_mark=True)
                    best_reward = reward_avg

                    self.model.save(f"{self.ckp_path}_{iter_i}")

                    # ckp_path = f"{saved_models_ckp_dir}/{iter_i}_ckp_{best_reward:.03f}"
                    # self.model.save(ckp_path)
                    rl_memory_obj.save()

                    save_mark = True

                if not self.print_title_mark:
                    log_string = ""
                    log_string += f"{'Iter':^7}" + " | "
                    # log_string += f"{'Err':^7}" + " | "
                    log_string += f"{'ValR':^7}" + " | "
                    log_string += f"{'TraR':^7}" + " | "
                    log_string += f"{'Win':^4}" + " | "
                    for info_k in info_dict:
                        log_string += f"{info_k:^7}" + " | "

                    for_bar.write(log_string)
                    with open(log_txt_path, "a") as log_txt_file:
                        log_txt_file.write(log_string + "\n")

                    self.print_title_mark = True

                writer.add_scalars("Reward", {"Eval": reward_avg, "Train": train_reward_avg}, iter_i)
                # writer.add_scalar("Eval Reward", reward_avg, iter_i)
                # writer.add_scalar("Train Reward", train_reward_avg, iter_i)
                # writer.add_scalar("Total Error", error_avg, iter_i)
                writer.add_scalar("Success Rate", success_rate, iter_i)
                for info_k in info_dict:
                    writer.add_scalar(info_k, info_dict[info_k], iter_i)

                log_string = ""
                log_string += f"{str(iter_i) + ('*' if save_mark else ''):^7s}" + " | "
                # log_string += f"{error_avg:^7.04f}" + " | "
                log_string += f"{reward_avg:^7.03f}" + " | "
                log_string += f"{train_reward_avg:^7.03f}" + " | "
                log_string += f"{success_rate:^4.02f}" + " | "
                for info_k in info_dict:
                    log_string += f"{info_dict[info_k]:^7.03f}" + " | "
                for_bar.write(log_string)
                with open(log_txt_path, "a") as log_txt_file:
                    log_txt_file.write(log_string + "\n")

                # log_string = (f"min_obs = {rl_memory_obj.min_obs.round(2)} |" +
                #               f"max_obs = {rl_memory_obj.max_obs.round(2)} |")
                # for_bar.write(log_string)

                error_avg = 0
                # info_dict = {}
                info_dict.clear()
                model_update_counter = 0

                vis_ckp_dir = f"{self.vis_dir}/ckp"
                os.makedirs(vis_ckp_dir, exist_ok=True)

                vis_counter += 1

                self.vis_env(env_eval, vis_ckp_dir, random_seed=self.config.random_seed, vis_counter=vis_counter)

            child_game_steps_sum = 0
            if not self.bc_mark or rollout_counter > 0:
            # if not self.bc_mark:
                for p_idx in range(self.config.mp_num):
                    main_conn, child_conn = conn_list[p_idx]
                    conn_cache_dir = conn_cache_dir_list[p_idx]
                    recv_info = main_conn.recv()
                    child_game_steps, mem_item_list, mem_dict_list, cum_rwd_list_i = recv_info
                    child_game_steps_sum += child_game_steps

                    train_cum_reward_list += cum_rwd_list_i

                    for mem_idx in range(child_game_steps):
                        memory_item = mem_item_list[mem_idx]
                        memory_dict = mem_dict_list[mem_idx]

                        if self.config.vision_model:
                            # rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                            # next_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                            # memory_dict["rgb"] = rgb
                            # memory_dict["next_rgb"] = next_rgb
                            if required_step_num > CONN_IMG_NUM:
                                rgb = cv2.imread(f"{conn_cache_dir}/rgb/{mem_idx:08d}.jpg")
                                next_rgb = cv2.imread(f"{conn_cache_dir}/rgb/{mem_idx + 1:08d}.jpg")
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                                next_rgb = cv2.cvtColor(next_rgb, cv2.COLOR_BGR2RGB)
                                memory_dict["rgb"] = rgb
                                memory_dict["next_rgb"] = next_rgb

                                mask = np.load(f"{conn_cache_dir}/mask/{mem_idx:08d}.npy")
                                depth = np.load(f"{conn_cache_dir}/depth/{mem_idx:08d}.npy")
                                # mask = cv2.imread(f"{conn_cache_dir}/mask/{mem_idx:08d}.jpg", 0)
                                # depth = cv2.imread(f"{conn_cache_dir}/depth/{mem_idx:08d}.jpg", 0)
                                memory_dict["mask"] = mask
                                memory_dict["depth"] = depth

                        rl_memory_obj.push(*memory_item, **memory_dict)

                        # del memory_item, memory_dict

            if rollout_counter > 0:
                rollout_counter -= child_game_steps_sum
                if rollout_counter <= 0:
                    rl_memory_obj.save()

            if break_mark:
                break

        for child_p in child_p_list:
            child_p.join()
        # child_p.join()
        writer.close()
        log_txt_file.close()
        env_eval.close()

        for mc, cc in conn_list:

            mc.close()
            cc.close()

        rl_memory_obj.stop_mp()

        save_path = f"{self.model_save_dir}/final"
        self.model.save(save_path)

    def test(self, modal_corrupt=None, save_gif=True):
        # load_path = f"{self.saved_models_ckp_dir}/ckp"
        load_path = f"{self.model_save_dir}/best"
        self.model.load_state_dict(torch.load(load_path, weights_only=False))

        # env_vis = make_env(game_name=obj.game_name, render_mode="human", max_step=obj.max_game_step, **obj.game_mode_dict)
        # env_vis = make_env(game_name=self.game_name, surrol_mark=self.surrol_mark, render_mode=None, max_step=self.max_game_step, **self.game_mode_dict)
        env_vis = make_env(game_name=self.game_name, surrol_mark=self.surrol_mark, render_mode="rgb_array", max_step=self.max_game_step, **self.game_mode_dict)
        # env_vis = obj.env_vis

        vis_final_dir = f"{self.vis_dir}/final/mc={modal_corrupt}"
        os.makedirs(vis_final_dir, exist_ok=True)

        log_dir = f"{self.log_dir}/mc={modal_corrupt}"
        os.makedirs(log_dir, exist_ok=True)

        win_counter = 0
        cum_rwd_counter = 0

        # high_res = True
        high_res = False

        test_for_bar = tqdm.tqdm(range(self.config.test_num), leave=False, position=0)

        for i in test_for_bar:
            cum_rwd, vis_dict, step_n, win = self.vis_env(
                env_vis,
                vis_final_dir,
                random_seed=self.config.random_seed + i,
                vis_counter=i,
                high_res=high_res,
                modal_corrupt=modal_corrupt,
                save_gif=save_gif
            )
            win_counter += win
            cum_rwd_counter += cum_rwd

            test_for_bar.desc = f"Testing: {win_counter / self.config.test_num * 100: .02f}"

        win_rate = win_counter / self.config.test_num * 100
        avg_cum_rwd = cum_rwd_counter / self.config.test_num

        env_vis.close()

        log_txt_path = f"{log_dir}/result_log.txt"

        log_string = "\n"
        log_string += f"modal_corrupt: {modal_corrupt}\n"
        log_string += f"win_rate: {win_rate}\n"
        log_string += f"avg_rwd: {avg_cum_rwd}\n"

        with open(log_txt_path, "w") as f:
            f.write(log_string)

        table_path = f"{log_dir}/result.xlsx"
        data_dict = {
            "win_rate": win_rate,
            "avg_rwd": avg_cum_rwd,
        }
        df = pd.Series(data_dict)
        df.to_excel(table_path)

        print(log_string)


def main():
    args = get_args()
    # args = parser.parse_args()

    if not args.train and not args.test:
        raise ValueError("'--train' or '--test'?")

    print()
    print("==========================================================================================")
    print()

    print(f"Loading Config From: {args.config_file}")

    config = load_config(args.config_file)

    config = Config(config)

    config.config_file_path = args.config_file
    config.vis = args.vis
    config.training_mark = args.train
    config.testing_mark = args.test

    project = Project(config)

    if args.train:
        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Training Start at {date_time}")
        print()

        project.train()

        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Training Done at {date_time}")

    if args.test:
        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Testing Start at {date_time}")
        print()

        save_gif = True
        # save_gif = False

        # project.test(modal_corrupt=True)
        project.test(save_gif=save_gif)

        # for sigma in [0.01, 0.05, 0.1, 0.2]:
        #     project.test(modal_corrupt=f"gaussian_noise_sigma_{sigma}", save_gif=save_gif)
        #
        # if project.config.vision_model:
        #     project.test(modal_corrupt="random_crop", save_gif=save_gif)

        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Testing Done at {date_time}")
        print()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', required=True, help='path/to/target/config_file.yaml')
    parser.add_argument('--train', default=False, action="store_true", help='train mode')
    parser.add_argument('--test', default=False, action="store_true", help='test mode')
    parser.add_argument('--vis', default=True, action="store_true", help='vis mode')

    # parser = argparse.ArgumentParser(description='YAML Config Reader')
    # parser.add_argument('--demo.yaml', type=str, required=True, help='Path to the YAML demo.yaml file')
    return parser.parse_args()


if __name__ == '__main__':
    # draft()
    mp.set_start_method('spawn', force=True)
    main()
