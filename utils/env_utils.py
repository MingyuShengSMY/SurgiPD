import sys
import gymnasium as gym
import cv2
import numpy
import numpy as np
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append("SurRoL")
from surrol.tasks.needle_reach import NeedleReach
from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.needle_regrasp_bimanual import NeedleRegrasp
from surrol.tasks.peg_transfer import PegTransfer
from surrol.tasks.peg_transfer_bimanual import BiPegTransfer
from surrol.tasks.pick_and_place import PickAndPlace
from surrol.tasks.gauze_retrieve import GauzeRetrieve


def make_env(game_name, render_mode=None, max_step=1000, surrol_mark=True, **kwargs):
    if surrol_mark:
        if game_name == "NeedleReach":
            env = NeedleReach(render_mode=render_mode)
        elif game_name == "NeedlePick":
            env = NeedlePick(render_mode=render_mode)
        elif game_name == "NeedleRegrasp":
            env = NeedleRegrasp(render_mode=render_mode)
        elif game_name == "PegTransfer":
            env = PegTransfer(render_mode=render_mode)
        elif game_name == "BiPegTransfer":
            env = BiPegTransfer(render_mode=render_mode)
        elif game_name == "PickAndPlace":
            env = PickAndPlace(render_mode=render_mode)
        elif game_name == "GauzeRetrieve":
            env = GauzeRetrieve(render_mode=render_mode)
        else:
            raise ValueError
    else:
        if game_name == "Reacher-v5":
            env = gym.make(game_name, render_mode=render_mode, max_episode_steps=max_step, **kwargs)
        else:
            raise ValueError

    return env


def obs_dict2np(obs: dict, game_name: str, env: gym.Env = None, surrol_mark=True):
    if surrol_mark:
        robot_state_obs = obs["observation"]
        achieved_obs = obs["achieved_goal"]
        target_obs = obs["desired_goal"]

        pms1_jaw1_cp_any = 1.0 if len(env.p.getContactPoints(env.psm1.body, linkIndexA=6)) > 0 else 0.0
        pms1_jaw2_cp_any = 1.0 if len(env.p.getContactPoints(env.psm1.body, linkIndexA=7)) > 0 else 0.0
        claw_touch_any = np.array([pms1_jaw1_cp_any, pms1_jaw2_cp_any])
        next_obs = np.concatenate(
            [robot_state_obs, claw_touch_any],
            axis=-1)

        if env.psm2 is not None:
            pms2_jaw1_cp_any = 1.0 if len(env.p.getContactPoints(env.psm2.body, linkIndexA=6)) > 0 else 0.0
            pms2_jaw2_cp_any = 1.0 if len(env.p.getContactPoints(env.psm2.body, linkIndexA=7)) > 0 else 0.0
            claw2_touch_any = np.array([pms2_jaw1_cp_any, pms2_jaw2_cp_any])
            next_obs = np.concatenate(
                [next_obs, claw2_touch_any],
                axis=-1)

        if game_name in ["NeedlePick", "GauzeRetrieve"]:
            robot_claw_pos1 = robot_state_obs[:3]   # [6] is the tip angle,

            claw_to_target = target_obs - robot_claw_pos1
            claw_to_ach = achieved_obs - robot_claw_pos1
            ach_to_target = target_obs - achieved_obs

            claw_touch_needle = np.array([])
            next_obs = np.concatenate([next_obs, claw_to_target, claw_to_ach, ach_to_target, claw_touch_needle],
                                      axis=-1)

        elif game_name in ["NeedleRegrasp"]:
            robot_claw_pos1 = robot_state_obs[
                :3]  # [6] is the tip angle,
            robot_claw_pos2 = robot_state_obs[
                7:10]  # [13] is the tip angle

            needle_pos = robot_state_obs[14: 17]
            needle_p1 = robot_state_obs[23: 26]
            needle_p2 = robot_state_obs[29: 32]

            if game_name == "NeedleRegrasp":
                claw2_to_p = needle_p1 - robot_claw_pos2
                claw1_to_p = needle_p2 - robot_claw_pos1
            else:
                claw1_to_p = needle_p1 - robot_claw_pos1
                claw2_to_p = needle_p2 - robot_claw_pos2


            needle_to_target = target_obs - needle_pos

            claw_touch_needle = np.array([])
            next_obs = np.concatenate(
                [next_obs, claw_touch_needle, needle_to_target, claw2_to_p,
                 claw1_to_p], axis=-1)
        else:
            next_obs = next_obs

        next_obs = np.concatenate([next_obs, achieved_obs, target_obs], axis=-1)

    else:
        if isinstance(obs, tuple):
            next_obs = np.array(obs[0])
        else:
            next_obs = obs

    return next_obs


def surrol_render(env, h, w):
    img, depth, mask = env.ecm.render_image(width=w, height=h)
    mask = mask.astype(np.uint8)
    depth = (np.power(depth, 150) * 255.0).astype(np.uint8)

    return img, depth, mask


def step_env(env, action, game_name, step_n=None, max_step=None, render=False, h=224, w=224, surrol_mark=True):
    img, depth, mask = None, None, None

    terminated = False
    truncated = False
    win = False

    if surrol_mark:
        next_obs_dict, reward, done, info = env.step(action)

        if info["is_success"]:
            terminated = True
            # terminated = False
            truncated = False
            win = True
        else:
            if max_step is not None and step_n >= max_step:
                truncated = True
                terminated = False
                win = False

        if render:
            img, depth, mask = surrol_render(env, h, w)

    else:
        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        if terminated and not truncated:
            win = True

        if render:
            img = env.render()
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    next_obs = obs_dict2np(next_obs_dict, game_name=game_name, env=env, surrol_mark=surrol_mark)

    vis = {
        "rgb": img,
        "depth": depth,
        "mask": mask,
    }

    return next_obs, reward, vis, terminated, truncated, win, info, next_obs_dict


def reset_env(env, game_name, seed=None, render=False, h=224, w=224, surrol_mark=True):
    img, depth, mask = None, None, None

    if surrol_mark:
        env.seed(seed)
        obs = env.reset()
        if render:
            img, depth, mask = surrol_render(env, h, w)
    else:
        obs = env.reset(seed=seed)
        if render:
            img = env.render()
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    vis = {
        "rgb": img,
        "depth": depth,
        "mask": mask,
    }

    next_obs = obs_dict2np(obs, game_name=game_name, env=env, surrol_mark=surrol_mark)

    return next_obs, obs, vis


def action_denormalizer(action, game_name):
    if game_name == "Pendulum-v0":
        action *= 2

    return action


def obs_normalizer(obv):
    obv_normed = obv.astype(float)

    return obv_normed


def img_normalizer(img: list):
    if img[0] is None:
        return None
    img: np.ndarray = np.stack(img)  # k, h, w, c
    img = np.expand_dims(img.transpose([0, 3, 1, 2]), axis=0)  # [1, k, c, h, w]

    img_normed = img / 255.0

    return img_normed


def my_reward(env, custom_reward_obj, prev_obs, next_obs, reward, act, step_n, terminated, truncated):
    reward = custom_reward_obj.get_reward(env, prev_obs, next_obs, reward, act, step_n, terminated, truncated)

    return reward


def action_normalizer(action, game_name):

    if game_name == "Pendulum-v0":
        action /= 2

    return action
