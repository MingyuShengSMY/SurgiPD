import gc
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import psutil
import torch
import torchvision as tv
import torchvision.transforms.functional as TV_TF
import pympler.asizeof as asizeof

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from multiprocessing import shared_memory

from PIL import Image

from utils.utils import cv2_imshow
import sys


# mp.set_start_method('spawn', force=True)


def load_fun(rgb_root_dir, mem_dict, mem_keys, data_aug, rgb_pos, img_size=84, modal_corrupt=False, modal_corrupt_class=None,
             shared_array: np.ndarray | None = None, shm_slot_idx: int = None, shm_array_select_idx_dict: dict=None, *args, **kwargs):
    hist_idx_list: list = mem_dict["hist_idx_list"].tolist()
    next_hist_idx_list: list = mem_dict["next_hist_idx_list"].tolist()

    current_idx = hist_idx_list[-1]

    idx_set = list(set(hist_idx_list + next_hist_idx_list))

    mask_mark = kwargs.get("mask_mark")
    depth_mark = kwargs.get("depth_mark")

    mask, depth = None, None

    if rgb_root_dir is not None:

        idx_img_dict = {}

        for idx_i in idx_set:
            rgb_path = f"{rgb_root_dir}/rgb/{idx_i:08d}.jpg"
            img = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

            # rgb_path = f"{rgb_root_dir}/rgb/{idx_i:08d}.npy"
            # img = np.load(rgb_path, allow_pickle=True)

            # img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            # img = cv2.imread(rgb_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose([2, 0, 1]).astype(np.float32) / 255.0 * 2 - 1  # [c, h, w]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose([2, 0, 1]).astype(np.float32) / 255.0  # [c, h, w]

            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            img = img.transpose([2, 0, 1])  # [c, h, w]

            idx_img_dict[idx_i] = img

        rgb_list = [idx_img_dict[i] for i in hist_idx_list]
        next_rgb_list = [idx_img_dict[i] for i in next_hist_idx_list]

        rgb_list = np.stack(rgb_list)
        next_rgb_list = np.stack(next_rgb_list)

        if mask_mark:
            mask_path = f"{rgb_root_dir}/mask/{current_idx:08d}.npy"
            mask = np.load(mask_path)
            h, w = mask.shape
            mask = mask.reshape(1, 1, h, w)  # [t, c, h, w]
        if depth_mark:
            depth_path = f"{rgb_root_dir}/depth/{current_idx:08d}.npy"
            depth = np.load(depth_path)
            h, w = depth.shape
            depth = depth.reshape(1, 1, h, w)  # [t, c, h, w]

        if data_aug or rgb_pos:
            pad = img_size // 20

            dx = np.random.randint(0, 2 * pad + 1)
            dy = np.random.randint(0, 2 * pad + 1)

            # img_stack_list = [rgb_list, next_rgb_list]
            # img_stack = np.stack(img_stack_list, axis=0)

            rgb_pad = np.pad(rgb_list, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='edge')
            rgb_list_aug = rgb_pad[:, :, dx: dx + img_size, dy: dy + img_size]

            next_rgb_pad = np.pad(next_rgb_list, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='edge')
            next_rgb_list_aug = next_rgb_pad[:, :, dx: dx + img_size, dy: dy + img_size]

            if data_aug:
                rgb_list = rgb_list_aug
                next_rgb_list = next_rgb_list_aug

            if rgb_pos:
                shared_array[shm_array_select_idx_dict["rgb_pos"], shm_slot_idx, ...] = rgb_list_aug
                # mem_dict["rgb_pos"] = rgb_list_aug  # [k, c, h, w]

            if mask_mark:
                mask_pad = np.pad(mask, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='edge')
                mask = mask_pad[:, :, dx: dx + img_size, dy: dy + img_size]

            if depth_mark:
                depth_pad = np.pad(depth, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='edge')
                depth = depth_pad[:, :, dx: dx + img_size, dy: dy + img_size]

        shared_array[0, shm_slot_idx, ...] = rgb_list
        shared_array[1, shm_slot_idx, ...] = next_rgb_list

        if mask_mark:
            shared_array[shm_array_select_idx_dict["mask"], shm_slot_idx, ...] = mask
        if depth_mark:
            shared_array[shm_array_select_idx_dict["depth"], shm_slot_idx, ...] = depth

    return mem_dict


def worker_load_img(idx_queue: mp.Queue, img_queue: mp.Queue, free_slot_queue: mp.Queue, rgb_root_dir, mem_keys, data_aug, rgb_pos, img_size, modal_corrupt, shm_shape,
                    shm_dtype, shm_name, shm_array_select_idx_dict):
    pid = os.getpid()

    # modal_corruptor = modal_corruptor()
    modal_corruptor = None

    shm = shared_memory.SharedMemory(name=shm_name)

    shared_array = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)

    mask_mark = "mask" in mem_keys
    depth_mark = "depth" in mem_keys
    while True:
        idx, mem_dict_i = idx_queue.get()
        if idx is None:
            break

        shm_slot_idx = free_slot_queue.get()

        if shm_slot_idx < 0:
            break

        mem_dict = load_fun(rgb_root_dir, mem_dict_i, mem_keys, data_aug, rgb_pos, img_size,
                            modal_corrupt, modal_corruptor, shared_array, shm_slot_idx,
                            shm_array_select_idx_dict, mask_mark=mask_mark, depth_mark=depth_mark)

        # img_queue.put((idx, shm_slot_idx, mem_dict))
        img_queue.put((idx, shm_slot_idx))


class MemoryBuffer:
    def __init__(self,
                 memory_size=10000,
                 obs_dim=1,
                 act_dim=1,
                 continuous_act=True,
                 continuous_obs=True,
                 rwd_u=0.0,
                 rwd_v=1.0,
                 root_dir=".",
                 memory_extra_keys: list = None,
                 num_workers=0,
                 frame_k=1,
                 img_size=84,
                 rgb_in_disk=True,
                 img_c_dim=3,
                 data_aug=False,
                 rgb_pos=False,
                 modal_corrupt=False,
                 batch_size=256,
                 ):

        # [s, a, r, cum_r, s_, terminated]

        self.root_dir = f"{root_dir}"
        self.rgb_in_disk = rgb_in_disk
        self.img_c_dim = img_c_dim
        self.img_size = img_size
        self.frame_k = frame_k
        self.num_workers = 0 if (not self.rgb_in_disk or "rgb" not in memory_extra_keys) else num_workers

        self.mem_keys = ["obs", "act", "rwd", "next_obs", "trm", "trc"]
        self.extra_mem_keys = memory_extra_keys
        if self.extra_mem_keys is not None:
            self.mem_keys += self.extra_mem_keys

        # self.mem_dir = f"{self.root_dir}/mem"
        # os.makedirs(self.mem_dir, exist_ok=True)

        self.mem_dict = {
            "obs": np.empty((memory_size, obs_dim), dtype=np.float32 if continuous_obs else np.int64),
            "act": np.empty((memory_size, act_dim), dtype=np.float32 if continuous_act else np.int64),
            "rwd": np.empty((memory_size,)),
            "next_obs": np.empty((memory_size, obs_dim), dtype=np.float32 if continuous_obs else np.int64),
            "trm": np.empty((memory_size,), dtype=bool),
            "trc": np.empty((memory_size,), dtype=bool),
            "hist_idx_list": np.empty((memory_size, self.frame_k), dtype=np.uint64),
            "next_hist_idx_list": np.empty((memory_size, self.frame_k), dtype=np.uint64),
        }

        # counter_str_list = [f"{i:08d}" for i in range(memory_size)]

        self.rgb_dir = None
        # if "cum_rwd" in self.mem_keys:
        #     self.mem_dict["cum_rwd"] = self.mem_dict["rwd"].copy()
        if "prob" in self.mem_keys:
            self.mem_dict["prob"] = self.mem_dict["rwd"].copy()
        if "value" in self.mem_keys:
            self.mem_dict["value"] = self.mem_dict["rwd"].copy()
        if "rgb" in self.mem_keys:
            if self.rgb_in_disk:
                self.rgb_dir = f"{self.root_dir}/rgb"
                os.makedirs(self.rgb_dir, exist_ok=True)
            else:
                self.mem_dict["rgb"] = np.empty((memory_size, self.img_c_dim, self.img_size, self.img_size),
                                                dtype=np.uint8)
                self.mem_dict["next_rgb"] = self.mem_dict["rgb"]

            self.mem_keys.append("next_rgb")

        if "mask" in self.mem_keys:
            self.mask_dir = f"{self.root_dir}/mask"
            os.makedirs(self.mask_dir, exist_ok=True)
        if "depth" in self.mem_keys:
            self.depth_dir = f"{self.root_dir}/depth"
            os.makedirs(self.depth_dir, exist_ok=True)

        self.memory_size = memory_size
        self.memory_full_mark = False
        self.counter = 0
        self.total_counter = 0

        self.gae_gamma = 0.99
        self.gae_lambda = 0.95

        self.advantages = None
        self.v_targets = None

        self.rwd_u = rwd_u
        self.rwd_v = rwd_v

        self.task_queue: mp.Queue = None
        self.result_queue: mp.Queue = None
        self.free_slot_queue: mp.Queue = None
        self.workers = None
        self.data_aug = data_aug
        self.img_size = img_size

        self.modal_corrupt = modal_corrupt

        self.batch_size = batch_size

        self.mp_pre_load_factor = 1

        if self.num_workers > 0:
            # self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            # self.executor_map = self.executor.map

            self.mp_pre_load_factor = 4

            self.shm_len = batch_size * self.mp_pre_load_factor

            self.shm_free_slot_idx = np.arange(self.shm_len).tolist()

            shm_array_select_idx_dict = {}
            idx = 2
            if rgb_pos:
                shm_array_select_idx_dict["rgb_pos"] = idx
                idx += 1
            if "mask" in self.mem_keys:
                shm_array_select_idx_dict["mask"] = idx
                idx += 1
            if "depth" in self.mem_keys:
                shm_array_select_idx_dict["depth"] = idx
                idx += 1

            self.shm_array_select_idx_dict = shm_array_select_idx_dict

            shm_shape = (2 + (idx - 2), batch_size * self.mp_pre_load_factor, frame_k, img_c_dim, img_size, img_size)
            shm_dtype = np.uint8
            dummy = np.empty(shm_shape, dtype=shm_dtype)

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=dummy.nbytes
            )

            self.shared_array = np.ndarray(
                shm_shape,
                dtype=shm_dtype,
                buffer=self.shm.buf
            )

            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue(maxsize=self.shm_len)
            self.free_slot_queue = mp.Queue()
            # self.result_queue = mp.Queue(maxsize=256)
            # self.result_queue = mp.Queue()
            self.workers = [mp.Process(target=worker_load_img,
                                       args=(self.task_queue, self.result_queue, self.free_slot_queue, self.root_dir, self.mem_keys, data_aug, rgb_pos, self.img_size,
                                             self.modal_corrupt, shm_shape, shm_dtype, self.shm.name, shm_array_select_idx_dict))
                            for _ in range(num_workers)]
            for worker in self.workers:
                worker.start()

    # def push(self, obs, action, reward, cum_reward, next_obs, terminated, prob, value):
    def push(self, obs, act, rwd, next_obs, trm, trc, *args, **kwargs):
        """
        :param obs: (d,)
        :param act: (d,)
        :param rwd: float
        :param next_obs: (d,)
        :param trm: Bool
        :param trc: Bool
        :return:
        """

        done = trm or trc

        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        idx = self.counter
        hist_idx_list = [idx]
        for k in range(1, self.frame_k):
            if self.memory_full_mark and self.counter > 0:
                idx_i = (idx - k + self.memory_size) % self.memory_size
            else:
                idx_i = max(idx - k, 0)

            if idx_i != idx:
                prev_trm, prev_trc = self.mem_dict["trm"][idx_i], self.mem_dict["trc"][idx_i]
                if prev_trm or prev_trc:
                    idx_i = hist_idx_list[-1]

            hist_idx_list.append(idx_i)
        hist_idx_list = hist_idx_list[::-1]

        if done:
            next_hist_idx_list = hist_idx_list
        else:
            next_hist_idx_list = []
            for idx_i in hist_idx_list:
                idx_i = (idx_i + 1) % self.memory_size
                next_hist_idx_list.append(idx_i)

        self.mem_dict["obs"][idx] = obs.copy()
        self.mem_dict["act"][idx] = act.copy()
        self.mem_dict["rwd"][idx] = rwd
        self.mem_dict["next_obs"][idx] = next_obs.copy()
        self.mem_dict["trm"][idx] = trm
        self.mem_dict["trc"][idx] = trc
        self.mem_dict["hist_idx_list"][idx] = hist_idx_list.copy()
        self.mem_dict["next_hist_idx_list"][idx] = next_hist_idx_list.copy()

        for k in kwargs:
            v = kwargs[k]
            if k in self.mem_keys:
                if k == "rgb" or k == "next_rgb":
                    if self.rgb_in_disk:
                        img = v.astype(np.uint8)
                        # if img.shape[0] != self.img_size and img.shape[1] != self.img_size:
                        #     img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                        # path = f"{self.rgb_dir}/{idx + (k == "next_rgb"):08d}.npy"
                        # np.save(path, img)

                        path = f"{self.rgb_dir}/{idx + (k == "next_rgb"):08d}.jpg"
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(path, img)
                    else:
                        self.mem_dict[k][idx] = v.transpose([2, 0, 1]).astype(np.uint8).copy()
                elif k in ["mask", "depth"]:
                    path = f"{self.root_dir}/{k}/{idx:08d}.npy"
                    img = v.astype(np.uint8)
                    np.save(path, img)
                else:
                    self.mem_dict[k][idx] = deepcopy(kwargs[k])

        self.counter += 1
        self.total_counter += 1

        if self.counter >= self.memory_size:
            self.counter = 0
            self.memory_full_mark = True

    def get_item_dict(self, idx) -> dict:
        item_dict = {}
        for k in self.mem_dict:
            v = self.mem_dict[k][idx]

            if k == "rgb":
                v = self.mem_dict["rgb"][self.mem_dict["hist_idx_list"][idx]]
            if k == "next_rgb":
                v = self.mem_dict["rgb"][self.mem_dict["next_hist_idx_list"][idx]]

            item_dict[k] = v

        return item_dict

    def slice(self, index_list):
        batch_dict = {}
        batch_size = len(index_list)
        batch_index_list = []

        index_list_new = []

        # write_idx_array = np.arange(len(index_list))

        if self.num_workers > 0:
            while len(self.shm_free_slot_idx):
                self.free_slot_queue.put(self.shm_free_slot_idx.pop(0))

            shm_slot_idx_array = []

            for i in range(batch_size):
                idx, shm_slot_idx = self.result_queue.get()
                self.shm_free_slot_idx.append(shm_slot_idx)
                shm_slot_idx_array.append(shm_slot_idx)

                batch_index_list.append(idx)

                index_list_new.append(idx)

            shm_slot_idx_array = np.array(shm_slot_idx_array)

            index_list = index_list_new

            if "rgb" in self.mem_keys:
                batch_dict["rgb"] = self.shared_array[0, shm_slot_idx_array, ...]
            if "next_rgb" in self.mem_keys:
                batch_dict["next_rgb"] = self.shared_array[1, shm_slot_idx_array, ...]
            if "rgb_pos" in self.mem_keys:
                batch_dict["rgb_pos"] = self.shared_array[self.shm_array_select_idx_dict["rgb_pos"], shm_slot_idx_array, ...]
            if "mask" in self.mem_keys:
                batch_dict["mask"] = self.shared_array[self.shm_array_select_idx_dict["mask"], shm_slot_idx_array, 0:1, 0:1, ...]
            if "depth" in self.mem_keys:
                batch_dict["depth"] = self.shared_array[self.shm_array_select_idx_dict["depth"], shm_slot_idx_array, 0:1, 0:1, ...]

        mem_slice = self.get_item_dict(np.array(index_list))
        batch_dict.update(mem_slice)

        # for k in batch_dict:
        #     batch_dict[k] = np.stack(batch_dict[k])

        batch_dict["rwd"] = batch_dict["rwd"] * self.rwd_v + self.rwd_u

        for k in batch_dict:
            v = batch_dict[k]
            if isinstance(v, np.ndarray):
                # batch_dict[k] = torch.from_numpy(v).float().pin_memory()
                batch_dict[k] = torch.from_numpy(v).float()

        if self.advantages is not None:
            advantages = self.advantages[index_list]
            v_targets = self.v_targets[index_list]

            batch_dict["advantages"] = torch.from_numpy(advantages).float()
            batch_dict["v_targets"] = torch.from_numpy(v_targets).float()

        # e = time.time()
        # se = e - s

        return batch_index_list, batch_dict

    def sample(self, batch_size):

        assert self.memory_full_mark or self.counter, "No data available in memory"

        index_list = np.random.randint(0, self.size(), batch_size)

        batch_dict = self.slice(index_list)

        return batch_dict

    def get_batch_index_set(self, batch_size, batch_num=-1):

        assert self.memory_full_mark or self.counter, "No data available in memory"

        index_list = np.arange(self.size())

        np.random.shuffle(index_list)

        if batch_num == 1:
            batches = [index_list[:batch_size]]
        else:
            batches = [index_list[i:i + batch_size] for i in range(0, len(index_list), batch_size)]

        return batches

    def get_iter_index_set(self, batch_size, iter_num=-1, mp_pre_load_mark=False):

        assert self.memory_full_mark or self.counter, "No data available in memory"

        index_list = np.arange(self.size())

        origin_iter_num = iter_num
        if mp_pre_load_mark and self.mp_pre_load_factor > 1 and iter_num > 0 and self.num_workers > 0:
            iter_num = self.mp_pre_load_factor

        if iter_num == -1:
            np.random.shuffle(index_list)
            batches = [index_list[i:i + batch_size] for i in range(0, len(index_list), batch_size)]
            # batches = np.split(index_list, int(np.ceil(len(index_list) / batch_size).item()))
        else:
            index_list = np.random.choice(index_list, replace=True, size=(batch_size * iter_num,))
            batches = np.split(index_list, iter_num)

        if self.num_workers > 0:
            for idx in index_list:
                mem_dict_i = self.get_item_dict(idx)
                self.task_queue.put((idx, mem_dict_i))

            batches = batches[:origin_iter_num]

        return batches

    def stop_mp(self):
        if self.num_workers > 0:
            time.sleep(5)
            while not self.result_queue.empty():
                self.result_queue.get()

            while not self.task_queue.empty():
                self.task_queue.get()

            while not self.free_slot_queue.empty():
                self.free_slot_queue.get()

            for _ in range(self.num_workers):
                self.task_queue.put((None, None))
                self.free_slot_queue.put(-1)

            for p in self.workers:
                p.join()

            self.result_queue.close()
            self.result_queue.cancel_join_thread()

            self.task_queue.close()
            self.task_queue.cancel_join_thread()

            self.free_slot_queue.close()
            self.free_slot_queue.cancel_join_thread()

            self.shm.close()
            self.shm.unlink()

        return None

    def size(self):
        return self.counter if not self.memory_full_mark else self.memory_size

    def clear(self):
        self.counter = 0
        self.total_counter = 0
        self.memory_full_mark = False
        self.advantages = None
        self.v_targets = None

    def calculate_gae(self):

        advantages = []
        last_advantages = 0

        all_reward = self.mem_dict["rwd"][:self.size()]

        for i in reversed(range(self.size())):
            terminated = self.mem_dict["trm"][i] or self.mem_dict["trc"][i]
            reward = all_reward[i]
            if terminated or i >= self.size() - 1:
                next_value = 0
            else:
                next_value = self.mem_dict["value"][i + 1]

            td_error = reward + self.gae_gamma * next_value - self.mem_dict["value"][i]
            last_advantages = td_error + self.gae_gamma * self.gae_lambda * (1 - terminated) * last_advantages
            advantages.insert(0, last_advantages)

        advantages = np.array(advantages, dtype=np.float32)
        v_targets = advantages + self.mem_dict["value"][:self.size()]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        self.advantages = advantages
        self.v_targets = v_targets

        return advantages, v_targets

    def save(self):
        dir_str = f"{self.root_dir}"
        os.makedirs(dir_str, exist_ok=True)

        path = f"{dir_str}/total_counter.npy"
        np.save(path, self.total_counter, allow_pickle=True)

        path = f"{dir_str}/mem_dict.npz"
        np.savez(path, **self.mem_dict)

    def load(self):
        dir_str = f"{self.root_dir}"

        if not os.path.exists(dir_str):
            return False

        path1 = f"{dir_str}/total_counter.npy"
        path2 = f"{dir_str}/mem_dict.npz"

        if not os.path.exists(path1) or not os.path.exists(path2):
            return False

        self.total_counter = np.load(path1, allow_pickle=True).item()
        self.mem_dict = dict(np.load(path2, allow_pickle=True))

        self.counter = self.total_counter % self.memory_size

        self.memory_full_mark = self.total_counter >= self.memory_size

        return True
