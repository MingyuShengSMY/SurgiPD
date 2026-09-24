import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm


class MatchBoard(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # RPOT_POS = (0.55, -0.025, 0.6781)
    # GPOT_POS = (0.55, 0.03, 0.6781)
    # POT_ORN = (1.57,0,0)
    # BOARD_POS = (0.55, 0, 0.6781)
    BOARD_POS = (0.545, 0, 0.6781)
    BOARD_ORN = (1.57, 0, 0)
    # BOARD_SCALING = 0.03
    BOARD_SCALING = 0.028
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE = 3
    haptic = True
    ITEM_SCALING = 0.25 * SCALING * BOARD_SCALING / 0.03

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid=-1):

        self.reward_shaping = None
        self.match_board_id = None
        self.tray_id = None
        self.ecm = None
        self.item_target_goal_array = None

        self.item_target_pos_dict = {
            "0": None,
            "1": None,
            "2": None,
            "4": None,
            "5": None,
            "6": None,
            "a": None,
            "b": None,
            "c": None,
        }

        self.item_id_dict = {
            "0": None,
            "1": None,
            "2": None,
            "4": None,
            "5": None,
            "6": None,
            "a": None,
            "b": None,
            "c": None,
        }

        self.item_name_list = list(self.item_id_dict.keys())
        self.item_name_list.sort()

        # self.item_target_pos_dict = {
        #     "0": np.array([0, 0, 0]),
        #     "1": np.array([0, 0, 0]),
        #     "2": np.array([0, 0, 0]),
        #     "4": np.array([0, 0, 0]),
        #     "5": np.array([0, 0, 0]),
        #     "6": np.array([0, 0, 0]),
        #     "A": np.array([0, 0, 0]),
        #     "B": np.array([0, 0, 0]),
        #     "C": np.array([0, 0, 0]),
        # }

        item_space = 0.15 * self.BOARD_SCALING / 0.03

        self.item_init_pos_list = [
            np.array([-item_space, -0.1 * 1 - item_space, 0.06]),
            np.array([-item_space, 0.1 * 1 + item_space, 0.06]),
            # np.array([-item_space, 0.08 * 2 + item_space, 0.06]),
            np.array([0, -0.1 * 1 - item_space, 0.06]),
            np.array([0, 0.1 * 1 + item_space, 0.06]),
            # np.array([0, 0.08 * 2 + item_space, 0.06]),

            np.array([item_space, -0.1 * 1 - item_space, 0.06]),
            np.array([item_space, 0.1 * 1 + item_space, 0.06]),
            # np.array([item_space, 0.08 * 2 + item_space, 0.06]),

            np.array([item_space + 0.075, 0, 0.06]),
            np.array([item_space + 0.075, item_space, 0.06]),
            np.array([item_space + 0.075, -item_space, 0.06]),
        ]

        super(MatchBoard, self).__init__(render_mode, cid)
        self._view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(MatchBoard, self)._env_setup(goal_plot=False)
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True

        # camera
        if self._render_mode == 'human':
            # self.pbu.reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            self.pbu.reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                                  target=(-0.13, 0.03, -0.94))

        if self.ecm is None:
            self.ecm = Ecm((0.15, 0.0, 0.8524),  # p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                           scaling=self.SCALING, _p=self.p)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        if self.tray_id is None:
            self.tray_id = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                                           np.array(self.POSE_TRAY[0]) * self.SCALING,
                                           self.p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                                           globalScaling=self.SCALING,
                                           useFixedBase=1)
        else:
            self.p.resetBasePositionAndOrientation(self.tray_id, np.array(self.POSE_TRAY[0]) * self.SCALING,
                                                   self.p.getQuaternionFromEuler(self.POSE_TRAY[1]))
        self.obj_ids['fixed'].append(self.tray_id)  # 1

        if self.match_board_id is None:
            self.match_board_id = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/match_board.urdf'),
                                                  np.array(self.BOARD_POS) * self.SCALING,
                                                  self.p.getQuaternionFromEuler(self.BOARD_ORN),
                                                  globalScaling=self.BOARD_SCALING,
                                                  useFixedBase=1)
            self.item_target_pos_dict = {
                "0": np.array(self.p.getLinkState(self.match_board_id, 0)[0]),
                "1": np.array(self.p.getLinkState(self.match_board_id, 1)[0]),
                "2": np.array(self.p.getLinkState(self.match_board_id, 2)[0]),
                "4": np.array(self.p.getLinkState(self.match_board_id, 3)[0]),
                "5": np.array(self.p.getBasePositionAndOrientation(self.match_board_id)[0]),
                "6": np.array(self.p.getLinkState(self.match_board_id, 4)[0]),
                "a": np.array(self.p.getLinkState(self.match_board_id, 5)[0]),
                "b": np.array(self.p.getLinkState(self.match_board_id, 6)[0]),
                "c": np.array(self.p.getLinkState(self.match_board_id, 7)[0]),
            }

            item_target_goal_array = []
            for k in self.item_name_list:
                item_target_goal_array.append(self.item_target_pos_dict[k])

            self.item_target_goal_array = np.concatenate(item_target_goal_array, axis=-1)

        else:
            self.p.resetBasePositionAndOrientation(self.match_board_id, np.array(self.BOARD_POS) * self.SCALING,
                                                   self.p.getQuaternionFromEuler(self.BOARD_ORN))
        self.obj_ids['fixed'].append(self.match_board_id)

        # self.item_target_pos_dict = {
        #     "0": self.p.getLinkState(self.match_board_id, 0)[0],
        #     "1": self.p.getLinkState(self.match_board_id, 1)[0],
        #     "2": self.p.getLinkState(self.match_board_id, 2)[0],
        #     "4": self.p.getLinkState(self.match_board_id, 3)[0],
        #     "5": self.p.getBasePositionAndOrientation(self.match_board_id)[0],
        #     "6": self.p.getLinkState(self.match_board_id, 4)[0],
        #     "a": self.p.getLinkState(self.match_board_id, 5)[0],
        #     "b": self.p.getLinkState(self.match_board_id, 6)[0],
        #     "c": self.p.getLinkState(self.match_board_id, 7)[0],
        # }

        init_pos_list = self.item_init_pos_list.copy()
        self.np_random.shuffle(init_pos_list)

        item_orn = [0, 0, 0]

        for i in range(7):
            if i == 3:
                continue

            obj_str = str(i)

            fn = f'match_board/number_{obj_str}.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)

            pos_offset = init_pos_list.pop(0)

            if self.item_id_dict[obj_str] is None:
                self.item_id_dict[obj_str] = self.p.loadURDF(urdf_path,
                                                             np.array(self.BOARD_POS) * self.SCALING + pos_offset,
                                                             self.p.getQuaternionFromEuler(item_orn),
                                                             globalScaling=self.ITEM_SCALING)
            else:
                self.p.resetBasePositionAndOrientation(self.item_id_dict[obj_str],
                                                       np.array(self.BOARD_POS) * self.SCALING + pos_offset,
                                                       self.p.getQuaternionFromEuler(item_orn))
            self.obj_ids['rigid'].append(self.item_id_dict[obj_str])
        for i in range(3):
            obj_str = chr(i + ord('a'))
            fn = f'match_board/letter_{obj_str.upper()}.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)
            pos_offset = init_pos_list.pop(0)
            if self.item_id_dict[obj_str] is None:
                self.item_id_dict[obj_str] = self.p.loadURDF(urdf_path,
                                                             np.array(self.BOARD_POS) * self.SCALING + pos_offset,
                                                             self.p.getQuaternionFromEuler(item_orn),
                                                             globalScaling=self.ITEM_SCALING)
            else:
                self.p.resetBasePositionAndOrientation(self.item_id_dict[obj_str],
                                                       np.array(self.BOARD_POS) * self.SCALING + pos_offset,
                                                       self.p.getQuaternionFromEuler(item_orn))

            self.obj_ids['rigid'].append(self.item_id_dict[obj_str])

        self.obj_id, self.obj_link1 = self.obj_ids['fixed'][2], 1   # TODO complex task

        self.achieved_goal = np.empty((len(self.item_name_list) * 3,))
        self.observation = np.empty((len(self.item_name_list) * 15 + 7,))

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        robot_gripper_pos = robot_state[0: 3]
        self.observation[:7] = robot_state

        # obj_red_pos, _ = self.pbu.get_link_pose(self.obj_id_list[0], -1)
        # obj_red_pos = np.array(obj_red_pos)
        # obj_green_pos, _ = self.pbu.get_link_pose(self.obj_id_list[1], -1)
        # obj_green_pos = np.array(obj_green_pos)

        wp_pos_array_list = []
        wp_rel_array_list = []
        wp_orn_array_list = []
        obj_pos_array_list = []
        obj_orn_array_list = []
        obj_rel_array_list = []

        observation_idx_counter = 7

        for k_idx, k in enumerate(self.item_name_list):
            wp_pos, wp_orn = self.pbu.get_link_pose(self.item_id_dict[k], 0)
            # wp_pos, wp_orn = self.p.getLinkState(self.item_id_dict[k], 0)
            wp_pos = np.array(wp_pos)
            wp_orn = np.array(self.p.getEulerFromQuaternion(wp_orn))

            wp_rel = wp_pos - robot_gripper_pos

            self.observation[observation_idx_counter: observation_idx_counter + 3] = wp_pos
            observation_idx_counter += 3
            self.observation[observation_idx_counter: observation_idx_counter + 3] = wp_orn
            observation_idx_counter += 3
            self.observation[observation_idx_counter: observation_idx_counter + 3] = wp_rel
            observation_idx_counter += 3

            # wp_pos_array_list.append(wp_pos)
            # wp_orn_array_list.append(wp_orn)
            # wp_rel_array_list.append(wp_rel)

            obj_pos, obj_orn = self.pbu.get_link_pose(self.item_id_dict[k], -1)
            # obj_pos, obj_orn = self.p.getBasePositionAndOrientation(self.item_id_dict[k])
            obj_pos = np.array(obj_pos)
            obj_orn = np.array(self.p.getEulerFromQuaternion(obj_orn))

            obj_rel = self.item_target_pos_dict[k] - obj_pos

            self.achieved_goal[k_idx * 3: k_idx * 3 + 3] = obj_pos

            self.observation[observation_idx_counter: observation_idx_counter + 3] = obj_orn
            observation_idx_counter += 3
            self.observation[observation_idx_counter: observation_idx_counter + 3] = obj_rel
            observation_idx_counter += 3

            # obj_pos_array_list.append(obj_pos)
            # obj_orn_array_list.append(obj_orn)
            # obj_rel_array_list.append(obj_rel)

        # achieved_goal = np.array(obj_pos_array_list).ravel()

        # observation = np.concatenate(
        #     [robot_state] +
        #     wp_pos_array_list +
        #     wp_orn_array_list +
        #     wp_rel_array_list +
        #     obj_orn_array_list +
        #     obj_rel_array_list
        # ).ravel()

        obs = {
            'observation': self.observation,
            'achieved_goal': self.achieved_goal,
            'desired_goal': self.goal
        }
        return obs

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        reward = self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1

        d = goal_distance(achieved_goal, desired_goal)

        if self.reward_shaping is None:
            reward_shaping = 0
        else:
            reward_shaping = -d - self.reward_shaping

        self.reward_shaping = -d

        reward += reward_shaping * 10.0

        return reward
        # return self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal) / len(achieved_goal) * 3
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = self.item_target_goal_array.copy()

        self.reward_shaping = None

        return goal

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback(sphere_mark=False)
        # super()._sample_goal_callback(sphere_mark=True)
        self._waypoints = []

        task_idx_list = self.item_name_list.copy()
        self.np_random.shuffle(task_idx_list)
        # task_idx_list.sort()
        # task_idx_list.remove("1")
        # task_idx_list = ["1"] + task_idx_list
        # task_idx_list.remove("6")
        # task_idx_list = ["6"] + task_idx_list
        # task_idx_list.remove("b")
        # task_idx_list = ["b"] + task_idx_list
        # task_idx_list.remove("a")
        # task_idx_list = ["a"] + task_idx_list

        for i in task_idx_list:
            obj_id_i = self.item_id_dict[i]

            real_goal_i = self.item_target_pos_dict[i]

            pos_base_obj, orn_base_obj = self.p.getBasePositionAndOrientation(obj_id_i)
            pos_obj, orn_obj = self.pbu.get_link_pose(obj_id_i, 0)
            self._waypoint_z_init = pos_obj[2]
            orn = self.p.getEulerFromQuaternion(orn_obj)
            orn_eef = self.pbu.get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
            orn_eef = self.p.getEulerFromQuaternion(orn_eef)
            # yaw = orn[2]
            yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
                else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

            pos_base_obj = np.array(pos_base_obj)
            pos_obj = np.array(pos_obj)

            goal_i = real_goal_i + pos_obj - pos_base_obj
            goal_i[2] += 0.005 * self.SCALING

            lift_height = 0.02
            grasp_down = 0.001

            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (lift_height) * self.SCALING, yaw, 0.5])  # approach
            )
            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (grasp_down) * self.SCALING, yaw, 0.5])  # approach
            )
            for _ in range(3):
                self._waypoints.append(
                    np.array([pos_obj[0], pos_obj[1],
                              pos_obj[2] + (grasp_down) * self.SCALING, yaw, -0.5])  # grasp
                )
            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (lift_height) * self.SCALING, yaw, -0.5])  # lift up
            )
            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (lift_height) * self.SCALING, yaw, -0.5])  # move
            )

            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (0.005) * self.SCALING, yaw, -0.5])  # approach pot
            )

            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (0.005) * self.SCALING, yaw, 0.5])  # release
            )

            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (lift_height) * self.SCALING, yaw, 0.5])  # lift up
            )

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = self.pbu.get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            # if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(0.5):
                self._waypoints[i] = None
            break

        return action

    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])

    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)


if __name__ == "__main__":
    env = MatchBoard(render_mode='human')  # create one process and corresponding env

    while True:
        env.test(1000)
        time.sleep(2)
    env.close()
