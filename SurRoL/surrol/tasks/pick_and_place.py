import os
import time
import numpy as np

import pybullet as p_y_b
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    # get_link_pose,
    # reset_camera,
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm


class PickAndPlace(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    RPOT_POS = (0.55, -0.05, 0.6781)
    GPOT_POS = (0.55, 0.05, 0.6781)
    POT_ORN = (1.57, 0, 0)
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE = 3
    haptic = True

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid=-1):

        self.ch4_id_red = None
        self.ch4_id_green = None
        self.ch4_id_list = [None] * 6
        self.green_pot_id = None
        self.red_pot_id = None
        self.tray_pad_id = None
        self.ecm = None

        super(PickAndPlace, self).__init__(render_mode, cid)
        self._view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(PickAndPlace, self)._env_setup(goal_plot=False)
        # self.np_random.seed(4)  # for experiment reproduce

        self.has_object = True
        self._waypoint_goal = True

        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
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

        scaling = 0.15
        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        if self.tray_pad_id is None:
            self.tray_pad_id = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                                               np.array(self.POSE_TRAY[0]) * self.SCALING,
                                               self.p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                                               globalScaling=self.SCALING)
        else:
            self.p.resetBasePositionAndOrientation(self.tray_pad_id, np.array(self.POSE_TRAY[0]) * self.SCALING,
                                                   self.p.getQuaternionFromEuler(self.POSE_TRAY[1]))
        self.obj_ids['fixed'].append(self.tray_pad_id)  # 1
        # p.changeVisualShape(board, -1, textureUniqueId=metal)

        if self.red_pot_id is None:
            self.red_pot_id = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
                                              np.array(self.RPOT_POS) * self.SCALING,
                                              self.p.getQuaternionFromEuler(self.POT_ORN),
                                              useFixedBase=1)
        else:
            self.p.resetBasePositionAndOrientation(
                self.red_pot_id,
                np.array(self.RPOT_POS) * self.SCALING,
                self.p.getQuaternionFromEuler(self.POT_ORN),
            )
        self.obj_ids['fixed'].append(self.red_pot_id)  # 2

        if self.green_pot_id is None:
            self.green_pot_id = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
                                                np.array(self.GPOT_POS) * self.SCALING,
                                                self.p.getQuaternionFromEuler(self.POT_ORN),
                                                useFixedBase=1)
            self.p.changeVisualShape(self.green_pot_id, -1, rgbaColor=(0, 1, 0, 1), specularColor=(80, 80, 80))
        else:
            self.p.resetBasePositionAndOrientation(
                self.green_pot_id,
                np.array(self.GPOT_POS) * self.SCALING,
                self.p.getQuaternionFromEuler(self.POT_ORN),
            )
        self.obj_ids['fixed'].append(self.green_pot_id)  # 2

        # ch4
        # self.obj_id, self.obj_link = [], []
        yaw = (self.np_random.uniform() - 0.5) * np.pi
        pos = np.array(
            (workspace_limits[0].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,  # TODO: scaling
             workspace_limits[1].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,
             workspace_limits[2][0] + 0.005))
        orn = self.p.getQuaternionFromEuler((0, 0, yaw))
        if self.ch4_id_red is None:
            self.ch4_id_red = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
                                              pos,
                                              orn,
                                              useFixedBase=False,
                                              globalScaling=self.SCALING)
            self.p.changeVisualShape(self.ch4_id_red, -1, specularColor=(80, 80, 80))
        else:
            self.p.resetBasePositionAndOrientation(self.ch4_id_red, pos, orn)
        self.obj_ids['rigid'].append(self.ch4_id_red)  # 0

        yaw = (self.np_random.uniform() - 0.5) * np.pi
        pos_2 = pos
        while np.linalg.norm(pos_2 - pos) < 0.1:
            pos_2 = np.array(
                (workspace_limits[0].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,  # TODO: scaling
                 workspace_limits[1].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,
                 workspace_limits[2][0] + 0.005))
        orn = self.p.getQuaternionFromEuler((0, 0, yaw))
        if self.ch4_id_green is None:
            self.ch4_id_green = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
                                                pos_2,
                                                orn,
                                                useFixedBase=False,
                                                globalScaling=self.SCALING)
            self.p.changeVisualShape(self.ch4_id_green, -1, rgbaColor=(0, 1, 0, 1), specularColor=(80, 80, 80))
        else:
            self.p.resetBasePositionAndOrientation(self.ch4_id_green, pos_2, orn)
        self.obj_ids['rigid'].append(self.ch4_id_green)  # 0

        self.obj_id_list, self.obj_link_list = [self.ch4_id_red, self.ch4_id_green], [1, 1]

        old_pos_list = [pos, pos_2]

        for i in range(0, 6):
            pos_2 = pos
            yaw = (self.np_random.uniform() - 0.5) * np.pi
            orn = self.p.getQuaternionFromEuler((0, 0, yaw))

            while any(np.linalg.norm(pos_2 - pos_i) < 0.1 for pos_i in old_pos_list):
                pos_2 = np.array((workspace_limits[0].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,
                                  # TODO: scaling
                                  workspace_limits[1].mean() + (self.np_random.uniform() - 0.5) * 0.06 * self.SCALING,
                                  workspace_limits[2][0] + 0.005))

            old_pos_list.append(pos_2)

            if self.ch4_id_list[i] is None:
                pick_item = self.p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
                                            pos_2,
                                            orn,
                                            useFixedBase=False,
                                            globalScaling=self.SCALING)
                # if i <= 2:
                #     self.p.changeVisualShape(pick_item, -1, specularColor=(80, 80, 80))
                # else:
                # self.p.changeVisualShape(pick_item, -1, rgbaColor=(0, 1, 0, 1), specularColor=(80, 80, 80))

                self.p.changeVisualShape(pick_item, -1, rgbaColor=(0, 0, 1, 1), specularColor=(80, 80, 80))
                self.ch4_id_list[i] = pick_item
            else:
                self.p.resetBasePositionAndOrientation(self.ch4_id_list[i], pos_2, orn)
            self.obj_ids['rigid'].append(self.ch4_id_list[i])

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        robot_gripper_pos = robot_state[0: 3]

        # obj_red_pos, _ = self.pbu.get_link_pose(self.obj_id_list[0], -1)
        # obj_red_pos = np.array(obj_red_pos)
        # obj_green_pos, _ = self.pbu.get_link_pose(self.obj_id_list[1], -1)
        # obj_green_pos = np.array(obj_green_pos)

        wp_red_pos, wp_red_orn = self.pbu.get_link_pose(self.obj_id_list[0], self.obj_link_list[0])
        wp_red_pos = np.array(wp_red_pos)
        wp_red_orn = np.array(self.p.getEulerFromQuaternion(wp_red_orn))

        wp_green_pos, wp_green_orn = self.pbu.get_link_pose(self.obj_id_list[1], self.obj_link_list[1])
        wp_green_pos = np.array(wp_green_pos)
        wp_green_orn = np.array(self.p.getEulerFromQuaternion(wp_green_orn))

        # relative position state
        wp_red_rel = wp_red_pos - robot_gripper_pos
        wp_green_rel = wp_green_pos - robot_gripper_pos

        achieved_goal = np.concatenate(
            [wp_red_pos, wp_green_pos],
        )

        goal_rel = self.goal - achieved_goal

        observation = np.concatenate([
            robot_state, wp_red_rel, wp_green_rel, wp_red_orn, wp_green_orn, goal_rel
        ])
        obs = {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy()
        }
        return obs

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # workspace_limits = self.workspace_limits1
        # goal = np.array([workspace_limits[0].mean() + 0.01 * self.np_random.normal() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * self.np_random.normal() * self.SCALING,
        #                  workspace_limits[2][1] - 0.04 * self.SCALING])

        # if self.np_random.uniform() >= 0.5:
        goal_green = self.p.getBasePositionAndOrientation(self.green_pot_id)[0]
            # self.p.changeVisualShape(self.ch4_id_0, -1, rgbaColor=(0, 1, 0, 1), specularColor=(80, 80, 80))
        # else:
        goal_red = self.p.getBasePositionAndOrientation(self.red_pot_id)[0]
            # self.p.changeVisualShape(self.ch4_id_0, -1, rgbaColor=(1, 0, 0, 1), specularColor=(80, 80, 80))
        goal = np.array(goal_red + goal_green)

        goal[[2, 5]] += 0.005 * self.SCALING
        # goal[5] += 0.005 * self.SCALING

        # return goal.copy()
        return goal

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal) / 2
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback(sphere_mark=False)
        # super()._sample_goal_callback(sphere_mark=True)
        self._waypoints = []

        task_idx_list = [0, 1]
        self.np_random.shuffle(task_idx_list)

        for i in task_idx_list:
            obj_id_i = self.obj_id_list[i]
            obj_link_i = self.obj_link_list[i]

            goal_i = self.goal[i * 3: 3 + i * 3]

            pos_obj, orn_obj = self.pbu.get_link_pose(obj_id_i, obj_link_i)
            self._waypoint_z_init = pos_obj[2]
            orn = self.p.getEulerFromQuaternion(orn_obj)
            orn_eef = self.pbu.get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
            orn_eef = self.p.getEulerFromQuaternion(orn_eef)
            yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
                else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
            )
            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (-0.0007 + 0.0102 - 0.005) * self.SCALING, yaw, 0.5])  # approach
            )
            # for _ in range(1):
            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (-0.0007 + 0.0102 - 0.005) * self.SCALING, yaw, -0.5])  # grasp
            )
            self._waypoints.append(
                np.array([pos_obj[0], pos_obj[1],
                          pos_obj[2] + (-0.0007 + 0.0102 + 0.02) * self.SCALING, yaw, -0.5])  # lift up
            )
            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (-0.0007 + 0.0102 + 0.02) * self.SCALING, yaw, -0.5])  # move
            )

            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (0.0102 + 0.005) * self.SCALING, yaw, -0.5])  # approach pot
            )

            self._waypoints.append(
                np.array([goal_i[0], goal_i[1],
                          goal_i[2] + (0.0102 + 0.005 + 0.02) * self.SCALING, yaw, 0.5])  # release
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
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
            # if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.0):
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
    env = PickAndPlace(render_mode='human')  # create one process and corresponding env

    while True:
        env.test(horizon=200)

    env.close()
    time.sleep(2)
