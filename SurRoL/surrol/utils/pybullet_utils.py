"""
This file is modified from
https://github.com/columbia-robovision/decentralized-multiarm/blob/main/environment/rrt/pybullet_utils.py
"""
import time
import numpy as np
import pybullet as p_y_b
import pybullet_data
from collections import namedtuple, defaultdict, deque
from itertools import product, combinations

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
MAX_DISTANCE = 0

# Constraints

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild',
                                               'maxAppliedForce'])

# Joints

JOINT_TYPES = {
    p_y_b.JOINT_REVOLUTE: 'revolute',  # 0
    p_y_b.JOINT_PRISMATIC: 'prismatic',  # 1
    p_y_b.JOINT_SPHERICAL: 'spherical',  # 2
    p_y_b.JOINT_PLANAR: 'planar',  # 3
    p_y_b.JOINT_FIXED: 'fixed',  # 4
    p_y_b.JOINT_POINT2POINT: 'point2point',  # 5
    p_y_b.JOINT_GEAR: 'gear',  # 6
}

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

# Body and base

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

# Links

BASE_LINK = -1
STATIC_MASS = 0

# get_num_links = get_num_joints
# get_links = get_joints

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos', 'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness'])

# Camera

CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                       'viewMatrix', 'projectionMatrix', 'cameraUp',
                                       'cameraForward', 'horizontal', 'vertical',
                                       'yaw', 'pitch', 'dist', 'target'])


def split_7d(pose):
    return [list(pose[:3]), list(pose[3:])]


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def merge_pose_2d(pose):
    return pose[0] + pose[1]


def get_goal_test_fn(goal_conf, atol=0.001, rtol=0):
    def fn(self, conf):
        return np.allclose(conf, goal_conf, atol=atol, rtol=rtol)

    return fn


class PyBulletUtils:
    def __init__(self, _p=None):
        if _p is None:
            _p = p_y_b

        self.p = _p
        self.cid = self.p._client

        self.get_links = self.get_joints

    def step(self, duration=1.0):
        for i in range(int(duration * 240)):
            self.p.stepSimulation()

    def step_real(self, duration=1.0):
        for i in range(int(duration * 240)):
            self.p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def get_euler_from_quaternion(self, quaternion):
        return list(self.p.getEulerFromQuaternion(quaternion))

    def get_quaternion_from_euler(self, euler):
        return list(self.p.getQuaternionFromEuler(euler))

    def configure_pybullet(self, rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
        if not rendering:
            self.p.connect(self.p.DIRECT)
        else:
            self.p.connect(self.p.GUI_SERVER)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if not debug:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, True)
        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0, 0, -9.8)

    def remove_all_constraints(self, ):
        for cid in self.get_constraint_ids():
            self.p.removeConstraint(cid)

    def get_constraint_ids(self, ):
        """
        getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
        Note that the constraint unique ids may not be contiguous, since you may remove constraints.
        """
        return sorted([self.p.getConstraintUniqueId(i) for i in range(self.p.getNumConstraints())])

    def get_constraint_info(self, constraint):
        # there are four additional arguments
        return ConstraintInfo(*self.p.getConstraintInfo(constraint)[:11])

    def get_num_joints(self, body):
        return self.p.getNumJoints(body)

    def get_joints(self, body):
        return list(range(self.get_num_joints(body)))

    def get_joint(self, body, joint_or_name):
        if type(joint_or_name) is str:
            return self.joint_from_name(body, joint_or_name)
        return joint_or_name

    def get_joint_info(self, body, joint):
        return JointInfo(*self.p.getJointInfo(body, joint))

    def get_joints_info(self, body, joints):
        return [JointInfo(*self.p.getJointInfo(body, joint)) for joint in joints]

    def get_joint_name(self, body, joint):
        return self.get_joint_info(body, joint).jointName.decode('UTF-8')

    def get_joint_names(self, body):
        return [self.get_joint_name(body, joint) for joint in self.get_joints(body)]

    def joint_from_name(self, body, name):
        for joint in self.get_joints(body):
            if self.get_joint_name(body, joint) == name:
                return joint
        raise ValueError(body, name)

    def has_joint(self, body, name):
        try:
            self.joint_from_name(body, name)
        except ValueError:
            return False
        return True

    def joints_from_names(self, body, names):
        return tuple(self.joint_from_name(body, name) for name in names)

    def get_joint_state(self, body, joint):
        return JointState(*self.p.getJointState(body, joint))

    def get_joint_position(self, body, joint):
        return self.get_joint_state(body, joint).jointPosition

    def get_joint_torque(self, body, joint):
        return self.get_joint_state(body, joint).appliedJointMotorTorque

    def get_joint_positions(self, body, joints=None):
        return list(self.get_joint_position(body, joint) for joint in joints)

    def set_joint_position(self, body, joint, value):
        self.p.resetJointState(body, joint, value)

    def set_joint_positions(self, body, joints, values):
        assert len(joints) == len(values)
        for joint, value in zip(joints, values):
            self.set_joint_position(body, joint, value)

    def get_configuration(self, body):
        return self.get_joint_positions(body, self.get_movable_joints(body))

    def set_configuration(self, body, values):
        self.set_joint_positions(body, self.get_movable_joints(body), values)

    def get_full_configuration(self, body):
        # Cannot alter fixed joints
        return self.get_joint_positions(body, self.get_joints(body))

    def get_joint_type(self, body, joint):
        return self.get_joint_info(body, joint).jointType

    def is_movable(self, body, joint):
        return self.get_joint_type(body, joint) != self.p.JOINT_FIXED

    def get_movable_joints(self, body):  # 45 / 87 on pr2
        return [joint for joint in self.get_joints(body) if self.is_movable(body, joint)]

    def joint_from_movable(self, body, index):
        return self.get_joints(body)[index]

    def is_circular(self, body, joint):
        joint_info = self.get_joint_info(body, joint)
        if joint_info.jointType == self.p.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit

    def get_joint_limits(self, body, joint):
        """
        Obtain the limits of a single joint
        :param body: int
        :param joint: int
        :return: (int, int), lower limit and upper limit
        """
        if self.is_circular(body, joint):
            return CIRCULAR_LIMITS
        joint_info = self.get_joint_info(body, joint)
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit

    def get_joints_limits(self, body, joints):
        """
        Obtain the limits of a set of joints
        :param body: int
        :param joints: array type
        :return: a tuple of 2 arrays - lower limit and higher limit
        """
        lower_limit = []
        upper_limit = []
        for joint in joints:
            lower_limit.append(self.get_joint_info(body, joint).jointLowerLimit)
            upper_limit.append(self.get_joint_info(body, joint).jointUpperLimit)
        return lower_limit, upper_limit

    def get_min_limit(self, body, joint):
        return self.get_joint_limits(body, joint)[0]

    def get_max_limit(self, body, joint):
        return self.get_joint_limits(body, joint)[1]

    def get_max_velocity(self, body, joint):
        return self.get_joint_info(body, joint).jointMaxVelocity

    def get_max_force(self, body, joint):
        return self.get_joint_info(body, joint).jointMaxForce

    def get_joint_q_index(self, body, joint):
        return self.get_joint_info(body, joint).qIndex

    def get_joint_v_index(self, body, joint):
        return self.get_joint_info(body, joint).uIndex

    def get_joint_axis(self, body, joint):
        return self.get_joint_info(body, joint).jointAxis

    def get_joint_parent_frame(self, body, joint):
        joint_info = self.get_joint_info(body, joint)
        return joint_info.parentFramePos, joint_info.parentFrameOrn

    def violates_limit(self, body, joint, value):
        if not self.is_circular(body, joint):
            lower, upper = self.get_joint_limits(body, joint)
            if (value < lower) or (upper < value):
                return True
        return False

    def violates_limits(self, body, joints, values):
        return any(self.violates_limit(body, joint, value) for joint, value in zip(joints, values))

    def wrap_joint(self, body, joint, value):
        if self.is_circular(body, joint):
            return wrap_angle(value)
        return value

    def get_difference_fn(self, body, joints):
        def fn(q2, q1):
            difference = []
            for joint, value2, value1 in zip(joints, q2, q1):
                difference.append(circular_difference(value2, value1)
                                  if self.is_circular(body, joint)
                                  else (np.array(value2) - np.array(value1)))
            return list(difference)

        return fn

    def get_distance_fn(self, body, joints, weights=None):
        if weights is None:
            weights = 1 * np.ones(len(joints))
        difference_fn = self.get_difference_fn(body, joints)

        def fn(self, q1, q2):
            diff = np.array(difference_fn(q2, q1))
            return np.sqrt(np.dot(weights, diff * diff))

        return fn

    def get_sample_fn(self, body, joints):
        def fn(self, ):
            values = []
            for joint in joints:
                limits = CIRCULAR_LIMITS if self.is_circular(body, joint) \
                    else self.get_joint_limits(body, joint)
                values.append(np.random.uniform(*limits))
            return list(values)

        return fn

    def get_extend_fn(self, body, joints, resolutions=None):
        if resolutions is None:
            resolutions = 0.05 * np.ones(len(joints))
        difference_fn = self.get_difference_fn(body, joints)

        def fn(self, q1, q2):
            steps = np.abs(np.divide(difference_fn(q2, q1), resolutions))
            num_steps = int(max(steps))
            waypoints = []
            diffs = difference_fn(q2, q1)
            for i in range(num_steps):
                waypoints.append(
                    list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + q1))
            return waypoints

        return fn

    # Collision

    def pairwise_collision(self, body1, body2, max_distance=MAX_DISTANCE):  # 10000
        # getContactPoints
        # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0
        return self.p.getContactPoints(body1, body2) != ()

    def pairwise_link_collision(self, body1, link1, body2, link2, max_distance=MAX_DISTANCE):  # 10000
        # return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
        #                               linkIndexA=link1, linkIndexB=link2)) != 0  # getContactPoints
        return self.p.getContactPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexA=link1,
            linkIndexB=link2)

    def single_collision(self, body1, **kwargs):
        for body2 in self.get_bodies():
            if (body1 != body2) and self.pairwise_collision(body1, body2, **kwargs):
                return True
        return False

    def all_collision(self, **kwargs):
        bodies = self.get_bodies()
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                if self.pairwise_collision(bodies[i], bodies[j], **kwargs):
                    return True
        return False

    def get_moving_links(self, body, moving_joints):
        moving_links = list(moving_joints)
        for link in moving_joints:
            moving_links += self.get_link_descendants(body, link)
        return list(set(moving_links))

    def get_moving_pairs(self, body, moving_joints):
        moving_links = self.get_moving_links(body, moving_joints)
        for i in range(len(moving_links)):
            link1 = moving_links[i]
            ancestors1 = set(self.get_joint_ancestors(body, link1)) & set(moving_joints)
            for j in range(i + 1, len(moving_links)):
                link2 = moving_links[j]
                ancestors2 = set(self.get_joint_ancestors(
                    body, link2)) & set(moving_joints)
                if ancestors1 != ancestors2:
                    yield link1, link2

    def get_self_link_pairs(self, body, joints, disabled_collisions=set()):
        moving_links = self.get_moving_links(body, joints)
        fixed_links = list(set(self.get_links(body)) - set(moving_links))
        check_link_pairs = list(product(moving_links, fixed_links))
        if True:
            check_link_pairs += list(self.get_moving_pairs(body, joints))
        else:
            check_link_pairs += list(combinations(moving_links, 2))
        check_link_pairs = list(
            filter(lambda pair: not self.are_links_adjacent(body, *pair), check_link_pairs))
        check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                    (pair[::-1] not in disabled_collisions), check_link_pairs))
        return check_link_pairs

    def get_collision_fn(self, body, joints, obstacles, attachments, self_collisions, disabled_collisions):
        check_link_pairs = self.get_self_link_pairs(
            body, joints, disabled_collisions) if self_collisions else []
        moving_bodies = [body] + [attachment.child for attachment in attachments]
        if obstacles is None:
            obstacles = list(set(self.get_bodies()) - set(moving_bodies))
        # + list(combinations(moving_bodies, 2))
        check_body_pairs = list(product(moving_bodies, obstacles))

        def collision_fn(self, q):
            if self.violates_limits(body, joints, q):
                return True
            self.set_joint_positions(body, joints, q)
            for attachment in attachments:
                attachment.assign()
            for link1, link2 in check_link_pairs:
                if self.pairwise_link_collision(body, link1, body, link2):
                    return True
            return any(self.pairwise_collision(*pair) for pair in check_body_pairs)

        return collision_fn

    # Bodies

    def get_bodies(self, ):
        return [self.p.getBodyUniqueId(i)
                for i in range(self.p.getNumBodies())]

    def get_body_info(self, body):
        return BodyInfo(*self.p.getBodyInfo(body))

    def get_base_name(self, body):
        return self.get_body_info(body).base_name.decode(encoding='UTF-8')

    def get_body_name(self, body):
        return self.get_body_info(body).body_name.decode(encoding='UTF-8')

    def get_name(self, body):
        name = self.get_body_name(body)
        if name == '':
            name = 'body'
        return '{}{}'.format(name, int(body))

    def has_body(self, name):
        try:
            self.body_from_name(name)
        except ValueError:
            return False
        return True

    def body_from_name(self, name):
        for body in self.get_bodies():
            if self.get_body_name(body) == name:
                return body
        raise ValueError(name)

    def remove_body(self, body):
        return self.p.removeBody(body)

    def get_body_pos(self, body):
        return self.get_body_pose(body)[0]

    def get_body_quat(self, body):
        return self.get_body_pose(body)[1]  # [x,y,z,w]

    def set_pose(self, body, pose):
        (point, quat) = pose
        self.p.resetBasePositionAndOrientation(body, point, quat)

    def set_point(self, body, point):
        self.set_pose(body, (point, self.get_body_quat(body)))

    def set_quat(self, body, quat):
        self.set_pose(body, (self.get_body_pos(body), quat))

    def is_rigid_body(self, body):
        for joint in self.get_joints(body):
            if self.is_movable(body, joint):
                return False
        return True

    def is_fixed_base(self, body):
        return self.get_mass(body) == STATIC_MASS

    def dump_body(self, body):
        print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
            body, self.get_body_name(body), self.is_rigid_body(body), self.is_fixed_base(body)))
        for joint in self.get_joints(body):
            print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
                joint, self.get_joint_name(
                    body, joint), JOINT_TYPES[self.get_joint_type(body, joint)],
                self.is_circular(body, joint), self.get_joint_limits(body, joint)))
        print('Link id: {} | Name: {} | Mass: {}'.format(-1,
                                                         self.get_base_name(body), self.get_mass(body)))
        for link in self.get_links(body):
            print('Link id: {} | Name: {} | Parent: {} | Mass: {}'.format(
                link, self.get_link_name(body, link), self.get_link_name(
                    body, self.get_link_parent(body, link)),
                self.get_mass(body, link)))
            # print(get_joint_parent_frame(body, link))
            # print(map(get_data_geometry, get_visual_data(body, link)))
            # print(map(get_data_geometry, get_collision_data(body, link)))

    def dump_world(self, ):
        for body in self.get_bodies():
            self.dump_body(body)
            print()

    def remove_all_bodies(self, ):
        for i in self.get_body_ids():
            self.p.removeBody(i)

    def reset_body_base(self, body, pose):
        self.p.resetBasePositionAndOrientation(body, pose[0], pose[1])

    def get_body_infos(self, ):
        """ Return all body info in a list """
        return [self.get_body_info(i) for i in self.get_body_ids()]

    def get_body_names(self, ):
        """ Return all body names in a list """
        return [bi.body_name for bi in self.get_body_infos()]

    def get_body_id(self, name):
        return self.get_body_names().index(name)

    def get_body_ids(self, ):
        return sorted([self.p.getBodyUniqueId(i) for i in range(self.p.getNumBodies())])

    def get_body_pose(self, body):
        raw = self.p.getBasePositionAndOrientation(body)
        position = list(raw[0])
        orn = list(raw[1])
        return [position, orn]

    # Control

    def control_joint(self, body, joint, value):
        return self.p.setJointMotorControl2(bodyUniqueId=body,
                                           jointIndex=joint,
                                           controlMode=self.p.POSITION_CONTROL,
                                           targetPosition=value,
                                           targetVelocity=0,
                                           maxVelocity=self.get_max_velocity(body, joint),
                                           force=self.get_max_force(body, joint))

    def control_joints(self, body, joints, positions, velocity=1.0, acceleration=2.0):
        return self.p.setJointMotorControlArray(body,
                                               joints,
                                               self.p.POSITION_CONTROL,
                                               targetPositions=positions,
                                               positionGains=[velocity] * len(joints))

    def forward_kinematics(self, body, joints=None, positions=None, eef_link=None):
        """
        Skip the resetJointState step,
        as it's not recommend to do this while running simulation.
        (https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#)
        """
        eef_link = self.get_num_joints(body) - 1 if eef_link is None else eef_link  # not true for dVRK models
        if joints is not None:
            # not recommend to resetJointState while running simulation
            old_positions = self.get_joint_positions(body, joints)
            self.set_joint_positions(body, joints, positions)
            eef_pose = self.get_link_pose(body, eef_link)
            self.set_joint_positions(body, joints, old_positions)
        else:
            eef_pose = self.get_link_pose(body, eef_link)
        return eef_pose

    def inverse_kinematics(self, body, eef_link, position, orientation=None):
        """
        Modify the threshold and iteration parameters
        """
        if orientation is None:
            jv = self.p.calculateInverseKinematics(bodyUniqueId=body,
                                                  endEffectorLinkIndex=eef_link,
                                                  targetPosition=position,
                                                  residualThreshold=1e-12,
                                                  maxNumIterations=1000,
                                                   bodyIndex=self.cid
                                                )
        else:
            jv = self.p.calculateInverseKinematics(bodyUniqueId=body,
                                                  endEffectorLinkIndex=eef_link,
                                                  targetPosition=position,
                                                  targetOrientation=orientation,
                                                  residualThreshold=1e-12,
                                                  maxNumIterations=1000,
                                                   bodyIndex=self.cid)
        return jv

    def get_link_name(self, body, link):
        if link == BASE_LINK:
            return self.get_base_name(body)
        return self.get_joint_info(body, link).linkName.decode('UTF-8')

    def get_link_parent(self, body, link):
        if link == BASE_LINK:
            return None
        return self.get_joint_info(body, link).parentIndex

    def link_from_name(self, body, name):
        if name == self.get_base_name(body):
            return BASE_LINK
        for link in self.get_joints(body):
            if self.get_link_name(body, link) == name:
                return link
        raise ValueError(body, name)

    def has_link(self, body, name):
        try:
            self.link_from_name(body, name)
        except ValueError:
            return False
        return True

    def get_link_state(self, body, link):
        return LinkState(*self.p.getLinkState(body, link))

    def get_com_pose(self, body, link):  # COM = center of mass
        link_state = self.get_link_state(body, link)
        return list(link_state.linkWorldPosition), list(link_state.linkWorldOrientation)

    def get_link_inertial_pose(self, body, link):
        link_state = self.get_link_state(body, link)
        return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation

    def get_link_pose(self, body, link):
        if link == BASE_LINK:
            return self.get_body_pose(body)
        # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        link_state = self.get_link_state(body, link)
        return [list(link_state.worldLinkFramePosition), list(link_state.worldLinkFrameOrientation)]

    def get_all_link_parents(self, body):
        return {link: self.get_link_parent(body, link) for link in self.get_links(body)}

    def get_all_link_children(self, body):
        children = {}
        for child, parent in self.get_all_link_parents(body).items():
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
        return children

    def get_link_children(self, body, link):
        children = self.get_all_link_children(body)
        return children.get(link, [])

    def get_link_ancestors(self, body, link):
        parent = self.get_link_parent(body, link)
        if parent is None:
            return []
        return self.get_link_ancestors(body, parent) + [parent]

    def get_joint_ancestors(self, body, link):
        return self.get_link_ancestors(body, link) + [link]

    def get_movable_joint_ancestors(self, body, link):
        return list(filter(lambda j: self.is_movable(body, j), self.get_joint_ancestors(body, link)))

    def get_link_descendants(self, body, link):
        descendants = []
        for child in self.get_link_children(body, link):
            descendants.append(child)
            descendants += self.get_link_descendants(body, child)
        return descendants

    def are_links_adjacent(self, body, link1, link2):
        return (self.get_link_parent(body, link1) == link2) or \
            (self.get_link_parent(body, link2) == link1)

    def get_adjacent_links(self, body):
        adjacent = set()
        for link in self.get_links(body):
            parent = self.get_link_parent(body, link)
            adjacent.add((link, parent))
            # adjacent.add((parent, link))
        return adjacent

    def get_adjacent_fixed_links(self, body):
        return list(filter(lambda item: not self.is_movable(body, item[0]),
                           self.get_adjacent_links(body)))

    def get_fixed_links(self, body):
        edges = defaultdict(list)
        for link, parent in self.get_adjacent_fixed_links(body):
            edges[link].append(parent)
            edges[parent].append(link)
        visited = set()
        fixed = set()
        for initial_link in self.get_links(body):
            if initial_link in visited:
                continue
            cluster = [initial_link]
            queue = deque([initial_link])
            visited.add(initial_link)
            while queue:
                for next_link in edges[queue.popleft()]:
                    if next_link not in visited:
                        cluster.append(next_link)
                        queue.append(next_link)
                        visited.add(next_link)
            fixed.update(product(cluster, cluster))
        return fixed

    def get_dynamics_info(self, body, link=BASE_LINK):
        return DynamicsInfo(*self.p.getDynamicsInfo(body, link))

    def get_mass(self, body, link=BASE_LINK):
        return self.get_dynamics_info(body, link).mass

    def get_joint_inertial_pose(self, body, joint):
        dynamics_info = self.get_dynamics_info(body, joint)
        return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn

    def reset_camera(self, yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
        self.p.resetDebugVisualizerCamera(
            cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)

    def get_camera(self, ):
        return CameraInfo(*self.p.getDebugVisualizerCamera())

    def render_image(self, width, height, view_matrix, proj_matrix, shadow=1):
        (_, _, px, depth, mask) = self.p.getCameraImage(width=width,
                                                    height=height,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    shadow=shadow,
                                                    lightDirection=(10, 0, 10),
                                                    renderer=self.p.ER_BULLET_HARDWARE_OPENGL
                                                    )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array, depth, mask
