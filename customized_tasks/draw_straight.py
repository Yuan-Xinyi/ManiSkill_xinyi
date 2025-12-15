import math
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from typing import Any, Union

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors


@register_env("DrawStraightLine", max_episode_steps=50)
class DrawStraightLineEnv(BaseEnv):
    """
    Robot draws a straight line:
    Step 1 — reach start sphere
    Step 2 — move toward goal sphere
    """

    MAX_DOTS = 50
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.01
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    goal_thresh = 0.02
    radius = 0.05
    dist_thresh = 0.02
    
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.02, **kwargs):
        self.has_touched_start = None
        self.has_touched_goal = None
        self.max_deviation = None
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ----------------------------------------------------------
    # Simulation + Cameras
    # ----------------------------------------------------------
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.2, 1.0], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ----------------------------------------------------------
    # Scene
    # ----------------------------------------------------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.start_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[1, 0, 0, 1],
            name="start_site", body_type="kinematic",
            add_collision=False, initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.start_site)

        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1],
            name="goal_site", body_type="kinematic",
            add_collision=False, initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            X_MIN = -0.42
            X_MAX = 0.085
            Y_MIN = -0.65
            Y_MAX = 0.65

            # ----- random start + goal -----
            max_offset = self.radius
            center_x_min = X_MIN + max_offset
            center_x_max = X_MAX - max_offset
            center_y_min = Y_MIN + max_offset
            center_y_max = Y_MAX - max_offset
            if center_x_max <= center_x_min and center_y_max <= center_y_min:
                raise ValueError("The defined radius is too large for the table size.")

            center_x = torch.rand((b, 1), device=self.device) * (center_x_max - center_x_min) + center_x_min
            center_y = torch.rand((b, 1), device=self.device) * (center_y_max - center_y_min) + center_y_min
            xy = torch.cat([center_x, center_y], dim=1)
            
            # Sample radius uniformly from [0.05, self.radius]
            evaluate = False
            if evaluate:
                radius = torch.full((b, 1), self.radius, device=self.device)
            else:
                min_radius = 0.05
                radius = torch.rand((b, 1), device=self.device) * (self.radius - min_radius) + min_radius

            theta = torch.rand((b, 1), device=self.device) * 2 * math.pi

            offset = torch.cat([radius * torch.cos(theta),
                                radius * torch.sin(theta)], dim=1)
            start_xy = xy + offset
            goal_xy  = xy - offset

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xyz[:, :2] = start_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.start_site.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = goal_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.goal_site.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # ----- reset state flags -----
            if self.has_touched_start is None:
                self.has_touched_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.has_touched_start[env_idx] = False

            if self.has_touched_goal is None:
                self.has_touched_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.has_touched_goal[env_idx] = False

            # ----- init customized variables -----
            if self.max_deviation is None:
                self.max_deviation = torch.zeros(self.num_envs, device=self.device)
            self.max_deviation[env_idx] = 0

    # ----------------------------------------------------------
    # Evaluate (NO history here)
    # ----------------------------------------------------------
    def evaluate(self):
        tcp = self.agent.tcp.pose.p
        goal_pos = self.goal_site.pose.p
        goal_dist = torch.linalg.norm(tcp - goal_pos, dim=1)
        reached_goal = goal_dist < self.dist_thresh
        success = self.has_touched_start & reached_goal & (self.max_deviation < self.dist_thresh)
        # print(f'max deviation: {self.max_deviation.cpu().numpy()}')

        return {
            "has_reached_start": self.has_touched_start,
            "has_reached_goal": self.has_touched_goal,
            "max_deviation": self.max_deviation,
            "success": success,
        }

    # ----------------------------------------------------------
    # Observations
    # ----------------------------------------------------------
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                start_pose=self.start_site.pose.raw_pose,
                goal_pose=self.goal_site.pose.raw_pose,
                tcp_to_start_pos=self.start_site.pose.p - self.agent.tcp.pose.p,
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                start_to_goal_pos=self.goal_site.pose.p - self.start_site.pose.p,
            )
        return obs

    # ----------------------------------------------------------
    # Reward
    # ----------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp.pose.p
        start_pos = self.start_site.pose.p
        goal_pos = self.goal_site.pose.p

        # -----------------------------------------
        # 1) reach start
        # -----------------------------------------
        dist_to_start = torch.linalg.norm(tcp - start_pos, dim=1)
        self.has_touched_start |= (dist_to_start < self.dist_thresh)
        mask = self.has_touched_start.float()

        reach_start_reward = 2 * (1 - torch.tanh(5 * dist_to_start))  # range (0,4)
        reward = reach_start_reward.clone()
        reward[self.has_touched_start] =  4.0

        # -----------------------------------------
        # 2) approach & move to goal (only when has_touched_start=True)
        # -----------------------------------------
        dist_to_goal = torch.linalg.norm(goal_pos - tcp, dim=1)
        approach_reward = 2 * (1 - torch.tanh(5 * dist_to_goal))  # range (0,4)
        reward += mask * approach_reward

        # ----- movement along the line -----
        # Calculate deviation from the line segment [start_pos, goal_pos]
        line_vec = goal_pos - start_pos
        line_len_sq = (line_vec ** 2).sum(dim=1)
        
        point_vec = tcp - start_pos
        t = (point_vec * line_vec).sum(dim=1) / (line_len_sq + 1e-8)
        t_clamped = torch.clamp(t, 0.0, 1.0)
        
        closest_point = start_pos + t_clamped.unsqueeze(1) * line_vec
        deviation = torch.linalg.norm(tcp - closest_point, dim=1)
        update_deviation = (deviation > self.max_deviation) & self.has_touched_start
        self.max_deviation[update_deviation] = deviation[update_deviation]

        deviation_penalty = 10.0 * deviation
        reward -= mask * deviation_penalty

        # # ----- near goal bonus -----
        # close_bonus = (dist_to_goal < 0.05).float() * 3.0
        # reward += mask * close_bonus

        # ----- success -----
        success = self.has_touched_start & (dist_to_goal < self.dist_thresh) & (self.max_deviation < self.dist_thresh)
        reward[success] = 20.0
        qvel = self.agent.robot.get_qvel() 
        rot_penalty = torch.norm(qvel, dim=1)
        reward -= 0.1 * rot_penalty
        reward[success] -= rot_penalty[success]
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5