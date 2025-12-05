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


@register_env("DrawStraightLine", max_episode_steps=100)
class DrawStraightLineEnv(BaseEnv):
    """
    Robot draws a straight line:
    Step 1 — reach start sphere
    Step 2 — move toward goal sphere
    """

    MAX_DOTS = 100
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.01
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    goal_thresh = 0.01
    
    
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    def __init__(self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.02, **kwargs):
        self.has_touched_start = None
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
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

            # ----- random start + goal -----
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            start_xy = xy + sampler.sample(radius, 100)
            goal_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02

            xyz[:, :2] = start_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.start_site.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = goal_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.goal_site.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # ----- reset state flags -----
            # self.has_touched_start = torch.zeros(b, dtype=torch.bool, device=self.device)
            # Initialize has_touched_start on first reset
            if self.has_touched_start is None:
                self.has_touched_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            # Reset only selected environments
            self.has_touched_start[env_idx] = False


            # ----- init prev tcp -----
            tcp = self.agent.tcp.pose.p
            self.prev_tcp = tcp.clone()

    # ----------------------------------------------------------
    # Evaluate (NO history here)
    # ----------------------------------------------------------
    def evaluate(self):
        tcp = self.agent.tcp.pose.p
        goal_pos = self.goal_site.pose.p
        goal_dist = torch.linalg.norm(tcp - goal_pos, dim=1)
        reached_goal = goal_dist < 0.01
        success = self.has_touched_start & reached_goal

        return {
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
        self.has_touched_start |= (dist_to_start < 0.01)

        reach_start_reward = 2 * (1 - torch.tanh(5 * dist_to_start))
        reward = reach_start_reward.clone()
        reward[self.has_touched_start] = 2.0 + reach_start_reward[self.has_touched_start]


        # ==========================================================
        # 2) approach & move to goal (only when has_touched_start=True)
        # ==========================================================
        mask = self.has_touched_start.float()
        dist_to_goal = torch.linalg.norm(goal_pos - tcp, dim=1)
        approach_reward = 2 * (1 - torch.tanh(5 * dist_to_goal))
        reward += mask * approach_reward

        dist_to_goal_prev = torch.linalg.norm(self.prev_tcp - goal_pos, dim=1)
        dist_reduction = dist_to_goal_prev - dist_to_goal   
        dist_reduction_reward = torch.clamp(dist_reduction, min=0.0) * 10.0
        reward += mask * dist_reduction_reward

        move = tcp - self.prev_tcp
        dir_start = (start_pos - tcp)
        dir_goal  = (goal_pos  - tcp)
        dir_start = dir_start / (torch.norm(dir_start, dim=1, keepdim=True) + 1e-6)
        dir_goal  = dir_goal  / (torch.norm(dir_goal,  dim=1, keepdim=True) + 1e-6)

        desired_dir = dir_start.clone()
        desired_dir[self.has_touched_start] = dir_goal[self.has_touched_start]

        proj = torch.sum(move * desired_dir, dim=1)
        forward_reward = torch.clamp(proj, min=0.0)
        reward += mask * forward_reward * 20.0 

        # ----- near goal bonus -----
        close_bonus = (dist_to_goal < 0.05).float() * 3.0
        reward += mask * close_bonus

        # ----- success -----
        success = self.has_touched_start & (dist_to_goal < 0.01)
        reward[success] = 8.0

        self.prev_tcp = tcp.clone()
        return reward


    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 8.0