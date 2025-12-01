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


@register_env("DrawStraightLine-v1", max_episode_steps=300)
class DrawStraightLineEnv(BaseEnv):
    """
    Task:
      Robot must draw as long and as straight a line as possible.
      The start of the line is explicitly shown as a sphere.
    """

    MAX_DOTS = 300
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.01
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    goal_thresh = 0.025
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=320,
                height=240,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=960,
            fov=1.2,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
    
    
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

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.start_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[1, 0, 0, 1],
            name="start_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.start_site)
        
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            start_xy = xy + sampler.sample(radius, 100)
            goal_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = start_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.start_site.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = goal_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.goal_site.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            tcp = self.agent.tcp.pose.p              # (num_envs, 3)
            self.prev_tcp = tcp.clone()



    def evaluate(self):
        tcp = self.agent.tcp.pose.p          # (num_envs, 3)

        # distance to start
        start_pos = self.start_site.pose.p        # (num_envs, 3)
        start_dist = torch.linalg.norm(tcp - start_pos, dim=1)

        # distance to.goal_site
        goal_pos = self.goal_site.pose.p          # (num_envs, 3)
        goal_dist = torch.linalg.norm(tcp - goal_pos, dim=1)

        reached_start = start_dist < 0.02
        reached_goal = goal_dist < 0.02

        # must reach start first, then reach.goal_site
        success = reached_start & reached_goal

        return {
            "reached_start": reached_start,
            "reached_goal": reached_goal,
            "success": success,
        }


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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp.pose.p

        # ------------------------------------------------------
        # 1) reach start  (对应 StackCube 的 reaching reward)
        # ------------------------------------------------------
        start_pos = self.start_site.pose.p
        dist_to_start = torch.linalg.norm(tcp - start_pos, dim=1)
        reach_start_reward = 2 * (1 - torch.tanh(5 * dist_to_start))

        reward = reach_start_reward.clone()

        # 未到达 start：直接返回
        not_reached_start = ~info["reached_start"]
        if not_reached_start.any():
            self.prev_tcp = tcp.clone()
            return reward

        # ------------------------------------------------------
        # 2) approach.goal_site (对应 StackCube 的 place_reward)
        # ------------------------------------------------------
        goal_pos = self.goal_site.pose.p
        dist_to_goal = torch.linalg.norm(goal_pos - tcp, dim=1)
        approach_reward = 1 - torch.tanh(5 * dist_to_goal)

        reward[info["reached_start"]] = (4 + approach_reward)[info["reached_start"]]

        # ------------------------------------------------------
        # 3) forward movement bonus
        # ------------------------------------------------------
        move = tcp - self.prev_tcp
        goal_dir = goal_pos - tcp
        proj = torch.sum(move[:, :2] * goal_dir[:, :2], dim=1)
        forward_reward = torch.clamp(proj, min=0.0)

        reward[info["reached_start"]] += forward_reward[info["reached_start"]] * 5.0

        # ------------------------------------------------------
        # 4) success
        # ------------------------------------------------------
        reward[info["success"]] = 10.0

        self.prev_tcp = tcp.clone()
        return reward



    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
