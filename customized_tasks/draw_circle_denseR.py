import math
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig


@register_env("DrawCircle-denseR", max_episode_steps=300)
class DrawCircleEnv(BaseEnv):
    r"""
    **Task Description:**
    Instantiates a table with a white canvas on it and a goal circle with an outline.
    A robot with a stick is to draw the circle with a red line.

    **Randomizations:**
    - the goal circle's position on the xy-plane is randomized
    - the goal circle's z-rotation is randomized in range [0, 2 pi]

    **Success Conditions:**
    - the drawn points by the robot are within a euclidean distance of 0.05m with points on the goal circle
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/DrawCircle-v1_rt.mp4"

    MAX_DOTS = 300
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.01
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    THRESHOLD = 0.025
    RADIUS = 0.15
    NUM_POINTS = 100

    SUPPORTED_ROBOTS: ["panda_stick"]  # type: ignore
    agent: PandaStick

    def __init__(self, *args, robot_uids="panda_stick", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self._reward_mode = 'normalized_dense'

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=320, height=240, fov=1.2, near=0.01, far=100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=1280, height=960, fov=1.2, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        """Load table, canvas, dots, and circular goal points"""
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        # Build canvas
        self.canvas = self.scene.create_actor_builder()
        self.canvas.add_box_visual(
            half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
        )
        self.canvas.add_box_collision(half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2])
        self.canvas.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
        self.canvas = self.canvas.build_static(name="canvas")

        # Create circle points
        theta = torch.linspace(0, 2 * math.pi, self.NUM_POINTS, device=self.device)
        circle_points = torch.stack([
            self.RADIUS * torch.cos(theta),
            self.RADIUS * torch.sin(theta),
            torch.ones_like(theta) * (self.CANVAS_THICKNESS + 0.001)
        ], dim=1).cpu().numpy()
        self.original_circle_points = circle_points

        # Visualize circle points
        for i, p in enumerate(circle_points):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.002, material=sapien.render.RenderMaterial(base_color=[0,0,0,1]))
            builder.initial_pose = sapien.Pose(p=p)
            builder.build_kinematic(name=f"goal_circle_point_{i}")

        # Initialize dots
        self.dots = []
        color_choices = torch.randint(0, len(self.BRUSH_COLORS), (self.num_envs,))
        for i in range(self.MAX_DOTS):
            actors = []
            if len(self.BRUSH_COLORS) > 1:
                for env_idx in range(self.num_envs):
                    builder = self.scene.create_actor_builder()
                    builder.add_cylinder_visual(
                        radius=self.BRUSH_RADIUS,
                        half_length=self.DOT_THICKNESS/2,
                        material=sapien.render.RenderMaterial(base_color=self.BRUSH_COLORS[color_choices[env_idx]])
                    )
                    builder.set_scene_idxs([env_idx])
                    builder.initial_pose = sapien.Pose(p=[0,0,0.1])
                    actor = builder.build_kinematic(name=f"dot_{i}_{env_idx}")
                    actors.append(actor)
                self.dots.append(Actor.merge(actors))
            else:
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_visual(
                    radius=self.BRUSH_RADIUS,
                    half_length=self.DOT_THICKNESS/2,
                    material=sapien.render.RenderMaterial(base_color=self.BRUSH_COLORS[0])
                )
                builder.initial_pose = sapien.Pose(p=[0,0,0.1])
                actor = builder.build_kinematic(name=f"dot_{i}")
                self.dots.append(actor)

        # Initialize reference distance and triangles (circle xy)
        self.ref_dist = torch.zeros((self.num_envs, self.NUM_POINTS), device=self.device, dtype=torch.bool)
        self.triangles = torch.from_numpy(circle_points[:, :2]).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.draw_step = 0
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reset dots positions
            for dot in self.dots:
                dot.set_pose(sapien.Pose(p=[0,0,-self.DOT_THICKNESS], q=euler2quat(0, math.pi/2, 0)))

            # Reset ref_dist
            self.ref_dist[env_idx] = torch.zeros((b, self.NUM_POINTS), dtype=torch.bool, device=self.device)

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        robot_touching_table = (self.agent.tcp.pose.p[:,2] < self.CANVAS_THICKNESS + self.DOT_THICKNESS + 0.005)
        robot_brush_pos = torch.zeros((self.num_envs,3), device=self.device)
        robot_brush_pos[:,2] = -self.DOT_THICKNESS
        robot_brush_pos[robot_touching_table,:2] = self.agent.tcp.pose.p[robot_touching_table,:2]
        robot_brush_pos[robot_touching_table,2] = self.DOT_THICKNESS/2 + self.CANVAS_THICKNESS
        new_dot_pos = Pose.create_from_pq(robot_brush_pos, euler2quat(0, math.pi/2,0))
        self.dots[self.draw_step].set_pose(new_dot_pos)
        self.draw_step += 1

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def success_check(self):
        # 至少画了一半点才开始判定
        if self.draw_step > 0.5 * self.NUM_POINTS:
            # 计算覆盖率：ref_dist 里记录了哪些轨迹点被覆盖过
            coverage_ratio = self.ref_dist.float().mean(dim=1)

            # 判定条件：覆盖率大于阈值就算成功
            # 例如覆盖 >=80% 轨迹点
            success_flags = coverage_ratio > 0.8

            return success_flags

        # 没画够点，不判定
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)



    def compute_dense_reward(self, obs=None, action=None, info=None):
        """
        通用轨迹奖励:
        - 轨迹贴合奖励：鼓励笔尖接近目标轨迹
        - 覆盖奖励：新覆盖点加分
        - 进度奖励：覆盖率越高奖励越大，且鼓励向前推进
        - 动作惩罚：减少抖动
        - 连续性奖励：鼓励画点之间紧密相连
        """
        reward = torch.zeros(self.num_envs, device=self.device)

        # 笔尖位置
        brush_pos = self.agent.tcp.pose.p
        brush_xy = brush_pos[:, :2]
        brush_z = brush_pos[:, 2]

        # ---- (1) Z 限制 ----
        z_mask = torch.abs(brush_z - self.CANVAS_THICKNESS) < 0.02

        # ---- (2) 轨迹贴合奖励 ----
        dist = torch.cdist(brush_xy.unsqueeze(1), self.triangles)  # [num_envs, 1, NUM_POINTS]
        min_dist, min_idx = dist.min(dim=2)  # [num_envs, 1]
        min_dist = min_dist.squeeze(-1)      # [num_envs]
        min_idx = min_idx.squeeze(-1)        # [num_envs]，记录当前位置在轨迹上的索引

        sigma = 0.02
        shape_reward = torch.exp(- (min_dist ** 2) / (2 * sigma ** 2))
        reward += 1.0 * shape_reward * z_mask.float()

        # ---- (3) 覆盖奖励 ----
        near_goal = dist.squeeze(1) < self.THRESHOLD
        new_cover = torch.logical_and(near_goal, ~self.ref_dist)
        cover_reward = new_cover.float().sum(dim=1) * 0.3
        reward += cover_reward

        # 更新覆盖点
        self.ref_dist = torch.logical_or(self.ref_dist, near_goal)

        # ---- (4) 进度奖励 ----
        # 保存上一步的进度（轨迹索引），奖励推进
        if not hasattr(self, "last_progress"):
            self.last_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        progress = min_idx  # 当前点的轨迹索引
        progress_delta = (progress - self.last_progress).clamp(min=0)  # 只奖励前进
        reward += 0.05 * progress_delta.float()

        self.last_progress = progress  # 更新进度

        # 另外：整体覆盖率奖励，鼓励完成度
        coverage_ratio = self.ref_dist.float().mean(dim=1)
        reward += 1.0 * coverage_ratio

        # ---- (5) 动作惩罚 ----
        if action is not None:
            reward -= 0.01 * torch.norm(action, dim=1)

        # ---- (6) 连续性奖励 ----
        if self.draw_step > 1:
            prev_dot_pos = self.dots[self.draw_step-2].pose.p[:, :2]
            curr_dot_pos = brush_xy
            dot_dist = torch.norm(curr_dot_pos - prev_dot_pos, dim=1)

            continuity_reward = torch.exp(-50 * (dot_dist - 0.5 * self.DOT_THICKNESS)**2)
            reward += 0.5 * continuity_reward

        return reward



    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 8