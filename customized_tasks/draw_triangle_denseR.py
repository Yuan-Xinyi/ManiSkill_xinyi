import math
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

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

# ---------------- Curriculum Scheduler ----------------
class CurriculumScheduler:
    def __init__(self):
        self.curr_level = 0
        self.last_level = -1

    def update(self, step: int):
        """Update curriculum level based on training step"""
        if step < 2e4:
            self.curr_level = 0
        elif step < 4e4:
            self.curr_level = 1
        elif step < 6e4:
            self.curr_level = 2
        elif step < 8e4:
            self.curr_level = 3
        elif step < 1e5:
            self.curr_level = 4
        else:
            self.curr_level = 5

        if self.curr_level != self.last_level:
            print(f"[Curriculum] Step={step} → Entering Level {self.curr_level}")
            print(f"           Params: {self.get_params()}")
            self.last_level = self.curr_level

    def get_params(self):
        """Return parameters for current curriculum level"""
        if self.curr_level == 0:
            return dict(sigma=0.05, threshold=0.05,
                        w_shape=1.0, w_cover=0.3, w_progress=0.0,
                        w_cont=0.0, w_back=0.0)
        elif self.curr_level == 1:
            return dict(sigma=0.04, threshold=0.045,
                        w_shape=1.0, w_cover=0.5, w_progress=0.002,
                        w_cont=0.3, w_back=0.001)
        elif self.curr_level == 2:
            return dict(sigma=0.03, threshold=0.040,
                        w_shape=1.0, w_cover=0.5, w_progress=0.002,
                        w_cont=0.3, w_back=0.001)
        elif self.curr_level == 3:
            return dict(sigma=0.02, threshold=0.035,
                        w_shape=1.0, w_cover=0.5, w_progress=0.002,
                        w_cont=0.3, w_back=0.001)
        elif self.curr_level == 4:
            return dict(sigma=0.015, threshold=0.03,
                        w_shape=1.0, w_cover=0.5, w_progress=0.002,
                        w_cont=0.3, w_back=0.001)
        else:
            return dict(sigma=0.01, threshold=0.03,
                        w_shape=1.0, w_cover=0.5, w_progress=0.002,
                        w_cont=0.3, w_back=0.001)


@register_env("DrawTriangle-denseR", max_episode_steps=300)
class DrawTriangleEnv(BaseEnv):
    r"""
    Task: draw a triangle on canvas with a stick
    """

    MAX_DOTS = 300
    DOT_THICKNESS = 0.003
    CANVAS_THICKNESS = 0.02
    BRUSH_RADIUS = 0.01
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    NUM_POINTS = 250   # 用于显示和 reward 的固定点数

    SUPPORTED_ROBOTS: ["panda_stick"]  # type: ignore
    agent: PandaStick

    def __init__(self, *args, robot_uids="panda_stick", **kwargs):
        self.scheduler = CurriculumScheduler()
        self.global_step = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self._reward_mode = 'normalized_dense'

    # ---------------- Sim config ----------------
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
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig("render_camera", pose=pose,
                            width=1280, height=960, fov=1.2, near=0.01, far=100)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return [CameraConfig("base_camera", pose=pose, width=320, height=240, fov=1.2, near=0.01, far=100)]


    # ---------------- Load scene ----------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        # Canvas
        canvas = self.scene.create_actor_builder()
        canvas.add_box_visual(half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
                              material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]))
        canvas.add_box_collision(half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2])
        canvas.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
        self.canvas = canvas.build_static(name="canvas")

        # ---- Triangle points (替换 Circle points) ----
        scale = 1.5
        verts = torch.tensor([[0, 0.15*scale, self.CANVAS_THICKNESS + 0.001],
                              [-0.15*scale, -0.15*scale, self.CANVAS_THICKNESS + 0.001],
                              [0.15*scale, -0.15*scale, self.CANVAS_THICKNESS + 0.001]], device=self.device)

        # 沿三条边插值生成 NUM_POINTS
        edge_points = []
        n_each = self.NUM_POINTS // 3
        for i in range(3):
            start, end = verts[i], verts[(i+1) % 3]
            t = torch.linspace(0, 1, n_each, device=self.device).unsqueeze(1)
            t = t**2
            edge_points.append((1 - t) * start + t * end)
        triangle_points = torch.cat(edge_points, dim=0)

        self.original_triangle_points = triangle_points.cpu().numpy()
        self.triangles = triangle_points[:, :2].unsqueeze(0).repeat(self.num_envs, 1, 1)

        # ---- Goal visualization ----
        self.goal_points = []
        for i, p in enumerate(triangle_points.cpu().numpy()):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.002, 
                                      material=sapien.render.RenderMaterial(base_color=[0, 0, 0, 1]))
            builder.initial_pose = sapien.Pose(p=p)
            actor = builder.build_kinematic(name=f"goal_triangle_point_{i}")
            self.goal_points.append(actor)

        # 初始化 dots
        self.dots = []
        for i in range(self.MAX_DOTS):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=self.BRUSH_RADIUS,
                half_length=self.DOT_THICKNESS / 2,
                material=sapien.render.RenderMaterial(base_color=self.BRUSH_COLORS[0]),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            actor = builder.build_kinematic(name=f"dot_{i}")
            self.dots.append(actor)

        # Coverage buffer
        self.ref_dist = torch.zeros((self.num_envs, self.NUM_POINTS), device=self.device, dtype=torch.bool)

    # ---------------- Reset ----------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.draw_step = 0
        self.table_scene.initialize(env_idx)
        for dot in self.dots:
            dot.set_pose(sapien.Pose(p=[0, 0, -self.DOT_THICKNESS], q=euler2quat(0, math.pi/2, 0)))
        self.ref_dist[env_idx] = torch.zeros((len(env_idx), self.NUM_POINTS), device=self.device, dtype=torch.bool)

        # update curriculum
        self.scheduler.update(self.global_step)

    # ---------------- After control ----------------
    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        robot_touching = (self.agent.tcp.pose.p[:, 2] < self.CANVAS_THICKNESS + self.DOT_THICKNESS + 0.005)
        brush_pos = torch.zeros((self.num_envs, 3), device=self.device)
        brush_pos[:, 2] = -self.DOT_THICKNESS
        brush_pos[robot_touching, :2] = self.agent.tcp.pose.p[robot_touching, :2]
        brush_pos[robot_touching, 2] = self.DOT_THICKNESS / 2 + self.CANVAS_THICKNESS

        new_dot_pos = Pose.create_from_pq(brush_pos, euler2quat(0, math.pi / 2, 0))
        self.dots[self.draw_step].set_pose(new_dot_pos)
        self.draw_step += 1

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

        self.global_step += 1

    # ---------------- Reward ----------------
    def compute_dense_reward(self, obs=None, action=None, info=None):
        params = self.scheduler.get_params()
        sigma = params["sigma"]
        threshold = params["threshold"]
        w_shape, w_cover, w_progress, w_cont, w_back = (
            params["w_shape"], params["w_cover"], params["w_progress"], params["w_cont"], params["w_back"]
        )

        reward = torch.zeros(self.num_envs, device=self.device)

        brush_pos = self.agent.tcp.pose.p
        brush_xy = brush_pos[:, :2]
        brush_z = brush_pos[:, 2]

        # shape reward
        dist = torch.cdist(brush_xy.unsqueeze(1), self.triangles)
        min_dist, min_idx = dist.min(dim=2)
        min_dist = min_dist.squeeze(-1)
        min_idx = min_idx.squeeze(-1)

        ## this is the soft boundary to avoid suddenly 0 shape reward
        z_factor = torch.exp(- ((brush_z - self.CANVAS_THICKNESS) ** 2) / (2 * (0.02 ** 2)))
        shape_reward = w_shape * torch.exp(- (min_dist ** 2) / (2 * sigma ** 2)) * z_factor

        # coverage reward
        near_goal = dist.squeeze(1) < threshold
        new_cover = torch.logical_and(near_goal, ~self.ref_dist)
        cover_reward = (w_cover * new_cover.float().sum(dim=1))
        self.ref_dist = torch.logical_or(self.ref_dist, near_goal)

        # progress reward + back penalty
        if not hasattr(self, "last_progress"):
            self.last_progress = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        progress = min_idx
        progress_delta = (progress - self.last_progress)
        forward = torch.clamp(progress_delta, min=0)
        backward = torch.clamp(-progress_delta, min=0)

        progress_reward = w_progress * forward.float()
        back_penalty = - w_back * backward.float()

        self.last_progress = progress

        # coverage ratio
        coverage_ratio = self.ref_dist.float().mean(dim=1)
        coverage_bonus = 0.5 * coverage_ratio

        # continuity reward
        cont_reward = torch.zeros_like(reward)
        if self.draw_step > 1:
            prev_dot = self.dots[self.draw_step - 2].pose.p[:, :2]
            curr_dot = brush_xy
            dot_dist = torch.norm(curr_dot - prev_dot, dim=1)
            cont_reward = w_cont * torch.exp(-50 * (dot_dist - 0.5 * self.DOT_THICKNESS) ** 2)

        # action penalty
        action_penalty = torch.zeros_like(reward)
        if action is not None:
            action_penalty = -0.01 * torch.norm(action, dim=1)

        # final reward
        reward = shape_reward + cover_reward + progress_reward + back_penalty + coverage_bonus + cont_reward + action_penalty

        # --- logging info for wandb ---
        if info is not None:
            info["shape_reward"] = shape_reward.mean().item()
            info["cover_reward"] = cover_reward.mean().item()
            info["progress_reward"] = progress_reward.mean().item()
            info["back_penalty"] = back_penalty.mean().item()
            info["coverage_bonus"] = coverage_bonus.mean().item()
            info["continuity"] = cont_reward.mean().item()
            info["action_penalty"] = action_penalty.mean().item()
            info["curriculum_level"] = self.scheduler.curr_level

        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 8
