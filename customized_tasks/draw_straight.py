import torch
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat
import numpy as np
from mani_skill.envs.tasks.drawing.draw import TableTopFreeDrawEnv
from mani_skill.utils.registration import register_env


@register_env("DrawStraight-denseR", max_episode_steps=300)
class DrawStraightEnv(TableTopFreeDrawEnv):
    """
    A custom environment that encourages drawing a long straight line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 记录轨迹（全部三维，不拆 xy）
        self.prev_tcp_pos = None
        self.start_tcp_pos = None

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)

        # 初始位置（shape: N x 3）
        self.prev_tcp_pos = self.agent.tcp.pose.p.clone().detach()
        self.start_tcp_pos = self.agent.tcp.pose.p.clone().detach()

    # ------------------------------------
    #             Reward Function
    # ------------------------------------
    def compute_dense_reward(self, obs, action, info):
        tcp_pos = self.agent.tcp.pose.p  # (N, 3)

        # =======================================================
        # 1. plane_reward: 越接近平面越好
        # =======================================================
        z = tcp_pos[..., 2]
        raw_plane = 1 - torch.tanh(10 * (z - (self.CANVAS_THICKNESS + 0.002)))
        plane_reward = torch.clamp(raw_plane, 0, 1)  # 正规化到 0~1

        # =======================================================
        # 2. straight_reward: 方向是否一致（cosine → 0~1）
        # =======================================================
        v = tcp_pos - self.prev_tcp_pos
        v_xy = v[..., :2]
        v_norm = torch.norm(v_xy, dim=-1) + 1e-6
        d = v_xy / v_norm.unsqueeze(-1)  # 当前方向

        total_disp = tcp_pos - self.start_tcp_pos
        total_xy = total_disp[..., :2]
        total_norm = torch.norm(total_xy, dim=-1) + 1e-6
        d_ref = total_xy / total_norm.unsqueeze(-1)  # 整体方向

        # cosine similarity ∈ [-1,1]
        raw_cos = (d * d_ref).sum(dim=-1)
        straight_reward = torch.clamp((raw_cos + 1) / 2.0, 0, 1)  # 映射到 0~1

        # =======================================================
        # 3. forward_reward: 是否向前推进（长度越大越好）
        # =======================================================
        # raw_forward ∈ [0, +∞)
        raw_forward = torch.sum(v_xy * d_ref, dim=-1)
        raw_forward = torch.clamp(raw_forward, min=0.0)

        # 关键：归一化到 0~1，避免 reward 爆炸
        # 假设每步最大可能推进 2cm = 0.02 m，你可以根据任务再调
        forward_norm_scale = 0.02
        forward_reward = torch.clamp(raw_forward / forward_norm_scale, 0, 1)

        # =======================================================
        # 组合奖励（每项都是 0~1）
        # =======================================================
        reward = (
            10.0 * plane_reward +
            4.0 * straight_reward +
            1.0 * forward_reward
        )

        # debug info
        if info is not None:
            info["plane_reward"] = plane_reward.mean().item()
            info["straight_reward"] = straight_reward.mean().item()
            info["forward_reward"] = forward_reward.mean().item()

        # update prev pos
        self.prev_tcp_pos = tcp_pos.clone().detach()

        return reward


    def compute_normalized_dense_reward(self, obs, action, info):
        reward = self.compute_dense_reward(obs, action, info)
        # 简单 normalize，使 reward 永远在 [0,1] 区间
        return torch.tanh(reward / 5.0)

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info):
        # 统一返回三维 tcp_pose，不返回 xy，保持一致性
        tcp_pose = self.agent.tcp.pose.raw_pose
        return dict(tcp_pose=tcp_pose)
