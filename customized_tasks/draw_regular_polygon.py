import math
import torch
from mani_skill.utils.registration import register_env
from .base_draw_shape_env import BaseDrawShapeEnv


# ---------------- Regular Polygon Environment ----------------
class DrawRegularPolygonEnv(BaseDrawShapeEnv):
    """
    Task: draw a regular n-gon (polygon with num_edges sides) on canvas with a stick.
    The polygon's vertices lie on a circle of radius self.RADIUS.
    """

    def __init__(self, *args, num_edges=8, **kwargs):
        self.num_edges = num_edges
        super().__init__(*args, **kwargs)

    def _init_shape_points(self) -> torch.Tensor:
        """
        Return polygon points as (NUM_POINTS, 3).
        Vertices are equally spaced on a circle of radius self.RADIUS.
        """
        self.NUM_POINTS = 80
        points_per_edge = self.NUM_POINTS // self.num_edges

        # 顶点角度
        angles = torch.linspace(0, 2 * math.pi, self.num_edges + 1, device=self.device)[:-1]

        # 外接圆顶点
        vertices = torch.stack([
            self.RADIUS * torch.cos(angles),
            self.RADIUS * torch.sin(angles),
            torch.ones_like(angles) * (self.CANVAS_THICKNESS + 0.001),
        ], dim=1)

        # 沿每条边插值
        points = []
        for i in range(self.num_edges):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % self.num_edges]
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_points = v1 * (1 - t) + v2 * t
            points.append(edge_points)

        shape_points = torch.cat(points, dim=0)

        # 调整到固定 NUM_POINTS
        if shape_points.shape[0] > self.NUM_POINTS:
            shape_points = shape_points[:self.NUM_POINTS]
        elif shape_points.shape[0] < self.NUM_POINTS:
            pad = self.NUM_POINTS - shape_points.shape[0]
            shape_points = torch.cat([shape_points, shape_points[:pad]], dim=0)

        return shape_points


# ---------------- 注册几个常用环境 ----------------
@register_env("DrawTriangle-denseR", max_episode_steps=300)
class DrawTriangleEnv(DrawRegularPolygonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_edges=3, **kwargs)


@register_env("DrawOctagon-denseR", max_episode_steps=300)
class DrawOctagonEnv(DrawRegularPolygonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_edges=8, **kwargs)


@register_env("DrawHexadecagon-denseR", max_episode_steps=300)  # 16 边形
class DrawHexadecagonEnv(DrawRegularPolygonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_edges=16, **kwargs)

@register_env("DrawDodecagon-denseR", max_episode_steps=300)  # 12 边形
class DrawDodecagonEnv(DrawRegularPolygonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_edges=12, **kwargs)
