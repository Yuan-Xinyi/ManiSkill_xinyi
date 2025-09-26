import math
import torch
from mani_skill.utils.registration import register_env
from .base_draw_shape_env import BaseDrawShapeEnv


# ---------------- Circle Environment ----------------
@register_env("DrawCircle-denseR", max_episode_steps=300)
class DrawCircleEnv(BaseDrawShapeEnv):
    """
    Task: draw a circle on canvas with a stick
    """

    def _init_shape_points(self) -> torch.Tensor:
        """
        Return circle points as (NUM_POINTS, 3).
        """
        theta = torch.linspace(0, 2 * math.pi, self.NUM_POINTS, device=self.device)
        circle_points = torch.stack([
            self.RADIUS * torch.cos(theta),
            self.RADIUS * torch.sin(theta),
            torch.ones_like(theta) * (self.CANVAS_THICKNESS + 0.001),
        ], dim=1)
        return circle_points

