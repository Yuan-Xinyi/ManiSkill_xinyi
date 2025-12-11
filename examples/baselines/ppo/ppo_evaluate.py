from collections import defaultdict
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import wandb

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv




@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    run_folder: Optional[str] = None
    """path to the run folder containing checkpoints (e.g. runs/eg5_...)"""
    output_file: str = "evaluation_results.jsonl"
    """output jsonl file name"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_eval_envs: int = 1024
    """the number of parallel evaluation environments"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ---------------customized tasks---------------
    if args.env_id == "DrawStraightLine":
        from customized_tasks import draw_straight
    else:
        print(f"Customized task for env_id {args.env_id}.")

    # env setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    
    # env setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    
    # Find all checkpoints
    checkpoints = []
    if args.run_folder:
        if not os.path.exists(args.run_folder):
             raise ValueError(f"Run folder {args.run_folder} does not exist")
        for f in os.listdir(args.run_folder):
            if f.startswith("ckpt_") and f.endswith(".pt"):
                try:
                    iter_num = int(f.split("_")[1].split(".")[0])
                    checkpoints.append((iter_num, os.path.join(args.run_folder, f)))
                except ValueError:
                    pass
        checkpoints.sort() # Sort by iteration
    elif args.checkpoint:
        checkpoints = [(0, args.checkpoint)]
    else:
        raise ValueError("Please provide --run-folder or --checkpoint")

    # Load checkpoint once to get the agent structure
    dummy_env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=not args.eval_partial_reset)
    agent = Agent(dummy_env).to(device)
    dummy_env.close()

    radii = np.arange(0.1, 0.6 + 0.001, 0.05)
    
    # Open output file
    output_path = args.output_file
    if args.run_folder:
        output_path = os.path.join(args.run_folder, args.output_file)
    
    print(f"Results will be saved to {output_path}")

    for iter_num, ckpt_path in checkpoints:
        print(f"Evaluating checkpoint: {ckpt_path}")
        agent.load_state_dict(torch.load(ckpt_path))
        agent.eval()
        
        model_results = {
            "checkpoint": ckpt_path,
            "iteration": iter_num,
            "metrics": {}
        }

        for radius in radii:
            print(f"  Radius: {radius:.2f}")
            
            # Create eval envs
            eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
            if isinstance(eval_envs.action_space, gym.spaces.Dict):
                eval_envs = FlattenActionSpaceWrapper(eval_envs)
            eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
            
            eval_envs.base_env.radius = radius
            eval_obs, _ = eval_envs.reset(seed=args.seed)
            
            eval_metrics = defaultdict(list)
            
            # Evaluation loop
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, _, _, _, eval_infos = eval_envs.step(action)
                    
                    if "final_info" in eval_infos:
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            
            # Aggregate metrics
            radius_metrics = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean().item()
                radius_metrics[k] = mean
            
            model_results["metrics"][f"radius_{radius:.2f}"] = radius_metrics
            eval_envs.close()
        
        # Append to JSONL
        with open(output_path, "a") as f:
            f.write(json.dumps(model_results) + "\n")

    # logger.close()