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

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    num_eval_envs: int = 2048
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
    
    # Load checkpoint once to get the agent structure
    dummy_env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    if isinstance(dummy_env.action_space, gym.spaces.Dict):
        dummy_env = FlattenActionSpaceWrapper(dummy_env)
    dummy_env = ManiSkillVectorEnv(dummy_env, 1, ignore_terminations=not args.eval_partial_reset)
    agent = Agent(dummy_env).to(device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint from {args.checkpoint}")
    dummy_env.close()

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)

    radii = np.arange(0.1, 0.8 + 0.001, 0.05)
    all_results = {}

    for radius in radii:
        print(f"Evaluating with radius {radius:.2f}")
        
        video_folder = f"runs/{run_name}/videos/radius_{radius:.2f}"
        
        eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
        if isinstance(eval_envs.action_space, gym.spaces.Dict):
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
        
        eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
        
        # Set radius
        eval_envs.base_env.radius = radius
        
        print("Evaluating")
        agent.eval()
        eval_obs, _ = eval_envs.reset(seed=args.seed)
        
        # Save initial state for replay
        initial_states = eval_envs.base_env.get_state().clone()
        
        tcp_history = []
        action_history = [] # Store actions for replay
        eval_metrics = defaultdict(list)
        num_episodes = 0
        
        # Track success once
        success_once = torch.zeros(args.num_eval_envs, dtype=torch.bool, device=device)
        
        start_pos = eval_envs.base_env.start_site.pose.p.clone()
        goal_pos = eval_envs.base_env.goal_site.pose.p.clone()
        
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                action = agent.get_action(eval_obs, deterministic=True)
                action_history.append(action.clone()) # Save action
                eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(action)
                
                # Update success_once
                current_success = eval_envs.base_env.evaluate()['success']
                success_once |= current_success
                
                tcp_pos = eval_envs.base_env.agent.tcp.pose.p.clone()
                tcp_history.append(tcp_pos)

                if "final_info" in eval_infos:
                    mask = eval_infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
        
        # Calculate MSE
        tcp_history = torch.stack(tcp_history) # (T, N, 3)
        vec_sg = goal_pos - start_pos # (N, 3)
        len_sg_sq = (vec_sg ** 2).sum(dim=1) # (N,)
        mse_per_env = torch.zeros(args.num_eval_envs, device=device)
        
        for t in range(args.num_eval_steps):
            p = tcp_history[t]
            vec_sp = p - start_pos
            proj_coef = (vec_sp * vec_sg).sum(dim=1) / (len_sg_sq + 1e-8)
            closest_point = start_pos + vec_sg * proj_coef.unsqueeze(1)
            dist_sq = ((p - closest_point) ** 2).sum(dim=1)
            mse_per_env += dist_sq
            
        mse_per_env /= args.num_eval_steps
        
        # Use success_once for sorting
        success_per_env = success_once
        
        print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps")
        radius_results = {}
        for k, v in eval_metrics.items():
            mean = torch.stack(v).float().mean().item()
            radius_results[f"eval_{k}_mean"] = mean
            logger.add_scalar(f"eval/{k}", mean, 0)
            print(f"eval_{k}_mean={mean}")
        
        radius_results["mse_mean"] = mse_per_env.mean().item()
        print(f"MSE mean: {radius_results['mse_mean']}")
        
        all_results[f"radius_{radius:.2f}"] = radius_results

        if args.capture_video:
            sort_keys = []
            for i in range(args.num_eval_envs):
                s = 1 if success_per_env[i].item() else 0
                m = mse_per_env[i].item()
                sort_keys.append((-s, m, i))
            
            sort_keys.sort()
            
            print(f"Debug: Top 5 sort_keys: {sort_keys[:5]}")
            
            # Filter: only take successful ones first
            successful_indices = [x for x in sort_keys if x[0] == -1]
            print(f"Debug: Found {len(successful_indices)} successful environments.")
            
            if len(successful_indices) >= 8:
                top_indices = [x[2] for x in successful_indices[:8]]
            else:
                top_indices = [x[2] for x in successful_indices]
                
                if len(top_indices) == 0:
                     # Fallback: draw top 3 best failures to debug
                     top_indices = [x[2] for x in sort_keys[:3]]
            
            print(f"Top indices for radius {radius:.2f} (Success count: {len(successful_indices)}): {top_indices}")
            
            # Re-run top environments for video recording
            if len(top_indices) > 0:
                print(f"Re-running {len(top_indices)} environments for video recording...")
                os.makedirs(video_folder, exist_ok=True)
                
                # Stack actions for easier indexing: (T, N, ActionDim)
                saved_actions = torch.stack(action_history)
                
                for rank, env_idx in enumerate(top_indices):
                    current_seed = args.seed + env_idx
                    
                    # Create single env
                    rec_env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
                    if isinstance(rec_env.action_space, gym.spaces.Dict):
                        rec_env = FlattenActionSpaceWrapper(rec_env)
                    
                    rec_env = RecordEpisode(rec_env, output_dir=video_folder, save_trajectory=False, max_steps_per_video=args.num_eval_steps, video_fps=30)
                    rec_env = ManiSkillVectorEnv(rec_env, 1, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
                    
                    rec_env.base_env.radius = radius
                    
                    # Reset and restore state
                    rec_env.reset(seed=current_seed)
                    
                    env_state = initial_states[env_idx].unsqueeze(0)
                    rec_env.base_env.set_state(env_state)
                    
                    for t in range(args.num_eval_steps):
                        with torch.no_grad():
                            # Replay action
                            action = saved_actions[t, env_idx].unsqueeze(0)
                            _, _, _, _, infos = rec_env.step(action)
                            
                            # Check success directly from base_env for the single env
                            if rec_env.base_env.evaluate()['success'].item():
                                break
                    
                    rec_env.close()
                
                    generated_video = os.path.join(video_folder, "0.mp4")
                    if os.path.exists(generated_video):
                        target_name = os.path.join(video_folder, f"rank{rank}_env{env_idx}.mp4")
                        shutil.move(generated_video, target_name)
        
        eval_envs.close()

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    with open(f"runs/{run_name}/eval_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Evaluation results saved to runs/{run_name}/eval_metrics.json")

    logger.close()