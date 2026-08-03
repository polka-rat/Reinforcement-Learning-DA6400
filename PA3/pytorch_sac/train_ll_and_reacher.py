#!/usr/bin/env python3
import builtins
import os
import sys
import time
from functools import partial

# Force immediate terminal output even under tmux/ssh/redirection.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

print = partial(builtins.print, flush=True)


def _early_log(msg):
    # Mirror startup diagnostics to a file in case stdout is swallowed by runner tooling.
    print(msg)
    try:
        startup_log_path = os.environ.get('TRAIN_STARTUP_LOG', 'train_startup.log')
        with open(startup_log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass


_early_log(f"[startup] train.py launched at {time.strftime('%Y-%m-%d %H:%M:%S')}")
_early_log(f"[startup] pid={os.getpid()} cwd={os.getcwd()}")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pickle as pkl
import inspect
import csv
from pathlib import Path

# Prefer an offscreen MuJoCo backend on headless machines so evaluation video
# capture does not crash through GLFW when no X display is available.
if 'MUJOCO_GL' not in os.environ and not os.environ.get('DISPLAY'):
    os.environ['MUJOCO_GL'] = 'egl'
if os.environ.get('MUJOCO_GL') == 'egl' and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import hydra
from omegaconf import OmegaConf
from hydra.utils import get_class
try:
    # Available in hydra-core 1.x.
    from hydra.core.hydra_config import HydraConfig
except Exception:
    HydraConfig = None
from env_wrappers import PendulumWrapper
from env_wrappers import LunarLanderWrapper
from env_wrappers import ReacherWrapper

try:
    from dm_control import suite
    DMC_AVAILABLE = True
except ImportError:
    DMC_AVAILABLE = False
    suite = None


SAC_PARAM_DEFAULTS = {
    'discount': 0.99,
    'init_temperature': 0.1,
    'alpha_lr': 1e-4,
    'alpha_betas': [0.9, 0.999],
    'actor_lr': 1e-4,
    'actor_betas': [0.9, 0.999],
    'actor_update_frequency': 1,
    'critic_lr': 1e-4,
    'critic_betas': [0.9, 0.999],
    'critic_tau': 0.005,
    'critic_target_update_frequency': 2,
    'batch_size': 1024,
    'learnable_temperature': True,
}


def apply_missing_sac_param_defaults(agent_params, seed):
    """Backfill standard SAC parameters when Hydra did not compose agent defaults."""
    missing_keys = [key for key in SAC_PARAM_DEFAULTS if key not in agent_params]
    if missing_keys:
        print(
            f"[Seed {seed}] Warning: Hydra config is missing SAC params {missing_keys}. "
            "Applying built-in defaults."
        )
        for key in missing_keys:
            agent_params[key] = copy.deepcopy(SAC_PARAM_DEFAULTS[key])
    return agent_params


def get_reacher_mode(cfg):
    return str(getattr(cfg, 'reacher_mode', 'standard')).lower()


def get_default_reacher_evaluator_dir(cfg):
    reward_mode = str(getattr(getattr(cfg, 'reacher', None), 'reward_mode', 'rb')).lower()
    seed_root = Path('.', 'exp', 'reacher', reward_mode, f'seed_{cfg.seed}').resolve()
    evaluator_root = seed_root / 'evaluator'
    os.makedirs(evaluator_root, exist_ok=True)

    # Optimal-policy evaluator should log under the same numeric evaluator run
    # that contains the checkpoint, inside an evaluate/ subdir.
    checkpoint_runs = []
    for child in evaluator_root.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / 'checkpoint.pt').exists():
            checkpoint_runs.append((int(child.name), child))

    if checkpoint_runs:
        _, run_dir = max(checkpoint_runs, key=lambda item: item[0])
    else:
        # Fallback when evaluator run checkpoints are absent: use explicit
        # load_checkpoint parent if available, otherwise evaluator/0.
        load_checkpoint = str(getattr(cfg, 'load_checkpoint', '') or '').strip()
        if load_checkpoint:
            run_dir = Path(load_checkpoint).resolve().parent
        else:
            run_dir = evaluator_root / '0'

    return str((run_dir / 'evaluate').resolve())

def make_pendulum_env(cfg):
    """Function to make pendulum environment"""
    pendulum_cfg = getattr(cfg, 'pendulum', None)
    # Check top-level target_theta first (for easy override), then fall back to pendulum.target_theta
    target_theta = float(getattr(cfg, 'target_theta', 0.0))
    if target_theta == 0.0 and pendulum_cfg is not None:
        target_theta = float(getattr(pendulum_cfg, 'target_theta', 0.0))
    
    max_episode_steps = 1000
    if pendulum_cfg is not None:
        max_episode_steps = int(getattr(pendulum_cfg, 'max_episode_steps', 1000))

    base_env = gym.make('Pendulum-v1', max_episode_steps=max_episode_steps, render_mode='rgb_array' if cfg.save_video else None)

    env = PendulumWrapper(base_env, target_theta=target_theta)
    if hasattr(env, 'seed'):
        env.seed(cfg.seed)
    else:
        env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    return env

def make_lunarlander_env(cfg):
    """Function to make LunarLander-v3 environment."""
    lunarlander_cfg = getattr(cfg, 'lunarlander', None)
    continuous = True
    max_episode_steps = 1000
    reward_mode = 'default'
    hover_bonus = 200.0
    hover_x_threshold = 0.1
    hover_y_min = 0.4
    hover_y_max = 0.6
    hover_once_per_episode = True

    if lunarlander_cfg is not None:
        continuous = bool(getattr(lunarlander_cfg, 'continuous', True))
        max_episode_steps = int(getattr(lunarlander_cfg, 'max_episode_steps', 1000))
        reward_mode = str(getattr(lunarlander_cfg, 'reward_mode', 'default'))
        hover_bonus = float(getattr(lunarlander_cfg, 'hover_bonus', 200.0))
        hover_x_threshold = float(getattr(lunarlander_cfg, 'hover_x_threshold', 0.1))
        hover_y_min = float(getattr(lunarlander_cfg, 'hover_y_min', 0.4))
        hover_y_max = float(getattr(lunarlander_cfg, 'hover_y_max', 0.6))
        hover_once_per_episode = bool(
            getattr(lunarlander_cfg, 'hover_once_per_episode', True))

    base_env = gym.make(
        'LunarLander-v3',
        continuous=continuous,
        max_episode_steps=max_episode_steps,
        render_mode='rgb_array' if cfg.save_video else None,
    )
    env = LunarLanderWrapper(
        base_env,
        reward_mode=reward_mode,
        hover_bonus=hover_bonus,
        hover_x_threshold=hover_x_threshold,
        hover_y_min=hover_y_min,
        hover_y_max=hover_y_max,
        hover_once_per_episode=hover_once_per_episode,
    )

    if hasattr(env, 'seed'):
        env.seed(cfg.seed)
    else:
        env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    return env


class DMCToGymnasiumWrapper(gym.Wrapper):
    """Wrapper to convert DeepMind Control Suite environment to gymnasium API."""

    def __init__(self, env, max_episode_steps=1000):
        self._dmc_env = env
        self._max_steps = max_episode_steps
        self._step_count = 0

        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        import gymnasium.spaces as spaces
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape,
            dtype=action_spec.dtype,
        )

        obs_sizes = [spec.shape for spec in obs_spec.values()]
        total_obs_size = sum(np.prod(size) for size in obs_sizes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        self._obs_keys = list(obs_spec.keys())
        self._obs_slices = {}
        offset = 0
        for key, spec in obs_spec.items():
            size = int(np.prod(spec.shape))
            self._obs_slices[key] = slice(offset, offset + size)
            offset += size

    def _flatten_obs(self, obs_dict):
        obs_list = []
        for key in self._obs_keys:
            obs_list.append(obs_dict[key].flatten())
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        timestep = self._dmc_env.reset()
        obs = self._flatten_obs(timestep.observation)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        timestep = self._dmc_env.step(action)

        obs = self._flatten_obs(timestep.observation)
        reward = 0.0 if timestep.reward is None else float(timestep.reward)
        terminated = bool(timestep.last())
        truncated = self._step_count >= self._max_steps
        discount = 1.0 if timestep.discount is None else float(timestep.discount)
        info = {'discount': discount}
        return obs, reward, terminated, truncated, info

    def render(self):
        try:
            return self._dmc_env.physics.render(height=256, width=256, camera_id=0)
        except Exception:
            return None

    @property
    def _max_episode_steps(self):
        return self._max_steps

    def close(self):
        self._dmc_env.close()


def make_reacher_env(cfg):
    """Function to make Reacher environment from DeepMind Control Suite (easy version)."""
    if not DMC_AVAILABLE:
        raise ImportError(
            "dm_control is required for reacher environment. Install it with: pip install dm-control"
        )

    reacher_cfg = getattr(cfg, 'reacher', None)
    # Check top-level reward_mode first (for easy override), then fall back to reacher.reward_mode
    reward_mode = str(getattr(cfg, 'reward_mode', 'rb')).lower()
    if reward_mode == 'rb' and reacher_cfg is not None:
        reward_mode = str(getattr(reacher_cfg, 'reward_mode', 'rb')).lower()
    
    max_episode_steps = 1000
    target_distance_threshold = 0.01

    if reacher_cfg is not None:
        max_episode_steps = int(getattr(reacher_cfg, 'max_episode_steps', 1000))
        target_distance_threshold = float(
            getattr(reacher_cfg, 'target_distance_threshold', 0.01)
        )
        if str(getattr(cfg, 'reacher_mode', 'standard')) == 'optimal_policy_evaluator':
            max_episode_steps = int(
                getattr(reacher_cfg, 'optimal_policy_eval_episode_length', 5000)
            )

    base_env = suite.load(domain_name='reacher', task_name='easy', visualize_reward=False)
    # For Rc, ReacherWrapper is sole owner of the 1000-step timeout counted
    # from sub-episode start. Infinite inner budget prevents the inner wrapper
    # from firing truncation at misaligned global-1000 boundaries.
    inner_max_steps = 10**9 if reward_mode == 'rc' else max_episode_steps
    env = DMCToGymnasiumWrapper(base_env, max_episode_steps=inner_max_steps)
    env = ReacherWrapper(
        env,
        reward_mode=reward_mode,
        target_distance_threshold=target_distance_threshold,
    )
    print("&&&&&&& REACHER KA REWARD MODeL: ", reward_mode)
    if hasattr(env, 'seed'):
        env.seed(cfg.seed)
    else:
        env.reset(seed=cfg.seed)

    if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
        env.action_space.seed(cfg.seed)
    return env

class Workspace(object):
    def __init__(self, cfg):
        # Prefer Hydra's resolved output dir so per-run logs land under hydra.run.dir
        # even when hydra.job.chdir is false.
        # For multi-seed runs, use the video_dir as work_dir
        if getattr(cfg, '_multi_seed_run', False):
            self.work_dir = os.path.expanduser(cfg.video_dir)
            # Create directory for multi-seed runs
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            try:
                if HydraConfig is not None:
                    self.work_dir = HydraConfig.get().runtime.output_dir
                else:
                    self.work_dir = os.getcwd()
            except Exception:
                self.work_dir = os.getcwd()
        if cfg.env == "reacher" and get_reacher_mode(cfg) == 'optimal_policy_evaluator':
            self.work_dir = get_default_reacher_evaluator_dir(cfg)
            os.makedirs(self.work_dir, exist_ok=True)
        print(f'[Seed {cfg.seed}] workspace: {self.work_dir}')

        self.cfg = cfg

        # Disable struct mode to allow dynamic config keys
        OmegaConf.set_struct(cfg, False)

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent='sac')

        # Ensure expected metric files exist immediately for easier monitoring.
        self.train_csv_path = os.path.join(self.work_dir, 'train.csv')
        self.eval_csv_path = os.path.join(self.work_dir, 'eval.csv')
        for csv_path in (self.train_csv_path, self.eval_csv_path):
            if not os.path.exists(csv_path):
                open(csv_path, 'a').close()
        print(f'[Seed {cfg.seed}] train metrics file: {self.train_csv_path}')
        print(f'[Seed {cfg.seed}] eval metrics file: {self.eval_csv_path}')

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # Print device information at the start
        if 'cuda' in str(self.device):
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'[Seed {cfg.seed}] Using device: {self.device}')
            print(f'[Seed {cfg.seed}] GPU: {gpu_name}, Total Memory: {total_memory:.2f} GB')
        else:
            print(f'[Seed {cfg.seed}] Using device: {self.device}')

        if cfg.env == "pendulum":
            self.env = make_pendulum_env(cfg)
        elif cfg.env == "lunarlander":
            self.env = make_lunarlander_env(cfg)
        elif cfg.env == "reacher":
            self.env = make_reacher_env(cfg)
        else:
            self.env = utils.make_env(cfg)

        # Extract agent params - start with defaults and merge with config
        agent_params_cfg = OmegaConf.create({})
        try:
            if hasattr(cfg, 'agent') and hasattr(cfg.agent, 'params'):
                # First try to get all params without full resolution
                agent_params_cfg = OmegaConf.merge(
                    OmegaConf.create({}),
                    cfg.agent.params
                )
                print(f"[Seed {cfg.seed}] Extracted agent params keys: {list(agent_params_cfg.keys())}")
        except Exception as e:
            print(f"[Seed {cfg.seed}] Warning: Could not extract agent params from config: {e}")
            import traceback
            traceback.print_exc()
            pass

        obs_dim = self.env.observation_space.shape[0]
        
        # Determine action space properties
        is_discrete = hasattr(self.env.action_space, 'n')
        if is_discrete:
            action_dim = int(self.env.action_space.n)
            action_range = [0, int(self.env.action_space.n) - 1]
            agent_class = get_class('agent.sac.DiscreteSACAgent')
        else:
            action_dim = self.env.action_space.shape[0]
            action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
            agent_class = get_class('agent.sac.SACAgent')

        # Handle both discrete and continuous action spaces
        if is_discrete:
            
            # Resolve critic config
            if 'critic_cfg' in agent_params_cfg:
                critic_cfg_val = agent_params_cfg.critic_cfg
                # If it's a string reference (like "reacher_double_q_critic"), resolve it from cfg
                if isinstance(critic_cfg_val, str):
                    critic_cfg = getattr(cfg, critic_cfg_val, None)
                    if critic_cfg is None:
                        print(f"[Seed {cfg.seed}] Warning: Could not resolve critic_cfg={critic_cfg_val}, building default")
                        critic_cfg = OmegaConf.create({
                            'class': 'agent.critic.DiscreteDoubleQCritic',
                            'params': {
                                'obs_dim': obs_dim,
                                'action_dim': action_dim,
                                'hidden_dim': 512,
                                'hidden_depth': 2
                            }
                        })
                else:
                    critic_cfg = critic_cfg_val
            else:
                # Build default critic config
                if cfg.env == 'reacher':
                    critic_cfg = OmegaConf.create({
                        'class': 'agent.critic.DoubleQCritic',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_dim': 512,
                            'hidden_depth': 2
                        }
                    })
                else:
                    critic_cfg = OmegaConf.create({
                        'class': 'agent.critic.DiscreteDoubleQCritic',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_dim': 512,
                            'hidden_depth': 2
                        }
                    })
            
            # Resolve actor config
            if 'actor_cfg' in agent_params_cfg:
                actor_cfg_val = agent_params_cfg.actor_cfg
                # If it's a string reference, resolve it from cfg
                if isinstance(actor_cfg_val, str):
                    actor_cfg = getattr(cfg, actor_cfg_val, None)
                    if actor_cfg is None:
                        print(f"[Seed {cfg.seed}] Warning: Could not resolve actor_cfg={actor_cfg_val}, building default")
                        actor_cfg = OmegaConf.create({
                            'class': 'agent.actor.CategoricalActor',
                            'params': {
                                'obs_dim': obs_dim,
                                'action_dim': action_dim,
                                'hidden_depth': 2,
                                'hidden_dim': 512
                            }
                        })
                else:
                    actor_cfg = actor_cfg_val
            else:
                # Build default actor config
                actor_cfg = OmegaConf.create({
                    'class': 'agent.actor.CategoricalActor',
                    'params': {
                        'obs_dim': obs_dim,
                        'action_dim': action_dim,
                        'hidden_depth': 2,
                        'hidden_dim': 512
                    }
                })
        else:
            # Continuous action space - resolve critic config
            if 'critic_cfg' in agent_params_cfg:
                critic_cfg_val = agent_params_cfg.critic_cfg
                # If it's a string reference, resolve it from cfg
                if isinstance(critic_cfg_val, str):
                    critic_cfg = getattr(cfg, critic_cfg_val, None)
                    if critic_cfg is None:
                        print(f"[Seed {cfg.seed}] Warning: Could not resolve critic_cfg={critic_cfg_val}, building default")
                        critic_cfg = OmegaConf.create({
                            'class': 'agent.critic.DoubleQCritic',
                            'params': {
                                'obs_dim': obs_dim,
                                'action_dim': action_dim,
                                'hidden_dim': 512,
                                'hidden_depth': 2
                            }
                        })
                else:
                    critic_cfg = critic_cfg_val
            else:
                # Build default critic config
                if cfg.env == 'reacher':
                    critic_cfg = OmegaConf.create({
                        'class': 'agent.critic.DoubleQCritic',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_dim': 512,
                            'hidden_depth': 2
                        }
                    })
                else:
                    critic_cfg = OmegaConf.create({
                        'class': 'agent.critic.DoubleQCritic',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_dim': 512,
                            'hidden_depth': 2
                        }
                    })
            
            # Resolve actor config
            if 'actor_cfg' in agent_params_cfg:
                actor_cfg_val = agent_params_cfg.actor_cfg
                # If it's a string reference, resolve it from cfg
                if isinstance(actor_cfg_val, str):
                    actor_cfg = getattr(cfg, actor_cfg_val, None)
                    if actor_cfg is None:
                        print(f"[Seed {cfg.seed}] Warning: Could not resolve actor_cfg={actor_cfg_val}, building default")
                        actor_cfg = OmegaConf.create({
                            'class': 'agent.actor.DiagGaussianActor',
                            'params': {
                                'obs_dim': obs_dim,
                                'action_dim': action_dim,
                                'hidden_depth': 2,
                                'hidden_dim': 512,
                                'log_std_bounds': [-5, 2]
                            }
                        })
                else:
                    actor_cfg = actor_cfg_val
            else:
                # Build default actor config
                if cfg.env == 'reacher':
                    actor_cfg = OmegaConf.create({
                        'class': 'agent.actor.DiagGaussianActor',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_depth': 2,
                            'hidden_dim': 512,
                            'log_std_bounds': [-5, 2]
                        }
                    })
                else:
                    actor_cfg = OmegaConf.create({
                        'class': 'agent.actor.DiagGaussianActor',
                        'params': {
                            'obs_dim': obs_dim,
                            'action_dim': action_dim,
                            'hidden_depth': 2,
                            'hidden_dim': 512,
                            'log_std_bounds': [-5, 2]
                        }
                    })

        # Do not resolve critic/actor configs here (they have nested references),
        # Extract all agent params carefully
        agent_params = {}
        if hasattr(cfg, 'agent') and hasattr(cfg.agent, 'params'):
            # Get all keys without resolving values
            params_dict = OmegaConf.to_container(cfg.agent.params, resolve=False)
            if isinstance(params_dict, dict):
                for key, value in params_dict.items():
                    if key not in ['critic_cfg', 'actor_cfg']:  # Skip nested configs, we handle those separately
                        # Avoid propagating missing/placeholder values like "???"
                        if value is not None and value != '???':
                            agent_params[key] = value
        
        # Add obs_dim and action_dim that we computed
        agent_params['obs_dim'] = self.env.observation_space.shape[0]
        if hasattr(self.env.action_space, 'n'):
            agent_params['action_dim'] = int(self.env.action_space.n)
            agent_params['action_range'] = [0, int(self.env.action_space.n) - 1]
        else:
            agent_params['action_dim'] = self.env.action_space.shape[0]
            agent_params['action_range'] = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
        
        print(f"[Seed {cfg.seed}] Extracted agent params: {list(agent_params.keys())}")
        print(f"[Seed {cfg.seed}] Agent param values: {agent_params}")

        agent_params = apply_missing_sac_param_defaults(agent_params, cfg.seed)

        agent_params['device'] = str(self.device)
        agent_params['critic_cfg'] = critic_cfg
        agent_params['actor_cfg'] = actor_cfg
        
        # Verify all required parameters are present
        required_params = ['discount', 'init_temperature', 'alpha_lr', 'alpha_betas', 
                          'actor_lr', 'actor_betas', 'actor_update_frequency',
                          'critic_lr', 'critic_betas', 'critic_tau', 
                          'critic_target_update_frequency', 'batch_size', 'learnable_temperature']
        missing_params = [p for p in required_params if p not in agent_params]
        if missing_params:
            print(f"[Seed {cfg.seed}] ERROR: Missing required agent parameters: {missing_params}")
            print(f"[Seed {cfg.seed}] Available parameters: {list(agent_params.keys())}")
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        self._agent_class = agent_class
        self._agent_params_template = copy.deepcopy(agent_params)
        self.agent = agent_class(**agent_params)

        # Determine action shape for replay buffer
        if hasattr(self.env.action_space, 'n'):
            action_shape = (1,)  # Discrete: store as 1D array
        else:
            action_shape = self.env.action_space.shape  # Continuous

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          action_shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        # --- PEBBLE reward-learning components ---
        class RewardModel(nn.Module):
            """Reward model r(s, a) used for preference learning."""
            def __init__(self, obs_dim, action_dim, hidden_dim=256):
                super().__init__()
                in_dim = obs_dim + action_dim
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, obs, action):
                x = torch.cat([obs, action], dim=-1)
                return self.net(x).squeeze(-1)

        class PreferenceBuffer(object):
            """Stores trajectory segments (for query proposal) and labeled pairs."""
            def __init__(self, max_segments=2000):
                self.segments = []
                self.max_segments = max_segments
                self.prefs = []  # tuples: (seg_a, seg_b, label)

            def add_segment(self, segment):
                self.segments.append(segment)
                if len(self.segments) > self.max_segments:
                    self.segments.pop(0)

            def add_pref(self, seg_a, seg_b, label):
                self.prefs.append((seg_a, seg_b, int(label)))

            def num_segments(self):
                return len(self.segments)

            def num_prefs(self):
                return len(self.prefs)

            def sample_pref_batch(self, batch_size):
                if len(self.prefs) == 0:
                    return []
                k = min(batch_size, len(self.prefs))
                idx = np.random.choice(len(self.prefs), size=k, replace=False)
                return [self.prefs[i] for i in idx]

        pebble_cfg = getattr(cfg, 'pebble', None)
        self.use_pebble = bool(pebble_cfg and pebble_cfg.get('enabled', False))
        if self.use_pebble:
            obs_dim = int(self.env.observation_space.shape[0])
            act_dim = int(self.env.action_space.shape[0])

            self.segment_length = int(pebble_cfg.get('segment_length', 50))
            self.teacher_noise = float(pebble_cfg.get('teacher_noise', 0.0))
            self.pref_budget = int(pebble_cfg.get('budget', 2000))
            self.query_batch = int(pebble_cfg.get('query_batch_size', 16))
            self.query_interval = int(pebble_cfg.get('query_interval', 1000))
            self.reward_train_steps = int(pebble_cfg.get('reward_train_steps', 200))
            self.reward_batch_size = int(pebble_cfg.get('reward_batch_size', 64))
            self.relabel_interval = int(pebble_cfg.get('relabel_interval', 1000))
            self.pref_start_step = int(pebble_cfg.get('pref_start_step', int(cfg.num_seed_steps)))

            hidden = int(pebble_cfg.get('hidden_dim', 256))
            lr = float(pebble_cfg.get('lr', 3e-4))
            ensemble_size = int(pebble_cfg.get('ensemble_size', 3))

            self.reward_models = nn.ModuleList([
                RewardModel(obs_dim, act_dim, hidden_dim=hidden).to(self.device)
                for _ in range(ensemble_size)
            ])
            self.reward_opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.reward_models]
            self.pref_buffer = PreferenceBuffer(max_segments=int(pebble_cfg.get('max_segments', 2000)))
            self.pref_queries_used = 0
            self.reward_ready = False

            self.bce_logits = nn.BCEWithLogitsLoss()
            
            # For Reacher with PEBBLE: create three simulated teachers (one per reward mode: ra, rb, rc)
            self.reacher_reward_modes = None
            if cfg.env == 'reacher' and self.use_pebble:
                print(f"[Seed {cfg.seed}] Creating three simulated teachers for Reacher (ra, rb, rc)")
                # Store the three reward modes for preference generation
                self.reacher_reward_modes = ['ra', 'rb', 'rc']
                # We'll use these to compute returns in different reward spaces
            
            print(f"[Seed {cfg.seed}] PEBBLE enabled with {ensemble_size} reward models")
        # --- end pebble components ---

        self.video_recorder = VideoRecorder(
            cfg.video_dir if cfg.save_video else None)
        self.step = 0
        self.next_eval_step = 0
        self.next_video_step = cfg.video_frequency
        self.eval_pending = True
        self.reward_flip_done = False
        self.video_pending = False
        self.progress_print_frequency = int(getattr(cfg, 'progress_print_frequency', 1000))

        self._sync_reward_schedule_with_step(initial=True)

        if self.cfg.env == "reacher" and self._reacher_mode() == 'optimal_policy_evaluator':
            self._ensure_default_reacher_evaluator_checkpoint()
        if getattr(cfg, 'load_checkpoint', ''):
            self.load_checkpoint(cfg.load_checkpoint)

    def _reacher_mode(self):
        return get_reacher_mode(self.cfg)

    def _ensure_default_reacher_evaluator_checkpoint(self):
        """Auto-select the trained checkpoint for evaluator-only runs."""
        if getattr(self.cfg, 'load_checkpoint', ''):
            return

        reward_mode = str(getattr(getattr(self.cfg, 'reacher', None), 'reward_mode', 'rb')).lower()
        current_seed = self.cfg.seed
        
        seed_root = Path('.') / 'exp' / 'reacher' / reward_mode / f'seed_{current_seed}'
        evaluator_root = seed_root / 'evaluator'

        candidate_paths = []
        direct_checkpoint = seed_root / 'checkpoint.pt'
        if direct_checkpoint.exists():
            candidate_paths.append(direct_checkpoint)

        evaluator_checkpoint = evaluator_root / 'checkpoint.pt'
        if evaluator_checkpoint.exists():
            candidate_paths.append(evaluator_checkpoint)

        # Only search subdirectories within the current seed's directory
        for child in sorted(seed_root.iterdir()) if seed_root.exists() else []:
            if not child.is_dir() or child.name == 'evaluator':
                continue
            checkpoint = child / 'checkpoint.pt'
            if checkpoint.exists():
                candidate_paths.append(checkpoint)

        for child in sorted(evaluator_root.iterdir()) if evaluator_root.exists() else []:
            if not child.is_dir():
                continue
            checkpoint = child / 'checkpoint.pt'
            if checkpoint.exists():
                candidate_paths.append(checkpoint)

        if not candidate_paths:
            raise FileNotFoundError(
                f"No checkpoint.pt found for seed {current_seed} (reward_mode={reward_mode}) under "
                f"{seed_root} or {evaluator_root}\n"
                f"Searched paths:\n"
                f"  - {direct_checkpoint}\n"
                f"  - {evaluator_checkpoint}\n"
                f"  - subdirectories of {seed_root}\n"
                f"  - subdirectories of {evaluator_root}"
            )

        def checkpoint_sort_key(path):
            parent_name = path.parent.name
            grandparent_name = path.parent.parent.name if path.parent.parent else ''
            # Prefer evaluator checkpoints in numbered subdirs (e.g., evaluator/0/)
            if parent_name.isdigit() and grandparent_name == 'evaluator':
                return (2, int(parent_name))
            # Then prefer numbered subdirs in seed root (e.g., seed_8/0/)
            if parent_name.isdigit():
                return (1, int(parent_name))
            # Then prefer checkpoint at seed root (e.g., seed_8/checkpoint.pt)
            if parent_name == seed_root.name:
                return (0, -1)
            # Finally, evaluator root checkpoint (e.g., seed_8/evaluator/checkpoint.pt)
            if parent_name == 'evaluator':
                return (0, 0)
            return (0, parent_name)

        # Try checkpoints in order of preference
        sorted_candidates = sorted(candidate_paths, key=checkpoint_sort_key)
        
        for candidate in reversed(sorted_candidates):  # Try best candidates first
            try:
                # Quick check: verify checkpoint has required keys
                checkpoint = torch.load(candidate, map_location=self.device)
                if isinstance(checkpoint, dict) and 'agent' in checkpoint and 'replay_buffer' in checkpoint:
                    self.cfg.load_checkpoint = str(candidate.resolve())
                    print(f'[Seed {current_seed}] Auto-selected checkpoint from {reward_mode}: {self.cfg.load_checkpoint}')
                    return
                else:
                    missing_keys = set(['agent', 'replay_buffer']) - set(checkpoint.keys() if isinstance(checkpoint, dict) else [])
                    print(f'[Seed {current_seed}] Skipping incomplete checkpoint {candidate.name} (missing: {missing_keys})')
            except Exception as e:
                print(f'[Seed {current_seed}] Error loading {candidate.name}: {type(e).__name__}. Trying next candidate.')
                continue
        
        # If we get here, no complete checkpoint was found
        raise FileNotFoundError(
            f"No complete checkpoint.pt (with 'agent' and 'replay_buffer' keys) found for seed {current_seed} "
            f"(reward_mode={reward_mode}).\n"
            f"Checked {len(candidate_paths)} candidates:\n"
            + "\n".join(f"  - {str(c)}" for c in sorted_candidates)
        )

    def _resolve_checkpoint_path(self, path):
        """Resolve checkpoint paths relative to the current run directory."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.work_dir, path)

    def _serialize_cfg_for_checkpoint(self):
        """Best-effort config serialization for checkpoints.

        Some Hydra override patterns can leave string references that do not
        fully resolve at save time. In that case, preserve the config without
        forcing interpolation resolution rather than failing the whole run.
        """
        try:
            return OmegaConf.to_container(self.cfg, resolve=True)
        except Exception as exc:
            print(
                f"[Seed {self.cfg.seed}] Warning: Could not fully resolve config "
                f"for checkpoint save ({exc}). Saving unresolved config instead."
            )
            return OmegaConf.to_container(self.cfg, resolve=False)

    def _reward_flip_enabled(self):
        lunarlander_cfg = getattr(self.cfg, 'lunarlander', None)
        return bool(getattr(lunarlander_cfg, 'reward_flip_enabled', False))

    def _reward_flip_step(self):
        lunarlander_cfg = getattr(self.cfg, 'lunarlander', None)
        return int(getattr(lunarlander_cfg, 'reward_flip_step', 0))

    def _reward_flip_bonus(self):
        lunarlander_cfg = getattr(self.cfg, 'lunarlander', None)
        return float(getattr(lunarlander_cfg, 'reward_flip_bonus', -100.0))

    def _sync_reward_schedule_with_step(self, initial=False):
        if not hasattr(self.env, 'set_hover_bonus'):
            return
        if getattr(self.cfg, 'env', '') != 'lunarlander':
            return
        lunarlander_cfg = getattr(self.cfg, 'lunarlander', None)
        if lunarlander_cfg is None or str(getattr(lunarlander_cfg, 'reward_mode', 'default')) != 'hover_box':
            return
        if not self._reward_flip_enabled():
            return

        should_flip = self.step >= self._reward_flip_step()
        if should_flip and not self.reward_flip_done:
            new_bonus = self._reward_flip_bonus()
            self.env.set_hover_bonus(new_bonus)
            self.reward_flip_done = True
            action = 'Applying' if initial else 'Flipping'
            print(f'[Seed {self.cfg.seed}] {action} hover reward change at step {self.step}: hover_bonus -> {new_bonus}')

    def _set_hidden_dim_in_cfg(self, cfg, hidden_dim):
        if not OmegaConf.is_config(cfg):
            cfg_container = copy.deepcopy(cfg)
        else:
            cfg_container = OmegaConf.to_container(cfg, resolve=False)

        if not isinstance(cfg_container, dict):
            return cfg

        params = cfg_container.get('params', {})
        if not isinstance(params, dict):
            params = {}

        params['hidden_dim'] = int(hidden_dim)

        cfg_container['params'] = params
        return OmegaConf.create(cfg_container)

    def _try_rebuild_agent_with_alternate_hidden_dim(self, checkpoint):
        current_hidden_dim = None
        checkpoint_hidden_dim = None

        try:
            current_hidden_dim = int(self.agent.actor.state_dict()['trunk.0.weight'].shape[0])
        except Exception:
            pass

        try:
            checkpoint_hidden_dim = int(
                checkpoint['agent']['actor']['trunk.0.weight'].shape[0]
            )
        except Exception:
            pass

        # Strictly support the 512 <-> 1024 architecture alternatives.
        if current_hidden_dim not in (512, 1024):
            return False

        if checkpoint_hidden_dim in (512, 1024) and checkpoint_hidden_dim != current_hidden_dim:
            alt_hidden_dim = checkpoint_hidden_dim
        else:
            alt_hidden_dim = 512 if current_hidden_dim == 1024 else 1024

        rebuilt_params = copy.deepcopy(self._agent_params_template)
        rebuilt_params['actor_cfg'] = self._set_hidden_dim_in_cfg(
            rebuilt_params['actor_cfg'], alt_hidden_dim
        )
        rebuilt_params['critic_cfg'] = self._set_hidden_dim_in_cfg(
            rebuilt_params['critic_cfg'], alt_hidden_dim
        )

        self.agent = self._agent_class(**rebuilt_params)
        self._agent_params_template = rebuilt_params
        print(
            f"[Seed {self.cfg.seed}] Rebuilt agent for checkpoint compatibility "
            f"with hidden_dim={alt_hidden_dim}."
        )
        return True

    def save_checkpoint(self, path):
        resolved_path = self._resolve_checkpoint_path(path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        checkpoint = {
            'step': self.step,
            'agent': self.agent.state_dict(),
            'replay_buffer': self.replay_buffer.state_dict(),
            'next_eval_step': self.next_eval_step,
            'next_video_step': self.next_video_step,
            'eval_pending': self.eval_pending,
            'video_pending': self.video_pending,
            'reward_flip_done': self.reward_flip_done,
            'cfg': self._serialize_cfg_for_checkpoint(),
        }
        if hasattr(self.env, 'reward_state_dict'):
            checkpoint['env_reward_state'] = self.env.reward_state_dict()
        torch.save(checkpoint, resolved_path)
        print(f'[Seed {self.cfg.seed}] Saved checkpoint to {resolved_path}')

    def load_checkpoint(self, path):
        resolved_path = self._resolve_checkpoint_path(path)
        checkpoint = torch.load(resolved_path, map_location=self.device)
        
        # Diagnostic: show checkpoint structure
        checkpoint_keys = set(checkpoint.keys()) if isinstance(checkpoint, dict) else set()
        required_keys = {'agent', 'replay_buffer', 'step'}
        missing_keys = required_keys - checkpoint_keys
        
        if missing_keys:
            print(f"[Seed {self.cfg.seed}] WARNING: Checkpoint missing keys: {missing_keys}")
            print(f"[Seed {self.cfg.seed}] Available checkpoint keys: {sorted(checkpoint_keys)}")
        
        self.step = int(checkpoint.get('step', 0))
        self.next_eval_step = int(checkpoint.get('next_eval_step', self.step + self.cfg.eval_frequency))
        self.next_video_step = int(checkpoint.get('next_video_step', self.step + self.cfg.video_frequency))
        self.eval_pending = bool(checkpoint.get('eval_pending', False))
        self.video_pending = bool(checkpoint.get('video_pending', False))
        self.reward_flip_done = bool(checkpoint.get('reward_flip_done', False))
        
        # Check for agent key
        if 'agent' not in checkpoint:
            raise KeyError(
                f"Checkpoint at {resolved_path} is incomplete or corrupted - missing 'agent' key.\n"
                f"Available keys: {sorted(checkpoint_keys)}\n"
                f"This checkpoint may be from an incomplete or failed training run. "
                f"Try explicitly specifying a different checkpoint with load_checkpoint=path/to/checkpoint.pt"
            )
        
        try:
            self.agent.load_state_dict(checkpoint['agent'])
        except RuntimeError as exc:
            if 'size mismatch' not in str(exc):
                raise
            print(f"[Seed {self.cfg.seed}] Checkpoint shape mismatch detected: {exc}")
            rebuilt = self._try_rebuild_agent_with_alternate_hidden_dim(checkpoint)
            if not rebuilt:
                raise
            try:
                self.agent.load_state_dict(checkpoint['agent'])
            except RuntimeError:
                # If alternate width still fails, surface original mismatch.
                raise exc
        
        if 'replay_buffer' not in checkpoint:
            print(f"[Seed {self.cfg.seed}] WARNING: Checkpoint missing 'replay_buffer' key. "
                  f"Skipping replay buffer restoration. Available keys: {sorted(checkpoint_keys)}")
        else:
            try:
                self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])
            except Exception as e:
                print(f"[Seed {self.cfg.seed}] WARNING: Failed to load replay_buffer: {type(e).__name__}: {e}")
        
        if hasattr(self.env, 'load_reward_state_dict') and 'env_reward_state' in checkpoint:
            self.env.load_reward_state_dict(checkpoint['env_reward_state'])
        self._sync_reward_schedule_with_step(initial=True)
        print(f'[Seed {self.cfg.seed}] Loaded checkpoint from {resolved_path} at step {self.step}')
        
    def _evaluate_average_return(self, save_video=False):
        average_episode_reward = 0
        # Rc episodes only terminate on success (done=True). To avoid eval
        # hanging forever (especially at step 0 with an untrained policy), use
        # a configurable per-episode step budget for eval only. This does NOT
        # affect training — Rc training episodes are still unlimited.
        rc_mode = (
            self.cfg.env == "reacher"
            and hasattr(self.env, 'reward_mode')
            and str(self.env.reward_mode).lower() == 'rc'
        )
        reacher_cfg = getattr(self.cfg, 'reacher', None)
        eval_max_steps = int(getattr(reacher_cfg, 'eval_max_episode_steps', 5000)) if rc_mode else None

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0 and save_video))
            done = False
            episode_reward = 0
            eval_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                if save_video:
                    self.video_recorder.record(self.env)
                episode_reward += reward
                eval_step += 1
                if eval_max_steps is not None and eval_step >= eval_max_steps:
                    break

            average_episode_reward += episode_reward
            if save_video:
                video_path = f'seed_{self.cfg.seed}_train_steps_{self.step}.mp4'
                self.video_recorder.save(video_path)
                print(f"[Seed {self.cfg.seed}] Video saved to video/{video_path}")
        average_episode_reward /= self.cfg.num_eval_episodes
        return average_episode_reward
    
    # --- PEBBLE Methods ---
    def _segment_return_by_model(self, model, segment):
        s = torch.as_tensor(segment['states'], device=self.device).float()
        a = torch.as_tensor(segment['actions'], device=self.device).float()
        return model(s, a).sum()

    def _segment_return_ensemble(self, segment):
        returns = []
        with torch.no_grad():
            for m in self.reward_models:
                returns.append(self._segment_return_by_model(m, segment).detach())
        return torch.stack(returns, dim=0)

    def _predict_reward(self, obs, action):
        if (not self.use_pebble) or (not self.reward_ready):
            return None
        s = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        a = torch.as_tensor(action, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            vals = [m(s, a).detach() for m in self.reward_models]
            r = torch.stack(vals, dim=0).mean().item()
        return float(r)

    def _label_pair(self, seg_a, seg_b):
        """Generate preference label comparing two segments.
        
        For Reacher with multiple reward modes (teachers), compute preferences
        from all three reward formulations and use majority voting.
        """
        if self.cfg.env == 'reacher' and self.reacher_reward_modes:
            # For Reacher with three teachers, use majority voting across reward modes
            labels = []
            for mode in self.reacher_reward_modes:
                mode_a = float(seg_a.get('return_' + mode, seg_a.get('return', 0))) + np.random.normal(scale=self.teacher_noise)
                mode_b = float(seg_b.get('return_' + mode, seg_b.get('return', 0))) + np.random.normal(scale=self.teacher_noise)
                if abs(mode_a - mode_b) > 1e-8:
                    label = 1 if mode_a > mode_b else 0
                    labels.append(label)
            if len(labels) > 0:
                # Use majority voting: 1 if more than half prefer seg_a
                return int(1 if sum(labels) > len(labels) / 2 else 0)
        
        # Default single-teacher preference labeling
        ra = float(seg_a['return']) + np.random.normal(scale=self.teacher_noise)
        rb = float(seg_b['return']) + np.random.normal(scale=self.teacher_noise)
        if abs(ra - rb) < 1e-8:
            return int(np.random.choice([0, 1]))
        return int(1 if ra > rb else 0)

    def _add_episode_segments(self, ep_traj):
        if not self.use_pebble:
            return
        L = self.segment_length
        n = len(ep_traj)
        if n < 2:
            return
        step = max(1, L // 2)
        for start in range(0, max(1, n - L + 1), step):
            seg = ep_traj[start:start + L]
            if len(seg) < 2:
                continue
            states = np.stack([x['obs'] for x in seg], axis=0)
            actions = np.stack([x['action'] for x in seg], axis=0)
            rewards = np.asarray([x['reward'] for x in seg], dtype=np.float32)
            self.pref_buffer.add_segment({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'return': float(rewards.sum()),
            })

    def _query_preferences(self):
        if not self.use_pebble:
            return 0
        if self.pref_queries_used >= self.pref_budget:
            return 0
        if self.pref_buffer.num_segments() < 2:
            return 0

        # Build candidate pairs and score by model disagreement + entropy.
        candidates = []
        n_seg = self.pref_buffer.num_segments()
        n_candidates = min(256, n_seg * 2)
        for _ in range(n_candidates):
            i, j = np.random.choice(n_seg, size=2, replace=False)
            sa = self.pref_buffer.segments[i]
            sb = self.pref_buffer.segments[j]
            dr = self._segment_return_ensemble(sa) - self._segment_return_ensemble(sb)
            probs = torch.sigmoid(dr)
            mean_p = probs.mean()
            entropy = -(mean_p * torch.log(mean_p + 1e-8) + (1 - mean_p) * torch.log(1 - mean_p + 1e-8))
            disagreement = probs.std(unbiased=False)
            score = (entropy + disagreement).item()
            candidates.append((score, sa, sb))

        candidates.sort(key=lambda x: x[0], reverse=True)
        k = min(self.query_batch, self.pref_budget - self.pref_queries_used, len(candidates))
        added = 0
        for idx in range(k):
            _, sa, sb = candidates[idx]
            label = self._label_pair(sa, sb)
            self.pref_buffer.add_pref(sa, sb, label)
            added += 1

        self.pref_queries_used += added
        return added

    def _train_reward_models(self):
        if not self.use_pebble:
            return
        if self.pref_buffer.num_prefs() == 0:
            return

        last_loss = None
        for _ in range(self.reward_train_steps):
            batch = self.pref_buffer.sample_pref_batch(self.reward_batch_size)
            if len(batch) == 0:
                break

            for model, opt in zip(self.reward_models, self.reward_opts):
                logits_list = []
                labels_list = []
                for seg_a, seg_b, label in batch:
                    ra = self._segment_return_by_model(model, seg_a)
                    rb = self._segment_return_by_model(model, seg_b)
                    logits_list.append((ra - rb).view(1))
                    labels_list.append(torch.tensor([float(label)], device=self.device))

                logits = torch.cat(logits_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                loss = self.bce_logits(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                last_loss = loss

        if last_loss is not None:
            self.reward_ready = True
            self.logger.log('train/pebble/reward_loss', float(last_loss.item()), self.step)
            self.logger.log('train/pebble/prefs', self.pref_buffer.num_prefs(), self.step)
            self.logger.log('train/pebble/queries_used', self.pref_queries_used, self.step)

    def _relabel_replay_buffer(self):
        if not self.use_pebble or (not self.reward_ready):
            return
        n = len(self.replay_buffer)
        if n <= 0:
            return

        with torch.no_grad():
            obs = torch.as_tensor(self.replay_buffer.obses[:n], device=self.device).float()
            act = torch.as_tensor(self.replay_buffer.actions[:n], device=self.device).float()
            preds = []
            for m in self.reward_models:
                preds.append(m(obs, act).unsqueeze(0))
            rew = torch.cat(preds, dim=0).mean(dim=0).cpu().numpy().reshape(-1, 1)
            self.replay_buffer.rewards[:n] = rew.astype(np.float32)

    def _maybe_run_pebble(self):
        if not self.use_pebble:
            return
        if self.step < self.pref_start_step:
            return
        if self.step % max(1, self.query_interval) != 0:
            return

        added = self._query_preferences()
        if added > 0:
            self.logger.log('train/pebble/queries_added', added, self.step)
        self._train_reward_models()
        self._relabel_replay_buffer()
    # --- end PEBBLE methods ---
    
    def evaluate(self, save_video=False):
        # Reacher requirement (Q2.3): evaluate each trained SAC policy against
        # all three reward formulations every evaluation checkpoint.
        if self.cfg.env == "reacher" and hasattr(self.env, 'reward_mode'):
            eval_modes = ('ra', 'rb', 'rc')
            original_mode = str(self.env.reward_mode).lower()
            mode_rewards = {}

            for mode in eval_modes:
                self.env.reward_mode = mode
                mode_save_video = save_video and (mode == original_mode)
                mode_rewards[mode] = self._evaluate_average_return(
                    save_video=mode_save_video
                )
                self.logger.log(
                    f'eval/episode_reward_{mode}',
                    mode_rewards[mode],
                    self.step
                )

            # Restore the configured training reward mode.
            self.env.reward_mode = original_mode
            # Keep legacy metric name for compatibility with existing analysis code.
            self.logger.log('eval/episode_reward', mode_rewards[original_mode], self.step)
            self.logger.dump(self.step)
            return

        average_episode_reward = self._evaluate_average_return(save_video=save_video)
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def evaluate_optimal_policy(self, save_video=False):
        if self.cfg.env != "reacher":
            raise ValueError("optimal_policy_evaluator is only supported for env=reacher")
        if not hasattr(self.env, '_is_in_target'):
            raise ValueError("Reacher optimal policy evaluation requires ReacherWrapper")

        reacher_cfg = getattr(self.cfg, 'reacher', None)
        num_episodes = int(getattr(reacher_cfg, 'optimal_policy_eval_episodes', 500))
        episode_length = int(
            getattr(reacher_cfg, 'optimal_policy_eval_episode_length', 5000)
        )
        original_mode = str(getattr(self.env, 'reward_mode', 'rb')).lower()
        results = []
        eval_start_time = time.time()

        for episode in range(num_episodes):
            episode_start_time = time.time()
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0 and save_video))

            steps_to_goal = None
            steps_in_target = 0
            reached_goal = False
            steps_executed = 0

            for t in range(1, episode_length + 1):
                steps_executed = t
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, _, _, _ = self.env.step(action)

                if save_video and episode == 0:
                    self.video_recorder.record(self.env)

                in_target = bool(self.env._is_in_target(obs))
                if not reached_goal:
                    if in_target:
                        reached_goal = True
                        steps_to_goal = t
                        steps_in_target = 1
                else:
                    if in_target:
                        steps_in_target += 1

            if save_video and episode == 0:
                video_path = f'seed_{self.cfg.seed}_optimal_policy_eval.mp4'
                self.video_recorder.save(video_path)
                print(f"[Seed {self.cfg.seed}] Video saved to video/{video_path}")

            results.append({
                'episode': episode,
                'reward_mode': original_mode,
                'episode_length': episode_length,
                'steps_executed': steps_executed,
                'reached_goal': int(reached_goal),
                'steps_to_goal': steps_to_goal if steps_to_goal is not None else episode_length + 1,
                'steps_in_target': steps_in_target,
            })
            successful = [r for r in results if r['reached_goal']]
            running_success_rate = len(successful) / len(results)
            running_mean_steps_to_goal = (
                float(np.mean([r['steps_to_goal'] for r in successful]))
                if successful else float('nan')
            )
            running_mean_steps_in_target = (
                float(np.mean([r['steps_in_target'] for r in successful]))
                if successful else 0.0
            )
            elapsed = time.time() - eval_start_time
            episode_duration = time.time() - episode_start_time

            steps_to_goal_display = (
                steps_to_goal if steps_to_goal is not None else "not reached"
            )
            print(
                f"[Seed {self.cfg.seed}] Eval Ep {episode + 1}/{num_episodes} | "
                f"Steps Executed: {steps_executed}/{episode_length} | "
                f"Reached: {reached_goal} | Steps to Goal: {steps_to_goal_display} | "
                f"Steps in Target: {steps_in_target} | Success Rate: {running_success_rate:.3f} | "
                f"Mean Goal Steps: {running_mean_steps_to_goal:.2f} | "
                f"Mean Target Steps: {running_mean_steps_in_target:.2f} | "
                f"Ep Time: {episode_duration:.1f}s | Total Time: {elapsed:.1f}s"
            )

        results_path = os.path.join(self.work_dir, 'optimal_policy_eval.csv')
        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'episode',
                    'reward_mode',
                    'episode_length',
                    'steps_executed',
                    'reached_goal',
                    'steps_to_goal',
                    'steps_in_target',
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        successful = [r for r in results if r['reached_goal']]
        success_rate = len(successful) / max(1, len(results))
        mean_steps_to_goal = (
            float(np.mean([r['steps_to_goal'] for r in successful]))
            if successful else float('nan')
        )
        mean_steps_in_target = (
            float(np.mean([r['steps_in_target'] for r in successful]))
            if successful else 0.0
        )

        summary = {
            'episodes': len(results),
            'episode_length': episode_length,
            'success_rate': success_rate,
            'mean_steps_to_goal_success_only': mean_steps_to_goal,
            'mean_steps_in_target_success_only': mean_steps_in_target,
            'results_path': results_path,
        }

        print(f"[Seed {self.cfg.seed}] Optimal policy evaluation complete:")
        print(
            f"[Seed {self.cfg.seed}] success_rate={success_rate:.3f}, "
            f"mean_steps_to_goal={mean_steps_to_goal:.2f}, "
            f"mean_steps_in_target={mean_steps_in_target:.2f}"
        )
        print(f"[Seed {self.cfg.seed}] Saved detailed results to {results_path}")
        return summary

    def run(self):
        if self._reacher_mode() == 'optimal_policy_evaluator':
            if not getattr(self.cfg, 'load_checkpoint', ''):
                raise ValueError(
                    "reacher_mode=optimal_policy_evaluator requires load_checkpoint to be set"
                )
            self.evaluate_optimal_policy(save_video=self.cfg.save_video)
            return

        episode, episode_reward, done = 0, 0, True
        rc_episode_reward = 0
        rc_steps_to_goal = 0
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # Evaluate only after an episode finishes and a threshold has been crossed.
                if self.eval_pending:
                    self.logger.log('eval/episode', episode, self.step)
                    should_save_video = self.video_pending or self.step == int(self.cfg.num_train_steps)
                    if should_save_video:
                        print(f"[Seed {self.cfg.seed}] Saving video at step {self.step}")
                    self.evaluate(save_video=should_save_video)
                    self.eval_pending = False
                    # Update next_eval_step for subsequent evals if this was the initial eval at step 0
                    if self.step == 0:
                        self.next_eval_step = self.cfg.eval_frequency
                    if should_save_video:
                        self.video_pending = False
                        self.next_video_step += self.cfg.video_frequency
                _rc_mode = (
                    self.cfg.env == "reacher"
                    and hasattr(self.env, 'reward_mode')
                    and str(self.env.reward_mode).lower() == 'rc'
                )
                self.logger.log(
                    'train/episode_reward',
                    rc_episode_reward if _rc_mode else episode_reward,
                    self.step)

                if _rc_mode:
                    self.logger.log('train/rc_steps_to_goal', rc_steps_to_goal, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                rc_episode_reward = 0
                rc_steps_to_goal = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                
                # Initialize trajectory buffer for PEBBLE segment creation
                ep_traj = []
                
                # Print progress after each episode
                percent_to_next_eval = ((self.step % self.cfg.eval_frequency) / self.cfg.eval_frequency) * 100
                percent_total = (self.step / self.cfg.num_train_steps) * 100
                steps_remaining = self.cfg.num_train_steps - self.step
                print(f"[Seed {self.cfg.seed}] Ep {episode} | Step {self.step}/{int(self.cfg.num_train_steps)} | "
                      f"Total: {percent_total:.1f}% | To Next Eval: {percent_to_next_eval:.1f}% | "
                      f"Remaining: {steps_remaining}")

            # If we just finished seed collection, force a fresh episode before
            # training starts. Set done=True and jump back to the top so the
            # if done: block resets the episode cleanly without stepping the env.
            if self.step == self.cfg.num_seed_steps and not done:
                print(f"[Seed {self.cfg.seed}] Seed steps complete at step {self.step}, starting fresh episode.")
                if self.use_pebble and ep_traj:
                    # Preserve preference segments collected during the seed
                    # episode before forcing the fresh post-seed reset.
                    self._add_episode_segments(ep_traj)
                episode_reward = 0
                rc_episode_reward = 0
                episode_step = 0
                obs = self.env.reset()
                self.agent.reset()
                ep_traj = []
                episode += 1
                self.logger.log('train/episode', episode, self.step)
                print(f"[Seed {self.cfg.seed}] Ep {episode} | Step {self.step}/{int(self.cfg.num_train_steps)} | Fresh episode started.")

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            # For Rc mode, timeouts are in-episode (done=False) and handled by
            # the wrapper. Use the timeout_reset flag to mask bootstrap on those
            # steps, since episode_step grows past _max_episode_steps after the
            # first timeout and the simple episode_step check breaks.
            if info.get('timeout_reset', False):
                done_no_max = 0
            else:
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            rc_episode_reward += reward

            # Store transition in trajectory for PEBBLE segment creation
            ep_traj.append({
                'obs': np.array(obs, copy=True),
                'action': np.array(action, copy=True),
                'reward': float(reward),
                'next_obs': np.array(next_obs, copy=True)
            })

            # If using PEBBLE and reward model is trained, use learned reward
            store_reward = float(reward)
            if self.use_pebble:
                pred = self._predict_reward(obs, action)
                if pred is not None:
                    store_reward = pred

            self.replay_buffer.add(obs, action, store_reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            prev_step = self.step
            self.step += 1
            self._sync_reward_schedule_with_step()

            if info.get('timeout_reset', False) and self.use_pebble and ep_traj:
                # Rc timeouts are robot-only resets inside the same logical
                # episode. Flush the finished sub-trajectory here so PEBBLE
                # does not wait until eventual success to see any segments.
                self._add_episode_segments(ep_traj)
                ep_traj = []

            # Heartbeat for long episodes (e.g., Rc) so progress is visible
            # even when no episode boundary is reached for a while.
            if self.progress_print_frequency > 0 and self.step % self.progress_print_frequency == 0:
                percent_total = (self.step / self.cfg.num_train_steps) * 100
                print(
                    f"[Seed {self.cfg.seed}] Progress | Step {self.step}/{int(self.cfg.num_train_steps)} "
                    f"({percent_total:.1f}%) | Episode {episode} step {episode_step} | "
                    f"Episode return so far: {rc_episode_reward:.2f}"
                )

            if _rc_mode and done:
                rc_steps_to_goal = episode_step
                print(
                    f"[Seed {self.cfg.seed}] Rc training success | Episode {episode} | "
                    f"Reached task in {rc_steps_to_goal} steps"
                )

            # At episode end, process trajectory into preference segments for PEBBLE
            if done and self.use_pebble:
                self._add_episode_segments(ep_traj)

            # Periodic PEBBLE query/train/relabel phase
            self._maybe_run_pebble()

            # Queue eval only when we cross an eval boundary mid-episode.
            if prev_step < self.next_eval_step <= self.step:
                self.eval_pending = True
                while self.next_eval_step <= self.step:
                    self.next_eval_step += self.cfg.eval_frequency
            
            # Queue video saving only after crossing the next video boundary.
            if prev_step < self.next_video_step <= self.step:
                self.video_pending = True

            # Force periodic train-metric flushing so logs continue even if episodes are long.
            if self.step > self.cfg.num_seed_steps and self.step % int(self.cfg.log_frequency) == 0:
                self.logger.dump(self.step, save=True, ty='train')
        
        # Final evaluation after training completes
        print(f"[Seed {self.cfg.seed}] Training completed at {self.step} steps. Running final evaluation...")
        self.logger.log('eval/episode', episode, self.step)
        self.evaluate(save_video=True)
        self.logger.dump(self.step)
        if getattr(self.cfg, 'save_checkpoint', False):
            self.save_checkpoint(self.cfg.checkpoint_path)
        print(f"[Seed {self.cfg.seed}] Final evaluation complete.")


if 'config_name' in inspect.signature(hydra.main).parameters:
    _hydra_main = hydra.main(config_path='config', config_name='train', version_base=None)
else:
    _hydra_main = hydra.main(config_path='config/train.yaml', strict=True)


@_hydra_main
def main(cfg):
    # Disable struct mode to allow dynamic config keys
    OmegaConf.set_struct(cfg, False)
    
    # Support multi-seed sequential runs
    run_seeds = getattr(cfg, 'run_seeds', [])
    if run_seeds and len(run_seeds) > 0:
        print(f"\n{'='*60}")
        print(f"Running {len(run_seeds)} seeds sequentially")
        print(f"Seeds: {run_seeds}")
        print(f"{'='*60}\n")
        
        completed = []
        failed = []
        start_time = time.time()
        
        for idx, seed_val in enumerate(run_seeds, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(run_seeds)}] Starting seed {seed_val}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            try:
                # Update config with current seed
                cfg.seed = int(seed_val)
                
                # Build the video_dir path with environment-specific grouping
                # For pendulum: include target_theta; For reacher: include reward_mode
                if hasattr(cfg, 'experiment') and cfg.experiment:
                    if cfg.env == 'pendulum':
                        target_theta_val = float(getattr(cfg, 'target_theta', 0.0))
                        cfg.video_dir = f'./exp/{cfg.experiment}/theta_{target_theta_val}/seed_{seed_val}'
                    elif cfg.env == 'reacher':
                        reward_mode_val = str(getattr(cfg, 'reward_mode', 'rb')).lower()
                        cfg.video_dir = f'./exp/{cfg.experiment}/reward_{reward_mode_val}/seed_{seed_val}'
                    else:
                        cfg.video_dir = f'./exp/{cfg.experiment}/seed_{seed_val}'
                else:
                    if cfg.env == 'pendulum':
                        target_theta_val = float(getattr(cfg, 'target_theta', 0.0))
                        cfg.video_dir = f'./exp/{cfg.env}_pebble/theta_{target_theta_val}/seed_{seed_val}'
                    elif cfg.env == 'reacher':
                        reward_mode_val = str(getattr(cfg, 'reward_mode', 'rb')).lower()
                        cfg.video_dir = f'./exp/{cfg.env}_pebble/reward_{reward_mode_val}/seed_{seed_val}'
                    else:
                        cfg.video_dir = f'./exp/{cfg.env}_pebble/seed_{seed_val}'
                
                # Mark this as a multi-seed run so Workspace uses video_dir as work_dir
                cfg._multi_seed_run = True
                
                # Create and run workspace
                workspace = Workspace(cfg)
                workspace.run()
                completed.append(seed_val)
                print(f"\n[OK] Completed seed {seed_val}\n")
            except Exception as e:
                failed.append(seed_val)
                print(f"\n[FAIL] Failed seed {seed_val}: {e}\n")
                import traceback
                traceback.print_exc()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"MULTI-SEED RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Completed: {completed}")
        if failed:
            print(f"Failed: {failed}")
        print(f"Total time: {int(duration//3600)}h {int((duration%3600)//60)}m {int(duration%60)}s")
        print(f"{'='*60}\n")
    else:
        # Single seed run (original behavior)
        workspace = Workspace(cfg)
        workspace.run()


if __name__ == '__main__':
    main()
