#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl


# import dmc2gym
import gymnasium as gym
import hydra
import video
from video import VideoRecorder
import logger
from logger import Logger
import replay_buffer
from replay_buffer import ReplayBuffer
import utils
import env_wrappers
from env_wrappers import PendulumWrapper, LunarLanderWrapper
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

def make_pendulum_env(cfg):
    """Function to make pendulum environment"""
    pendulum_cfg = getattr(cfg, 'pendulum', None)
    target_theta = 0.0
    max_episode_steps = 1000
    reward_scale = 1.0

    if pendulum_cfg is not None:
        target_theta = float(getattr(pendulum_cfg, 'target_theta', 0.0))
        max_episode_steps = int(getattr(pendulum_cfg, 'max_episode_steps', 1000))
        reward_scale = float(getattr(pendulum_cfg, 'reward_scale', 1.0))

    # enable rgb_array render mode when saving video so env.render() returns frames
    if getattr(cfg, 'save_video', False):
        base_env = gym.make('Pendulum-v1', max_episode_steps=max_episode_steps, render_mode='rgb_array')
    else:
        base_env = gym.make('Pendulum-v1', max_episode_steps=max_episode_steps)

    env = PendulumWrapper(base_env, target_theta=target_theta, reward_scale=reward_scale)
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

    if lunarlander_cfg is not None:
        continuous = bool(getattr(lunarlander_cfg, 'continuous', True))
        max_episode_steps = int(getattr(lunarlander_cfg, 'max_episode_steps', 1000))

    # enable rgb_array render mode when saving video so env.render() returns frames
    if getattr(cfg, 'save_video', False):
        base_env = gym.make(
            'LunarLander-v3',
            continuous=continuous,
            max_episode_steps=max_episode_steps,
            render_mode='rgb_array',
        )
    else:
        base_env = gym.make(
            'LunarLander-v3',
            continuous=continuous,
            max_episode_steps=max_episode_steps,
        )
    env = LunarLanderWrapper(base_env)

    if hasattr(env, 'seed'):
        env.seed(cfg.seed)
    else:
        env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    return env

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                     save_tb=cfg.log_save_tb,
                     log_frequency=cfg.log_frequency,
                     agent=cfg.agent.name,
                     console=False)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        if cfg.env == "pendulum":
            self.env = make_pendulum_env(cfg)
        elif cfg.env == "lunarlander":
            self.env = make_lunarlander_env(cfg)
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        # checkpoint directory inside the run directory
        self.ckpt_dir = os.path.join(self.work_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def save_checkpoint(self):
        try:
            ckpt = {
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict(),
                'actor_opt': self.agent.actor_optimizer.state_dict(),
                'critic_opt': self.agent.critic_optimizer.state_dict(),
                'log_alpha': self.agent.log_alpha.detach().cpu(),
                'log_alpha_opt': self.agent.log_alpha_optimizer.state_dict(),
                'step': int(self.step)
            }
            path = os.path.join(self.ckpt_dir, f'ckpt_{self.step}.pt')
            torch.save(ckpt, path)
            return path
        except Exception:
            return None

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        try:
            self.agent.actor.load_state_dict(ckpt['actor'])
            self.agent.critic.load_state_dict(ckpt['critic'])
            # optimizers' state dicts can be loaded if optimizers exist
            try:
                self.agent.actor_optimizer.load_state_dict(ckpt['actor_opt'])
                self.agent.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            except Exception:
                pass
            try:
                self.agent.log_alpha.data = ckpt.get('log_alpha', self.agent.log_alpha).to(self.device)
                if 'log_alpha_opt' in ckpt:
                    try:
                        self.agent.log_alpha_optimizer.load_state_dict(ckpt['log_alpha_opt'])
                    except Exception:
                        pass
            except Exception:
                pass
            self.step = int(ckpt.get('step', self.step))
        except Exception as e:
            raise e

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        # Create a progress bar for timesteps (one per workspace / experiment).
        pbar = None
        try:
            if tqdm is not None:
                pbar = tqdm(total=int(self.cfg.num_train_steps), initial=self.step, desc='timesteps')
        except Exception:
            pbar = None
        
        # Evaluate at the beginning of training (step 0)
        self.logger.log('eval/episode', episode, self.step)
        self.evaluate()
        
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        if getattr(self.cfg, 'save_final_checkpoint', False):
            ckpt_path = self.save_checkpoint()
            if ckpt_path is not None:
                print(f'Saved final checkpoint: {ckpt_path}')

@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    project_root = hydra.utils.get_original_cwd()

    # If cfg.run_experiments is true, run multiple experiments (seeds x thetas)
    if getattr(cfg, 'run_experiments', True):
        # default lists (can be overridden by cfg.run_seeds / cfg.run_thetas)
        default_thetas = [0, -10, 30, -60]
        default_seeds = [52, 8, 42, 32, 31, 10, 98] #[44, 63, 11, 15, 90, 21, 0, 12, 52, 8]

        thetas = default_thetas
        seeds = default_seeds
        checkpoint_seed = default_seeds[0]

        alpha_thetas = [-60, 90, 120, -150]
        init_temperatures = [0.01, 0.03, 0.05]

        orig_cwd = os.getcwd()
        try:
            for seed in seeds:

                # Q2

                for theta in thetas:
                    # set experiment-specific fields on cfg
                    cfg.seed = int(seed)
                    cfg.save_final_checkpoint = int(seed) == int(checkpoint_seed)
                    if 'pendulum' not in cfg:
                        cfg.pendulum = {}
                    # set target theta
                    try:
                        cfg.pendulum.target_theta = float(np.deg2rad(theta))
                    except Exception:
                        # if cfg.pendulum is a dict-like
                        cfg.pendulum['target_theta'] = float(np.deg2rad(theta))

                    cfg.experiment = f'theta_{theta}_seed_{seed}'

                    run_dir = os.path.join(project_root, 'exp', 'theta_sweep', f'theta_{theta}', f'seed_{seed}')
                    os.makedirs(run_dir, exist_ok=True)
                    print(f"Starting run theta={theta} seed={seed} -> {run_dir}")
                    os.chdir(run_dir)
                    try:
                        workspace = Workspace(cfg)
                        workspace.run()
                    finally:
                        os.chdir(orig_cwd)


                # Q5i

                for theta in alpha_thetas:
                    for init_temp in init_temperatures:
                        # set experiment-specific fields on cfg
                        cfg.seed = int(seed)
                        cfg.save_final_checkpoint = False
                        cfg.pendulum.target_theta = float(np.deg2rad(theta))

                        # set fixed temperature
                        cfg.agent.params.init_temperature = float(init_temp)
                        cfg.agent.params.learnable_temperature = False

                        cfg.experiment = f'theta_{theta}_alpha_{init_temp}_seed_{seed}'

                        run_dir = os.path.join(project_root, 'exp', 'alpha_sweep', f'theta_{theta}', f'alpha_{init_temp}', f'seed_{seed}')
                        os.makedirs(run_dir, exist_ok=True)
                        print(f"Starting run theta={theta} alpha={init_temp} seed={seed} -> {run_dir}")
                        os.chdir(run_dir)
                        try:
                            workspace = Workspace(cfg)
                            workspace.run()
                        finally:
                            os.chdir(orig_cwd)

                # Q5ii: Reward scaling experiment for theta=90 with 10x and 0.1x scaling
                # theta = 90
                # reward_scales = [10.0, 0.1]
                
                # for reward_scale in reward_scales:
                #     # Experiment 1: Fixed alpha = 0.05 with scaling
                #     cfg.seed = int(seed)
                #     cfg.save_final_checkpoint = False
                #     cfg.pendulum.target_theta = float(np.deg2rad(theta))
                #     cfg.pendulum.reward_scale = float(reward_scale)
                    
                #     # set fixed temperature at alpha=0.05
                #     cfg.agent.params.init_temperature = 0.05
                #     cfg.agent.params.learnable_temperature = False
                    
                #     cfg.experiment = f'theta_{theta}_scale_{reward_scale}_alpha_0.05_seed_{seed}'
                    
                #     run_dir = os.path.join(project_root, 'exp', 'fixed_0.05', f'{reward_scale}', f'seed_{seed}')
                #     os.makedirs(run_dir, exist_ok=True)
                #     print(f"Starting run theta={theta} scale={reward_scale} alpha=0.05 (fixed) seed={seed} -> {run_dir}")
                #     os.chdir(run_dir)
                #     try:
                #         workspace = Workspace(cfg)
                #         workspace.run()
                #     finally:
                #         os.chdir(orig_cwd)
                    
                #     # Experiment 2: Automated alpha tuning with scaling
                #     cfg.seed = int(seed)
                #     cfg.save_final_checkpoint = int(seed) == int(checkpoint_seed)
                #     cfg.pendulum.target_theta = float(np.deg2rad(theta))
                #     cfg.pendulum.reward_scale = float(reward_scale)
                    
                #     # enable learnable temperature (automated tuning)
                #     cfg.agent.params.learnable_temperature = True
                    
                #     cfg.experiment = f'theta_{theta}_scale_{reward_scale}_alpha_auto_seed_{seed}'
                    
                #     run_dir = os.path.join(project_root, 'exp', 'auto', f'{reward_scale}', f'seed_{seed}')
                #     os.makedirs(run_dir, exist_ok=True)
                #     print(f"Starting run theta={theta} scale={reward_scale} alpha=auto seed={seed} -> {run_dir}")
                #     os.chdir(run_dir)
                #     try:
                #         workspace = Workspace(cfg)
                #         workspace.run()
                #     finally:
                #         os.chdir(orig_cwd)

        finally:
            os.chdir(orig_cwd)
    else:
        workspace = Workspace(cfg)
        workspace.run()


if __name__ == '__main__':
    main()
