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

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

# import dmc2gym
import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from hydra.utils import get_class
from env_wrappers import PendulumWrapper
from env_wrappers import LunarLanderWrapper

def make_pendulum_env(cfg):
    """Function to make pendulum environment"""
    pendulum_cfg = getattr(cfg, 'pendulum', None)
    target_theta = 0.0
    max_episode_steps = 1000

    if pendulum_cfg is not None:
        target_theta = float(getattr(pendulum_cfg, 'target_theta', 0.0))
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

    if lunarlander_cfg is not None:
        continuous = bool(getattr(lunarlander_cfg, 'continuous', True))
        max_episode_steps = int(getattr(lunarlander_cfg, 'max_episode_steps', 1000))

    base_env = gym.make(
        'LunarLander-v3',
        continuous=continuous,
        max_episode_steps=max_episode_steps,
        render_mode='rgb_array' if cfg.save_video else None,
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
        print(f'[Seed {cfg.seed}] workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)

        requested_device = str(cfg.device).lower()
        if requested_device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print(f"[Seed {cfg.seed}] Requested CUDA but CUDA is unavailable. Falling back to MPS.")
            else:
                self.device = torch.device('cpu')
                print(f"[Seed {cfg.seed}] Requested CUDA but CUDA is unavailable. Falling back to CPU.")
        elif requested_device == 'mps' and not torch.backends.mps.is_available():
            self.device = torch.device('cpu')
            print(f"[Seed {cfg.seed}] Requested MPS but MPS is unavailable. Falling back to CPU.")
        else:
            self.device = torch.device(cfg.device)

        # Keep Hydra config in sync so instantiated modules receive a valid device.
        cfg.device = str(self.device)

        # Print device information at the start
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'[Seed {cfg.seed}] Using device: {self.device}')
            print(f'[Seed {cfg.seed}] GPU: {gpu_name}, Total Memory: {total_memory:.2f} GB')
        elif self.device.type == 'mps':
            print(f'[Seed {cfg.seed}] Using device: {self.device} (Apple Silicon GPU)')
        else:
            print(f'[Seed {cfg.seed}] Using device: {self.device}')

        if cfg.env == "pendulum":
            self.env = make_pendulum_env(cfg)
            self.eval_env = make_pendulum_env(cfg)
        elif cfg.env == "lunarlander":
            self.env = make_lunarlander_env(cfg)
            self.eval_env = make_lunarlander_env(cfg)
        else:
            self.env = utils.make_env(cfg)
            self.eval_env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        
        # Handle both discrete and continuous action spaces
        if hasattr(self.env.action_space, 'n'):
            # Discrete action space
            cfg.agent.params.action_dim = int(self.env.action_space.n)
            cfg.agent.params.action_range = [0, int(self.env.action_space.n) - 1]
            agent_class = get_class('agent.sac.DiscreteSACAgent')
            critic_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.discrete_double_q_critic, resolve=True)
            )
            actor_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.categorical_actor, resolve=True)
            )
        else:
            # Continuous action space
            cfg.agent.params.action_dim = self.env.action_space.shape[0]
            cfg.agent.params.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
            agent_class = get_class('agent.sac.SACAgent')
            critic_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.double_q_critic, resolve=True)
            )
            actor_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.diag_gaussian_actor, resolve=True)
            )

        agent_params = OmegaConf.to_container(cfg.agent.params, resolve=True)
        agent_params['critic_cfg'] = critic_cfg
        agent_params['actor_cfg'] = actor_cfg
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

        self.video_recorder = VideoRecorder(
            cfg.video_dir if cfg.save_video else None)
        self.step = 0
        self.next_eval_step = cfg.eval_frequency
        self.last_eval_step = -1

    def evaluate(self, eval_step, save_video=False):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0 and save_video))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.eval_env.step(action)
                if save_video:
                    self.video_recorder.record(self.eval_env)
                episode_reward += reward

            average_episode_reward += episode_reward
            if save_video and episode == 0:
                video_path = f'seed_{self.cfg.seed}_train_steps_{eval_step}.mp4'
                self.video_recorder.save(video_path)
                print(f"[Seed {self.cfg.seed}] Video saved to video/{video_path}")
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        eval_step)
        self.logger.dump(eval_step)
        self.last_eval_step = int(eval_step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        # Log evaluation at step 0 before any training starts.
        self.logger.log('eval/episode', 0, 0)
        self.evaluate(eval_step=0, save_video=False)
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                
                # Print progress after each episode
                percent_to_next_eval = ((self.step % self.cfg.eval_frequency) / self.cfg.eval_frequency) * 100
                percent_total = (self.step / self.cfg.num_train_steps) * 100
                steps_remaining = self.cfg.num_train_steps - self.step
                print(f"[Seed {self.cfg.seed}] Ep {episode} | Step {self.step}/{int(self.cfg.num_train_steps)} | "
                      f"Total: {percent_total:.1f}% | To Next Eval: {percent_to_next_eval:.1f}% | "
                      f"Remaining: {steps_remaining}")

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

            # Evaluate immediately at exact boundaries: 10k, 20k, ...
            while self.next_eval_step <= self.step:
                target_eval_step = int(self.next_eval_step)
                self.logger.log('eval/episode', episode, target_eval_step)
                should_save_video = (
                    target_eval_step % int(self.cfg.video_frequency) == 0
                    or target_eval_step == int(self.cfg.num_train_steps)
                )
                if should_save_video:
                    print(f"[Seed {self.cfg.seed}] Saving video at step {target_eval_step}")
                self.evaluate(eval_step=target_eval_step, save_video=should_save_video)
                self.next_eval_step += int(self.cfg.eval_frequency)
        
        # Final evaluation after training completes
        if self.last_eval_step != int(self.step):
            print(f"[Seed {self.cfg.seed}] Training completed at {self.step} steps. Running final evaluation...")
            self.logger.log('eval/episode', episode, self.step)
            self.evaluate(eval_step=self.step, save_video=True)
            self.logger.dump(self.step)
            print(f"[Seed {self.cfg.seed}] Final evaluation complete.")


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
