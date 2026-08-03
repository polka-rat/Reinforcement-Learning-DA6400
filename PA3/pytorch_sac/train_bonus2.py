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
        # --- end pebble components ---

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
            self.logger.log('train/pebble_reward_loss', float(last_loss.item()), self.step)
            self.logger.log('train/pebble_prefs', self.pref_buffer.num_prefs(), self.step)
            self.logger.log('train/pebble_queries_used', self.pref_queries_used, self.step)

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
            self.logger.log('train/pebble_queries_added', added, self.step)
        self._train_reward_models()
        self._relabel_replay_buffer()

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
        
        # trajectory buffer for current episode (for PEBBLE segments)
        ep_traj = []  # list of (obs, action, reward, next_obs)

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
                ep_traj = []

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

            # store transition in trajectory for PEBBLE segment creation
            ep_traj.append({
                'obs': np.array(obs, copy=True),
                'action': np.array(action, copy=True),
                'reward': float(reward),
                'next_obs': np.array(next_obs, copy=True)
            })

            # If using PEBBLE and reward model is trained, use learned reward.
            store_reward = float(reward)
            if self.use_pebble:
                pred = self._predict_reward(obs, action)
                if pred is not None:
                    store_reward = pred

            self.replay_buffer.add(obs, action, store_reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass

            # At episode end, process trajectory into preference segments.
            if done and self.use_pebble:
                self._add_episode_segments(ep_traj)

            # Periodic PEBBLE query/train/relabel phase.
            self._maybe_run_pebble()

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

    # If cfg.run_experiments is true, run multiple experiments (seeds x thetas x budgets)
    if getattr(cfg, 'run_experiments', True):
        # Run PEBBLE experiments for selected target thetas and feedback budgets
        pebble_thetas = [-60, 90, 120]
        pebble_budgets = [1000, 1500, 2500]  # different feedback query budgets
        # sensible seed set (can be overridden in cfg)
        default_seeds = [98, 42]
        seeds = default_seeds
        orig_cwd = os.getcwd()
        try:
            for budget in pebble_budgets:
                for seed in seeds:
                    for theta in pebble_thetas:
                        # set experiment-specific fields on cfg
                        cfg.seed = int(seed)
                        cfg.save_final_checkpoint = False
                        if 'pendulum' not in cfg:
                            cfg.pendulum = {}
                        # set target theta (radians)
                        try:
                            cfg.pendulum.target_theta = float(np.deg2rad(theta))
                        except Exception:
                            cfg.pendulum['target_theta'] = float(np.deg2rad(theta))

                        # enable pebble in config (keeps default hyperparams from config/train.yaml)
                        if 'pebble' not in cfg:
                            cfg.pebble = {}
                        cfg.pebble.enabled = True
                        cfg.pebble.budget = int(budget)

                        cfg.experiment = f'pebble_budget_{budget}_theta_{theta}_seed_{seed}'

                        run_dir = os.path.join(project_root, 'exp', 'pebble', f'budget_{budget}', f'theta_{theta}', f'seed_{seed}')
                        os.makedirs(run_dir, exist_ok=True)
                        print(f"Starting PEBBLE run budget={budget} theta={theta} seed={seed} -> {run_dir}")
                        os.chdir(run_dir)
                        try:
                            workspace = Workspace(cfg)
                            workspace.run()
                        finally:
                            os.chdir(orig_cwd)
        finally:
            os.chdir(orig_cwd)
    else:
        workspace = Workspace(cfg)
        workspace.run()


if __name__ == '__main__':
    main()
