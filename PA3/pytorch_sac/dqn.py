import csv
import os
import random
import time
from collections import deque

import imageio
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.asarray, zip(*batch))
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).long()
        reward = torch.from_numpy(reward).float()
        next_state = torch.from_numpy(next_state).float()
        done = torch.from_numpy(done).float()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DQNTrainer:
    def __init__(
        self,
        env,
        seed,
        buffer_capacity=200000,
        batch_size=256,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        lr=3e-4,
        max_episode_steps=1000,
        hidden_dim=256,
    ):
        self.env = env
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_episode_steps = max_episode_steps

        self._set_seed(seed)
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.buffer = ReplayBuffer(buffer_capacity)
        self.policy_net = DQN(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(dim=1).item()

    def _get_greedy_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(dim=1).item()

    def _train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        current_q = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(dim=1)[0]
            target_q = reward + self.gamma * next_q * (1 - done)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _record_eval_video(self, eval_env, video_path, max_steps, random_policy=False):
        frames = []
        state, _ = eval_env.reset(seed=self.seed)
        for _ in range(max_steps):
            if random_policy:
                action = eval_env.action_space.sample()
            else:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            state, _, terminated, truncated, _ = eval_env.step(action)
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break
        if frames:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, frames, fps=30)

    def _evaluate_greedy(self, eval_env, num_eval_episodes, random_policy=False):
        returns = []
        for ep in range(num_eval_episodes):
            state, _ = eval_env.reset(seed=self.seed + 10_000 + ep)
            ep_return = 0.0
            for _ in range(self.max_episode_steps):
                if random_policy:
                    action = eval_env.action_space.sample()
                else:
                    action = self._get_greedy_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                ep_return += reward
                if terminated or truncated:
                    break
            returns.append(ep_return)
        return float(np.mean(returns))

    def train(
        self,
        num_episodes=1500,
        max_train_steps=500000,
        target_update_steps=1000,
        replay_factor=1,
        eval_frequency_steps=10000,
        num_eval_episodes=20,
        out_dir="dqn_lunarlander",
        save_video=False,
        video_frequency_steps=30000,
        video_dir_name="dqn_videos",
    ):
        run_dir = os.path.join(out_dir, f"seed_{self.seed}")
        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "train.csv")
        eval_csv_path = os.path.join(run_dir, "eval.csv")
        weights_path = os.path.join(run_dir, "policy_net.pt")
        video_dir = os.path.join(run_dir, video_dir_name)
        if save_video:
            os.makedirs(video_dir, exist_ok=True)
        eval_env = self.env.spec.make(render_mode="rgb_array" if save_video else None)

        total_env_steps = 0
        episode_rewards = []
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["episode", "episode_reward", "episode_steps", "total_env_steps", "epsilon"],
        )
        csv_writer.writeheader()
        eval_csv_file = open(eval_csv_path, "w", newline="")
        eval_csv_writer = csv.DictWriter(
            eval_csv_file, fieldnames=["episode", "episode_reward", "step"]
        )
        eval_csv_writer.writeheader()

        # Evaluate at step 0 before training starts.
        eval_reward = self._evaluate_greedy(
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            random_policy=True,
        )
        eval_csv_writer.writerow({"episode": 0, "episode_reward": eval_reward, "step": 0})
        eval_csv_file.flush()
        if save_video:
            step0_video_path = os.path.join(video_dir, f"seed_{self.seed}_train_steps_0.mp4")
            self._record_eval_video(
                eval_env=eval_env,
                video_path=step0_video_path,
                max_steps=self.max_episode_steps,
                random_policy=True,
            )
            print(f"[DQN seed {self.seed}] Saved video: {step0_video_path}")

        for episode in range(1, num_episodes + 1):
            if total_env_steps >= max_train_steps:
                break
            episode_start_time = time.time()
            state, _ = self.env.reset(seed=self.seed + episode)
            episode_reward = 0.0
            episode_steps = 0
            episode_loss_sum = 0.0
            episode_loss_count = 0

            for _ in range(self.max_episode_steps):
                action = self._get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(state, action, reward, next_state, done)

                for _ in range(replay_factor):
                    loss = self._train_step()
                    if loss is not None:
                        episode_loss_sum += loss
                        episode_loss_count += 1

                total_env_steps += 1
                if total_env_steps % target_update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if total_env_steps % eval_frequency_steps == 0:
                    eval_reward = self._evaluate_greedy(
                        eval_env=eval_env, num_eval_episodes=num_eval_episodes
                    )
                    eval_csv_writer.writerow(
                        {
                            "episode": episode,
                            "episode_reward": eval_reward,
                            "step": total_env_steps,
                        }
                    )
                    eval_csv_file.flush()
                if save_video and total_env_steps % video_frequency_steps == 0:
                    video_path = os.path.join(
                        video_dir, f"seed_{self.seed}_train_steps_{total_env_steps}.mp4"
                    )
                    self._record_eval_video(
                        eval_env=eval_env,
                        video_path=video_path,
                        max_steps=self.max_episode_steps,
                    )
                    print(f"[DQN seed {self.seed}] Saved video: {video_path}")

                state = next_state
                episode_reward += reward
                episode_steps += 1
                if done or total_env_steps >= max_train_steps:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(episode_reward)
            csv_writer.writerow(
                {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "episode_steps": episode_steps,
                    "total_env_steps": total_env_steps,
                    "epsilon": self.epsilon,
                }
            )
            csv_file.flush()
            duration = time.time() - episode_start_time
            avg_loss = (episode_loss_sum / episode_loss_count) if episode_loss_count > 0 else float("nan")
            buffer_fill_ratio = len(self.buffer) / float(self.buffer.buffer.maxlen)
            print(
                f"| train | E: {episode} | S: {total_env_steps} | R: {episode_reward:.4f} | "
                f"D: {duration:04.1f} s | EPS: {self.epsilon:.4f} | "
                f"LOSS: {avg_loss:.4f} | BFR: {buffer_fill_ratio:.4f} | ESTEP: {episode_steps}"
            )

        torch.save(self.policy_net.state_dict(), weights_path)
        csv_file.close()
        eval_csv_file.close()
        if eval_env is not None:
            eval_env.close()
        return episode_rewards
