#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym

from dqn import DQNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on discrete LunarLander-v3.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 63, 11], help="Seed list")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Upper bound on episodes")
    parser.add_argument("--max_train_steps", type=int, default=500000, help="Stop exactly at this many env steps")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Episode cap")
    parser.add_argument("--target_update_steps", type=int, default=1000, help="Target net update interval")
    parser.add_argument("--eval_frequency_steps", type=int, default=10000, help="Eval interval in env steps")
    parser.add_argument("--num_eval_episodes", type=int, default=20, help="Episodes per evaluation")
    parser.add_argument("--replay_factor", type=int, default=1, help="Gradient updates per env step")
    parser.add_argument("--buffer_capacity", type=int, default=200000, help="Replay capacity")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Per-episode epsilon decay")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Q network hidden size")
    parser.add_argument("--out_dir", type=str, default="exp/dqn_lunarlander", help="Output root directory")
    parser.add_argument("--save_video", action="store_true", help="Save eval rollout videos during training")
    parser.add_argument("--video_frequency_steps", type=int, default=30000, help="Video save interval in env steps")
    parser.add_argument("--video_dir_name", type=str, default="dqn_videos", help="Video folder name under each seed dir")
    return parser.parse_args()


def make_env(max_episode_steps):
    return gym.make("LunarLander-v3", continuous=False, max_episode_steps=max_episode_steps)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== DQN training started | seed={seed} ===")
        env = make_env(args.max_episode_steps)
        trainer = DQNTrainer(
            env=env,
            seed=seed,
            buffer_capacity=args.buffer_capacity,
            batch_size=args.batch_size,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            lr=args.lr,
            max_episode_steps=args.max_episode_steps,
            hidden_dim=args.hidden_dim,
        )
        trainer.train(
            num_episodes=args.num_episodes,
            max_train_steps=args.max_train_steps,
            target_update_steps=args.target_update_steps,
            eval_frequency_steps=args.eval_frequency_steps,
            num_eval_episodes=args.num_eval_episodes,
            replay_factor=args.replay_factor,
            out_dir=args.out_dir,
            save_video=args.save_video,
            video_frequency_steps=args.video_frequency_steps,
            video_dir_name=args.video_dir_name,
        )
        env.close()
        print(f"=== DQN training complete | seed={seed} ===")

    print("\nAll requested DQN runs completed.")


if __name__ == "__main__":
    main()
