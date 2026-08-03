import gymnasium as gym
import numpy as np
import torch
from torch import nn
from dqn import Trainer

GAMMA = 0.99
TRUNC_LENGTH = 2000

# creating the modified environment
env = gym.make('MountainCar-v0', max_episode_steps=TRUNC_LENGTH)
seeds = [44, 63, 11, 15, 90, 21, 0, 12, 52, 8, 42, 32, 31, 10, 98]

import gc
import os

# Create logging directory
os.makedirs("logs_4d", exist_ok=True)

# Target network refresh rate sweep
# Current default is 20, so we test: 10, 15, 20, 25, 30
replay_factors = [1]
target_update_steps_list = [15]  # 2 smaller, 2 larger

total_runs = len(replay_factors) * len(target_update_steps_list) * len(seeds)
completed = 0

print(f"Total training runs: {total_runs}")
print(f"Replay factors (ρ): {replay_factors}")
print(f"Target update steps: {target_update_steps_list}")
print(f"Seeds: {len(seeds)} seeds")
print("=" * 80)

results = {}




for rho in replay_factors:
    results[rho] = {}
    
    for target_steps in target_update_steps_list:
        results[rho][target_steps] = []
        
        for seed_idx, seed in enumerate(seeds):
            completed += 1
            progress_pct = (completed / total_runs) * 100
            
            print(f"\n[{progress_pct:.1f}%] Run {completed}/{total_runs}")
            print(f"  ρ={rho}, target_update_steps={target_steps}, seed={seed} ({seed_idx+1}/{len(seeds)})")
            
            try:
                env = gym.make('MountainCar-v0', max_episode_steps=TRUNC_LENGTH)
                
                # Try to use wandb, fall back to local logging if it fails
                use_wandb = True
                try:
                    import wandb
                    import threading
                    
                    wandb_success = [False]
                    def init_wandb():
                        try:
                            wandb.init(project="RL_PA2", name=f"target_sweep_rho_{rho}_steps_{target_steps}_seed_{seed}", mode="online")
                            wandb_success[0] = True
                        except:
                            pass
                    
                    wandb_thread = threading.Thread(target=init_wandb, daemon=True)
                    wandb_thread.start()
                    wandb_thread.join(timeout=5)  # 5 second timeout
                    
                    if not wandb_success[0]:
                        use_wandb = False
                except Exception as e:
                    print(f"    ⚠ W&B initialization failed, using offline logging")
                    use_wandb = False
                
                dqn_trainer = Trainer(
                    env=env,
                    gamma=GAMMA,
                    seed=int(seed),
                    use_wandb=use_wandb,
                    wandb_project="RL_PA2",
                    wandb_run_name=f"target_sweep_rho_{rho}_steps_{target_steps}_seed_{seed}",
                    trunc_length=TRUNC_LENGTH
                )
                
                train_rewards = dqn_trainer.train(num_episodes=1500, target_update_steps=target_steps, replay_factor=rho)
                results[rho][target_steps].append(train_rewards)
                
                
                # Save results after each run for recovery
                import pickle
                with open(f"logs_4d/rho_{rho}_steps_{target_steps}_seed_{seed}.pkl", "wb") as f:
                    pickle.dump(train_rewards, f)
                
                # Save aggregated results pkl periodically after each seed completes
                with open("logs_4d/aggregated_results_checkpoint.pkl", "wb") as f:
                    pickle.dump(results, f)
                
                env.close()
                if use_wandb:
                    try:
                        wandb.finish()
                    except:
                        pass
                
                del dqn_trainer, env
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"  ✓ Complete!")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[rho][target_steps].append(None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("All training complete!")

# Save aggregated results
import pickle
with open("logs_4d/target_network_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Results saved to target_network_results.pkl")