# PA3

Changes made on top of the pytorch_sac implementation

## 1) Environment Wrappers

File: `pytorch_sac/env_wrappers.py`

### PendulumWrapper
- Wraps `Pendulum-v1` from Gymnasium.
- Supports configurable target angle via `target_theta`.
- Uses a custom reward function:
  - `reward = -(angle_error^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)`
  - where `angle_error = angle_normalize(theta - target_theta)`.
- Normalizes Gymnasium step/reset output into the older training loop format:
  - `reset() -> obs`
  - `step() -> (obs, reward, done, info)`

### LunarLanderWrapper
- Wraps `LunarLander-v3`.
- Keeps the environment default reward by default (via `_reward(env_reward)`). Can define custom reward function.
- Also normalizes Gymnasium API output into `(obs, reward, done, info)`.


## 2) Environment Loading Changes

File: `pytorch_sac/train.py`

### `make_pendulum_env(cfg)`
- Creates `Pendulum-v1` with configurable episode length.
- Reads from config:
  - `pendulum.target_theta`
  - `pendulum.max_episode_steps`
- Applies `PendulumWrapper`.
- Seeds environment and action space.

### `make_lunarlander_env(cfg)`
- Creates `LunarLander-v3`.
- Reads from config (when present):
  - `lunarlander.continuous` (continuous/discrete mode)
  - `lunarlander.max_episode_steps`
- Applies `LunarLanderWrapper`.
- Seeds environment and action space.

### Workspace env selection
- `env == "pendulum"` -> uses `make_pendulum_env`.
- `env == "lunarlander"` -> uses `make_lunarlander_env`.
- otherwise falls back to original `utils.make_env(cfg)` path.

## 3) Config Changes

File: `pytorch_sac/config/train.yaml`

- Added Pendulum-specific config block:
  - `pendulum.target_theta`
  - `pendulum.max_episode_steps` (set to 1000)

Example:

```yaml
pendulum:
  target_theta: 0.0
  max_episode_steps: 1000
```


## 4) Logging Behavior

- Train/eval scalar logs are produced by `pytorch_sac/logger.py`.
- Outputs are saved in each run directory:
  - `train.csv`
  - `eval.csv`
  - `tb/` (when `log_save_tb: true`)
- Console logs use compact fields such as:
  - `E` episode, `S` step, `R` episode reward,
  - `D` duration, `BR` batch reward,
  - `ALOSS`, `CLOSS`, `TLOSS`, `TVAL`, `AENT`.

## 5) Environment Setup Notes

File: `env.yml`

```bash
conda env create -f env.yml
```

## 6) Training Scripts for Pendulum-v1

Main training scripts are in `pytorch_sac/`:

- `pytorch_sac/train.py`
  - Base SAC runner for the main experiments.
  - Used for the standard theta sweep and fixed-alpha comparisons.
  - Saves runs under paths like `exp/theta_sweep/...` and `exp/alpha_sweep/...`.

- `pytorch_sac/train_bonus.py`
  - PEBBLE-enabled training script.

- `pytorch_sac/train_last.py`
  - Reward-scale sweep runner.
  - Compares auto-alpha vs fixed-alpha behavior across different reward scales.
  - Saves runs under `exp/scale_sweep/...`.


## 7) Reacher Environment & How to Run

This project includes a `ReacherWrapper` (in `pytorch_sac/env_wrappers.py`) that supports three reward formulations:

- `ra`: Dense reward — `1.0` when fingertip in target, otherwise `-distance - 0.01*|action|^2`. Fixed-length episodes (T=1000).
- `rb`: Sparse reward — `1.0` when in target, otherwise `0.0`. Fixed-length episodes (T=1000).
- `rc`: Variable-length episodic formulation — base reward `-1.0` each step; episode terminates early on success, and timeouts apply a reset penalty and a robot-only reset (see wrapper for details).

Running examples (Hydra overrides):

- Run Reacher with sparse reward (`rb`):

```bash
python -m pytorch_sac.train_ll_and_reacher env=reacher reward_mode=rb
```

- Run Reacher with dense reward (`ra`) and custom target threshold:

```bash
python -m pytorch_sac.train_ll_and_reacher env=reacher reward_mode=ra reacher.target_distance_threshold=0.02
```

- Run Reacher with `rc` mode (variable-length) and PEBBLE enabled:

```bash
python -m pytorch_sac.train_ll_and_reacher env=reacher reward_mode=rc pebble.enabled=true pebble.budget=2000
```

- Run multiple seeds sequentially (overrides `run_seeds` list):

```bash
python -m pytorch_sac.train_ll_and_reacher env=reacher +run_seeds=[44,63,11]
```

- Enable video recording for an experiment:

```bash
python -m pytorch_sac.train_ll_and_reacher env=reacher save_video=true
```

Notes and tips:

- The `reacher` config block in `pytorch_sac/config/train.yaml` contains convenient defaults for `reward_mode`, `max_episode_steps` and evaluation settings — override with Hydra CLI args as shown above.
- Use `pebble` overrides to experiment with preference-based tuning (enable with `pebble.enabled=true` and adjust `pebble.budget`, `segment_length`, etc.).


## 9) LunarLander Usage (hovering + changing alpha)

`LunarLanderWrapper` (in `pytorch_sac/env_wrappers.py`) provides an optional "hover box" reward shaping mode plus configurable reward-flip scheduling used in experiments.

- Key lunarlander config options (Hydra names):
  - `lunarlander.reward_mode`: `default` or `hover_box` (adds `hover_bonus` when inside box)
  - `lunarlander.hover_bonus`, `lunarlander.hover_x_threshold`, `lunarlander.hover_y_min`, `lunarlander.hover_y_max`
  - `lunarlander.reward_flip_enabled` (bool), `lunarlander.reward_flip_step` (step to flip), `lunarlander.reward_flip_bonus` (value to add/subtract when flipping)
  - `lunarlander.continuous` (use continuous action space)

- Alpha (temperature) control for SAC:
  - Use `agent.params.learnable_temperature=true` to enable a learnable temperature (changing alpha / auto-alpha).
  - Use `agent.params.learnable_temperature=false` and set `agent.params.init_temperature=<value>` to freeze alpha to a fixed value (e.g., `0.01`).
  - Tweak `agent.params.alpha_lr` and `agent.params.alpha_betas` when using learnable temperature.

Running examples:

- Run LunarLander with hover-box reward and hover bonus:

```bash
python -m pytorch_sac.train_ll_and_reacher env=lunarlander lunarlander.reward_mode=hover_box lunarlander.hover_bonus=200
```

- Run with fixed alpha (α=0.01):

```bash
python -m pytorch_sac.train_ll_and_reacher env=lunarlander agent.params.learnable_temperature=false agent.params.init_temperature=0.01
```

- Run with learnable (changing) alpha / auto-alpha:

```bash
python -m pytorch_sac.train_ll_and_reacher env=lunarlander agent.params.learnable_temperature=true agent.params.init_temperature=0.1 agent.params.alpha_lr=1e-4
```

- Enable the reward-flip schedule used in experiments (flip reward at step N):

```bash
python -m pytorch_sac.train_ll_and_reacher env=lunarlander lunarlander.reward_flip_enabled=true lunarlander.reward_flip_step=250000 lunarlander.reward_flip_bonus=-100
```

## 10) Discrete SAC for LunarLander

File: `pytorch_sac/train_discrete_SAC.py`

- Training script that supports both continuous and discrete action spaces.
- For discrete action spaces (LunarLander with `continuous=false`):
  - Uses `DiscreteSACAgent` with a `CategoricalActor` and `DiscreteDoubleQCritic`.
  - Action dimension is set to `env.action_space.n`.
- Also supports the standard continuous SAC path (Pendulum, continuous LunarLander).
- Runs via Hydra; config in `pytorch_sac/config/train.yaml`.

Running examples:

- Run discrete SAC on LunarLander (set `continuous=false`):

```bash
python -m pytorch_sac.train_discrete_SAC env=lunarlander lunarlander.continuous=false
```

## 11) Discrete DQN for LunarLander

File: `pytorch_sac/train_dqn_lunarlander.py`, `pytorch_sac/dqn.py`

- Full DQN implementation (`dqn.py`) with:
  - `DQN`: 3-layer feedforward Q-network (ReLU activations, hidden dim 256).
  - `DQNTrainer`: training loop with epsilon-greedy exploration, target network (Polyak-style updates), replay buffer, CSV logging, and video recording.
- Training script (`train_dqn_lunarlander.py`) uses `LunarLander-v3` in discrete mode (`continuous=False`).
- Configurable hyperparameters via CLI arguments (seeds, learning rate, epsilon schedule, buffer capacity, replay factor, etc.).
- Output saved under `exp/dqn_lunarlander/`.

Running examples:

- Run DQN with default settings:

```bash
python -m pytorch_sac.train_dqn_lunarlander
```

- Run DQN with custom hyperparameters:

```bash
python -m pytorch_sac.train_dqn_lunarlander --seeds 42 123 --lr 1e-3 --epsilon_decay 0.99 --num_episodes 10000
```

- Enable video recording of evaluation rollouts:

```bash
python -m pytorch_sac.train_dqn_lunarlander --save_video
```

## 12) Analysis

- Pendulum:
All analysis and plots for Pendulum is done in `main_final.ipynb`

- Lunarlander:
Analysis of performance of changing alpha v/s fixed alpha is done in `lunarlander_hovering_results.ipynb`

- Reacher:
Analysis of performance of each reward (tested against itself) and Evaluation of other metrics (Steps taken to Goal/ Time spent in Goal) is done in `reacher_evaluations.ipynb`

- PEBBLE:
Analysis of performance for various thetas is done in `pebble_analysis.ipynb`



