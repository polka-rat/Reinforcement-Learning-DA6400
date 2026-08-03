# PA2

## Setup

#### The environment uses Python 3.12
```bash
conda env create -f pa2env.yml
conda activate rl_pa2
```
## `dqn.py`
Code implementing the DQN, with provision for changing any parameters required for Q2, 3, 4.

###
`ReplayBuffer`
- stores and samples transitions, implemented with a deque.
###
`DQN`
- 2 hidden-layer MLP with 32 neurons each.
- RELU activation functions between linear layers.
###
`Trainer`
- class for training the DQN, contains all the main functionality required for training -
    - includes WandB logging (`_init_wandb`)
    - provides implementation for all the following questions. For vanilla case, hyperparameters are described in the report.
    - action selected by Epsilon-greedy
    - replay factor, batch_size, target refresh rate etc, are variables that can be changed for the following questions.
    - main training loop in `train`

###
`RankBasedPERBuffer`
- implements rank based prioritized experience replay
- new transitions assigned max TD error to ensure they are sampled at least once
- transitions are sorted based on TD error with highest TD error given rank 1 and lower TD errors with larger ranks to signify their importance
- probabaility of sampling a transition: P(i) ∝ (1 / rank(i))^α
###
`DQN`
- 2 hidden-layer MLP with 32 neurons each.
- RELU activation functions between linear layers.
###
`Trainer`
- class for training the DQN like the vanilla DQN along with added functionality to handle PER -
    - choose the appropriate buffer depending on whether `use_per` is set or not
    - update TD error priorities
    - ensure α is annealed from 0.5 to 0 as described in the shared report.
    - β has not been set and left as 0 according to the report
    - main training loop in `train`
---

# Other Files used for computation and analysis

## `main.ipynb`
- Contains Q2, Q3, and Q4 d (i).

## `q4_replay_factors.ipynb`
- Contains implementation of Q4 a.

## `main_targetnet_sweep.py`
- Contains implementation of Q4 d (ii).

## `extract_q4_results.ipynb`
- Contains data processing and analysis of logs for Q4 (Extraction for Q4 d (ii) got a bit messy as the 3 of us individually ran seeds of our own with different output directories)

## `dqn_bonus.py`
- Contains original DQN with added PER implementation for bonus question.

## `bonus.ipynb`
- Contains bonus question's experiments and analysis.

## `vid.mov`
- Optimal performance of the DQN agent for the Mountain Car environment.