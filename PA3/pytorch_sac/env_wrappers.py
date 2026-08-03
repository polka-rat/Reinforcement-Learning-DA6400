import gymnasium as gym
import os
import numpy as np


class PendulumWrapper(gym.Wrapper):
    """Pendulum wrapper with configurable target angle and reward hook."""

    def __init__(self, env, target_theta=0.0, cosine_weight=2.0, reward_scale=1.0):
        super().__init__(env)
        self.target_theta = float(target_theta)
        self.cosine_weight = float(cosine_weight)
        self.reward_scale = float(reward_scale)

    def _angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _reward(self, theta, theta_dot, action):
        angle_error = self._angle_normalize(theta - self.target_theta)
        torque = np.clip(action, self.action_space.low, self.action_space.high)

        base_reward = -(angle_error ** 2 + 0.01 * (theta_dot ** 2) + 0.0008 * np.sum(torque ** 2))
        cos_bonus = 0.0
        if abs(angle_error) <= (np.pi / 2):
            cos_bonus = self.cosine_weight * np.cos(angle_error)

        return base_reward

    def _compute_reward(self, obs, action):
        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]

        return float(self._reward(theta, theta_dot, action)) * self.reward_scale

    @property
    def _max_episode_steps(self):
        return getattr(self.env, '_max_episode_steps', None)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step(self, action):
        step_out = self.env.step(action)

        if len(step_out) == 5:
            obs, _, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step_out

        reward = self._compute_reward(obs, action)
        return obs, reward, done, info


class LunarLanderWrapper(gym.Wrapper):
    """LunarLander wrapper with optional hover-box reward shaping."""

    def __init__(self,
                 env,
                 reward_mode='default',
                 hover_bonus=200.0,
                 hover_x_threshold=0.1,
                 hover_y_min=0.4,
                 hover_y_max=0.6,
                 hover_once_per_episode=True):
        super().__init__(env)
        self.reward_mode = reward_mode
        self.hover_bonus = float(hover_bonus)
        self.hover_x_threshold = float(hover_x_threshold)
        self.hover_y_min = float(hover_y_min)
        self.hover_y_max = float(hover_y_max)
        self.hover_once_per_episode = bool(hover_once_per_episode)
        self._hover_bonus_awarded = False

    def set_hover_bonus(self, hover_bonus):
        self.hover_bonus = float(hover_bonus)

    def set_reward_mode(self, reward_mode):
        self.reward_mode = str(reward_mode)

    def reward_state_dict(self):
        return {
            'reward_mode': self.reward_mode,
            'hover_bonus': self.hover_bonus,
            'hover_bonus_awarded': self._hover_bonus_awarded,
        }

    def load_reward_state_dict(self, state_dict):
        self.reward_mode = str(state_dict.get('reward_mode', self.reward_mode))
        self.hover_bonus = float(state_dict.get('hover_bonus', self.hover_bonus))
        self._hover_bonus_awarded = bool(
            state_dict.get('hover_bonus_awarded', self._hover_bonus_awarded))

    def _in_hover_box(self, obs):
        x = float(obs[0])
        y = float(obs[1])
        return abs(x) < self.hover_x_threshold and self.hover_y_min < y < self.hover_y_max

    def _reward(self, obs, env_reward):
        reward = float(env_reward)
        if self.reward_mode != 'hover_box':
            return reward

        if self._in_hover_box(obs):
            if not self.hover_once_per_episode or not self._hover_bonus_awarded:
                reward += self.hover_bonus
                self._hover_bonus_awarded = True
        return reward

    def _compute_reward(self, obs, env_reward):
        return self._reward(obs, env_reward)

    @property
    def _max_episode_steps(self):
        return getattr(self.env, '_max_episode_steps', None)

    def reset(self, **kwargs):
        self._hover_bonus_awarded = False
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step(self, action):
        step_out = self.env.step(action)

        if len(step_out) == 5:
            obs, env_reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            reward = self._compute_reward(obs, env_reward)
            return obs, reward, done, info

        obs, env_reward, done, info = step_out
        reward = self._compute_reward(obs, env_reward)
        return obs, reward, done, info


class ReacherWrapper(gym.Wrapper):
    """Reacher wrapper with three reward formulations (Ra, Rb, Rc).
    
    Ra: Dense reward. Fixed T=1000 episodes.
        reward = 1 if in target, else -|distance| - |action|^2
    
    Rb: Sparse reward. Fixed T=1000 episodes.
        reward = 1 if in target, else 0
    
    Rc: Variable length episodes.
        reward = 0 if in target, else -1
        - Terminates early if target reached with low velocity (both reset)
                - If timeout at 1000 steps: apply reset penalty, robot-only reset,
                    keep same goal, continue same episode
    """

    def __init__(self, env, reward_mode='rb', target_distance_threshold=0.01):
        """
        Args:
            env: Base reacher environment (should be DMCToGymnasiumWrapper)
            reward_mode: 'ra', 'rb', or 'rc'
            target_distance_threshold: Distance threshold for being "in target"
        """
        super().__init__(env)
        self.reward_mode = str(reward_mode).lower()
        self.target_distance_threshold = float(target_distance_threshold)
        self._step_count = 0
        self._target_pos = None
        self._came_from_timeout = False  # FIX: tracks whether last episode ended via Rc timeout

        # Episode tracking for Rc mode
        self._episode_return = 0.0
        self._episode_length = 0
        self._reset_penalty_rc = -20.0  # Penalty when timeout without completing task
        
        self._identify_to_target_indices()
        
        if self.reward_mode not in ['ra', 'rb', 'rc']:
            raise ValueError(f"reward_mode must be 'ra', 'rb', or 'rc', got {self.reward_mode}")
    
    def _identify_to_target_indices(self):
        obs_slices = getattr(self.env, '_obs_slices', {})
        if 'to_target' in obs_slices:
            self._to_target_slice = obs_slices['to_target']
        else:
            # Fallback for the standard flattened dm_control Reacher layout:
            # [position(2), velocity(2), to_target(2)]
            self._to_target_slice = slice(4, 6)  # obs layout: [position(2), velocity(2), to_target(2)]

        if 'velocity' in obs_slices:
            self._velocity_slice = obs_slices['velocity']
        else:
            self._velocity_slice = slice(2, 4)  # obs layout: [position(2), velocity(2), to_target(2)]
    

    def _get_fingertip_pos(self, obs):
        """Deprecated helper kept for backwards compatibility."""
        del obs
        return None

    def _get_target_pos(self, obs):
        """Deprecated helper kept for backwards compatibility."""
        del obs
        return None

    def _get_to_target_vec(self, obs):
        """Extract the finger-to-target vector from the flattened observation."""
        return np.asarray(obs[self._to_target_slice], dtype=np.float32)

    def _distance_to_target(self, obs):
        """Compute finger-to-target distance from the observation."""
        distance = np.linalg.norm(self._get_to_target_vec(obs))
        return float(distance)

    def _get_in_target_threshold(self):
        """Use the sum of target and finger radii when physics is available."""
        dmc_env = getattr(self.env, "_dmc_env", None)
        if dmc_env is None:
            return self.target_distance_threshold

        physics = dmc_env.physics
        try:
            radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
            return float(radii)
        except Exception:
            return self.target_distance_threshold

    def _is_in_target(self, obs):
        """Check if fingertip is close to target."""
        distance = self._distance_to_target(obs)
        return distance < self._get_in_target_threshold()

    def _get_velocity(self, obs):
        """Extract joint velocities from observation."""
        return np.asarray(obs[self._velocity_slice], dtype=np.float32)

    def _get_target_geom_xy(self):
        """Read the target's xy position from dm_control physics joint state."""
        dmc_env = getattr(self.env, "_dmc_env", None)
        if dmc_env is None:
            return None
        physics = dmc_env.physics
        try:
            # In dm_control reacher, the target moves via slider joints (target_x, target_y).
            # qpos is the actual runtime position — geom_pos is a static body-frame offset.
            target_x = float(physics.named.data.qpos['target_x'])
            target_y = float(physics.named.data.qpos['target_y'])
            return np.array([target_x, target_y], dtype=np.float32)
        except Exception:
            return None

    def _compute_reward_ra(self, obs, action):
        """Reward Ra: 1 if in target, else -distance - |action|^2"""
        if self._is_in_target(obs):
            return 1.0

        distance = self._distance_to_target(obs)
        action_cost = 0.01*np.sum(action ** 2)
        return -distance - action_cost

    def _compute_reward_rb(self, obs, action):
        """Reward Rb: 1 if in target, else 0"""
        del action
        if self._is_in_target(obs):
            return 1.0
        return 0.0

    def _compute_reward_rc(self, obs, action):
        """Reward Rc: always -1. Goal is signalled by termination, not reward."""
        del obs, action
        # STRICT ENFORCEMENT: RC reward must be exactly -1.0
        return -1.0

    def _should_terminate_rc(self, obs):
        """For mode Rc: terminate if fingertip reaches target with near-zero velocity."""
        velocity = self._get_velocity(obs)
        in_target = self._is_in_target(obs)
        low_velocity = np.linalg.norm(velocity) < 0.05  # joint velocity in rad/s (0.05 was Cartesian scale, not joint)
        return in_target and low_velocity

    def _compute_reward(self, obs, action):
        """Compute reward based on mode.
        
        STRICT ENFORCEMENT FOR RC:
        - Base reward is always -1.0
        - No modifications to base reward in this function
        - Timeout penalty (-20) is applied in step() only
        """
        if self.reward_mode == 'ra':
            return self._compute_reward_ra(obs, action)
        elif self.reward_mode == 'rb':
            return self._compute_reward_rb(obs, action)
        elif self.reward_mode == 'rc':
            # STRICT: Return -1.0 with NO modifications
            return self._compute_reward_rc(obs, action)
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    @property
    def _max_episode_steps(self):
        return getattr(self.env, '_max_episode_steps', None)

    def reset(self, **kwargs):
        """Reset environment.

        For Ra/Rb: Full reset (robot + target).
        For Rc: Full reset (robot + target) at true episode boundaries (success only).
        Timeout handling is done inside step() as an in-episode robot-only reset —
        the outer training loop never sees a done=True on timeout.
        """
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._came_from_timeout = False

        # Full reset: reset both robot and target
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]

        # Store target position for Rc mode directly from physics.
        self._target_pos = self._get_target_geom_xy()
        return obs

    def _partial_reset_robot_only(self):
        """For Rc mode: reset only the robot arm, keep target in same location.
        
        Accesses the dm_control physics engine to:
        1. Reset the robot to a random configuration
        2. Restore the target to its saved position
        """
        # Get the underlying dm_control environment
        dmc_env = self.env._dmc_env
        
        # Full reset to randomize everything
        dmc_env.reset()
        
        # Now we need to restore the target position
        # In dm_control's reacher environment, the target is the last 2 DOF
        # Access physics state: [joint_positions, joint_velocities]
        physics = dmc_env.physics
        
        try:
            if self._target_pos is not None:
                # Restore target position via joint qpos — geom_pos is a static
                # body-frame offset and does not move the target at runtime.
                physics.named.data.qpos['target_x'] = float(self._target_pos[0])
                physics.named.data.qpos['target_y'] = float(self._target_pos[1])
                # Zero out target velocities so it stays fixed
                physics.named.data.qvel['target_x'] = 0.0
                physics.named.data.qvel['target_y'] = 0.0
                physics.forward()
            
            # CRITICAL: reset the inner wrapper's step counter so it doesn't
            # fire truncated=True at misaligned global-1000 boundaries.
            self.env._step_count = 0

            # Get observation after partial reset
            obs = self._get_obs_from_physics(dmc_env)
            return obs
        except Exception as e:
            # Fallback: if physics manipulation fails, do full reset
            print(f"Warning: Partial reset failed ({e}), falling back to full reset")
            print(f"Available qpos keys: {list(physics.named.data.qpos.axes.row.names)}")
            self.env._step_count = 0  # reset here too
            obs = dmc_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            return obs
    
    def _get_obs_from_physics(self, dmc_env):
        """Extract observation from dm_control physics state."""
        # This mimics what the environment's task.get_observation does
        physics = dmc_env.physics
        obs_dict = dmc_env.task.get_observation(physics)
        
        # Flatten observation dictionary using our wrapper's flattening logic
        obs_list = []
        for key in self.env._obs_keys:
            if key in obs_dict:
                obs_list.append(obs_dict[key].flatten())
        
        return np.concatenate(obs_list).astype(np.float32)

    def step(self, action):
        """Step environment.
        
        Episode length handling:
        - Ra/Rb: Fixed length 1000, no early termination
        - Rc: Variable length, early termination on target reach,
              timeout at 1000 triggers robot-only reset with penalty -20
              while continuing the same episode
        """
        self._step_count += 1
        step_out = self.env.step(action)

        if len(step_out) == 5:
            obs, _, env_terminated, env_truncated, info = step_out
        else:
            obs, _, env_terminated, env_truncated, info = step_out[0], None, step_out[2], False, step_out[3]

        # For Rc mode, ReacherWrapper is sole authority on timeouts.
        # Keep inner wrapper's step count mirrored to ours so it never fires
        # its own truncated=True at misaligned global-1000 boundaries.
        if self.reward_mode == 'rc':
            self.env._step_count = self._step_count

        # Get base reward from wrapper (for RA/RB) or direct computation (for RC)
        reward = self._compute_reward(obs, action)
        
        terminated = False
        truncated = False
        reset_penalty_applied = False
        
        if self.reward_mode in ['ra', 'rb']:
            # Fixed length episodes: no early termination, only truncate at 1000 steps
            # Reward is used as-is from _compute_reward
            truncated = self._step_count >= 1000
            
        elif self.reward_mode == 'rc':
            # STRICT RC ENFORCEMENT:
            # - Base reward is exactly -1.0 (already computed above)
            # - Check for early termination (target reached)
            # - On timeout: apply ADDITIONAL -20 penalty, then robot-only reset
            
            # Variable length episodes
            if self._should_terminate_rc(obs):
                # Early termination: target reached with low velocity
                terminated = True
                # No reward modification on termination
            elif self._step_count >= 1000:
                # Timeout: apply ADDITIONAL -20 penalty on this step
                # Current reward is -1.0, add -20 to get -21.0 total
                reward += self._reset_penalty_rc  # reward = -1.0 + (-20.0) = -21.0
                reset_penalty_applied = True
                obs = self._partial_reset_robot_only()
                self._step_count = 0
                terminated = False
                truncated = False
                info['timeout_reset'] = True
        
        # Track episode metrics for Rc mode
        if self.reward_mode == 'rc':
            self._episode_return += reward
            self._episode_length += 1
            
            # Log episode metrics in info when episode ends
            if terminated or truncated:
                info['episode_return'] = float(self._episode_return)
                info['episode_length'] = int(self._episode_length)
                info['reset_penalty_applied'] = bool(reset_penalty_applied)
        
        done = bool(terminated or truncated)
        return obs, reward, done, info