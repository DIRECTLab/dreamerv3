import functools
import logging
import threading

import elements
import embodied
import numpy as np
import gymnasium as gym
from gym.spaces import Box
import torch

# Shared backend singleton
_BACKEND = None
_BACKEND_LOCK = None

def get_backend(task, env_args):
    """Get or create the shared IsaacLab backend."""
    global _BACKEND, _BACKEND_LOCK
    import threading
    if _BACKEND_LOCK is None:
        _BACKEND_LOCK = threading.Lock()
    
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = IsaacLabBackend(task, env_args)
        return _BACKEND


class IsaacLabBackend:
    """Shared backend that manages the actual IsaacLab environment."""
    
    def __init__(self, task, env_args):
        self._env = gym.make(task, cfg=env_args['config'], render_mode="rgb_array", **env_args)
        self._num_envs = self._env.env.scene.num_envs
        
        # Pending actions buffer - collect actions from all wrappers before stepping
        self._pending_actions = {}
        self._pending_resets = set()
        self._last_obs = None
        self._last_reward = None
        self._last_done = None
        self._last_info = None
        self._step_count = 0
        
        # Perform initial reset
        self.reset()
        
    @property
    def num_envs(self):
        return self._num_envs
    
    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space
    
    def set_action(self, env_id, action, reset):
        """Store action for a specific sub-environment."""
        self._pending_actions[env_id] = action
        if reset:
            self._pending_resets.add(env_id)
    
    def step_if_ready(self):
        """Step the environment once all actions are collected."""
        if len(self._pending_actions) < self._num_envs:
            return False  # Not all actions collected yet
        
        # If all envs request reset, do a full reset
        if len(self._pending_resets) == self._num_envs:
            self.reset()
            return True
        
        # Stack actions in order
        actions = np.stack([self._pending_actions[i] for i in range(self._num_envs)])
        actions = torch.from_numpy(actions)
        # Move to env device if available
        if hasattr(self._env, 'device'):
            actions = actions.to(self._env.device)
        elif hasattr(self._env, 'unwrapped') and hasattr(self._env.unwrapped, 'device'):
            actions = actions.to(self._env.unwrapped.device)
        
        # Step the environment
        obs, reward, done, timeouts, info = self._env.step(actions)
        
        self._last_obs = obs['policy'].cpu().numpy() if isinstance(obs['policy'], torch.Tensor) else obs['policy']
        self._last_reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        self._last_done = np.logical_or(
            done.cpu().numpy() if isinstance(done, torch.Tensor) else done,
            timeouts.cpu().numpy() if isinstance(timeouts, torch.Tensor) else timeouts
        )
        self._last_info = info
        self._step_count += 1
        
        # Clear pending
        self._pending_actions.clear()
        self._pending_resets.clear()
        
        return True
    
    def get_obs(self, env_id):
        """Get observation for a specific sub-environment."""
        return {
            'obs': self._last_obs[env_id],
            'reward': self._last_reward[env_id],
            'done': self._last_done[env_id],
        }
    
    def reset(self):
        """Reset all environments."""
        obs, info = self._env.reset()
        self._last_obs = obs['policy'].cpu().numpy() if isinstance(obs['policy'], torch.Tensor) else obs['policy']
        self._last_reward = np.zeros(self._num_envs, dtype=np.float32)
        self._last_done = np.zeros(self._num_envs, dtype=bool)
        self._last_info = info
        self._pending_actions.clear()
        self._pending_resets.clear()


class IsaacLabEnv(embodied.Env):
    """Wrapper that presents a single sub-environment view of the shared backend."""
    
    def __init__(self, env_id, backend, obs_key='image', act_key='action'):
        self._env_id = env_id
        self._backend = backend
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._is_first = True
    
    @functools.cached_property
    def obs_space(self):
        # Single env observation space (no num_envs dimension)
        obs_shape = self._backend.observation_space.shape[1]
        spaces = {
            self._obs_key: elements.Space(
                np.float32, obs_shape,
                self._backend.observation_space.low[0],
                self._backend.observation_space.high[0]
            ),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        return spaces
    
    @functools.cached_property
    def act_space(self):
        act_shape = self._backend.action_space.shape[1]
        spaces = {
            self._act_key: elements.Space(
                np.float32, act_shape,
                self._backend.action_space.low[0],
                self._backend.action_space.high[0]
            ),
            'reset': elements.Space(bool),
        }
        return spaces
    
    def step(self, action):
        reset = action.get('reset', False)
        act = action.get(self._act_key, np.zeros(self._backend.action_space.shape[1]))
        
        # Submit action to backend
        self._backend.set_action(self._env_id, act, reset)
        
        # Try to step (will only succeed when all envs have submitted)
        self._backend.step_if_ready()
        
        # Get our observation
        data = self._backend.get_obs(self._env_id)
        
        is_first = self._is_first or self._done
        self._done = data['done']
        self._is_first = False
        
        return {
            self._obs_key: np.asarray(data['obs'], dtype=np.float32),
            'reward': np.float32(data['reward']),
            'is_first': is_first,
            'is_last': self._done,
            'is_terminal': self._done,
        }
    
    def close(self):
        pass  # Backend handles cleanup


def get_env(task, env_args, env_id=0):
    """Factory function that returns a single sub-environment wrapper."""
    backend = get_backend(task, env_args)
    return IsaacLabEnv(env_id, backend)
