import functools
import logging
import threading

import elements
import embodied
import numpy as np
import gymnasium as gym
from gym.spaces import Box

# Global registry to store created environments
_ENV_REGISTRY = {}
_ENV_LOCK = threading.Lock()


def get_isaaclab_env_factory(task, env_args, index=0, **kwargs):
    """Module-level function that returns cached factory instances."""
    key = (task, index)
    with _ENV_LOCK:
        if key not in _ENV_REGISTRY:
            _ENV_REGISTRY[key] = IsaacLabEnvFactory(task, env_args, index, **kwargs)
        return _ENV_REGISTRY[key]


class IsaacLabEnvFactory:
    """Factory that ensures the environment is only created once globally."""
    
    def __init__(self, task, env_args, index=0, **kwargs):
        self._task = task
        self._env_args = env_args["env_args"]
        self._kwargs = kwargs
        self._index = index
        self._env = None
    
    def _get_or_create_env(self):
        if self._env is None:
            self._env = IsaacLabEnv(self._task, self._env_args, **self._kwargs)
        return self._env
    
    def __call__(self):
        return self._get_or_create_env()
    
    @property
    def obs_space(self):
        return self._get_or_create_env().obs_space
    
    @property
    def act_space(self):
        return self._get_or_create_env().act_space
    
    def step(self, action):
        return self._get_or_create_env().step(action)
    
    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
    
    def render(self):
        return self._get_or_create_env().render()

class IsaacLabEnv(embodied.Env):
  def __init__(self, env, env_args, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, cfg=env_args['config'], render_mode="rgb_array", **env_args)
    else:
      assert not kwargs, kwargs
      self._env = env
      
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    
  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._info = self._env.step(action)
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return elements.Space(np.int32, (), 0, space.n)
    return elements.Space(space.dtype, space.shape, space.low, space.high)
