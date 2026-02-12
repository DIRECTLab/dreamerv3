import functools
import logging
import threading

import elements
import embodied
import numpy as np
import gymnasium as gym
from gym.spaces import Box
import torch

_ENV = None

def get_env(task, env_args):
  global _ENV
  if _ENV is None:
     _ENV = IsaacLabEnv(task, env_args)
  return _ENV

class IsaacLabEnv(embodied.Env):
  def __init__(self, env, env_args, obs_key='image', act_key='action', **kwargs):
    self._env = gym.make(env, cfg=env_args['config'], render_mode="rgb_array", **env_args)

    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

    self._all_keys = set([self._obs_key, self._act_key, 'reward', 'is_first', 'is_last', 'is_terminal'])
    
  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    num_envs = self._env.env.scene.num_envs
    spaces = {
      self._obs_key: self._env.observation_space,
      'reward': Box(-np.inf, np.inf, (num_envs,), dtype=np.float32),
      'is_first': Box(False, True, (num_envs,), dtype=bool),
      'is_last': Box(False, True, (num_envs,), dtype=bool),
    }
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return spaces

  @functools.cached_property
  def act_space(self):

    spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      _obs = obs[0]['policy']
      return self._obs(_obs, 0.0, is_first=True)
 
    action = action[self._act_key]
    action = torch.from_numpy(action)
    obs, reward, done, timeouts, self._info = self._env.step(action)
    _obs = obs['policy']
    self._done = np.logical_or(done.cpu().numpy(), timeouts.cpu().numpy())

    return self._obs(
        _obs, reward,
        is_last=self._done,
        is_terminal=self._done
    )
  
  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    
    if isinstance(obs, torch.Tensor):
      obs = obs.cpu()
    
    if isinstance(reward, torch.Tensor):
      reward = reward.cpu().numpy()

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
