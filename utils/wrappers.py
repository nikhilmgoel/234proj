import numpy as np
import gym
from gym import spaces
from viewer import SimpleImageViewer
from collections import deque


class MaxAndSkipEnv(gym.Wrapper):
    """
    Wrapper from Berkeley's Assignment
    Takes a max pool over the last n states
    """
    def __init__(self, custom_reset, env=None, skip=4):
        """Return only every `skip`-th frame"""
        self.reset_custom = custom_reset
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self, i):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.reset_custom(i)
        self._obs_buffer.append(obs)
        return obs

    def reset(self, i):
        """okie"""
        return self._reset(i)


class PreproWrapper(gym.Wrapper):
    """
    Wrapper for Pong to apply preprocessing
    Stores the state into variable self.obs
    """
    def __init__(self, env, prepro, shape, custom_reset, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        self.reset_custom = custom_reset
        super(PreproWrapper, self).__init__(env)
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=high, shape=shape)
        self.high = high

    def _step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info

    def _reset(self, i):
        self.env.reset()
        self.obs = self.prepro(self.reset_custom(i))
        return self.obs

    def reset(self, i):
        """okie"""
        return self._reset(i)
