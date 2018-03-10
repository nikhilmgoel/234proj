import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np

class NavigateEnv(gym.Env):
  metadata = {
  	'render.modes': ['human'], 
  	'video.frames_per_second' : 50
  }

  def __init__(self):
  	self.massbot = 1.0
  	self.tau = 0.02  # seconds between state updates
  	self.min_x_position = -1.2
  	self.max_x_position = 0.6
  	self.min_y_position = 0.0
  	self.max_y_position = 0.0
  	self.max_speed = 0.07
  	self.min_step = 0
  	self.max_step = 2.0

  	self.action_space = spaces.Box(
  		low=np.array(-self.max_step, -self.max_step), 
  		high=np.array(self.max_step, self.max_step))
  	self.observation_space = spaces.Box(
    	low=np.array(self.min_x_position, self.min_y_position),
    	high=np.array(self.max_x_position, self.max_y_position))

	self.seed() # does this start agent at same game state with each run?
	self.viewer = None
	self.state = None

  def step(self, action):
  	assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
  	return #next_state, reward, is_terminal, debug_info
    
  def reset(self):
  	self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
  	return np.array(self.state)
    
  def render(self, mode='human', close=False):
  	return
    