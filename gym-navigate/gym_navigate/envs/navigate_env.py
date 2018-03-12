import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from gym.envs.classic_control import rendering

import sys
sys.path.append('../../')
import read_data

class NavigateEnv(gym.Env):
  metadata = {
  	'render.modes': ['human', 'rgb_array'], 
  	'video.frames_per_second' : 28
  }

  def __init__(self):

    # Bot dynamics
    self.min_position = np.array([0, 0])
    self.max_position = np.array([239, 200])
    self.goal_position = np.array([131, 0])
    self.start_position = np.array([131, 200])
    self.min_step = 0
    self.max_step = 2.0
    self.frames_per_step = 8
    self.distance_to_goal = 0.0
    self.prev_distance_to_goal = 0.0

    # If below -1 reward for n steps in a row, terminate game
    self.fault = [0,5] # [current count, n]

    # Viewer
    self.screen_width = 700
    self.screen_height = 1000
    self.world_width = self.max_position[0] - self.min_position[0]
    self.scale = self.screen_width / self.world_width
    self.grid_boxes = 100
    self.bot_width = 40
    self.bot_height = 20

    # Spaces
    self.action_space = spaces.Box(
  		low=np.array([-self.max_step, -self.max_step]), 
  		high=np.array([self.max_step, self.max_step]))
    self.observation_space = spaces.Box(
    	low=np.array([self.min_position[0], self.min_position[1]]),
    	high=np.array([self.max_position[0], self.max_position[1]]))

    self.viewer = None
    self.state = None
    self.bot_position = None

    self.episodes = []
    for s in read_data.VALID_SCENES:
      scene = read_data.load_processed_scene(s)
      for e in range(5):
        new_ep = []
        for f in range(e * 1792, (e+1) * 1792):
          new_ep.append(scene[f])
        self.episodes.append(new_ep)

    self.game_index = -1
    self.tick = -1


  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    self.tick += 1

    position = self.bot_position
    position = (position[0] + action[0], position[1] + action[1])

    if (position[0] > self.max_position[0]): position[0] = self.max_position[0]
    if (position[1] > self.max_position[1]): position[1] = self.max_position[1]
    if (position[0] < self.min_position[0]): position[0] = self.min_position[0]
    if (position[1] < self.min_position[1]): position[1] = self.min_position[1]

    done = bool(position[0] >= self.goal_position[0] and position[1] >= self.goal_position[1])
    if self.tick > 1792:
      done = True

    self.bot_position = position

    # nikhil - another possible reward model to try: reward increases inversely proportional to gap between agent and goal
    # calculate reward
    reward = 0
    self.distance_to_goal = np.linalg.norm(position - self.goal_position)

    # check for closing on goal
    if (self.distance_to_goal > self.prev_distance_to_goal):
      reward += -1.0
      self.fault[0]+=1
    elif (self.distance_to_goal == self.prev_distance_to_goal):
      reward += 0.0
    else:
      reward += 1.0

    # check for collision
    if self.state[position[0], position[1]] == 1:
      reward += -100

    # nikhil - see if adding a final reward for reaching the goal affects performance
    # nikhil - penalize each step taken?
    #if done:
        #reward = 1.0
    #reward-= math.pow(action[0],2)*0.1

    self.state = self.load_frame(self.game_index, self.tick)
    self.state[position[0], position[1]] = 0
    return self.state, reward, done, {}


  def reset(self, index):
    """ set our state as the empty circle of death state with our
    actor starting at the start position (coming in from white plaza)

    Args:
      index (int): the index episode to pull
    """
    # load up a game
    self.state = self.load_frame(index, 0)
    self.game_index = index
    self.tick = 0

    # put the bot at the starting position
    self.bot_position = self.start_position
    self.state[self.bot_position[0], self.bot_position[1]] = 1.

    return np.array(self.state)


  def load_frame(self, index, tick):
    """ set our state as the empty circle of death state with our
    actor starting at the start position (coming in from white plaza)

    Args:
      index (int): the index episode to pull
    """
    return self.episodes[index][tick]


  def _height(self, xs): #TODO
    return np.sin(3 * xs)*.45+.55


  def render(self, mode='human'):

    if self.viewer is None:
      self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

      # list of evenly spaced x and y coordinates for the grid boxes
      xs = np.linspace(self.min_position[0], self.max_position[1], self.grid_boxes)
      ys = self._height(xs)
      xys = list(zip((xs-self.min_position[0])*self.scale, ys*self.scale))

    
      # nikhil - may use later when creating a line that trails the bot to show the entire path taken
      # track = rendering.make_polyline(xys)
      # track.set_linewidth(4)
      # self.viewer.add_geom(track)
      
      clearance = 10

      # bot
      l,r,t,b = -self.bot_width / 2, self.bot_width / 2, self.bot_height, 0
      bot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
      bot.add_attr(rendering.Transform(translation=(0, clearance)))
      self.bot_trans = rendering.Transform()
      bot.add_attr(self.bot_trans)
      self.viewer.add_geom(bot)
      
      # GOOOOOOAAAAAAAALLLLLLL #TODO
      flagx = np.linalg.norm(self.goal_position - self.min_position)*self.scale 
      flagy1 = self._height(self.goal_position[1])*self.scale
      flagy2 = flagy1 + 50
      flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
      self.viewer.add_geom(flagpole)
      flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
      flag.set_color(.8,.8,0)
      self.viewer.add_geom(flag)

      pos = self.state[0]

      # nikhil - may use later to show rotational movement of the bot
      #self.bot_trans.set_translation((np.linalg.norm(pos-self.min_position[0]))*self.scale, self._height(pos[0])*self.scale) #TODO
      #self.bot_trans.set_rotation(math.cos(3 * pos[0])) #TODO

      return self.viewer.render(return_rgb_array = mode == 'rgb_array')


  def close(self):
    if self.viewer: self.viewer.close()
  