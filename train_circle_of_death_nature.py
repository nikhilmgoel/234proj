import gym
import gym_navigate
from utils.preprocess import rescale
from utils.wrappers import MaxAndSkipEnv, PreproWrapper

from schedule import LinearExploration, LinearSchedule
from nature import NatureQN

from configs.baseline_network import config

import read_data

"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    custom_set_episode = env.set_episode
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=rescale, shape=(read_data.SCALED_HEIGHT, read_data.SCALED_WIDTH, 1), custom_set_episode=custom_set_episode)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
