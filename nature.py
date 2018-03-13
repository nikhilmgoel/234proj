import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear


from configs.baseline_network import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n
        out = state
        with tf.variable_scope(scope, reuse=reuse):
            first_layer = layers.conv2d(state, num_outputs = 32, kernel_size = 8, stride=4, biases_initializer=layers.xavier_initializer())
            second_layer = layers.conv2d(first_layer, num_outputs = 64, kernel_size = 4, stride=2, biases_initializer=layers.xavier_initializer())
            third_layer = layers.conv2d(second_layer, num_outputs = 64, kernel_size = 3, biases_initializer=layers.xavier_initializer())
            flattened = layers.flatten(third_layer)
            last = layers.fully_connected(flattened, num_outputs = 512, biases_initializer=layers.xavier_initializer())
            out = layers.fully_connected(last, num_outputs = num_actions, biases_initializer=layers.xavier_initializer(), activation_fn=None)

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
