import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from schedule import LinearExploration, LinearSchedule

from configs.baseline_network import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        # state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        self.batch_size = self.config.batch_size
        self.img_height = 240
        self.img_width = 201
        self.nchannels = 1
        self.state_history = self.config.state_history

        shape = [None, self.img_height, self.img_width, self.nchannels*self.state_history]
        self.s = tf.placeholder('uint8', shape)
        self.sp = tf.placeholder('uint8', shape)

        shape = [None]
        self.a = tf.placeholder('int32', shape)
        self.r = tf.placeholder('float32', shape)
        self.done_mask = tf.placeholder('bool', shape)
        self.lr = tf.placeholder('float32')

        ##############################################################
        ######################## END YOUR CODE #######################


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

        flattened = tf.contrib.layers.flatten(state, scope=scope)
        # get number of params 
        # print (int(flattened.get_shape()[1]) * num_actions + num_actions)
        out = tf.contrib.layers.fully_connected(flattened, num_actions, reuse=reuse, scope=scope)

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        var_hold = {}
        for q_var in q_vars:
            var_hold['target_' + q_var.name] = q_var
        ops = []
        for t_var in target_q_vars:
            ops.append(t_var.assign(var_hold[t_var.name]))

        self.update_target_op = tf.group(*ops)


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        action_one_hots = tf.one_hot(self.a, num_actions)
        qsa = tf.reduce_sum(tf.multiply(q, action_one_hots), axis=1)

        q_target_sa = tf.scalar_mul(self.config.gamma, tf.reduce_max(target_q, axis=1))
        dones = 1.0 - tf.cast(self.done_mask, tf.float32)
        q_target_sa = tf.multiply(dones, q_target_sa)
        q_target_sa = tf.add(self.r, q_target_sa)

        self.loss = tf.reduce_mean(tf.square(tf.subtract(qsa, q_target_sa)))


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        trainer = tf.train.AdamOptimizer(self.lr)
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        grads_and_vars_start = trainer.compute_gradients(self.loss, var_list=q_vars)
        if self.config.grad_clip is True:
            grads_and_vars = []
            for grad, var in grads_and_vars_start:
                grads_and_vars.append((tf.clip_by_norm(grad, self.config.clip_val), var))
        else:
            grads_and_vars = grads_and_vars_start
        self.train_op = trainer.apply_gradients(grads_and_vars)
        self.grad_norm = tf.global_norm([grad for grad, var in grads_and_vars])


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
