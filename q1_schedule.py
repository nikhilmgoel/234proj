import numpy as np
from utils.test_env import EnvTest


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        modify self.epsilon such that 
            for t = 0, self.epsilon = self.eps_begin
            for t = self.nsteps, self.epsilon = self.eps_end
            linear decay between the two

            self.epsilon should never go under self.eps_end

        Args:
            t: (int) nth frames
        """
        if t == 0:
            self.epsilon = self.eps_begin
        elif 0 < t < self.nsteps:
            self.epsilon = self.eps_begin - (t)*(float(self.eps_begin - self.eps_end)/self.nsteps)
        else: 
            self.epsilon = self.eps_end


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        with probability self.epsilon, return a random action
            else, return best_action

            you can access the environment stored in self.env
            and epsilon with self.epsilon

            you may want to use env.action_space.sample() to generate 
            a random action   

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return best_action



def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    
    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")


def your_test():
    """
    Use this to implement your own tests
    """
    pass


if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()