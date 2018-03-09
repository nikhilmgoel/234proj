from gym.envs.registration import register

register(
    id='NavigateEnv-v0',
    entry_point='gym_navigate.envs:NavigateEnv',
)