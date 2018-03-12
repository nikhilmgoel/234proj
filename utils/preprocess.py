import numpy as np

def rescale(state):
    """
    Preprocess state (1948, 1630, 1) image into
    a (239, 200, 1) image in grey scale
    """
    state = np.reshape(state, [239, 200, 1]).astype(np.float32)

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)


def blackandwhite(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    # erase background
    state[state==144] = 0
    state[state==109] = 0
    state[state!=0] = 1

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2, 0] # downsample by factor of 2

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)