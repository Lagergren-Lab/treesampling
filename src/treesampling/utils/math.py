import numpy as np


def logsubexp(l1, l2):
    dx = -l1 + l2
    if np.isclose(dx, 0):
        exp_x = 1
    else:
        assert l1 >= l2, f"l1: {l1}, l2: {l2}"
        exp_x = np.exp(dx)

    res = l1 + np.log(1 - exp_x)
    return res


def gumbel_max_trick_sample(log_probs):
    gumbels = np.random.gumbel(size=len(log_probs))
    sample = np.argmax(log_probs + gumbels)
    return sample
