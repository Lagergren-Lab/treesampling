import math
import numpy as np


def logsubexp(l1, l2):
    assert l1 >= l2, f"l1: {l1}, l2: {l2}"
    return l1 + math.log(1 - math.exp(-l1 + l2))


def gumbel_max_trick_sample(log_probs):
    gumbels = np.random.gumbel(size=len(log_probs))
    sample = np.argmax(log_probs + gumbels)
    return sample
