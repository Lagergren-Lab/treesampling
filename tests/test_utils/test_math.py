import numpy as np
from scipy.stats import chisquare

from treesampling.utils.math import StableOp

def test_stable_op_random_choice():
    """
    Check that the random choice function from the class StableOp
    outputs the same results in log scale and in linear scale
    """
    so = StableOp(log_probs=False)
    so_log = StableOp(log_probs=True)
    arr = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isclose(np.sum(arr), 1.0)
    sample_size = 10000
    freqs = np.zeros(arr.size)
    freqs_log = np.zeros(arr.size)
    for _ in range(sample_size):
        choice = so.random_choice(arr)
        choice_log = so_log.random_choice(np.log(arr))

        freqs[choice] += 1
        freqs_log[choice_log] += 1

    test0 = chisquare(freqs, f_exp=sample_size * arr)
    test_log = chisquare(freqs_log, f_exp=sample_size * arr)
    print(freqs)
    print(test0.pvalue)
    print(freqs_log)
    print(test_log.pvalue)
    assert test0.pvalue > 0.05
    assert test_log.pvalue > 0.05

    # sanity check
    rnd_test = chisquare(freqs, f_exp=np.array([0.25, 0.25, 0.25, 0.25]) * sample_size)
    print(rnd_test.pvalue)
    assert rnd_test.pvalue < 0.05



