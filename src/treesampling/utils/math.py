import numpy as np
import scipy.special as sp
import numpy.linalg as la

class StableOp:

    def __init__(self, log_probs=False):
        self.log_probs = log_probs

    def add(self, items: list):
        if self.log_probs:
            return np.logaddexp.reduce(items)
        else:
            return np.sum(items)

    def mul(self, items: list):
        if self.log_probs:
            return np.sum(items)
        else:
            return np.prod(items)

    def div(self, a, b):
        if self.log_probs:
            return a - b
        else:
            return a / b

    def sub(self, a, b):
        if self.log_probs:
            return logdiffexp(a, b)
        else:
            return a - b

    def zero(self):
        return -np.inf if self.log_probs else 0.0

    def one(self):
        return 0.0 if self.log_probs else 1.0

    def random_choice(self, arr: np.ndarray) -> int:
        if self.log_probs:
            return gumbel_max_trick_sample(arr)
        else:
            return np.random.choice(arr.size, p=arr)

    def normalize(self, arr, axis=None):
        if self.log_probs:
            return arr - sp.logsumexp(arr, axis=axis, keepdims=True)
        else:
            return arr / np.sum(arr, axis=axis, keepdims=True)


def logdiffexp(l1, l2):
    """
    Subtraction in linear scale of log terms.
    """
    if np.isclose(l1, l2, atol=1e-8):
        # includes also case l1 == l2 == - np.inf
        res = - np.inf
    elif l1 == -np.inf:
        res = -np.inf
    elif l2 == -np.inf:
        res = l1
    else:
        assert l1 > l2, f"l1: {l1}, l2: {l2}"
        dx = -l1 + l2
        exp_x = np.exp(dx)
        res = l1 + np.log1p(-exp_x)

    return res


def gumbel_max_trick_sample(log_probs: np.ndarray) -> int:
    # check that input log probs are normalized
    assert np.isclose(sp.logsumexp(log_probs), 0.0), (f"sum of log probs should be 0.0, but is "
                                                             f": {sp.logsumexp(log_probs)}")
    gumbels = np.random.gumbel(size=len(log_probs))
    sample = np.argmax(log_probs + gumbels)
    return sample


def generate_random_matrix(n, condition_number=10):
    # Construct random matrix P with specified condition number
    #
    #  Bierlaire, M., Toint, P., and Tuyttens, D. (1991).
    #  On iterative algorithms for linear ls problems with bound constraints.
    #  Linear Algebra and Its Applications, 143, 111–143.
    # https://gist.github.com/bstellato/23322fe5d87bb71da922fbc41d658079

    cond_P = condition_number
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P / 4., log_cond_P * (n + 1) / (4 * (n - 1)), log_cond_P / (2. * (n - 1)))
    exp_vec = exp_vec[:n]
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    P = U.dot(S).dot(V.T)
    P = P.dot(P.T)
    return P