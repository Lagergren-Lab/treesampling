import numpy as np
import numpy.linalg as la


def logsubexp(l1, l2):
    """
    Subtraction in linear scale of log terms.
    """
    if np.isclose(l1, l2):
        # includes also case l1 == l2 == - np.inf
        res = - np.inf
    else:
        assert l1 > l2, f"l1: {l1}, l2: {l2}"
        dx = -l1 + l2
        exp_x = np.exp(dx)
        res = l1 + np.log(1 - exp_x)

    return res


def gumbel_max_trick_sample(log_probs):
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