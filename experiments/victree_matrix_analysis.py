"""
Take an interesting matrix from VICTree execution, compute the total weight via brute force and use it for
assessing sampling correctness for the CastawayRST algorithm.
"""
import logging
import time

import numpy as np
import scipy.special as sp

from treesampling.algorithms.castaway import importance_sample
# from treesampling.algorithms import CastawayRST
from treesampling.algorithms.castaway_reboot import Castaway2RST
from treesampling.algorithms.kulkarni import kulkarni_rst
from treesampling.algorithms.wilson import wilson_rst_from_matrix
from treesampling.utils.evaluation import analyse_true_dist, get_victree_demo_matrix, cheeger_constant, get_sampler_pmf
from treesampling.utils.graphs import parlist_to_newick, crasher_matrix, block_matrix, mat_minor, laplacian, \
    crasher2_matrix


def run_analysis(matrix, sample_size=5000, temp=False):
    norm_matrix = matrix - sp.logsumexp(matrix, axis=0)
    norm_matrix[:, 0] = -np.inf
    logging.info(f"Conductance: {cheeger_constant(norm_matrix, root=0, log_probs=True)[0]}")
    logging.info(f"Condition number: {np.linalg.cond(mat_minor(laplacian(np.exp(norm_matrix)), 0, 0))}")
    logging.info(f"Normalized matrix:\n{np.array_str(norm_matrix, max_line_width=100, precision=3, suppress_small=True)}")
    true_pmf, mst, edge_freq = analyse_true_dist(matrix)
    n_samples = castaway_dist(matrix, true_pmf, sample_size=sample_size)
    # test Wilson
    # wilson_dist(norm_matrix, true_pmf, sample_size=n_samples)
    # test Kulkarni
    try:
        kulkarni_dist(matrix, n_samples, true_pmf)
    except IndexError as ie:
        logging.info(f"Error in kulkarni sampling, skipping")
    # importance samples with tempering
    if temp:
        tempering_analysis(matrix, true_pmf)

def wilson_dist(matrix, true_pmf, sample_size=5000, time_limit=300):
    print("Wilson:")
    trees = {}
    avg_time = 0.
    start_alg = time.time()
    for i in range(sample_size):
        start = time.time()
        tree = tuple(wilson_rst_from_matrix(matrix, log_probs=True))
        end = time.time()
        avg_time = ((avg_time * i) + (end - start)) / (i + 1)
        # running estimation
        eta = avg_time * sample_size
        if time.time() - start_alg > 10 and eta > time_limit:
            print(f"ETA ({eta}s) exceeds time limit ({time_limit}s)")
            break
        elif i % 100 == 0:
            print(f"Sample {i} ({end - start:.3f}s/sample, ETA: {eta:.3f}s)")

        nwk = parlist_to_newick(tree)
        if nwk not in trees:
            trees[nwk] = 0
        trees[nwk] += 1

    for nwk, freq in sorted([(k, v / 5000) for k, v in trees.items()], key=lambda u: u[1], reverse=True):
        if nwk in true_pmf:
            print(f"tree: {nwk} ({freq} | true: {true_pmf[nwk][1]})")
        else:
            print(f"tree: {nwk} ({freq} | true: 0.0)")
    return None


def tempering_analysis(matrix, true_pmf, temp_vals: list = None):
    if temp_vals is None:
        temp_vals = [2, 5, 10, 50, 100]
    for temp in temp_vals:
        print(f"temp: {temp}")
        try:
            trees = importance_sample(matrix, 100, temp=temp, log_probs=True)
        except Exception as e:
            print(f"Error: {e} -- skipping temperature {temp}")
            continue

        tot_weight = sp.logsumexp([w for w in trees.values()])
        max_print = len(true_pmf)
        i = 0
        for tree in sorted(trees, key=trees.get, reverse=True):
            nwk = parlist_to_newick(tree)
            freq = trees[tree] - tot_weight

            # if true weight is available, compare it
            true_weight = 0. if nwk not in true_pmf else true_pmf[nwk][1]
            print(f"tree: {nwk} ({np.exp(freq)} | true: {true_weight} | iw: {freq})")
            i += 1
            if i >= max_print:
                break


def kulkarni_dist(matrix, n_samples, true_pmf):
    print("Kulkarni:")
    kulkarni_pmf = {}
    for i in range(n_samples):
        tree = tuple(kulkarni_rst(matrix, 0, log_probs=True, debug=False))
        nwk = parlist_to_newick(tree)
        if nwk not in kulkarni_pmf:
            kulkarni_pmf[nwk] = 0
        kulkarni_pmf[nwk] += 1 / n_samples
    for nwk, freq in sorted([(k, v) for k, v in kulkarni_pmf.items()], key=lambda u: u[1], reverse=True):
        if nwk in true_pmf:
            print(f"tree: {nwk} ({freq} | true: {true_pmf[nwk][1]})")
        else:
            print(f"tree: {nwk} ({freq} | true: 0.0)")


def castaway_dist(matrix, true_pmf, sample_size=5000):
    # sample trees with CastawayRST
    # sampler = CastawayRST(matrix, 0, log_probs=True, trick=False, debug=True)
    sampler = Castaway2RST(matrix, 0, log_probs=True, trick=False, debug=False)
    # tree = sampler.castaway_rst()
    # print(f"CastawayRST tree: {parlist_to_newick(tree)}")
    castaway_pmf = get_sampler_pmf(matrix, lambda x: sampler.sample_tree_as_list(), n=sample_size)
    edge_freq = np.zeros_like(matrix)
    for _, f, t in castaway_pmf.values():
        for i in range(1, len(t)):
            edge_freq[t[i], i] += f

    print(f"Crasher trick: {sampler.wx._init_crashers}")
    print("Edge frequencies:")
    print(edge_freq)
    print("Edge frequencies sum (must be 1 everywhere except idx=0):")
    print(np.sum(edge_freq, axis=0))
    for nwk, (w, freq, t) in sorted([(k, v) for k, v in castaway_pmf.items()], key=lambda x: x[1][1], reverse=True):
        if nwk in true_pmf:
            print(f"tree: {nwk} ({freq} | true: {true_pmf[nwk][1]})")
        else:
            print(f"tree: {nwk} ({freq} | true: 0.0)")
    return sample_size


def main():
    # Example matrix from VICTree
    # matrix = get_victree_demo_matrix()
    # matrix = crasher_matrix(7, 1, -40)
    matrix = crasher2_matrix(7, 1, -70)
    # matrix = block_matrix(7, n_blocks=2, low_weight=-40, log_probs=True, noise_ratio=0.2, root=0)
    run_analysis(matrix, 10000)

if __name__ == "__main__":
    np.random.seed(42)
    logging.basicConfig(level=logging.DEBUG)
    main()
