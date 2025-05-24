"""
In this experiment, Castaway is tested against both Wilson (to determine time issues) and Kulkarni (to determine
feasibility issues with determinant). This experiment is done on graphs with 100 nodes.
Comparison is done by sampling 50 trees and calculating the proportion of mass covered by the sampled trees (when
the determinant can be computed). When determinant cannot be computed, only the time is computed.
The trees are sampled in parallel to speed up the process using multiprocessing.

tree_ratio: (true) probability mass covered by the sampled trees
crasher_issue: 1 if crashers detected in castaway, 0 otherwise
"""
import logging
import math
import sys
import time
from datetime import datetime
import multiprocessing as mp
from typing import Any

import numpy as np
import scipy.special as sp
from tqdm import tqdm

from treesampling.algorithms.castaway_reboot import Castaway2RST
from treesampling.algorithms.kulkarni import kulkarni_rst
from treesampling.algorithms.wilson import wilson_rst_from_matrix
from treesampling.utils.graphs import crasher_matrix, laplacian, mat_minor, tuttes_tot_weight, tree_weight


def sample_tree(sampler: callable, matrix: np.ndarray = None, id: int = None) -> tuple[tuple[int] | None, float, int]:
    """
    Wrapper function to sample a tree and measure the time taken
    :param sampler: function to sample a tree
    :param matrix: np.ndarray, matrix to sample from
    :return: tree as parent list and time taken to sample
    """
    start = time.time()
    tree_list = sampler(matrix) if matrix is not None else sampler()
    tree = tuple(tree_list) if tree_list is not None else tree_list
    end = time.time()
    return tree, end - start, id

def kulkarni_sampler(matrix: np.ndarray) -> list[int] | None:
    try:
        return kulkarni_rst(matrix, root=0, log_probs=True, debug=False)
    except IndexError:
        return None

def wilson_sampler(matrix: np.ndarray) -> list[int] | None:
    return wilson_rst_from_matrix(matrix, log_probs=True)

wilson_trees = [None]
wilson_times = [None]

def wilson_callback(res: Any) -> None:
    """
    Callback function to handle the result of Wilson's sampling
    :param res: result from the multiprocessing pool
    """
    # print(f"Wilson callback received result: {res}")
    tree, t, i = res
    wilson_trees[i] = tree
    wilson_times[i] = t


def main():
    # parameters
    low_weights = [-2, -5, -10, -20, -50]
    # low_weights = [-1, -20, -30, -40]
    # low_weights = [-6]
    time_limit: float = np.inf # the limit is 2 times the castaway time for the same (low_weight, n_nodes) pair
    n_sample = 64
    num_components = 2
    n_nodes = 100
    # n_nodes = 8
    num_seeds = 2

    columns = [
        "low_weight", "n_nodes", "seed",
        "wilson_time_avg", "kulkarni_time_avg", "castaway_time_avg",
        "wilson_time_std", "kulkarni_time_std", "castaway_time_std",
        "n_sample",
        # compare distribution for assessing accuracy (edge marginals and co-occurrence for complete statistics)
        "num_components",
        # "j_conductance", "cheegers_const", # random-walk-conductance and cheeger constant (too slow for 100 nodes)
        "log_det", "cond_number_log10", # bruteforce tot-log-weight and log-determinant of laplacian (should be equal)
        "crasher_issue", # 1 if crashers detected, 0 otherwise
        "wilson_pmass_covered", "kulkarni_pmass_covered", "castaway_pmass_covered" # probability mass covered (0 to 1)
    ]
    # append records to csv
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = timestamp + ".csv"
    with open(filename, "w") as f:
        f.write(",".join(columns) + "\n")
    # generate matrices with different values of low weight (in log-scale)
    bar = tqdm(total=len(low_weights) * num_seeds)
    for low_weight in low_weights:
        logging.debug(f"Low weight: {low_weight}")
        # multiple graphs iterations
        wilson_enabled = True
        for seed in range(num_seeds):
            bar.set_description(f"low_weight={low_weight}, seed={seed}")
            logging.debug(f"seed: {seed}")
            np.random.seed(seed)
            # generate crasher matrix
            matrix = crasher_matrix(n_nodes, log_eps=low_weight, num_components=num_components)
            matrix[:, 1:] = matrix[:, 1:] - sp.logsumexp(matrix[:, 1:], axis=0)
            logging.debug(f"Norm matrix:\n{np.array_str(matrix, max_line_width=100, precision=3, suppress_small=True)}")

            # get total weight with determinant
            exp_matrix = np.zeros((n_nodes, n_nodes))
            exp_matrix[:, 1:] = np.exp(matrix[:, 1:])
            log_det = np.nan
            if (det:=np.linalg.det(mat_minor(laplacian(exp_matrix), 0, 0))) > 0:
                log_det = np.log(det)
            # save conditioning number to double check with positive determinant correctness
            cond_number_log10 = np.log10(np.linalg.cond(mat_minor(laplacian(exp_matrix), 0, 0)))

            # parallelize castaway
            castaway_obj = Castaway2RST(matrix, root=0, log_probs=True, trick=low_weight > -10)
            castaway_sampler = castaway_obj.sample_tree_as_list
            crasher_issue = bool(castaway_obj.wx.crashers)
            with mp.Pool(processes=mp.cpu_count()) as pool:
                resPool = pool.starmap(sample_tree, [(castaway_sampler,) for _ in range(n_sample)])
            castaway_times = [resPool[i][1] for i in range(n_sample)]
            castaway_trees = [resPool[i][0] for i in range(n_sample)]
            castaway_time_avg = np.mean(castaway_times)
            castaway_time_std = np.std(castaway_times)
            time_limit = 2 * castaway_time_avg

            # parallelize wilson sampler with timeout
            global wilson_trees, wilson_times
            wilson_trees = [None] * n_sample
            wilson_times = [None] * n_sample
            wilson_time_avg = np.nan
            wilson_time_std = np.nan
            if wilson_enabled:
                # Run Wilson's in parallel with timeout based on castaway time
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    async_results = [pool.apply_async(sample_tree, args=(wilson_sampler, matrix, i), callback=wilson_callback)
                                     for i in range(n_sample)]
                    stop_after = time_limit * math.ceil(n_sample / mp.cpu_count()) + 5  # add some buffer time
                    logging.debug(f"Waiting for Wilson's sampling to finish with time limit {stop_after:.2f} seconds")
                    time.sleep(stop_after)
                    # print(f"wilson trees after callbacks {wilson_trees}")
                    pool.terminate()
                # count None values, if more than 0.5 * n_sample, disable wilson
                wilson_times = [t for t in wilson_times if t is not None]
                if len(wilson_times) < 0.5 * n_sample:
                    wilson_enabled = False
                    logging.debug(f"Less than 50% of Wilson's trees sampled in time limit ({len(wilson_times)} < {0.5 * n_sample}), disabling Wilson's sampling")
                else:
                    wilson_trees = [t for t in wilson_trees if t is not None]
                    wilson_time_avg = np.mean(np.array(wilson_times))
                    wilson_time_std = np.std(np.array(wilson_times))
                    logging.debug(f"Wilson's sampling completed: {len(wilson_trees)} trees sampled in {wilson_time_avg:.2f} seconds (std: {wilson_time_std:.2f})")

            # parallelize kulkarni with IndexError check

            kulkarni_trees = []
            kulkarni_times = []
            kulkarni_time_avg = np.nan
            kulkarni_time_std = np.nan
            with mp.Pool(processes=mp.cpu_count()) as pool:
                resPool = pool.starmap(sample_tree, [(kulkarni_sampler, matrix)] * n_sample)
            for i, (tree, t, _) in enumerate(resPool):
                if tree is not None:
                    kulkarni_trees.append(tree)
                    kulkarni_times.append(t)
            if len(kulkarni_trees) < 0.5 * n_sample:
                logging.debug("Less than 50% of Kulkarni's trees sampled, disabling Kulkarni's sampling")
            else:
                kulkarni_time_avg = np.mean(kulkarni_times)
                kulkarni_time_std = np.std(kulkarni_times)

            # compare distributions with log_det as total weight
            castaway_pmass_covered = np.exp(sp.logsumexp([tree_weight(tree, matrix, log_probs=True) for tree in list(set(castaway_trees))]) - log_det)
            if wilson_enabled and len(wilson_trees) > 0:
                wilson_pmass_covered = np.exp(sp.logsumexp([tree_weight(tree, matrix, log_probs=True) for tree in list(set(wilson_trees))]) - log_det)
            else:
                wilson_pmass_covered = 0
            if len(kulkarni_trees) > 0:
                kulkarni_pmass_covered = np.exp(sp.logsumexp([tree_weight(tree, matrix, log_probs=True) for tree in list(set(kulkarni_trees))]) - log_det)
            else:
                kulkarni_pmass_covered = 0

            # computing conductance could be slow for large graphs, so we skip it
            # j_conductance, _, _, _ = jens_conductance(matrix, root=0, log_probs=True)
            # cheegers_const, _, _, _ = cheeger_constant(matrix, root=0, log_probs=True)

            bar.update(1)

            # save record
            with open(filename, "a") as f:
            # "low_weight", "n_nodes", "seed",
            # "wilson_time_avg", "kulkarni_time_avg", "castaway_time_avg",
            # "wilson_time_std", "kulkarni_time_std", "castaway_time_std",
            # "n_sample",
            # "num_components",
            # "log_det", "cond_number_log10" # bruteforce tot-log-weight and log-determinant of laplacian (should be equal)
            # "crasher_issue", # 1 if crashers detected, 0 otherwise
            # "wilson_pmass_covered", "kulkarni_pmass_covered", "castaway_pmass_covered"
                rec = [
                    low_weight, n_nodes, seed,
                    wilson_time_avg, kulkarni_time_avg, castaway_time_avg,
                    wilson_time_std, kulkarni_time_std, castaway_time_std,
                    n_sample,
                    num_components,
                    log_det, cond_number_log10,
                    int(crasher_issue),
                    wilson_pmass_covered, kulkarni_pmass_covered, castaway_pmass_covered
                ]
                assert len(rec) == len(columns), f"record length {len(rec)} != columns length {len(columns)}"
                f.write(",".join([str(r) for r in rec]) + "\n")
            logging.debug("saved record: " + str(rec))

    bar.close()


if __name__ == "__main__":
    print(f"Parallelizing on {mp.cpu_count()}")
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO)
    main()
