"""
In this experiment, Castaway is tested against both Wilson (to determine time issues) and Kulkarni (to determine
feasibility issues with determinant).
tree_ratio: (true) probability mass covered by the sampled trees
crasher_issue: 1 if crashers detected in castaway, 0 otherwise
X_tot_weight_dist: distance between the total weight of the sampled trees and the true total weight
edge_freq_mse: mean square error of the edge frequencies between the sampled trees and the true tree dist
"""
import logging
import time
from datetime import datetime

import numpy as np
import scipy.special as sp
from tqdm import tqdm

from treesampling.algorithms.castaway_reboot import Castaway2RST
from treesampling.algorithms.kulkarni import kulkarni_rst
from treesampling.algorithms.wilson import wilson_rst_from_matrix
from treesampling.utils.evaluation import analyse_true_dist, get_sampler_pmf, jens_conductance, cheeger_constant
from treesampling.utils.graphs import crasher_matrix, laplacian, mat_minor, block_matrix


def compute_edge_statistics(pmf, tot_weight, reweighted: bool = False, log_p: bool = False) -> tuple[np.ndarray, np.ndarray]:
    n = len(next(iter(pmf.values()))[2])
    marginals = np.zeros((n, n)) - np.inf
    cooccurrences = np.zeros((n,) * 4) - np.inf

    for nwk, (w, freq, t) in pmf.items():
        tree_norm_log_weight = w - tot_weight if reweighted else np.log(freq)
        for v, u in enumerate(t):
            if u != -1:
                marginals[u, v] = np.logaddexp(marginals[u, v], tree_norm_log_weight)
                for v2, u2 in enumerate(t):
                    if u2 != -1:
                        cooccurrences[u, v, u2, v2] = np.logaddexp(cooccurrences[u, v, u2, v2], tree_norm_log_weight)
                        cooccurrences[u2, v2, u, v] = np.logaddexp(cooccurrences[u2, v2, u, v], tree_norm_log_weight)

    if not log_p:
        marginals = np.exp(marginals)
        cooccurrences = np.exp(cooccurrences)
    return marginals, cooccurrences


def evaluate_sampler(eval_pmf, true_pmf, reweighted: bool = False, true_edge_log_marginals: np.ndarray = None,
                     true_edge_log_cooccurrence: np.ndarray = None) -> dict:
    # compute tot_weight difference, edge marginals mse and co-occurrence mse
    n = len(next(iter(eval_pmf.values()))[2])
    true_tot_weight = -np.inf
    eval_tot_weight = -np.inf

    # compute total weight of the trees
    pmass_covered = 0
    for nwk, (w, freq, t) in true_pmf.items():
        # tot weight
        true_tot_weight = np.logaddexp(true_tot_weight, w)
        if nwk in eval_pmf:
            # consider only true negatives (i.e. eval_trees that are not in true_pmf are not considered, assumed to be ~0)
            eval_tot_weight = np.logaddexp(eval_tot_weight, eval_pmf[nwk][0])
            pmass_covered += freq

    eval_edge_log_marginals, eval_edge_log_cooccurrence = compute_edge_statistics(eval_pmf, eval_tot_weight, reweighted=reweighted, log_p=True)
    if true_edge_log_marginals is None or true_edge_log_cooccurrence is None:
        true_edge_log_marginals, true_edge_log_cooccurrence = compute_edge_statistics(true_pmf, true_tot_weight, reweighted=reweighted, log_p=True)

    # compare edge marginals and co-occurrences
    exp_true_log_marginals = np.exp(true_edge_log_marginals)
    exp_true_log_cooccurrence = np.exp(true_edge_log_cooccurrence)
    marginals_mse = np.sum((exp_true_log_marginals - np.exp(eval_edge_log_marginals)) ** 2) / (n * n)
    cooccurrence_mse = np.sum((exp_true_log_cooccurrence - np.exp(eval_edge_log_cooccurrence)) ** 2) / (n * n * n * n)

    # marginals and co-occurrences KL divergence (avoid inf values)
    mm = ~(np.isneginf(true_edge_log_marginals) | np.isneginf(eval_edge_log_marginals))  # marginals mask
    cm = ~(np.isneginf(true_edge_log_cooccurrence) | np.isneginf(eval_edge_log_cooccurrence))  # co-occurrence mask
    marginals_kl = np.sum(exp_true_log_marginals[mm] * (true_edge_log_marginals[mm] - eval_edge_log_marginals[mm]))
    cooccurrence_kl = np.sum(exp_true_log_cooccurrence[cm] * (true_edge_log_cooccurrence[cm] - eval_edge_log_cooccurrence[cm]))

    return {
        'tot_weight_diff': eval_tot_weight - true_tot_weight,
        'pmass_covered': pmass_covered,
        'edge_marginals_mse': marginals_mse,
        'edge_cooccurrence_mse': cooccurrence_mse,
        'edge_marginals_kl': marginals_kl,
        'edge_cooccurrence_kl': cooccurrence_kl
    }


def main():
    # parameters
    low_weights = [-2, -3, -5, -7, -10, -20, -30, -40, -50, -100]
    # low_weights = [-2, -20, -30, -40]
    # low_weights = [-6]
    time_limit = 1  # max 1s per tree (for wilson)
    n_sample = 1000
    num_components = 2
    n_nodes = 7
    num_seeds = 10
    enable_wilson = True

    columns = [
        "low_weight", "n_nodes", "seed", "wilson_time", "kulkarni_time", "castaway_time", "n_sample",
        # compare distribution for assessing accuracy (edge marginals and co-occurrence for complete statistics)
        "num_components", "wilson_edge_marg_mse", "kulkarni_edge_marg_mse", "castaway_edge_marg_mse",
        "wilson_edge_cooccurrence_mse", "kulkarni_edge_cooccurrence_mse", "castaway_edge_cooccurrence_mse",
        "wilson_marginals_kl", "kulkarni_marginals_kl", "castaway_marginals_kl",
        "wilson_cooccurrence_kl", "kulkarni_cooccurrence_kl", "castaway_cooccurrence_kl",
        "j_conductance", "cheegers_const", # random-walk-conductance and cheeger constant
        "tot_weight", "log_det", # bruteforce tot-log-weight and log-determinant of laplacian (should be equal)
        "wilson_tot_weight_diff", "kulkarni_tot_weight_diff", "castaway_tot_weight_diff", # log-diff
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
        for seed in range(num_seeds):
            logging.debug(f"seed: {seed}")
            np.random.seed(seed)
            # generate crasher matrix
            # matrix = crasher_matrix(n_nodes, log_eps=low_weight, num_components=num_components)
            matrix = block_matrix(n_nodes, n_blocks=num_components, low_weight=low_weight, log_probs=True, noise_ratio=0.1, root=0)
            matrix[:, 1:] = matrix[:, 1:] - sp.logsumexp(matrix[:, 1:], axis=0)
            logging.debug(f"Norm matrix:\n{np.array_str(matrix, max_line_width=100, precision=3, suppress_small=True)}")

            # get true pmf
            true_pmf, mst_nwk, edge_freq = analyse_true_dist(matrix)
            tot_weight = sp.logsumexp([w for w, _, _ in true_pmf.values()])
            exp_matrix = np.zeros((n_nodes, n_nodes))
            exp_matrix[:, 1:] = np.exp(matrix[:, 1:])
            log_det = np.nan
            if (det:=np.linalg.det(mat_minor(laplacian(exp_matrix), 0, 0))) > 0:
                log_det = np.log(det)
            true_edge_log_marginals, true_edge_log_cooccurrence = compute_edge_statistics(true_pmf, tot_weight, reweighted=True, log_p=True)

            # wilson
            wilson_time = np.nan
            wilson_eval = {}
            if enable_wilson:
                start = time.time()
                wilson_pmf = get_sampler_pmf(matrix, lambda x: wilson_rst_from_matrix(x, log_probs=True), n=n_sample, time_limit=time_limit)
                if wilson_pmf == {}:
                    logging.debug("Wilson sampling exceeded time limit, skipping")
                    enable_wilson = False
                    continue

                wilson_time = time.time() - start
                logging.debug(f"Wilson time: {wilson_time}")
                wilson_eval = evaluate_sampler(wilson_pmf, true_pmf, reweighted=True,
                                               true_edge_log_marginals=true_edge_log_marginals,
                                               true_edge_log_cooccurrence=true_edge_log_cooccurrence)

            # kulkarni
            try:
                start = time.time()
                kulkarni_pmf = get_sampler_pmf(matrix, lambda x: kulkarni_rst(x, log_probs=True), n=n_sample)
                kulkarni_time = time.time() - start
                kulkarni_eval = evaluate_sampler(kulkarni_pmf, true_pmf, reweighted=True,
                                                 true_edge_log_marginals=true_edge_log_marginals,
                                                 true_edge_log_cooccurrence=true_edge_log_cooccurrence)
            except IndexError as ie:
                kulkarni_time = np.nan
                kulkarni_eval = {}
                logging.debug(f"Error in kulkarni sampling, skipping")

            # castaway
            castaway_sampler = Castaway2RST(matrix, root=0, log_probs=True, trick=low_weight > -10)
            # rec if crashers exists
            crasher_issue = 1 if castaway_sampler.wx.crashers else 0
            start = time.time()
            castaway_pmf = get_sampler_pmf(matrix, lambda x: castaway_sampler.sample_tree_as_list(), n=n_sample)
            castaway_time = time.time() - start
            castaway_eval = evaluate_sampler(castaway_pmf, true_pmf, reweighted=True,
                                             true_edge_log_marginals=true_edge_log_marginals,
                                             true_edge_log_cooccurrence=true_edge_log_cooccurrence)

            j_conductance, _, _, _ = jens_conductance(matrix, root=0, log_probs=True)
            cheegers_const, _, _, _ = cheeger_constant(matrix, root=0, log_probs=True)

            bar.update(1)

            # save record
            with open(filename, "a") as f:
                rec = [low_weight, n_nodes, seed, wilson_time, kulkarni_time, castaway_time, n_sample, num_components,
                       wilson_eval.get('edge_marginals_mse', np.nan), kulkarni_eval.get('edge_marginals_mse', np.nan),
                       castaway_eval.get('edge_marginals_mse', np.nan),
                       wilson_eval.get('edge_cooccurrence_mse', np.nan), kulkarni_eval.get('edge_cooccurrence_mse', np.nan),
                       castaway_eval.get('edge_cooccurrence_mse', np.nan),
                       wilson_eval.get('edge_marginals_kl', np.nan), kulkarni_eval.get('edge_marginals_kl', np.nan),
                       castaway_eval.get('edge_marginals_kl', np.nan),
                       wilson_eval.get('edge_cooccurrence_kl', np.nan), kulkarni_eval.get('edge_cooccurrence_kl', np.nan),
                       castaway_eval.get('edge_cooccurrence_kl', np.nan),
                       j_conductance, cheegers_const, tot_weight, log_det,
                       wilson_eval.get('tot_weight_diff', np.nan), kulkarni_eval.get('tot_weight_diff', np.nan),
                       castaway_eval.get('tot_weight_diff', np.nan),
                       crasher_issue,
                       wilson_eval.get('pmass_covered', np.nan), kulkarni_eval.get('pmass_covered', np.nan),
                       castaway_eval.get('pmass_covered', np.nan)]
                assert len(rec) == len(columns), f"record length {len(rec)} != columns length {len(columns)}"
                f.write(",".join([str(r) for r in rec]) + "\n")
            logging.debug("saved record: " + str(rec))

    bar.close()


if __name__ == "__main__":
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO)
    main()
