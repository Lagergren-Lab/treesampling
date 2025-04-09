import time
import numpy as np
from tqdm import tqdm

from treesampling.algorithms.wilson import wilson_rst_from_matrix
from treesampling.algorithms.kulkarni import kulkarni_rst
from treesampling.utils.graphs import crasher_matrix
from treesampling.algorithms.castaway_reboot import Castaway2RST

def main():
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_time.csv"
    # Time analysis for crasher vs kulkarni
    k_values = [7, 10, 15]
    log_eps_values = [-2, -3, -5, -7, -10, -30, -40, -50, -100]
    n_seeds = 10
    n_sample = 500
    with open(filename, 'w') as f:
        f.write('k,seed,log_eps,wilson_time,kulkarni_time,castaway_time,trick,crashers\n')
    bar = tqdm(total=n_seeds * len(k_values) * len(log_eps_values))
    for k in k_values:
        for log_eps in log_eps_values:
            # print(f"Running k={k}, log_eps={log_eps}")
            bar.set_description(f"k={k},log_eps={log_eps}")
            kulkarni_disabled = False
            wilson_disabled = log_eps < -5
            trick = log_eps > -10
            for seed in range(n_seeds):
                np.random.seed(seed)
                matrix = crasher_matrix(k, 2, log_eps=log_eps)

                if not wilson_disabled:
                    bar.set_description(f"k={k},log_eps={log_eps},wilson...")
                    wilson_time = time.time()
                    for i in range(n_sample):
                        _ = wilson_rst_from_matrix(matrix, log_probs=True)
                    wilson_time = time.time() - wilson_time
                else:
                    wilson_time = np.nan

                if not kulkarni_disabled:
                    bar.set_description(f"k={k},log_eps={log_eps},kulkarni...")
                    kulkarni_time = time.time()
                    for i in range(n_sample):
                        try:
                            _ = kulkarni_rst(matrix, 0, log_probs=True, debug=False)
                        except IndexError:
                            kulkarni_disabled = True
                            break
                    kulkarni_time = time.time() - kulkarni_time if not kulkarni_disabled else np.nan
                else:
                    kulkarni_time = np.nan

                bar.set_description(f"k={k},log_eps={log_eps},castaway{'(trick)' if trick else ''}...")
                sampler = Castaway2RST(matrix, 0, log_probs=True, trick=trick, debug=False)
                crashers = sampler.wx.crashers
                castaway_time = time.time()
                for i in range(n_sample):
                    _ = sampler.sample_tree_as_list()
                castaway_time = time.time() - castaway_time

                # store results
                with open(filename, 'a') as f:
                    f.write(f"{k},{seed},{log_eps},{wilson_time / n_sample},{kulkarni_time / n_sample},{castaway_time / n_sample},{int(trick)},{int(bool(crashers))}\n")

                bar.update(1)

if __name__ == "__main__":
    main()
