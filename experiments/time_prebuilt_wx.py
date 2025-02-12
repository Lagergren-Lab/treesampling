import time

import numpy as np

from treesampling.algorithms import CastawayRST, random_spanning_tree_log
from treesampling.utils.graphs import random_uniform_graph, random_tree_skewed_graph, tree_to_newick


def main():
    n_samples = 100
    n_nodes = 20
    cache_size = 100
    tree_skewness = 5  # 1 is a uniform tree, the higher the more skewed towards one single tree
    print(f"Running {n_samples} samples on a graph with {n_nodes} nodes and cache size {cache_size}")
    # graph = random_uniform_graph(n_nodes, log_probs=True, normalize=False)
    graph, origin_tree = random_tree_skewed_graph(n_nodes, tree_skewness, root=0)
    prebuilt_wx_times = []
    sampler = CastawayRST(graph, 0, log_probs=True, trick=True, cache_size=cache_size)
    trees_newick = {}
    for _ in range(n_samples):
        start_time = time.time()
        tree = sampler.sample_tree()
        end_time = time.time() - start_time
        tree_newick = tree_to_newick(tree)
        if tree_newick in trees_newick:
            trees_newick[tree_newick] += 1
        else:
            trees_newick[tree_newick] = 1
        prebuilt_wx_times.append(end_time)

    print("first 10 trees\n", sorted(trees_newick.items(), key=lambda x: x[1], reverse=True)[:10])
    print("origin tree skewed graph\n", tree_to_newick(origin_tree))
    print("cache hits\n", sorted(sampler.wx._cache_hits.items(), key=lambda x: x[1], reverse=True))

    scratch_wx_times = []
    for _ in range(n_samples):
        start_time = time.time()
        tree = random_spanning_tree_log(graph, 0, trick=True)
        end_time = time.time() - start_time
        scratch_wx_times.append(end_time)


    print("With trick")
    total_prebuilt_wx_time = sum(prebuilt_wx_times)
    total_scratch_wx_time = sum(scratch_wx_times)
    print(f"Prebuilt WX: {total_prebuilt_wx_time / n_samples} +/- {np.std(prebuilt_wx_times)}")
    print(f"Scratch WX: {total_scratch_wx_time / n_samples} +/- {np.std(scratch_wx_times)}")
    print(f"Speedup over {n_samples} samples (in seconds): {total_scratch_wx_time - total_prebuilt_wx_time}")

    # without trick
    sampler = CastawayRST(graph, 0, log_probs=True, trick=False, cache_size=cache_size)
    for _ in range(n_samples):
        start_time = time.time()
        tree = sampler.sample_tree()
        end_time = time.time() - start_time
        prebuilt_wx_times.append(end_time)

    scratch_wx_times = []
    for _ in range(n_samples):
        start_time = time.time()
        tree = random_spanning_tree_log(graph, 0, trick=False)
        end_time = time.time() - start_time
        scratch_wx_times.append(end_time)

    print("Without trick")
    total_prebuilt_wx_time = sum(prebuilt_wx_times)
    total_scratch_wx_time = sum(scratch_wx_times)
    print(f"Prebuilt WX: {total_prebuilt_wx_time / n_samples} +/- {np.std(prebuilt_wx_times)}")
    print(f"Scratch WX: {total_scratch_wx_time / n_samples} +/- {np.std(scratch_wx_times)}")
    print(f"Speedup over {n_samples} samples (in seconds): {total_scratch_wx_time - total_prebuilt_wx_time}")


if __name__=='__main__':
    main()