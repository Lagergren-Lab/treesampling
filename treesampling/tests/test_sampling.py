from treesampling import algorithms
from treesampling.utils.graphs import *


def test_log_random_uniform_graph():
    log_graph = random_uniform_graph(5, log_probs=True)
    weight_matrix = nx.to_numpy_array(log_graph)
    # all weights except diagonals are < 0
    diag_mask = np.eye(weight_matrix.shape[0], dtype=bool)
    assert np.all(weight_matrix[diag_mask] == 0)
    assert np.all(weight_matrix[~diag_mask] < 0)


def test_random_k_trees_graph():
    root = 0
    graph = random_tree_skewed_graph(5, 100)
    norm_graph = normalize_graph_weights(graph)

    # Kirchhoff theorem for directed graphs, to get total weight of all trees
    tot_weight = tuttes_tot_weight(norm_graph, root)
    print(f"total weight: {tot_weight}")
    log_graph = reset_adj_matrix(graph, np.log(nx.to_numpy_array(graph)))

    print(nx.to_numpy_array(norm_graph))

    print(f"MST: {tree_to_newick(nx.maximum_spanning_arborescence(norm_graph))}")

    sample_size = 10000
    sample = {}
    acc = 0
    num = 0
    for i in range(sample_size):
        tree = algorithms.jens_rst(log_graph, root=root)
        tree_nwk = tree_to_newick(tree)
        if tree_nwk not in sample:
            weight = np.exp(graph_weight(tree, log_probs=True))
            sample[tree_nwk] = weight
            num += 1
            acc += weight
            print(f"new tree {tree_nwk}:{weight}. TOT weight = {acc} ({acc / tot_weight})")
    print(f"tot weight by sampling: {acc}")
    print(f"tot unique trees {num}")


def test_laplacian():
    n_nodes = 6

    # laplacian for number of undirected trees (no weight)
    weights = np.ones((n_nodes, n_nodes))
    graph = nx.from_numpy_array(weights)
    lap_mat = nx.laplacian_matrix(graph, weight='weight').toarray()
    print(lap_mat)
    assert np.isclose(np.linalg.det(lap_mat[1:, 1:]), cayleys_formula(n_nodes))

    # laplacian for total weight of undirected trees
    weights = np.random.random((n_nodes, n_nodes))
    graph = nx.from_numpy_array(weights)
    norm_graph = normalize_graph_weights(graph)
    lap2 = nx.laplacian_matrix(norm_graph).toarray()
    lap_tot_weight = np.linalg.det(lap2[1:, 1:])

    tot_weight = 0
    for pruf_seq in itertools.product(range(n_nodes), repeat=n_nodes - 2):
        tree = nx.from_prufer_sequence(list(pruf_seq))
        for e in tree.edges():
            tree.edges()[e]['weight'] = norm_graph.edges()[e]['weight']
        tw = graph_weight(tree)
        tot_weight += tw

    assert np.isclose(tot_weight, lap_tot_weight)

    # laplacian for directed rooted trees
    # FIXME: incorrect way of enumerating arborescences
    tot_weight0 = 0
    for pruf_seq in itertools.product(range(n_nodes), repeat=n_nodes - 2):
        if pruf_seq[-1] == 0:
            tree = nx.from_prufer_sequence(list(pruf_seq))
            for e in tree.edges():
                tree.edges()[e]['weight'] = norm_graph.edges()[e]['weight']
            tw = graph_weight(tree)
            tot_weight0 += tw

    laplacian_tot_weight = tuttes_tot_weight(norm_graph, 0)
    # assert np.isclose(tot_weight0, laplacian_tot_weight)


def test_uniform_graph_sampling():
    n_nodes = 5
    adj_mat = np.ones((n_nodes, n_nodes))
    graph = nx.from_numpy_array(adj_mat)
    norm_graph = normalize_graph_weights(graph)

    sample_size = 5000
    sample_dict = {}
    for s in range(sample_size):
        tree = algorithms.jens_rst(norm_graph, root=0)
        tree_nwk = tree_to_newick(tree)
        if tree_nwk not in sample_dict:
            sample_dict[tree_nwk] = 0
        sample_dict[tree_nwk] = sample_dict[tree_nwk] + 1
    print("\n")
    for k, v in sorted(sample_dict.items(), key=lambda p: p[1], reverse=True):
        print(k, v)
    print(len(sample_dict))






