import itertools
import warnings
from operator import mul
from functools import reduce
import numpy as np
import networkx as nx
from scipy.linalg import lu

from treesampling.utils.math import StableOp


# copied from VICTree
def tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert nx.is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)

def parlist_to_newick(parlist):
    tree = nx.DiGraph()
    for i, p in enumerate(parlist):
        if p != -1:
            tree.add_edge(p, i)
    return tree_to_newick(tree)

# taken from VICTree
def enumerate_rooted_trees(n_nodes, root=0, weighted_graph: nx.DiGraph | None = None) -> [nx.DiGraph]:
    """
    Generate all Prufer sequences to enumerate rooted trees
    :param n_nodes: number of nodes
    :param root: root label
    :param weighted_graph: graph (optional)
    :return: list of nx.DiGraphs with unique rooted trees
    """
    trees = []
    for pruf_seq in itertools.product(range(n_nodes), repeat=n_nodes - 2):
        unrooted_tree = nx.from_prufer_sequence(list(pruf_seq))
        # hang tree from root to generate associated arborescence
        rooted_tree = nx.dfs_tree(unrooted_tree, root)
        # if weighted graph is provided, return trees with weights
        if weighted_graph is not None:
            for e in rooted_tree.edges():
                rooted_tree.edges()[e]['weight'] = weighted_graph.edges()[e]['weight']
        trees.append(rooted_tree)
    return trees

def block_matrix(n_nodes: int, n_blocks: int = 2, log_probs: bool = False, low_weight: float = 1e-3,
                 root: int = None, noise_ratio: float = 0.) -> np.ndarray:
    """
    Generate a symmetric block matrix with weight of 1 for intra-block connections and low_weight for inter-block connections.
    :param n_nodes: int, number of nodes
    :param n_blocks: int, number of blocks
    :param log_probs: bool, if True, weights are log probabilities
    :param low_weight: float, weight for inter-block connections
    :param root: int, root index if any. root is detached from the blocks
    :return: np.ndarray, block matrix
    """

    op = StableOp(log_probs)
    if log_probs:
        noise =  np.random.rand(n_nodes, n_nodes) * abs(low_weight) * noise_ratio
    else:
        noise = np.random.rand(n_nodes, n_nodes) * low_weight * noise_ratio
    weights = np.zeros((n_nodes, n_nodes)) + low_weight + noise

    # divide nodes (exclude root) into k components
    component_nodes = n_nodes - (1 if root is not None else 0)
    nodes_per_component = component_nodes // n_blocks
    components = [-1] + [i // nodes_per_component for i in range(nodes_per_component * n_blocks)]
    # fill remaining nodes with last component
    components = components + [n_blocks - 1] * (component_nodes - len(components))
    # assign lower weight to arcs between components
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j or (root is not None and j == root):
                weights[i, j] = op.zero()
            # set 1 for arcs within components
            elif components[i] == components[j]:
                weights[i, j] = op.one()
    return weights

def random_uniform_graph(n_nodes, log_probs=False, normalize=None) -> nx.DiGraph:
    """
    Generate a random graph with random weights drawn from uniform distribution [0, 1].
    :param n_nodes: int, number of nodes
    :param log_probs: bool, if True, weights are log probabilities
    :param normalize: bool, if True, normalize weights
    :return: nx.DiGraph
    """
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    for u, v in graph.edges():
        if u == v:
            w = 0
        else:
            w = np.random.random()
        graph.edges()[u, v]['weight'] = w if not log_probs else np.log(w)

    if normalize is None:
        print("normalization will switch to True by default in the future")
        normalize = False
    if normalize:
        graph = normalize_graph_weights(graph, log_probs=log_probs)
    return graph


def random_weakly_connected_graph(n_nodes, log_probs=False, weak_weight=1e-3) -> nx.DiGraph:
    graph = nx.DiGraph()
    weights = np.random.random((n_nodes, n_nodes))
    component2 = [i for i in range(n_nodes) if i % 2 == 0]
    # divide into two components
    for i in range(n_nodes):
        for j in range(n_nodes):
            # set low weight for arcs between components
            if (i in component2) ^ (j in component2):
                weights[i, j] = weights[i, j] * weak_weight
    np.fill_diagonal(weights, 0)
    if log_probs:
        weights = np.log(weights)
    graph = reset_adj_matrix(graph, weights)
    return graph

def random_weakly_connected_k_subgraphs(n_nodes, k: int = 2, log_probs=False, weak_weight=1e-3,
                                        normalized: bool = True) -> nx.DiGraph:
    op = StableOp(log_probs)
    graph = nx.DiGraph()
    weights = np.random.random((n_nodes, n_nodes))
    if log_probs:
        weights = np.log(weights)

    # divide nodes into k components
    nodes_per_component = n_nodes // k
    components = [i // nodes_per_component for i in range(nodes_per_component * k)]
    # fill remaining nodes with last component
    components = components + [k - 1] * (n_nodes - len(components))
    # assign lower weight to arcs between components
    for i in range(n_nodes):
        for j in range(n_nodes):
            # set low weight for arcs between components
            if components[i] != components[j]:
                weights[i, j] = op.mul([weights[i, j], weak_weight])
    # remove self connections
    np.fill_diagonal(weights, op.zero())
    # reset graph with new weights
    graph = reset_adj_matrix(graph, weights)
    # normalize graph weights
    if normalized:
        graph = normalize_graph_weights(graph, log_probs=log_probs)
    return graph

def random_block_matrix_graph(n_nodes: int, n_blocks: int = 2, log_probs: bool = False, p: float = 0.5,
                              symmetric: bool = True) -> nx.DiGraph:
    """
    Generate a random block matrix graph with random weights drawn from uniform distribution [0, 1] and connections
    between blocks drawn from Bernoulli distribution with parameter p.
    :param n_nodes: int, number of nodes
    :param n_blocks: int, number of blocks
    :param log_probs: bool, if True, weights are log probabilities
    :param p: float, probability of connection between blocks
    :param symmetric: bool, if True, the graph matrix is symmetric
    :return: nx.DiGraph with block matrix structure
    """

    op = StableOp(log_probs)
    graph = nx.DiGraph()
    weights = np.random.random((n_nodes, n_nodes))
    if log_probs:
        weights = np.log(weights)

    # divide nodes into k components
    nodes_per_component = n_nodes // n_blocks
    components = [i // nodes_per_component for i in range(nodes_per_component * n_blocks)]
    # fill remaining nodes with last component
    components = components + [n_blocks - 1] * (n_nodes - len(components))
    # assign lower weight to arcs between components
    for i in range(n_nodes):
        for j in range(n_nodes):
            if symmetric and i > j:
                weights[i, j] = weights[j, i]
            # set low weight for arcs between components
            elif components[i] != components[j]:
                random_flip = op.one() if np.random.random() < p else op.zero()
                weights[i, j] = op.mul([weights[i, j], random_flip])
    # remove self connections
    np.fill_diagonal(weights, op.zero())
    # reset graph with new weights
    graph = reset_adj_matrix(graph, weights)
    # normalize graph weights
    graph = normalize_graph_weights(graph, log_probs=log_probs)
    return graph


def random_tree_skewed_graph(n_nodes, skewness, root: int | None = None,
                             log_probs=False) -> tuple[nx.DiGraph, nx.DiGraph]:
    """
    Generate an adjacency matrix by sampling 1 random spanning tree
    and adding k to the weight of each arc on that tree (uses nx random spanning tree function)
    The result is an adjacency matrix which has most probability mass on one tree.
    :param n_nodes: number of nodes
    :param skewness: unbalance of edge weights
    :return: tuple of resulting weighted graph and tree towards whom the graph is skewed
    """
    adj_matrix = np.ones((n_nodes, n_nodes))
    # remove self connections
    np.fill_diagonal(adj_matrix, 0)
    # if rooted, remove all arcs going in root
    if root is not None:
        adj_matrix[:, root] = 0
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    tree = nx.random_tree(n_nodes, create_using=nx.DiGraph)
    for e in tree.edges():
        adj_matrix[e] += skewness
    if log_probs:
        adj_matrix = np.log(adj_matrix)
    graph = reset_adj_matrix(graph, adj_matrix)
    return graph, tree


def normalize_graph_weights(graph, log_probs=False, rowwise=False) -> nx.DiGraph:
    """
    Normalize graph weights to sum to 1.
    :param graph: nx.DiGraph with weights
    :param log_probs: if True, weights are log probabilities
    :param rowwise: if True, normalize rowwise (axis=1), default is columnwise (axis=0) i.e. in-edges sum to 1
    :return: copy of graph with normalized weights (self loops set to 0)
    """
    op = StableOp(log_probs)
    adj_mat = nx.to_numpy_array(graph)
    # if no self loops in graph, set diagonal to -inf
    adj_mat[np.eye(adj_mat.shape[0], dtype=bool)] = op.zero()
    axis = 1 if rowwise else 0
    norm_adj_mat = op.normalize(adj_mat, axis=axis)
    norm_graph = reset_adj_matrix(graph, norm_adj_mat)
    return norm_graph


def reset_adj_matrix(graph: nx.DiGraph, matrix: np.ndarray) -> nx.DiGraph:
    weights_dict = [(i, j, matrix[i, j]) for i, j in itertools.product(range(matrix.shape[0]), repeat=2)]
    new_graph = graph.copy()
    new_graph.add_weighted_edges_from(weights_dict)
    return new_graph

def tree_weight(tree: list[int] | tuple[int], weight_matrix: np.ndarray, log_probs: bool = False) -> float:
    # takes tree (as list of parent nodes, where -1 indicates root node)
    op = StableOp(log_probs)
    arcs_idx = list(zip(*[(i, j) for j, i in enumerate(tree) if i != -1]))  # [(i1, i2, ..., in-1), (j1, j2, ..., jn-1)]
    return op.mul(weight_matrix[arcs_idx[0], arcs_idx[1]].tolist())

def graph_weight(graph: nx.DiGraph, log_probs=False):
    if log_probs:
        # sum of weights
        w = graph.size(weight='weight')
    else:
        # product of weights
        w = reduce(mul, list(graph.edges()[e]['weight'] for e in graph.edges()), 1)

    return w


def mat_minor(mat, row, col):
    """
    Minor of a matrix, removing row and col.
    :param mat: 2D numpy array
    :param row: row index to be removed
    :param col: col index to be removed
    :return:
    """
    M, N = mat.shape
    ridx = np.hstack([np.arange(0, row), np.arange(row + 1, M)])
    cidx = np.hstack([np.arange(0, col), np.arange(col + 1, N)])
    return mat[ridx[:, np.newaxis], cidx]


def adjoint(mat):
    """
    Adjoint of a matrix
    :param mat: 2D numpy array
    :return:
    """
    M, N = mat.shape
    adj = np.zeros_like(mat)
    for i in range(M):
        for j in range(N):
            adj[i, j] = (-1)**(i + j) * np.linalg.det(mat_minor(mat, i, j))
    return adj

def tuttes_determinant(X: np.ndarray) -> float:
    """
    Tutte's determinant of a graph (with root in 0)
    :param X: weight matrix
    :return: Tutte's determinant
    """
    A = np.copy(X)
    L = laplacian(A)
    return np.linalg.det(L[1:, 1:])


def tuttes_tot_weight(graph: nx.DiGraph|np.ndarray, root, weight='weight', contracted_arcs=None, deleted_arcs=None):
    """
    Ref: https://arxiv.org/pdf/1904.12221.pdf
    :param graph: directed graph with weights
    :param root: root node of every spanning tree
    :param weight: weight attribute name (if graph is nx.DiGraph)
    :param contracted_arcs: list of arcs to be contracted
    :param deleted_arcs: list of arcs to be deleted
    :return: the total weight sum of all spanning arborescence
     weights (that is the product of the tree arc weights)
    """
    contracted_arcs = contracted_arcs if contracted_arcs else []
    deleted_arcs = deleted_arcs if deleted_arcs else []
    if isinstance(graph, np.ndarray):
        A = graph.copy()
    elif isinstance(graph, nx.DiGraph):
        A = nx.to_numpy_array(graph, weight=weight)
    else:
        raise ValueError("graph must be either nx.DiGraph or np.ndarray")
    # contract arcs
    for arc in contracted_arcs:
        A[:, arc[1]] = 0
        A[arc[0], arc[1]] = 1
    # delete arcs
    for arc in deleted_arcs:
        A[arc[0], arc[1]] = 0
    L1 = laplacian(A)
    # L1 is also known as Kirchoff matrix (as in Colbourn 1996)
    L1r = mat_minor(L1, row=root, col=root)

    # det
    det = np.linalg.det(L1r)
    # check if matrix is almost singular (instable computations)
    # if (cond_num :=np.linalg.cond(L1r)) > 1 / np.finfo(L1.dtype).eps:
    #     # test if det is equal to the LU decomposition to check det accuracy
    P, L, U = lu(L1r)
    sign = np.linalg.det(P)
    det_lu = sign * np.prod(np.diag(U))
    #     if not np.isclose(det, det_lu):
    #         warnings.warn("LU decomposition failed")
    #     else:
    #         warnings.warn(f"LU decomposition {det_lu} is close to the determinant {det}")
    #     # raise ValueError(f"Matrix is ill-conditioned ({cond_num}), cannot compute determinant")
    #     warnings.warn(f"Matrix is ill-conditioned ({cond_num}), cannot compute determinant")
    return det_lu


def laplacian(A):
    """
    Kirchoff matrix of a graph (Laplacian)
    :param A: adjacency matrix
    :return: Kirchoff matrix
    """
    np.fill_diagonal(A, 0)
    Din = np.diag(np.sum(A, axis=0))
    return Din - A

def kirchhoff_tot_weight(graph, minor_row=0, minor_col=0):
    """
    Total sum of weights for undirected spanning trees in given (weighted) graph
    :param graph: nx.graph, can have weights
    :param minor_row: row to be removed for minor calculation
    :param minor_col: col to be removed for minor calculation
    :return:
    """
    norm_graph = normalize_graph_weights(graph)
    lap2 = nx.laplacian_matrix(norm_graph).toarray()
    lap_tot_weight = np.linalg.det(mat_minor(lap2, minor_row, minor_col))
    return lap_tot_weight


def cayleys_formula(n):
    assert n > 1
    return n**(n-2)

def nxtree_from_list(t: list) -> nx.DiGraph:
    """
    Convert a tree parent list to nx.DiGraph. Root has parent -1.
    :param t: list of length n_nodes
    :return: nx.DiGraph
    """
    tree = nx.DiGraph()
    for i, p in enumerate(t[1:]):
        tree.add_edge(p, i + 1)
    return tree

def prufer_to_list(prufer: list|tuple) -> tuple[int]:
    """
    Convert a Prufer sequence to a rooted tree.
    :param prufer: Prufer sequence
    :return: tuple of parents list
    """
    nx_tree = nx.from_prufer_sequence(prufer)
    rooted_tree = nx.dfs_tree(nx_tree, 0)
    parent = [-1] * nx_tree.number_of_nodes()
    for u, v in rooted_tree.edges:
        parent[v] = u

    return tuple(parent)


def brute_force_tot_weight(X: np.ndarray) -> float:
    """
    Brute force method to compute total weight of all trees in a graph
    :param X: adjacency matrix
    :return: total weight of all trees
    """
    n = X.shape[0]
    tot_weight = 0
    for prufer_tree in itertools.product(range(n), repeat=n-2):
        tree = prufer_to_list(prufer_tree)
        rooted_tree = tree
        assert rooted_tree[0] == -1
        weight = tree_weight(rooted_tree, X)
        tot_weight += weight
    return tot_weight

def crasher_matrix(n: int, num_components: int = 2, log_eps: float = -10) -> np.ndarray:
    """
    Generate a matrix with one crasher node for each component and a set of non-component nodes.
    The size of the components is (n - 1) // (num_components + 1) as the same number of nodes belong to the non-component subgraph.
    Assumes root is 0 and does not belong to any component nor the remaining nodes.
    - low connections from root
    - low connections between components
    - high connections within components
    - high connection from crasher to any other node in the component
    :param n: int, number of nodes
    :param num_components: int, number of blocks
    :param log_eps: float, weight for inter-block connections (in log scale)
    :return: np.ndarray, block matrix
    """
    root = 0

    op = StableOp(log_probs=True)
    # add variability to the weights
    weights = np.zeros((n, n)) + log_eps + np.random.rand(n, n) * abs(log_eps) / 10

    # divide nodes into num_components + 1 subgraphs
    k = num_components + 1
    nodes_per_component = (n - 1) // k
    components = [-1] + [i // nodes_per_component for i in range(n-1)]  # root is not in components
    # select the crasher for each component to be the first node in the component
    crashers = [i * nodes_per_component + 1 for i in range(num_components)]
    # fill remaining nodes with last component (excluding root)
    components = components + [k - 1] * (n - 1 - len(components))
    # assign lower weight to arcs between components
    for i in range(n):
        for j in range(n):
            if i == j or j == root:
                weights[i, j] = op.zero()
            # set 1 for arcs within components and arcs from crashers to any other node in the component or outside components
            elif components[i] == components[j] or (i in crashers and (components[i] == components[j] or components[j] == k - 1)):
                weights[i, j] = op.one()
    return weights

def crasher2_matrix(n: int, num_components: int = 2, log_eps: float = -10) -> np.ndarray:
    """
    Generate a matrix with two crasher nodes for each component and a set of non-component nodes.
    The size of the components is (n - 1) // (num_components + 1) as the same number of nodes belong to the non-component subgraph.
    Assumes root is 0 and does not belong to any component nor the remaining nodes.
    - low connections from root
    - low connections between components
    - high connections within components
    - high connection from crasher to any other node in the component
    :param n: int, number of nodes
    :param num_components: int, number of blocks (n // (num_components + 1) > 2 in order
        for 2 crashers to be selected)
    :param log_eps: float, weight for inter-block connections (in log scale)
    :return: np.ndarray, block matrix
    """
    root = 0

    op = StableOp(log_probs=True)
    # add variability to the weights
    weights = np.zeros((n, n)) + log_eps + np.random.rand(n, n) * abs(log_eps) / 10

    # divide nodes into num_components + 1 subgraphs
    k = num_components + 1
    nodes_per_component = (n - 1) // k
    components = [-1] + [i // nodes_per_component for i in range(n-1)]  # root is not in components
    # select the crasher for each component to be the first node in the component
    crashers = [i * nodes_per_component + 1 for i in range(num_components)] + [i * nodes_per_component + 2 for i in range(num_components)]
    # fill remaining nodes with last component (excluding root)
    components = components + [k - 1] * (n - 1 - len(components))
    # assign lower weight to arcs between components
    for i in range(n):
        for j in range(n):
            if i == j or j == root:
                weights[i, j] = op.zero()
            # set 1 for arcs within components and arcs from crashers to any other node in the component or outside components
            elif (components[i] != k-1 and components[i] == components[j]) or (i in crashers and (components[i] == components[j] or components[j] == k - 1)):
                weights[i, j] = op.one()
    return weights

def prufer_to_rooted_parent(prufer):
    nx_tree = nx.from_prufer_sequence(prufer)
    rooted_tree = nx.dfs_tree(nx_tree, 0)
    parent = [-1] * nx_tree.number_of_nodes()
    for u, v in rooted_tree.edges:
        parent[v] = u

    return tuple(parent), tree_to_newick(rooted_tree)
