import random
import itertools
from operator import mul
from functools import reduce
import numpy as np
import networkx as nx


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


def random_uniform_graph(n_nodes, log_probs=False) -> nx.DiGraph:
    graph = nx.complete_graph(n_nodes, create_using=nx.DiGraph)
    for u, v in graph.edges():
        if u == v:
            w = 0
        else:
            w = random.random()
        graph.edges()[u, v]['weight'] = w if not log_probs else np.log(w)
    return graph


def random_tree_skewed_graph(n_nodes, skewness, root: int | None = None) -> tuple[nx.DiGraph, nx.DiGraph]:
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
    graph = reset_adj_matrix(graph, adj_matrix)
    return graph, tree


def normalize_graph_weights(graph, log_probs=False, rowwise=False) -> nx.DiGraph:
    adj_mat = nx.to_numpy_array(graph)
    axis = 1 if rowwise else 0
    if not log_probs:
        adj_mat = adj_mat / adj_mat.sum(axis=axis, keepdims=True)
    else:
        adj_mat = adj_mat - np.logaddexp.reduce(adj_mat, axis=axis, keepdims=True)
    norm_graph = reset_adj_matrix(graph, adj_mat)
    return norm_graph


def reset_adj_matrix(graph: nx.DiGraph, matrix: np.ndarray) -> nx.DiGraph:
    weights_dict = [(i, j, matrix[i, j]) for i, j in itertools.product(range(matrix.shape[0]), repeat=2)]
    new_graph = graph.copy()
    new_graph.add_weighted_edges_from(weights_dict)
    return new_graph


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


def tuttes_tot_weight(graph: nx.DiGraph, root, weight='weight', contracted_edge=None):
    """
    Ref: https://arxiv.org/pdf/1904.12221.pdf
    :param graph: directed graph with weights
    :param root: root node of every spanning tree
    :return: the total weight sum of all spanning arborescence
     weights (that is the product of the tree arc weights)
    """

    A = nx.to_numpy_array(graph, weight=weight)
    if contracted_edge is not None:
        A[:, contracted_edge[1]] = 0
        A[contracted_edge[0], contracted_edge[1]] = 1
    np.fill_diagonal(A, 0)
    Din = np.diag(np.sum(A, axis=0))
    L1 = Din - A
    # L1 is also known as Kirchoff matrix (as in Colbourn 1996)
    L1r = mat_minor(L1, row=root, col=root)

    return np.linalg.det(L1r)


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
