from treesampling.utils.graphs import enumerate_rooted_trees, cayleys_formula


def test_enumerate_rooted_trees():
    n_nodes = 5
    trees = enumerate_rooted_trees(n_nodes)
    tot_trees = cayleys_formula(n_nodes)
    assert len(trees) == tot_trees
