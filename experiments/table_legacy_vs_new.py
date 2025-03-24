import logging

import networkx as nx
import numpy as np
from treesampling.algorithms import castaway_legacy
from treesampling.algorithms import castaway
from treesampling.algorithms.castaway import wx_dict_to_array


def main():
    logging.basicConfig(level=logging.INFO)
    K = 4
    # generate random graph
    weights = np.random.uniform(0, 1, size=(K, K))
    weights[:, 0] = 0
    weights[np.diag_indices(K)] = 0

    graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)

    # compute wx_table with castaway legacy
    wx_table_legacy = wx_dict_to_array(castaway_legacy._compute_wx_table(graph, x_set=list(range(1, K))), K)

    # compute wx_table with CastawayRST
    cast_obj = castaway.CastawayRST(weights, root=0, trick=False, debug=True)
    wx_table_new = cast_obj.wx.to_array()

    # compare
    print("Must be equal")
    print("legacy:")
    print(wx_table_legacy)
    print("new")
    print(wx_table_new)

if __name__ == "__main__":
    main()
