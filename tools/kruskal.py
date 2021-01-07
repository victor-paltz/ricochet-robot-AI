
import numpy as np
from tools.UnionFind import UnionFind


def kruskal_groups(data, dist, nb_groups, d_max=np.float("inf")):

    n = len(data)
    current_nb_groups = n

    if current_nb_groups <= nb_groups:
        return [[x] for x in data] + [[] for _ in range(current_nb_groups-nb_groups)]

    d_sorted = sorted((dist(data[i], data[j]), i, j)
                      for i in range(n) for j in range(i))

    uf = UnionFind(list(range(n)))

    for d, i, j in d_sorted:
        if d > d_max:
            break
        if uf.union(i, j):
            current_nb_groups -= 1
            if current_nb_groups == nb_groups:
                break

    final_groups = uf.groups()

    out = []
    for gp in final_groups:
        out.append([data[i] for i in gp])

    return [x[1] for x in sorted((-len(gp), gp) for gp in out)]
