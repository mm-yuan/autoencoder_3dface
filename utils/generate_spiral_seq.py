from sklearn.neighbors import KDTree
import numpy as np


def get_adj(A):
    adj = []
    for x in A:
        adj_x = []
        dx = x.todense()
        for i in range(x.shape[0]):
            adj_x.append(dx[i].nonzero()[1])
        adj.append(adj_x)
    
    return adj

def _next_ring(vert, adj, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        after_last_ring = False
        for vh2 in adj[vh1][0]:
            if after_last_ring:
                if is_new_vertex(vh2):
                    res.append(vh2)
            if vh2 in last_ring:
                after_last_ring = True
        for vh2 in adj[vh1][0]:
            if vh2 in last_ring:
                break
            if is_new_vertex(vh2):
                res.append(vh2)
    return res


def extract_spirals(vert, A, spiral_length, dilation=1):
    # output: spirals.size() = [N, spiral_length]
    spirals = []
    adj = get_adj(A)
    for vh0 in range(vert.shape[0]):
        reference_one_ring = []
        for vh1 in adj[vh0][0]:
            reference_one_ring.append(vh1)
        spiral = [vh0]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(vert, adj, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < spiral_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(vert, adj, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(vert, metric='euclidean')
            spiral = kdt.query(np.expand_dims(vert[spiral[0]],
                                              axis=0),
                               k=spiral_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:spiral_length * dilation][::dilation])
    # if dilation > 1:
    #     dilated_spirals = []
    #     for i in range(len(spirals)):
    #         dilated_spirals.append(spirals[i][::dilation])
    #     spirals = dilated_spirals
    return spirals
