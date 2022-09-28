# Code is modified from:
# https://github.com/uber-research/poet/blob/master/poet_distributed/
# under Apache license 



import numpy as np

def euclidean_distance(x, y):

    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)


def compute_novelty_vs_archive(archived_optimizers, optimizers, niche, k, low, high):
    distances = []
    niche.update_pata_ec(archived_optimizers, optimizers, low, high)
    for point in archived_optimizers.values():
        distances.append(euclidean_distance(point.pata_ec, niche.pata_ec))

    for point in optimizers.values():
        distances.append(euclidean_distance(point.pata_ec, niche.pata_ec))


    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()
