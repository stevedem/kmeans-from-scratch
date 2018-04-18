import pandas as pd


def silhouette_coefficient(df, clusters, n_clusters):
    import numpy as np
    import pandas as pd

    unique_clusters = np.unique(clusters)
    clusters = np.array(clusters)
    
    data = df.copy(deep=True)
    if 'species' in data.columns:
        data.drop('species', axis=1, inplace=True)
    if 'clusters' in data.columns:
        data.drop('clusters', axis=1, inplace=True)

    # Calculate the distance between every unique pair of points and store them for fast look-up
    distances = dict()
    for xi in range(0, len(data)):
        for xii in range(0, len(data)):
            if xi != xii:
                pair = tuple(sorted((xi, xii)))
                if pair not in distances.keys():
                    distances[pair] = np.linalg.norm(data.iloc[xi] - data.iloc[xii])
            else:
                distances[(xi, xii)] = 0.0

    # Calculate the silhouette coefficient
    sil_coefs = list()
    for xi, cj in zip(range(0, len(data)), clusters):

        # Calculate ai
        points_in_same_cluster = np.where(clusters == cj)[0]

        ai_distances = list()
        for xk in points_in_same_cluster:
            if xi != xk:
                pair = tuple(sorted((xi, xk)))
                ai_distances.append(distances[pair])

        ai_dist = np.array(ai_distances)
        if len(ai_dist) == 0:
            ai = 0
        else:
            ai = np.nanmean(ai_dist)

        # Calculate bi
        other_cluster_distances = list()
        for cluster in range(0, n_clusters):
            if cj != cluster and cluster in unique_clusters:
                points_in_dif_cluster = np.where(clusters == cluster)[0]

                bi_distances = list()
                for xj in points_in_dif_cluster:
                    if xi != xj:
                        pair = tuple(sorted((xi, xj)))
                        bi_distances.append(distances[pair])
                bi_dist = np.array(bi_distances)

                if len(bi_dist) == 0:
                    other_cluster_distances.append(0)
                else:
                    bi_j = np.nanmean(bi_dist)
                    other_cluster_distances.append(bi_j)

        # If points were only assigned to one cluster, bi = 0
        if len(other_cluster_distances) == 0:
            bi = 0
        else:
            bi = min(other_cluster_distances)

        # Calculate silhouette coefficient
        # If divide by 0, the sil coef is -1
        si = (bi - ai) / max([ai, bi])
        if pd.isnull(si):
            sil_coefs.append(-1)
        else:
            sil_coefs.append(si)

    # Calculate average silhouette coefficient
    nd_sil_coefs = np.array(sil_coefs)
    avg_sil_coef = nd_sil_coefs.mean()

    return avg_sil_coef
