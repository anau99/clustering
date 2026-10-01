from sklearn.cluster import AgglomerativeClustering


def agglomerative(X, k):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    return model.fit_predict(X)