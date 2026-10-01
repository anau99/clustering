from sklearn.cluster import SpectralClustering


def spectral(X, k):
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        random_state=42
    )
    return model.fit_predict(X)