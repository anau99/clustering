from sklearn.cluster import BisectingKMeans


def bisecting_kmeans(X, k):
    model = BisectingKMeans(
        n_clusters=k,
        random_state=42
    )
    return model.fit_predict(X)