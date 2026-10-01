from sklearn.cluster import KMeans


def kmeans(X, k):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",
        random_state=42
    )
    return model.fit_predict(X)