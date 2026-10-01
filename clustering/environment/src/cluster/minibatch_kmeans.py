from sklearn.cluster import MiniBatchKMeans


def minibatch_kmeans(X, k):
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1024
    )
    return model.fit_predict(X)