from sklearn.cluster import Birch


def birch(X, k):
    model = Birch(
        n_clusters=k
    )
    return model.fit_predict(X)