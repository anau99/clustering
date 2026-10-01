from sklearn.mixture import GaussianMixture


def gaussian_mixture(X, k):
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42
    )
    return model.fit_predict(X)