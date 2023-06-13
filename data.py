import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def make_gaussian(n, p, support, rng=0, noise=0):
    """Generate sparse synthetic data
    Parameters:
    ---
    n: int, number of samples
    p: int, dimension size
    support: int < p number of dimension with non-zero impact
    rng: int, the seed
    noise: magnitude of the Gaussian noise added to the labels
    Returns
    ---
    X_train: array (.8 n,p)
    X_test: array (.2 n, p)
    y_train: array .8n
    y_test: array .2n
    """
    # Gaussian of size [n * p]
    rng = np.random.default_rng(rng)
    X = rng.normal(size=(n, p))
    preprocess = StandardScaler()
    X = preprocess.fit_transform(X)

    # vector of size [p] with [support] non-zeros coefficients
    w = np.zeros(p)
    w[rng.choice(np.arange(p), replace=False, size=support)] = rng.uniform(size=support) 
    y = X @ w + noise*rng.normal(size=n)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=42)
    return X_train, X_test, y_train, y_test
