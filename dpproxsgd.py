import numpy as np
from lasso import lasso_obj, soft_thresholding, min_quadra, myclip, grad
from pathlib import Path
from numpy import linalg as LA
from sklearn.base import BaseEstimator

from tqdm import trange


            

class Proxsgd(BaseEstimator):
    def __init__(self,
    alpha=1,
    max_iter=1,
    q=1,
    eta=1,
    scale_noise=0,
    n=1,
    p=2,
    theo_best=0,
    noisy=1,
    save_scores = False,
    save_every = 1,
    rng=0,
    L = 1) -> None:
        self.alpha =alpha
        self.max_iter = max_iter
        self.save_scores = save_scores
        self.save_every = save_every
        self.q = q
        self.eta = eta
        self.scale_noise = scale_noise

        self.noisy = noisy
        self.rng = np.random.default_rng(rng)
        self.theo_best = theo_best
        self.L=L
        self.n = n
        self.p = p
    def fit(self, X, y, Xtest=None, ytest=None):
        self.z_ = np.ones(self.p)*10*self.alpha

        scores = []
        gnorms = []
        if Xtest is not None:
            scores_test = []

        # solving thanks to SGDRegressor as baseline, and using batch_size data point per iterations

        for it in trange(self.max_iter, leave=False):

            if self.q == 0: # decentralized
                i = self.rng.integers(len(X))
                g = grad(X[i], y[i], self.z_)
                gnorms.append(LA.norm(g))
                g =  myclip(g, self.L)
                g = g + self.noisy * self.scale_noise * self.rng.normal(size=self.z_.shape)
                self.z_ = self.z_ - (self.eta * g)/self.n
                self.z_ = np.array([soft_thresholding(zi, self.alpha*self.eta/self.n) for zi in self.z_])

            elif self.q == 1: # centralized (would be fairer to have u,x of size p in this case)
                g = grad(X, y, self.z_)
                gnorms.append(LA.norm(g))
                g =  myclip(g, self.L)
                g = g + self.noisy * self.scale_noise * self.rng.normal(size=self.z_.shape)
                self.z_ = self.z_ - (self.eta * g)/self.n
                self.z_ = np.array([soft_thresholding(zi, self.alpha*self.eta/self.n) for zi in self.z_])

            elif self.q:
                ids = self.rng.integers(len(X), size=(self.q))
                g = grad(X[ids], y[ids], self.z_)
                gnorms.append(LA.norm(g))
                g =  myclip(g, self.L)
                g = g + self.noisy * self.scale_noise * self.rng.normal(size=self.z_.shape)
                self.z_ = self.z_ - (self.eta * g)/self.n
                self.z_ = np.array([soft_thresholding(zi, self.alpha*self.eta/self.n) for zi in self.z_])


            scores.append(lasso_obj(X, y, self.alpha, self.z_))            
            # if len(scores)>2 and (scores[-1]-self.theo_best)/(scores[-2]-self.theo_best) > 1.1:
            #     self.scores_ = np.array(scores)
            #     self.gnorms_ = np.array(gnorms)
            #     return

            if self.save_scores and it%self.save_every == 0:
                # keep the score every regularly
                # scores.append(reg.score(X, y))
                scores_test.append(lasso_obj(Xtest, ytest, self.alpha, self.z_))
                # if len(scores_test)>2 and (scores_test[-1]-self.theo_best)/(scores_test[-2]-self.theo_best) > 1.1:
                #     self.scores_ = np.array(scores_test)
                #     return
        
        if self.save_scores:
            self.scores_ = np.array(scores_test)
        else:
            self.scores_ = np.array(scores)
        self.gnorms_ = np.array(gnorms)
        
    def save_score(self, fn=None):

        if fn is None:
            interesting_params = ["eta", "random_state"]
            params = self.get_params()

            print([params[k] for k in interesting_params])

            fn = Path("./models")/Path(self.prefix)/ Path("-".join(f"{k}={params[k]}" for k in interesting_params))
            fn.with_suffix(".npy")

        np.save(fn, self.scores_)


