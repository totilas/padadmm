import numpy as np
from lasso import lasso_obj, soft_thresholding, min_quadra, myclip
from pathlib import Path
from numpy import linalg as LA
from sklearn.base import BaseEstimator

from tqdm import trange


            

class MyADMM(BaseEstimator):
    def __init__(self,
    alpha=1,
    max_iter=1,
    q=1,
    lamb=1,
    gamma=1,
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
        self.lamb = lamb
        self.gamma = gamma
        self.scale_noise = scale_noise

        self.noisy = noisy
        self.rng = np.random.default_rng(rng)
        self.theo_best = theo_best
        self.L=L
        self.n = n
        self.p = p
    def fit(self, X, y, Xtest=None, ytest=None):
        self.z_ = np.ones(self.p)*10*self.alpha
        self.u_ = np.zeros((self.n, self.p))
        self.x_ = np.zeros((self.n, self.p))

        scores = []
        gnorms = []
        if Xtest is not None:
            scores_test = []


        for it in trange(self.max_iter, leave=False):

            if self.q == 0: # decentralized
                i = self.rng.integers(len(X))
                self.x_[i] = min_quadra(X[i], y[i], self.gamma, 2*self.z_ - self.u_[i])
                temp = self.u_[i].copy()
                gnorms.append(LA.norm(self.x_[i] - self.z_))
                du =  myclip(self.x_[i] - self.z_, self.L)
                self.u_[i] = self.u_[i] + 2 * self.lamb * du
                self.u_[i] = self.u_[i] + self.noisy * self.lamb * self.scale_noise * self.rng.normal(size=self.u_[i].shape)
                self.z_ = self.z_ + (self.u_[i]-temp)/len(X)
                self.z_ = np.array([soft_thresholding(zi, 2*self.alpha*self.gamma/self.n) for zi in self.z_])

            elif self.q == 1: # centralized (would be fairer to have u,x of size p in this case)
                self.z_ = np.mean(self.u_, axis=0)
                # print(f"thresholding parameter : {2*self.alpha*self.gamma} and alpha {self.alpha}")

                self.z_ = np.array([soft_thresholding(zi, 2*self.alpha*self.gamma/self.n) for zi in self.z_])
                for i in range(len(X)):
                    self.x_[i] = min_quadra(X[i], y[i], 2/self.gamma, 2*self.z_ - self.u_[i])
                    gnorms.append(LA.norm(self.x_[i] - self.z_))
                    du = myclip(self.x_[i] - self.z_, self.L)
                    self.u_[i] = self.u_[i] + 2 * self.lamb * du 
                    self.u_[i] = self.u_[i] + self.noisy * self.lamb * self.scale_noise * self.rng.normal(size=self.u_[i].shape)
            elif self.q:
                ids = self.rng.integers(len(X), size=(self.q))
                delta = 0
                for i in ids:
                    self.x_[i] = min_quadra(X[i], y[i],1/self.gamma, 2*self.z_ - self.u_[i])
                    temp = self.u_[i].copy()
                    gnorms.append(LA.norm(self.x_[i] - self.z_))
                    du = myclip(self.x_[i] - self.z_, self.L)
                    self.u_[i] = self.u_[i] + 2 * self.lamb * du
                    self.u_[i] = self.u_[i] + self.noisy * self.lamb * self.scale_noise * self.rng.normal(size=self.u_[i].shape)
                    delta += self.u_[i]-temp
                self.z_ = self.z_ + delta/len(X)
                self.z_ = np.array([soft_thresholding(zi, 2*self.alpha*self.gamma/self.n) for zi in self.z_])


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
            interesting_params = ["lamb", "alpha", "gamma", "random_state"]
            params = self.get_params()

            print([params[k] for k in interesting_params])

            fn = Path("./models")/Path(self.prefix)/ Path("-".join(f"{k}={params[k]}" for k in interesting_params))
            fn.with_suffix(".npy")

        np.save(fn, self.scores_)


