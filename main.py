import numpy as np
from data import make_gaussian
from sklearn.linear_model import LassoCV
from conversion import eps_delta_to_zcdp
from dpadmm import MyADMM
from lasso import lasso_obj
import pandas as pd
from dpproxsgd import Proxsgd

from tqdm import trange


#######################
# Load the data
#######################
n = 1000
p = 64
X_train, X_test, y_train, y_test = make_gaussian(n, p, 8, rng=18,noise=1e-1)

#######################
# Set privacy & Lasso params
#######################
base = LassoCV()
base.fit(X_train, y_train)
print("alpha", base.alpha_)
print("coef", base.coef_)

alpha = base.alpha_
theo_best = lasso_obj(X_test, y_test, alpha, base.coef_)
print("best theo", theo_best)


delta = 1e-6
epss = np.geomspace(.01, 5, num = 7)
myepss = [eps_delta_to_zcdp(eps, delta) for eps in epss]
save_every = 1
save_expes = True
max_iter = 1000


n_trials = 10

#######################
# Run ADMM 
#######################
inter_admm = np.zeros((len(epss), n_trials))
inter_sgd = np.zeros((len(epss), n_trials))

for i,eps in enumerate(myepss):



    print(epss[i])

    L_admm = 1.6
    scale_noise_admm = 2 * L_admm**2 * max_iter/(eps*n*n)

    L_sgd = 13
    scale_noise_sgd = 2 * L_sgd**2 * max_iter/(eps*n*n)

    for j in trange(n_trials):
        

        
        admm = MyADMM(**{
            'alpha': alpha,
            'lamb': 0.464159, # values found in the main_gridsearch
            'gamma': 1.584893,
            'max_iter': max_iter,
            'q':int(n/10),
            'n':n,
            'p':p,
            'theo_best':theo_best,
            'L': L_admm,
            'noisy': 1,
            "scale_noise": scale_noise_admm,
            'save_scores': True, # Save score on test dataset
            'save_every':1,
            'rng':j
        })

        
        msdg = Proxsgd(**{
            'alpha': alpha,
            'max_iter': max_iter,
            'q':int(n/10),
            'eta':0.31,
            'n':n,
            'p':p,
            'theo_best':theo_best,
            'L': L_sgd,
            'noisy': 1,
            "scale_noise": scale_noise_sgd,
            'save_scores': True, # Save score on test dataset
            'save_every':1,
            'rng':j
        })


        admm.fit(X_train, y_train, X_test, y_test)
        msdg.fit(X_train, y_train, X_test, y_test)

        #print(pd.DataFrame(admm.scores_))
        print(pd.DataFrame(msdg.scores_))

        inter_admm[i][j] = np.min(admm.scores_)
        inter_sgd[i][j] = np.min(msdg.scores_)


np.save("admm.npy", inter_admm)
np.save("prox.npy", inter_sgd)




