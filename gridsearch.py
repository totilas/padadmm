import numpy as np
from data import make_gaussian
from sklearn.linear_model import LassoCV
from conversion import eps_delta_to_zcdp
from dpadmm import MyADMM
from lasso import lasso_obj
from sklearn.model_selection import ParameterGrid
from dpproxsgd import Proxsgd
import tqdm

#######################
# Load the data
#######################
n = 1000
p = 64
X_train, X_test, y_train, y_test = make_gaussian(n, p, 8, rng=0, noise=1e-1)

#######################
# Set privacy & Lasso params
#######################
base = LassoCV() # ths sklearn baseline
base.fit(X_train, y_train)
print("alpha", base.alpha_)
print("coef", base.coef_)

alpha = base.alpha_
theo_best = lasso_obj(X_test, y_test, alpha, base.coef_)
print("best theo", theo_best)


delta = 1e-6
epss = np.logspace(-2, 1, num = 6)
myepss = [eps_delta_to_zcdp(eps, delta) for eps in epss]
save_every = 1
save_expes = True
L = 100
max_iter = 1000

eps = myepss[0]

def score_obj(estimator, X, y):
    return np.min(estimator.scores_)

def score_t(estimator, X, y):
    # print(estimator.scores_)
    return np.argmin(estimator.scores_)

def score_clip(estimator, X, y):
    return np.mean(estimator.gnorms_)

#######################
# Run ADMM 
#######################
if True:


    # grid search for best parameters PUT YOUR GRID SEARCH PARAMETERS HERE
    param_grid = {'alpha': [alpha],
                'lamb': np.geomspace(.05, 1, 4),
                'gamma': np.logspace(-1, 2, 6),
                'max_iter': [max_iter],
                'q':[int(n/10)],
                'n':[n],
                'p':[p],
                'theo_best':[theo_best],
                'L':[L],
                'noisy':[0],
                'save_scores':[False],
                'save_every':[1]}


    lamb = 1
    gamma = 1
    scale_noise = 2 * L**2 * max_iter/(eps*n)

    if True:
        grid_result = []
        for params in tqdm.tqdm(ParameterGrid(param_grid)):
            estimator = MyADMM(**params)
            estimator.fit(X_train, y_train)
            obj = np.min(estimator.scores_)
            t = np.argmin(estimator.scores_)
            clip = np.mean(estimator.gnorms_)

            grid_result.append({"lamb": params["lamb"], "gamma": params["gamma"], "obj": obj, "t": t, "clip": clip})

        import pandas as pd

        results = pd.DataFrame(sorted(grid_result, key=lambda x: x["obj"])).sort_values("obj")
        print(results)
        results.to_csv("admm.csv")



# #######################
# # Run SGD
# #######################

if True:

    # grid search for best parameters PUT YOUR GRID SEARCH PARAMETERS HERE
    scale_noise = 2 * L**2 * max_iter/(eps*n)
    eta0 = 1

    sgd_param_grid = {'eta':np.logspace(-1, 1, 5),
                    'q': [int(n/10)],
                    'n':[n],
                    'p':[p],
                    'max_iter':[max_iter],
                    'save_scores':[False],
                    'theo_best':[theo_best],
                    'L':[L],
                    'save_every':[10],
                    'alpha':[alpha]
                    }

    grid_result = []
    for params in tqdm.tqdm(ParameterGrid(sgd_param_grid)):
        estimator = Proxsgd(**params)
        estimator.fit(X_train, y_train)
        obj = np.min(estimator.scores_)
        t = np.argmin(estimator.scores_)
        clip = np.mean(estimator.gnorms_)

        grid_result.append({"eta": params["eta"], "obj": obj, "t": t, "clip": clip})

    import pandas as pd

    results = pd.DataFrame(sorted(grid_result, key=lambda x: x["obj"])).sort_values("obj")
    print(results)
    results.to_csv("sdg.csv")