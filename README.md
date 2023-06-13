# padadmm

Code of the paper From Noisy Fixed-Point Iterations to Private ADMM for Centralized and Federated Learninghttps://arxiv.org/abs/2302.12559

Run `main.py` to redo the experiment of the figure 1 of the paper. Then run `draw.py` to generate the matplotlib figure.

The code is structured as follows:
- `data.py` generate the synthetic data
- `conversion.py` implements the conversion from eps delta differential privacy and Renyi DP
- `dpadmm.py` implements the DP-ADMM of the paper. Note that the algorithm is implement for the centralized, federated and decentralized version
- `dpproxsgd.py` implements the baseline with DP-Prox SGD
- `gridsearch.py` for tuning the parameters with the grid search
- `lasso.py` some utils function specific the Lasso objective


To cite the paper:
```
@article{DBLP:journals/corr/abs-2302-12559,
  author       = {Edwige Cyffers and
                  Aur{\'{e}}lien Bellet and
                  Debabrota Basu},
  title        = {From Noisy Fixed-Point Iterations to Private {ADMM} for Centralized
                  and Federated Learning},
  journal      = {CoRR},
  volume       = {abs/2302.12559},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2302.12559},
  doi          = {10.48550/arXiv.2302.12559},
  eprinttype    = {arXiv},
  eprint       = {2302.12559},
  timestamp    = {Tue, 28 Feb 2023 14:02:05 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2302-12559.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```