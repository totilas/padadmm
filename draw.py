import numpy as np
import matplotlib.pyplot as plt



admm = np.load("admm.npy")
sgd = np.load("prox.npy")
print(admm)

epss = np.geomspace(.01, 20, num = 7)

plt.errorbar(epss, np.mean(admm, axis=1),  2*np.std(admm, axis=1), label=r"Private ADMM", color="b", capthick=1, capsize = 4)
plt.errorbar(epss, np.mean(sgd, axis=1),  2*np.std(sgd, axis=1), label=r"Private SGD", color="g", capthick=1, capsize = 4)
epss = np.geomspace(.01, 20, num = 7)

plt.legend()
plt.ylim(1e-3,1e4)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Epsilon')
plt.ylabel('Objective function')
plt.savefig('prox.pdf', bbox_inches='tight', pad_inches=0)

plt.show()