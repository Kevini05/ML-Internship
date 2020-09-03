import utils
import numpy as np
import matplotlib.pyplot as plt

E0 = np.linspace(0.0001,0.9999,901)
res = utils.cut_off_epsilon(E0,0.001,0.25)
plt.semilogy(E0,res)
plt.show()