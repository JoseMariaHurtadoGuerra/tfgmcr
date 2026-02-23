# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
print(sys.executable)


E = np.linspace(0.1, 5, 200)   # GeV por ejemplo
sigma = 1e-38 * E              # toy model

plt.plot(E, sigma)
plt.xlabel("Eν [GeV]")
plt.ylabel("σ [cm²]")
plt.grid(True)
plt.show()

