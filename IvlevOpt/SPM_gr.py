import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = 'Sparse_test2.csv'
data = pd.read_csv(filename, sep=";", header=None)
m_size = int(np.sqrt(data.size))
matrix = np.array(data, dtype=complex)
plt.spy(matrix, aspect="equal")
plt.title("Size: "+str(m_size)+'x'+ str(m_size)+"; Non zero: " + str(np.count_nonzero(matrix)))
plt.show()